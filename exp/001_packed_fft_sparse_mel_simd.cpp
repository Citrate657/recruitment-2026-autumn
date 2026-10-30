// Experiment 001
// Baseline: src/melspec.cpp at repository HEAD.
// Main changes from the baseline:
//   - precomputed FFT plan and reusable per-thread workspaces;
//   - packed half-length real FFT;
//   - sparse Mel filterbank and fused frame tiles;
//   - SIMD/FMA-friendly inner loops and OpenMP frame parallelism.

#include "melspec.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

#ifndef MELSPEC_FRAME_TILE
#define MELSPEC_FRAME_TILE 256
#endif

constexpr int kFrameTile = MELSPEC_FRAME_TILE;

void validate_fft_size(int n_fft) {
    if (n_fft < 2 || (n_fft & (n_fft - 1)) != 0)
        throw std::runtime_error("n_fft must be a power of two");
}

struct FftPlan {
    explicit FftPlan(int size) : n(static_cast<std::size_t>(size)) {
        validate_fft_size(size);
        bit_reverse.resize(n);

        int bits = 0;
        for (std::size_t value = n; value > 1; value >>= 1) ++bits;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t value = i;
            std::size_t reversed = 0;
            for (int bit = 0; bit < bits; ++bit) {
                reversed = (reversed << 1) | (value & 1U);
                value >>= 1;
            }
            bit_reverse[i] = reversed;
        }

        // Keep every stage's twiddles contiguous so the inner butterfly loop
        // can stream through them instead of following a strided large table.
        for (std::size_t length = 2; length <= n; length <<= 1) {
            stage_offsets.push_back(twiddle_real.size());
            const std::size_t half = length / 2;
            for (std::size_t j = 0; j < half; ++j) {
                const double angle = -2.0 * kPi * static_cast<double>(j) /
                                     static_cast<double>(length);
                twiddle_real.push_back(std::cos(angle));
                twiddle_imag.push_back(std::sin(angle));
            }
        }
    }

    std::size_t n;
    std::vector<std::size_t> bit_reverse;
    std::vector<std::size_t> stage_offsets;
    std::vector<double> twiddle_real;
    std::vector<double> twiddle_imag;
};

// Cooley-Tukey DIT FFT. The caller has already placed the input in
// bit-reversed order, so no separate permutation pass is needed.
void fft_from_bit_reversed(double* __restrict real, double* __restrict imag,
                           const FftPlan& plan) {
    int stage = 0;
    for (std::size_t length = 2; length <= plan.n; length <<= 1, ++stage) {
        const std::size_t half = length / 2;
        const std::size_t offset = plan.stage_offsets[stage];
        const double* twiddle_real = plan.twiddle_real.data() + offset;
        const double* twiddle_imag = plan.twiddle_imag.data() + offset;

        for (std::size_t block = 0; block < plan.n; block += length) {
#ifdef _OPENMP
#pragma omp simd
#endif
            for (std::size_t j = 0; j < half; ++j) {
                const std::size_t lower = block + j;
                const std::size_t upper = lower + half;
                const double lower_real = real[lower];
                const double lower_imag = imag[lower];
                const double upper_real = real[upper];
                const double upper_imag = imag[upper];

                const double product_real =
                    upper_real * twiddle_real[j] - upper_imag * twiddle_imag[j];
                const double product_imag =
                    upper_real * twiddle_imag[j] + upper_imag * twiddle_real[j];

                real[lower] = lower_real + product_real;
                imag[lower] = lower_imag + product_imag;
                real[upper] = lower_real - product_real;
                imag[upper] = lower_imag - product_imag;
            }
        }
    }
}

struct RealFftWorkspace {
    explicit RealFftWorkspace(int n_fft, int power_bins)
        : real(static_cast<std::size_t>(n_fft / 2)),
          imag(static_cast<std::size_t>(n_fft / 2)),
          power(static_cast<std::size_t>(power_bins)) {
    }

    std::vector<double> real;
    std::vector<double> imag;
    std::vector<double> power;
};

struct SparseMelFilter {
    int begin;
    int count;
    int value_offset;
};

struct SparseMelFilterbank {
    std::vector<SparseMelFilter> filters;
    std::vector<double> values;
    int first_bin = 0;
    int last_bin = -1;
};

SparseMelFilterbank build_sparse_mel_filterbank(int sr, int n_fft, int n_mels,
                                                double f_min, double f_max) {
    const int n_bins = n_fft / 2 + 1;
    const double mel_min = hz_to_mel(f_min);
    const double mel_max = hz_to_mel(f_max);

    std::vector<double> hz_points(static_cast<std::size_t>(n_mels + 2));
    for (int i = 0; i <= n_mels + 1; ++i) {
        hz_points[i] =
            mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1));
    }

    SparseMelFilterbank result;
    result.filters.reserve(static_cast<std::size_t>(n_mels));
    result.first_bin = n_bins;

    for (int m = 0; m < n_mels; ++m) {
        const double f_l = hz_points[m];
        const double f_c = hz_points[m + 1];
        const double f_r = hz_points[m + 2];
        const double enorm = 2.0 / (f_r - f_l);

        int begin = n_bins;
        int end = 0;
        const int value_offset = static_cast<int>(result.values.size());
        for (int k = 0; k < n_bins; ++k) {
            const double frequency =
                static_cast<double>(k) * sr / static_cast<double>(n_fft);
            const double rising = (frequency - f_l) / (f_c - f_l);
            const double falling = (f_r - frequency) / (f_r - f_c);
            const double value = std::max(0.0, std::min(rising, falling));
            if (value > 0.0) {
                begin = std::min(begin, k);
                end = k + 1;
                result.values.push_back(value * enorm);
            }
        }

        const int count = end - begin;
        result.filters.push_back({begin, count, value_offset});
        if (count > 0) {
            result.first_bin = std::min(result.first_bin, begin);
            result.last_bin = std::max(result.last_bin, end - 1);
        }
    }
    return result;
}

void fill_bit_reversed_packed_frame(const std::vector<double>& signal,
                                    const std::vector<double>& window,
                                    int start, const FftPlan& plan,
                                    RealFftWorkspace& workspace) {
    for (std::size_t i = 0; i < plan.n; ++i) {
        const std::size_t destination = plan.bit_reverse[i];
        // Multiplying by 0.5 here removes the corresponding factor from the
        // real-FFT recombination step.
        workspace.real[destination] =
            0.5 * signal[static_cast<std::size_t>(start) + 2 * i] *
            window[2 * i];
        workspace.imag[destination] =
            0.5 * signal[static_cast<std::size_t>(start) + 2 * i + 1] *
            window[2 * i + 1];
    }
}

void real_fft_power(const std::vector<double>& recombination_real,
                    const std::vector<double>& recombination_imag,
                    RealFftWorkspace& workspace) {
    const std::size_t packed_size = workspace.real.size();
    const std::size_t power_bins = workspace.power.size();

    for (std::size_t k = 0; k < power_bins; ++k) {
        const std::size_t positive_index = (k == packed_size) ? 0 : k;
        const std::size_t negative_index = (k == 0) ? 0 : packed_size - k;

        const double positive_real = workspace.real[positive_index];
        const double positive_imag = workspace.imag[positive_index];
        const double conjugate_real = workspace.real[negative_index];
        const double conjugate_imag = -workspace.imag[negative_index];

        const double difference_real = positive_real - conjugate_real;
        const double difference_imag = positive_imag - conjugate_imag;
        const double twiddle_real = recombination_real[k];
        const double twiddle_imag = recombination_imag[k];

        const double product_real =
            twiddle_real * difference_real - twiddle_imag * difference_imag;
        const double product_imag =
            twiddle_real * difference_imag + twiddle_imag * difference_real;

        // X[k] = E[k] + W_N^k O[k]. With the input packed at half scale,
        // the usual 1/2 factors cancel and the expression becomes direct.
        const double out_real = positive_real + conjugate_real + product_imag;
        const double out_imag = positive_imag + conjugate_imag - product_real;
        workspace.power[k] = out_real * out_real + out_imag * out_imag;
    }
}

int available_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// HTK mel scale
// ─────────────────────────────────────────────────────────────────────────────

double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public real-FFT helper
// ─────────────────────────────────────────────────────────────────────────────

void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    validate_fft_size(n);
    const FftPlan plan(n);

    std::vector<double> real(static_cast<std::size_t>(n));
    std::vector<double> imag(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < plan.n; ++i)
        real[i] = frame[plan.bit_reverse[i]];

    fft_from_bit_reversed(real.data(), imag.data(), plan);

    out.resize(static_cast<std::size_t>(n / 2 + 1));
    for (int k = 0; k <= n / 2; ++k)
        out[k] = {real[static_cast<std::size_t>(k)],
                  imag[static_cast<std::size_t>(k)]};
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel filterbank
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<double>> build_mel_filterbank(int sr, int n_fft,
                                                      int n_mels, double f_min,
                                                      double f_max) {
    const int n_bins = n_fft / 2 + 1;
    const double mel_min = hz_to_mel(f_min);
    const double mel_max = hz_to_mel(f_max);

    std::vector<double> hz_points(static_cast<std::size_t>(n_mels + 2));
    for (int i = 0; i <= n_mels + 1; ++i) {
        hz_points[i] =
            mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1));
    }

    std::vector<std::vector<double>> weights(
        static_cast<std::size_t>(n_mels),
        std::vector<double>(static_cast<std::size_t>(n_bins), 0.0));

    for (int m = 0; m < n_mels; ++m) {
        const double f_l = hz_points[m];
        const double f_c = hz_points[m + 1];
        const double f_r = hz_points[m + 2];
        const double enorm = 2.0 / (f_r - f_l);

        for (int k = 0; k < n_bins; ++k) {
            const double frequency =
                static_cast<double>(k) * sr / static_cast<double>(n_fft);
            const double rising = (frequency - f_l) / (f_c - f_l);
            const double falling = (f_r - frequency) / (f_r - f_c);
            weights[m][k] = std::max(0.0, std::min(rising, falling)) * enorm;
        }
    }
    return weights;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel spectrogram computation
// ─────────────────────────────────────────────────────────────────────────────

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    validate_fft_size(n_fft);
    if (hop_length <= 0)
        throw std::runtime_error("hop_length must be positive");
    if (n_mels <= 0 || sr <= 0 || f_max <= f_min)
        throw std::runtime_error("invalid mel-spectrogram parameters");

    const int n_samples = static_cast<int>(y.size());
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    const int packed_size = n_fft / 2;
    const FftPlan fft_plan(packed_size);
    const SparseMelFilterbank mel_fb =
        build_sparse_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    if (mel_fb.last_bin < 0)
        throw std::runtime_error("mel filterbank has no nonzero bins");

    // For the packed real FFT, X[k] needs exp(-2*pi*i*k/n_fft). Store only
    // the bins that can contribute to a nonzero Mel filter.
    const int power_bins = mel_fb.last_bin + 1;
    std::vector<double> recombination_real(
        static_cast<std::size_t>(power_bins));
    std::vector<double> recombination_imag(
        static_cast<std::size_t>(power_bins));
    for (int k = 0; k < power_bins; ++k) {
        const double angle =
            -2.0 * kPi * static_cast<double>(k) / static_cast<double>(n_fft);
        recombination_real[k] = std::cos(angle);
        recombination_imag[k] = std::sin(angle);
    }

    std::vector<double> window(static_cast<std::size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        window[i] =
            0.5 * (1.0 - std::cos(2.0 * kPi * i / static_cast<double>(n_fft)));
    }

    const int thread_capacity = available_threads();
    std::vector<RealFftWorkspace> workspaces;
    workspaces.reserve(static_cast<std::size_t>(thread_capacity));
    for (int i = 0; i < thread_capacity; ++i)
        workspaces.emplace_back(n_fft, power_bins);

    output.assign(static_cast<std::size_t>(n_mels) * n_frames, 0.0);
    std::vector<double> mel_tile(static_cast<std::size_t>(n_mels) *
                                 std::min(kFrameTile, n_frames));

    for (int frame_begin = 0; frame_begin < n_frames;
         frame_begin += kFrameTile) {
        const int tile_frames = std::min(kFrameTile, n_frames - frame_begin);
        const int workers = std::min(thread_capacity, tile_frames);

#ifdef _OPENMP
#pragma omp parallel num_threads(workers)
#endif
        {
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            RealFftWorkspace& workspace =
                workspaces[static_cast<std::size_t>(thread_id)];

            // Give each thread a contiguous range of frames. This avoids
            // false sharing when neighboring threads write the Mel tile.
            const int chunk = (tile_frames + workers - 1) / workers;
            const int local_begin = thread_id * chunk;
            const int local_end = std::min(tile_frames, local_begin + chunk);
            for (int local_t = local_begin; local_t < local_end; ++local_t) {
                const int start = (frame_begin + local_t) * hop_length;
                fill_bit_reversed_packed_frame(y, window, start, fft_plan,
                                               workspace);
                fft_from_bit_reversed(workspace.real.data(),
                                      workspace.imag.data(), fft_plan);
                real_fft_power(recombination_real, recombination_imag,
                               workspace);

                for (int m = 0; m < n_mels; ++m) {
                    const SparseMelFilter& filter =
                        mel_fb.filters[static_cast<std::size_t>(m)];
                    double accumulator = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : accumulator)
#endif
                    for (int k = 0; k < filter.count; ++k) {
                        accumulator += mel_fb.values[static_cast<std::size_t>(
                                           filter.value_offset + k)] *
                                       workspace.power[static_cast<std::size_t>(
                                           filter.begin + k)];
                    }
                    mel_tile[static_cast<std::size_t>(m) * tile_frames +
                             local_t] = accumulator;
                }
            }

#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static)
#endif
            for (int m = 0; m < n_mels; ++m) {
                const auto source = mel_tile.begin() +
                                    static_cast<std::size_t>(m) * tile_frames;
                const auto destination =
                    output.begin() + static_cast<std::size_t>(m) * n_frames +
                    frame_begin;
                std::copy_n(source, tile_frames, destination);
            }
        }
    }
}
