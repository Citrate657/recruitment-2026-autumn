// Experiment 005
// Baseline: src/melspec.cpp at repository HEAD.
// Main changes from experiment 003:
//   - add a three-rank MPI layer over the CPU+dual-GPU node kernel;
//   - assign contiguous global frame ranges to each rank;
//   - gather rank-local Mel slabs to rank zero for final output.

#include "melspec.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef MELSPEC_SLURM_MULTI
using MPI_Comm = int;
constexpr int MPI_COMM_WORLD = 0;
constexpr int MPI_DOUBLE = 0;
constexpr int MPI_FLOAT = 0;

inline int slurm_multi_gpu_rank() {
    const char* value = std::getenv("SLURM_PROCID");
    return value ? std::atoi(value) : 0;
}

inline int slurm_multi_gpu_size() {
    const char* value = std::getenv("SLURM_NTASKS");
    return value ? std::atoi(value) : 1;
}

inline double slurm_multi_gpu_time() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

#define MPI_Initialized(flag) (*(flag) = 1)
#define MPI_Init(argc, argv) 0
#define MPI_Comm_rank(comm, rank) (*(rank) = slurm_multi_gpu_rank())
#define MPI_Comm_size(comm, size) (*(size) = slurm_multi_gpu_size())
#define MPI_Wtime() slurm_multi_gpu_time()
#define MPI_Barrier(comm) 0
#define MPI_Abort(comm, code) (std::_Exit(code), 0)
#define MPI_Finalize() 0
#define MPI_Gatherv(...) 0
#else
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

#ifndef MELSPEC_FRAME_TILE
#define MELSPEC_FRAME_TILE 256
#endif

#ifndef MELSPEC_GPU_TILE
#define MELSPEC_GPU_TILE 1024
#endif

#ifndef MELSPEC_CPU_PERCENT
#define MELSPEC_CPU_PERCENT 2
#endif

constexpr int kFrameTile = MELSPEC_FRAME_TILE;
constexpr int kGpuTile = MELSPEC_GPU_TILE;
constexpr int kDefaultCpuPercent = MELSPEC_CPU_PERCENT;
std::string g_driver_output_path;
bool g_driver_mode = false;
int g_driver_rank = 0;
int g_driver_rank_count = 1;
int g_driver_global_frames = 0;
int g_driver_local_frames = 0;
int g_driver_local_frame_offset = 0;

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

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

void check_cufft(cufftResult result, const char* operation) {
    if (result != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed");
}

int available_threads();

__global__ void window_frames_kernel(const double* signal,
                                     const double* window,
                                     int frame_start, int frame_count,
                                     int batch_frames, int n_fft,
                                     int hop_length, double* framed) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_frames * n_fft;
    if (index >= total) return;

    const int frame = index / n_fft;
    const int sample = index - frame * n_fft;
    if (frame >= frame_count) {
        framed[index] = 0.0;
    } else {
        framed[index] =
            signal[(frame_start + frame) * hop_length + sample] *
            window[sample];
    }
}

__global__ void power_transpose_kernel(const cufftDoubleComplex* spectrum,
                                       int frame_count, int n_bins,
                                       double* power_transposed) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = frame_count * n_bins;
    if (index >= total) return;

    const int frame = index / n_bins;
    const int bin = index - frame * n_bins;
    const cufftDoubleComplex value = spectrum[index];
    power_transposed[bin * frame_count + frame] =
        value.x * value.x + value.y * value.y;
}

__global__ void sparse_mel_kernel(const double* power_transposed,
                                  const double* coefficients,
                                  const int* starts, const int* counts,
                                  const int* offsets, int frame_count,
                                  int n_mels, double* mel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_mels * frame_count;
    if (index >= total) return;

    const int mel_index = index / frame_count;
    const int frame = index - mel_index * frame_count;
    double accumulator = 0.0;
    const int count = counts[mel_index];
    const int coefficient_offset = offsets[mel_index];
    const int bin_start = starts[mel_index];
    for (int k = 0; k < count; ++k) {
        accumulator +=
            coefficients[coefficient_offset + k] *
            power_transposed[(bin_start + k) * frame_count + frame];
    }
    mel[index] = accumulator;
}

void run_gpu_range(const std::vector<double>& signal,
                   const std::vector<double>& window, int frame_begin,
                   int frame_count, int hop_length, int n_fft, int n_mels,
                   const SparseMelFilterbank& mel_fb, int device,
                   int output_stride, int output_frame_offset,
                   std::vector<double>& output, std::exception_ptr& error) {
    try {
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        const int n_bins = n_fft / 2 + 1;
        const int signal_sample_count =
            frame_count > 0
                ? (frame_count - 1) * hop_length + n_fft
                : 0;
        const std::size_t framed_elements =
            static_cast<std::size_t>(kGpuTile) * n_fft;
        const std::size_t spectrum_elements =
            static_cast<std::size_t>(kGpuTile) * n_bins;
        const std::size_t mel_elements =
            static_cast<std::size_t>(kGpuTile) * n_mels;

        double* device_signal = nullptr;
        double* device_window = nullptr;
        double* device_framed = nullptr;
        cufftDoubleComplex* device_spectrum = nullptr;
        double* device_power = nullptr;
        double* device_mel = nullptr;
        double* device_coefficients = nullptr;
        int* device_starts = nullptr;
        int* device_counts = nullptr;
        int* device_offsets = nullptr;
        double* pinned_mel = nullptr;
        cufftHandle plan = 0;
        cudaStream_t stream = nullptr;

        check_cuda(cudaMalloc(&device_signal,
                              signal_sample_count * sizeof(double)),
                   "cudaMalloc signal");
        check_cuda(cudaMalloc(&device_window, n_fft * sizeof(double)),
                   "cudaMalloc window");
        check_cuda(cudaMalloc(&device_framed,
                              framed_elements * sizeof(double)),
                   "cudaMalloc framed");
        check_cuda(cudaMalloc(&device_spectrum,
                              spectrum_elements * sizeof(cufftDoubleComplex)),
                   "cudaMalloc spectrum");
        check_cuda(cudaMalloc(&device_power,
                              spectrum_elements * sizeof(double)),
                   "cudaMalloc power");
        check_cuda(cudaMalloc(&device_mel, mel_elements * sizeof(double)),
                   "cudaMalloc mel");
        check_cuda(cudaMalloc(&device_coefficients,
                              mel_fb.values.size() * sizeof(double)),
                   "cudaMalloc coefficients");
        check_cuda(cudaMalloc(&device_starts, n_mels * sizeof(int)),
                   "cudaMalloc starts");
        check_cuda(cudaMalloc(&device_counts, n_mels * sizeof(int)),
                   "cudaMalloc counts");
        check_cuda(cudaMalloc(&device_offsets, n_mels * sizeof(int)),
                   "cudaMalloc offsets");
        check_cuda(cudaMallocHost(&pinned_mel,
                                  mel_elements * sizeof(double)),
                   "cudaMallocHost");

        std::vector<int> starts(static_cast<std::size_t>(n_mels));
        std::vector<int> counts(static_cast<std::size_t>(n_mels));
        std::vector<int> offsets(static_cast<std::size_t>(n_mels));
        for (int m = 0; m < n_mels; ++m) {
            starts[m] = mel_fb.filters[m].begin;
            counts[m] = mel_fb.filters[m].count;
            offsets[m] = mel_fb.filters[m].value_offset;
        }

        const double* signal_ptr = signal.data() + frame_begin * hop_length;
        check_cuda(cudaMemcpy(device_signal, signal_ptr,
                              signal_sample_count * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "copy signal");
        check_cuda(cudaMemcpy(device_window, window.data(),
                              n_fft * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "copy window");
        check_cuda(cudaMemcpy(device_coefficients, mel_fb.values.data(),
                              mel_fb.values.size() * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "copy coefficients");
        check_cuda(cudaMemcpy(device_starts, starts.data(),
                              n_mels * sizeof(int), cudaMemcpyHostToDevice),
                   "copy starts");
        check_cuda(cudaMemcpy(device_counts, counts.data(),
                              n_mels * sizeof(int), cudaMemcpyHostToDevice),
                   "copy counts");
        check_cuda(cudaMemcpy(device_offsets, offsets.data(),
                              n_mels * sizeof(int), cudaMemcpyHostToDevice),
                   "copy offsets");

        int fft_size = n_fft;
        check_cufft(cufftPlanMany(&plan, 1, &fft_size, nullptr, 1, n_fft,
                                  nullptr, 1, n_bins, CUFFT_D2Z, kGpuTile),
                    "cufftPlanMany");
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        constexpr int threads = 256;
        for (int tile_begin = 0; tile_begin < frame_count;
             tile_begin += kGpuTile) {
            const int tile_frames =
                std::min(kGpuTile, frame_count - tile_begin);
            const int framed_blocks =
                static_cast<int>((framed_elements + threads - 1) / threads);
            window_frames_kernel<<<framed_blocks, threads, 0, stream>>>(
                device_signal, device_window, tile_begin, tile_frames,
                kGpuTile, n_fft, hop_length, device_framed);
            check_cuda(cudaGetLastError(), "window_frames_kernel");

            check_cufft(cufftExecD2Z(plan, device_framed, device_spectrum),
                        "cufftExecD2Z");

            const int power_total = tile_frames * n_bins;
            const int power_blocks = (power_total + threads - 1) / threads;
            power_transpose_kernel<<<power_blocks, threads, 0, stream>>>(
                device_spectrum, tile_frames, n_bins, device_power);
            check_cuda(cudaGetLastError(), "power_transpose_kernel");

            const int mel_total = n_mels * tile_frames;
            const int mel_blocks = (mel_total + threads - 1) / threads;
            sparse_mel_kernel<<<mel_blocks, threads, 0, stream>>>(
                device_power, device_coefficients, device_starts,
                device_counts, device_offsets, tile_frames, n_mels,
                device_mel);
            check_cuda(cudaGetLastError(), "sparse_mel_kernel");

            check_cuda(cudaMemcpyAsync(
                           pinned_mel, device_mel,
                           mel_total * sizeof(double),
                           cudaMemcpyDeviceToHost, stream),
                       "copy mel to host");
            check_cuda(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize");

            for (int m = 0; m < n_mels; ++m) {
                std::copy_n(
                    pinned_mel + static_cast<std::size_t>(m) * tile_frames,
                    tile_frames,
                    output.begin() +
                        static_cast<std::size_t>(m) * output_stride +
                        frame_begin - output_frame_offset + tile_begin);
            }
        }

        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        check_cufft(cufftDestroy(plan), "cufftDestroy");
        check_cuda(cudaFreeHost(pinned_mel), "cudaFreeHost");
        check_cuda(cudaFree(device_signal), "cudaFree signal");
        check_cuda(cudaFree(device_window), "cudaFree window");
        check_cuda(cudaFree(device_framed), "cudaFree framed");
        check_cuda(cudaFree(device_spectrum), "cudaFree spectrum");
        check_cuda(cudaFree(device_power), "cudaFree power");
        check_cuda(cudaFree(device_mel), "cudaFree mel");
        check_cuda(cudaFree(device_coefficients), "cudaFree coefficients");
        check_cuda(cudaFree(device_starts), "cudaFree starts");
        check_cuda(cudaFree(device_counts), "cudaFree counts");
        check_cuda(cudaFree(device_offsets), "cudaFree offsets");
    } catch (...) {
        error = std::current_exception();
    }
}

void run_cpu_range(const std::vector<double>& signal,
                   const std::vector<double>& window, int frame_begin,
                   int frame_count, int hop_length, int n_fft, int n_mels,
                   const SparseMelFilterbank& mel_fb, const FftPlan& fft_plan,
                   const std::vector<double>& recombination_real,
                   const std::vector<double>& recombination_imag,
                   int output_stride, int output_frame_offset,
                   std::vector<double>& output) {
    if (frame_count <= 0) return;

    const int thread_capacity = available_threads();
    const int power_bins = static_cast<int>(recombination_real.size());
    std::vector<RealFftWorkspace> workspaces;
    workspaces.reserve(static_cast<std::size_t>(thread_capacity));
    for (int i = 0; i < thread_capacity; ++i)
        workspaces.emplace_back(n_fft, power_bins);

    std::vector<double> mel_tile(static_cast<std::size_t>(n_mels) *
                                 std::min(kFrameTile, frame_count));
    for (int local_frame_begin = 0; local_frame_begin < frame_count;
         local_frame_begin += kFrameTile) {
        const int tile_frames =
            std::min(kFrameTile, frame_count - local_frame_begin);
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
            const int chunk = (tile_frames + workers - 1) / workers;
            const int local_begin = thread_id * chunk;
            const int local_end = std::min(tile_frames, local_begin + chunk);
            for (int local_t = local_begin; local_t < local_end; ++local_t) {
                const int frame = frame_begin + local_frame_begin + local_t;
                fill_bit_reversed_packed_frame(signal, window,
                                               frame * hop_length, fft_plan,
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
                        accumulator +=
                            mel_fb.values[static_cast<std::size_t>(
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
                    output.begin() + static_cast<std::size_t>(m) *
                                         output_stride +
                    frame_begin - output_frame_offset + local_frame_begin;
                std::copy_n(source, tile_frames, destination);
            }
        }
    }
}

void pwrite_exact_mpi(int fd, const void* data, std::size_t size,
                      off_t offset) {
    const auto* bytes = static_cast<const char*>(data);
    std::size_t written = 0;
    while (written < size) {
        const ssize_t count =
            pwrite(fd, bytes + written, size - written, offset + written);
        if (count <= 0) {
            std::fprintf(stderr, "pwrite failed: %s\n", std::strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        written += static_cast<std::size_t>(count);
    }
}

void write_distributed_output_mpi(const std::string& path,
                                  const std::vector<float>& local_float,
                                  int n_mels, int n_frames,
                                  int local_frame_offset, int local_frames,
                                  int rank) {
    const std::size_t total_bytes =
        static_cast<std::size_t>(n_mels) * n_frames * sizeof(float);
#ifdef MELSPEC_SLURM_MULTI
    const std::uint64_t run_token = []() {
        const char* value = std::getenv("MELSPEC_RUN_TOKEN");
        return value ? std::strtoull(value, nullptr, 10)
                     : static_cast<std::uint64_t>(1);
    }();
    const std::string ready_path = path + ".ready";
#endif
    if (rank == 0) {
        const int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (fd < 0 || ftruncate(fd, static_cast<off_t>(total_bytes)) != 0) {
            std::fprintf(stderr, "cannot prepare output %s: %s\n", path.c_str(),
                         std::strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        close(fd);
#ifdef MELSPEC_SLURM_MULTI
        const int ready_fd =
            open(ready_path.c_str(), O_CREAT | O_WRONLY, 0644);
        if (ready_fd < 0) std::_Exit(1);
        pwrite_exact_mpi(ready_fd, &run_token, sizeof(run_token), 0);
        close(ready_fd);
#endif
    }
    int fd = -1;
#ifdef MELSPEC_SLURM_MULTI
    if (rank != 0) {
        bool ready = false;
        for (int attempt = 0; attempt < 60000; ++attempt) {
            const int ready_fd = open(ready_path.c_str(), O_RDONLY);
            if (ready_fd >= 0) {
                std::uint64_t observed = 0;
                const ssize_t count =
                    pread(ready_fd, &observed, sizeof(observed), 0);
                close(ready_fd);
                if (count == static_cast<ssize_t>(sizeof(observed)) &&
                    observed == run_token) {
                    ready = true;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!ready) std::_Exit(1);
    }
#else
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (fd < 0) fd = open(path.c_str(), O_WRONLY);
    if (fd < 0) {
        std::fprintf(stderr, "cannot open output %s: %s\n", path.c_str(),
                     std::strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int m = 0; m < n_mels; ++m) {
        const std::size_t row_offset =
            (static_cast<std::size_t>(m) * n_frames + local_frame_offset) *
            sizeof(float);
        pwrite_exact_mpi(
            fd,
            local_float.data() + static_cast<std::size_t>(m) * local_frames,
            static_cast<std::size_t>(local_frames) * sizeof(float),
            static_cast<off_t>(row_offset));
    }
    close(fd);
#ifndef MELSPEC_SLURM_MULTI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
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
    const int n_frames =
        g_driver_mode ? g_driver_global_frames
                      : 1 + (n_samples - n_fft) / hop_length;
    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) MPI_Init(nullptr, nullptr);

    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    const int frames_per_rank = n_frames / rank_count;
    const int extra_frames = n_frames % rank_count;
    const int local_frames =
        g_driver_mode
            ? g_driver_local_frames
            : frames_per_rank + (rank < extra_frames ? 1 : 0);
    const int local_frame_offset =
        g_driver_mode
            ? g_driver_local_frame_offset
            : rank * frames_per_rank + std::min(rank, extra_frames);
    const double profile_start = MPI_Wtime();

    const SparseMelFilterbank mel_fb =
        build_sparse_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    if (mel_fb.last_bin < 0)
        throw std::runtime_error("mel filterbank has no nonzero bins");

    std::vector<double> window(static_cast<std::size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        window[i] =
            0.5 * (1.0 - std::cos(2.0 * kPi * i / static_cast<double>(n_fft)));
    }

    const bool driver_mode = !g_driver_output_path.empty();
    const int output_stride = driver_mode ? local_frames : n_frames;
    const int output_frame_offset =
        driver_mode ? local_frame_offset : 0;
    output.assign(static_cast<std::size_t>(n_mels) * output_stride, 0.0);

    int cpu_percent = kDefaultCpuPercent;
    if (const char* configured = std::getenv("MELSPEC_CPU_PERCENT"))
        cpu_percent = std::atoi(configured);
    cpu_percent = std::max(0, std::min(cpu_percent, 50));
    const int cpu_frames = local_frames * cpu_percent / 100;
    const int gpu_frames = local_frames - cpu_frames;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    device_count = std::min(device_count, 2);
    if (device_count <= 0)
        throw std::runtime_error("no CUDA devices are available");

    std::vector<int> frame_begins(static_cast<std::size_t>(device_count));
    std::vector<int> frame_counts(static_cast<std::size_t>(device_count));
    const int frames_per_device = gpu_frames / device_count;
    const int extra_device_frames = gpu_frames % device_count;
    int frame_cursor = local_frame_offset;
    for (int device = 0; device < device_count; ++device) {
        frame_begins[device] = frame_cursor;
        frame_counts[device] =
            frames_per_device + (device < extra_device_frames ? 1 : 0);
        frame_cursor += frame_counts[device];
    }

    const int power_bins = mel_fb.last_bin + 1;
    const FftPlan fft_plan(n_fft / 2);
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

    std::vector<std::exception_ptr> errors(
        static_cast<std::size_t>(device_count));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(device_count));
    for (int device = 0; device < device_count; ++device) {
        workers.emplace_back([&, device]() {
            run_gpu_range(y, window, frame_begins[device],
                          frame_counts[device], hop_length, n_fft, n_mels,
                          mel_fb, device, output_stride,
                          output_frame_offset, output, errors[device]);
        });
    }

    run_cpu_range(y, window, local_frame_offset + gpu_frames, cpu_frames,
                  hop_length, n_fft, n_mels, mel_fb, fft_plan,
                  recombination_real, recombination_imag, output_stride,
                  output_frame_offset, output);

    for (std::thread& worker : workers) worker.join();
    for (const std::exception_ptr& error : errors) {
        if (error) std::rethrow_exception(error);
    }
    const double profile_compute = MPI_Wtime();

    std::vector<float> local_slab(
        static_cast<std::size_t>(n_mels) * local_frames);
    for (int m = 0; m < n_mels; ++m) {
        const auto source =
            output.begin() + static_cast<std::size_t>(m) * output_stride +
            (driver_mode ? 0 : local_frame_offset);
        std::copy_n(source, local_frames,
                    local_slab.begin() +
                        static_cast<std::size_t>(m) * local_frames);
    }

    if (!g_driver_output_path.empty()) {
        const double profile_write_start = MPI_Wtime();
        write_distributed_output_mpi(
            g_driver_output_path, local_slab, n_mels, n_frames,
            local_frame_offset, local_frames, rank);
        const double profile_write_end = MPI_Wtime();
        if (rank == 0 && std::getenv("MELSPEC_PROFILE_MPI")) {
            std::fprintf(
                stderr,
                "mpi_profile compute=%.6f distributed_write=%.6f\n",
                profile_compute - profile_start,
                profile_write_end - profile_write_start);
        }
        MPI_Finalize();
        std::_Exit(0);
    }

    std::vector<int> receive_counts(static_cast<std::size_t>(rank_count));
    std::vector<int> receive_offsets(static_cast<std::size_t>(rank_count));
    int gathered_size = 0;
    for (int source_rank = 0; source_rank < rank_count; ++source_rank) {
        const int source_frames =
            frames_per_rank + (source_rank < extra_frames ? 1 : 0);
        receive_counts[source_rank] = n_mels * source_frames;
        receive_offsets[source_rank] = gathered_size;
        gathered_size += receive_counts[source_rank];
    }

    std::vector<float> gathered;
    if (rank == 0) gathered.resize(static_cast<std::size_t>(gathered_size));
    const int local_slab_count = n_mels * local_frames;
    MPI_Gatherv(local_slab.data(), local_slab_count, MPI_FLOAT,
                rank == 0 ? gathered.data() : nullptr, receive_counts.data(),
                receive_offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    const double profile_gather = MPI_Wtime();

    if (rank == 0) {
        std::vector<double> final_output(
            static_cast<std::size_t>(n_mels) * n_frames, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int m = 0; m < n_mels; ++m) {
            for (int source_rank = 0; source_rank < rank_count;
                 ++source_rank) {
                const int source_frames =
                    frames_per_rank + (source_rank < extra_frames ? 1 : 0);
                const int source_frame_offset =
                    source_rank * frames_per_rank +
                    std::min(source_rank, extra_frames);
                const auto source =
                    gathered.begin() + receive_offsets[source_rank] +
                    static_cast<std::size_t>(m) * source_frames;
                const auto destination =
                    final_output.begin() +
                    static_cast<std::size_t>(m) * n_frames +
                    source_frame_offset;
                std::copy_n(source, source_frames, destination);
            }
        }
        output.swap(final_output);
    }
    const double profile_repack = MPI_Wtime();
    if (rank == 0 && std::getenv("MELSPEC_PROFILE_MPI")) {
        std::fprintf(
            stderr,
            "mpi_profile compute=%.6f gather_wait=%.6f repack=%.6f\n",
            profile_compute - profile_start,
            profile_gather - profile_compute,
            profile_repack - profile_gather);
    }

    MPI_Finalize();
    if (rank_count > 1 && rank != 0) std::_Exit(0);
}

#ifdef MELSPEC_DRIVER_MAIN

namespace {

void read_pcm_driver_mpi(const std::string& path,
                         std::vector<double>& signal) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input: " + path);
    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<int16_t> pcm(static_cast<std::size_t>(size / sizeof(int16_t)));
    if (!input.read(reinterpret_cast<char*>(pcm.data()), size))
        throw std::runtime_error("Failed to read PCM: " + path);
    signal.resize(pcm.size());
    for (std::size_t i = 0; i < pcm.size(); ++i)
        signal[i] = static_cast<double>(pcm[i]) / 32768.0;
}

struct DriverArgsMpiGpu {
    std::string input;
    std::string output;
    int sr = 44100;
    int n_fft = 2048;
    int hop_length = 512;
    int n_mels = 128;
    double f_min = 0.0;
    double f_max = 22050.0;
};

DriverArgsMpiGpu parse_driver_args_mpi_gpu(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("missing input or output path");
    DriverArgsMpiGpu args;
    args.input = argv[1];
    args.output = argv[2];
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc)
            throw std::runtime_error(std::string("missing value for ") +
                                     argv[i]);
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--sr")
            args.sr = std::stoi(value);
        else if (key == "--n_fft")
            args.n_fft = std::stoi(value);
        else if (key == "--hop_length")
            args.hop_length = std::stoi(value);
        else if (key == "--n_mels")
            args.n_mels = std::stoi(value);
        else if (key == "--f_min")
            args.f_min = std::stod(value);
        else if (key == "--f_max")
            args.f_max = std::stod(value);
        else
            throw std::runtime_error("unknown argument: " + key);
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 &&
            (std::string(argv[1]) == "--help" ||
             std::string(argv[1]) == "-h")) {
            std::cout
                << "Usage: " << argv[0] << " <input_pcm> <output_bin>"
                << " [--sr SR] [--n_fft N_FFT] [--hop_length HOP]"
                << " [--n_mels N_MELS] [--f_min F_MIN] [--f_max F_MAX]\n";
            return 0;
        }
        const DriverArgsMpiGpu args = parse_driver_args_mpi_gpu(argc, argv);
        int rank = 0;
        int rank_count = 1;
        MPI_Init(nullptr, nullptr);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

        std::ifstream metadata(args.input, std::ios::binary | std::ios::ate);
        if (!metadata)
            throw std::runtime_error("Cannot open input: " + args.input);
        const std::int64_t sample_count =
            static_cast<std::int64_t>(metadata.tellg()) / sizeof(int16_t);
        const int global_frames =
            1 + static_cast<int>((sample_count - args.n_fft) /
                                 args.hop_length);
        const int frames_per_rank = global_frames / rank_count;
        const int extra_frames = global_frames % rank_count;
        const int local_frames =
            frames_per_rank + (rank < extra_frames ? 1 : 0);
        const int local_frame_offset =
            rank * frames_per_rank + std::min(rank, extra_frames);

        std::vector<double> signal;
        read_pcm_driver_mpi(args.input, signal);
        g_driver_output_path = args.output;
        g_driver_mode = true;
        g_driver_rank = rank;
        g_driver_rank_count = rank_count;
        g_driver_global_frames = global_frames;
        g_driver_local_frames = local_frames;
        g_driver_local_frame_offset = local_frame_offset;

        std::vector<double> output;
        compute_melspectrogram(signal, args.sr, args.n_fft, args.hop_length,
                               args.n_mels, args.f_min, args.f_max, output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}

#endif
