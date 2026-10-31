// Experiment 008
// Single-node CPU optimization derived from exp001.
//
// Changes:
//   - parallel PCM normalization;
//   - one OpenMP region for all frame chunks instead of one region per tile;
//   - direct float32 Mel output, eliminating the full double output matrix,
//     the tile staging matrix and the final serial conversion pass.

#include "exp001_packed_fft_sparse_mel_simd.cpp"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args008 {
    std::string input;
    std::string output;
    int sr = 44100;
    int n_fft = 2048;
    int hop_length = 512;
    int n_mels = 128;
    double f_min = 0.0;
    double f_max = 22050.0;
};

double profile_now_008() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

Args008 parse_args_008(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("missing input or output path");
    Args008 args;
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

void read_pcm_008(const std::string& path, std::vector<double>& signal,
                  double& io_seconds, double& normalize_seconds) {
    const double io_start = profile_now_008();
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input: " + path);
    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(int16_t)) != 0)
        throw std::runtime_error("Invalid PCM file size: " + path);

    std::vector<int16_t> pcm(static_cast<std::size_t>(size / sizeof(int16_t)));
    if (!input.read(reinterpret_cast<char*>(pcm.data()), size))
        throw std::runtime_error("Failed to read PCM: " + path);
    const double io_end = profile_now_008();

    signal.resize(pcm.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < pcm.size(); ++i)
        signal[i] = static_cast<double>(pcm[i]) / 32768.0;
    normalize_seconds = profile_now_008() - io_end;
    io_seconds = io_end - io_start;
}

void compute_melspectrogram_float_008(
    const std::vector<double>& signal, int sr, int n_fft, int hop_length,
    int n_mels, double f_min, double f_max, std::vector<float>& output) {
    validate_fft_size(n_fft);
    if (hop_length <= 0)
        throw std::runtime_error("hop_length must be positive");
    if (n_mels <= 0 || sr <= 0 || f_max <= f_min)
        throw std::runtime_error("invalid mel-spectrogram parameters");

    const int n_samples = static_cast<int>(signal.size());
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    const int packed_size = n_fft / 2;
    const FftPlan fft_plan(packed_size);
    const SparseMelFilterbank mel_fb =
        build_sparse_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    if (mel_fb.last_bin < 0)
        throw std::runtime_error("mel filterbank has no nonzero bins");

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

    const int workers = std::min(available_threads(), n_frames);
    std::vector<RealFftWorkspace> workspaces;
    workspaces.reserve(static_cast<std::size_t>(workers));
    for (int i = 0; i < workers; ++i)
        workspaces.emplace_back(n_fft, power_bins);

    output.resize(static_cast<std::size_t>(n_mels) * n_frames);

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
        const int chunk = (n_frames + workers - 1) / workers;
        const int local_begin = thread_id * chunk;
        const int local_end = std::min(n_frames, local_begin + chunk);

        for (int frame = local_begin; frame < local_end; ++frame) {
            const int start = frame * hop_length;
            fill_bit_reversed_packed_frame(signal, window, start, fft_plan,
                                           workspace);
            fft_from_bit_reversed(workspace.real.data(),
                                  workspace.imag.data(), fft_plan);
            real_fft_power(recombination_real, recombination_imag, workspace);

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
                output[static_cast<std::size_t>(m) * n_frames + frame] =
                    static_cast<float>(accumulator);
            }
        }
    }
}

void write_float_008(const std::string& path,
                     const std::vector<float>& output) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open output: " + path);
    file.write(reinterpret_cast<const char*>(output.data()),
               static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) throw std::runtime_error("Failed to write output: " + path);
}

}  // namespace

#ifndef MELSPEC_DRIVER_INCLUDE_ONLY
int main(int argc, char** argv) {
    try {
        const Args008 args = parse_args_008(argc, argv);
        const bool profile = std::getenv("MELSPEC_PROFILE_CPU") != nullptr;

        const double run_start = profile_now_008();
        std::vector<double> signal;
        double read_io_seconds = 0.0;
        double normalize_seconds = 0.0;
        read_pcm_008(args.input, signal, read_io_seconds, normalize_seconds);
        const double read_end = profile_now_008();

        std::vector<float> output;
        compute_melspectrogram_float_008(
            signal, args.sr, args.n_fft, args.hop_length, args.n_mels,
            args.f_min, args.f_max, output);
        const double compute_end = profile_now_008();

        write_float_008(args.output, output);
        const double write_end = profile_now_008();

        if (profile) {
            const std::size_t frames =
                signal.size() >= static_cast<std::size_t>(args.n_fft)
                    ? 1 + (signal.size() - args.n_fft) / args.hop_length
                    : 0;
            std::fprintf(
                stderr,
                "exp008_profile read_io=%.6f normalize=%.6f "
                "compute_float=%.6f write=%.6f total=%.6f "
                "samples=%zu frames=%zu\n",
                read_io_seconds, normalize_seconds, compute_end - read_end,
                write_end - compute_end, write_end - run_start, signal.size(),
                frames);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
#endif
