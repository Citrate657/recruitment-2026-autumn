// Experiment 007
// Profiling driver around the unchanged exp001 kernel.
// It separates PCM read/normalization, spectrogram compute, float conversion
// and output write so deeper single-node optimizations can target measured
// costs rather than guesses.

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

struct Args007 {
    std::string input;
    std::string output;
    int sr = 44100;
    int n_fft = 2048;
    int hop_length = 512;
    int n_mels = 128;
    double f_min = 0.0;
    double f_max = 22050.0;
};

double profile_now_007() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

Args007 parse_args_007(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("missing input or output path");
    Args007 args;
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

void read_pcm_007(const std::string& path, std::vector<double>& signal) {
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
    signal.resize(pcm.size());
    for (std::size_t i = 0; i < pcm.size(); ++i)
        signal[i] = static_cast<double>(pcm[i]) / 32768.0;
}

void write_float_007(const std::string& path, const std::vector<double>& data,
                     std::vector<float>& output) {
    output.resize(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        output[i] = static_cast<float>(data[i]);

    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open output: " + path);
    file.write(reinterpret_cast<const char*>(output.data()),
               static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) throw std::runtime_error("Failed to write output: " + path);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args007 args = parse_args_007(argc, argv);
        const bool profile = std::getenv("MELSPEC_PROFILE_CPU") != nullptr;

        const double run_start = profile_now_007();
        std::vector<double> signal;
        read_pcm_007(args.input, signal);
        const double read_end = profile_now_007();

        std::vector<double> mel;
        compute_melspectrogram(signal, args.sr, args.n_fft, args.hop_length,
                               args.n_mels, args.f_min, args.f_max, mel);
        const double compute_end = profile_now_007();

        std::vector<float> output_f32;
        write_float_007(args.output, mel, output_f32);
        const double write_end = profile_now_007();

        if (profile) {
            std::fprintf(
                stderr,
                "exp007_profile read_normalize=%.6f compute=%.6f "
                "convert_write=%.6f total=%.6f samples=%zu frames=%zu\n",
                read_end - run_start, compute_end - read_end,
                write_end - compute_end, write_end - run_start,
                signal.size(),
                signal.size() >= static_cast<std::size_t>(args.n_fft)
                    ? 1 + (signal.size() - args.n_fft) / args.hop_length
                    : 0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
