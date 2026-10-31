// Experiment 009
// Three-node CPU implementation combining exp004's native Slurm distribution
// with exp008's direct float32 pipeline.

#define MELSPEC_DRIVER_INCLUDE_ONLY
#include "exp008_single_node_cpu_pipelined.cpp"
#undef MELSPEC_DRIVER_INCLUDE_ONLY

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int env_int_009(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

void read_local_pcm_009(const std::string& path, std::int64_t first_sample,
                        int sample_count, std::vector<double>& signal,
                        double& io_seconds, double& normalize_seconds) {
    const double io_start = profile_now_008();
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input: " + path);
    input.seekg(static_cast<std::streamoff>(first_sample * sizeof(int16_t)),
                std::ios::beg);

    std::vector<int16_t> pcm(static_cast<std::size_t>(sample_count));
    if (!input.read(reinterpret_cast<char*>(pcm.data()),
                    static_cast<std::streamsize>(sample_count *
                                                 sizeof(int16_t)))) {
        throw std::runtime_error("Failed to read local PCM range: " + path);
    }
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

void pwrite_exact_009(int fd, const void* data, std::size_t size,
                      off_t offset) {
    const auto* bytes = static_cast<const char*>(data);
    std::size_t written = 0;
    while (written < size) {
        const ssize_t count =
            pwrite(fd, bytes + written, size - written, offset + written);
        if (count <= 0)
            throw std::runtime_error(std::string("pwrite failed: ") +
                                     std::strerror(errno));
        written += static_cast<std::size_t>(count);
    }
}

std::uint64_t run_token_009() {
    const char* value = std::getenv("MELSPEC_RUN_TOKEN");
    return value ? std::strtoull(value, nullptr, 10)
                 : static_cast<std::uint64_t>(1);
}

void write_distributed_output_009(
    const std::string& path, const std::vector<float>& local, int n_mels,
    int global_frames, int local_frame_offset, int local_frames, int rank) {
    const std::size_t total_bytes =
        static_cast<std::size_t>(n_mels) * global_frames * sizeof(float);
    const std::string ready_path = path + ".ready";
    const std::uint64_t token = run_token_009();

    if (rank == 0) {
        const int output_fd =
            open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (output_fd < 0 ||
            ftruncate(output_fd, static_cast<off_t>(total_bytes)) != 0) {
            throw std::runtime_error("cannot prepare output " + path + ": " +
                                     std::strerror(errno));
        }
        close(output_fd);

        const int ready_fd =
            open(ready_path.c_str(), O_CREAT | O_WRONLY, 0644);
        if (ready_fd < 0)
            throw std::runtime_error("cannot open ready file " + ready_path);
        pwrite_exact_009(ready_fd, &token, sizeof(token), 0);
        close(ready_fd);
    } else {
        bool ready = false;
        for (int attempt = 0; attempt < 60000; ++attempt) {
            const int ready_fd = open(ready_path.c_str(), O_RDONLY);
            if (ready_fd >= 0) {
                std::uint64_t observed = 0;
                const ssize_t count =
                    pread(ready_fd, &observed, sizeof(observed), 0);
                close(ready_fd);
                if (count == static_cast<ssize_t>(sizeof(observed)) &&
                    observed == token) {
                    ready = true;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!ready)
            throw std::runtime_error("timed out waiting for output token");
    }

    const int fd = open(path.c_str(), O_WRONLY);
    if (fd < 0)
        throw std::runtime_error("cannot open output " + path + ": " +
                                 std::strerror(errno));
    for (int m = 0; m < n_mels; ++m) {
        const std::size_t byte_offset =
            (static_cast<std::size_t>(m) * global_frames +
             local_frame_offset) *
            sizeof(float);
        pwrite_exact_009(
            fd, local.data() + static_cast<std::size_t>(m) * local_frames,
            static_cast<std::size_t>(local_frames) * sizeof(float),
            static_cast<off_t>(byte_offset));
    }
    close(fd);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args008 args = parse_args_008(argc, argv);
        if (args.n_fft <= 0 || args.hop_length <= 0 || args.n_mels <= 0)
            throw std::runtime_error("invalid mel-spectrogram parameters");

        const int rank = env_int_009("SLURM_PROCID", 0);
        const int rank_count = env_int_009("SLURM_NTASKS", 1);
        if (rank_count != 3)
            throw std::runtime_error("exp009 requires exactly three Slurm ranks");

        std::ifstream metadata(args.input, std::ios::binary | std::ios::ate);
        if (!metadata)
            throw std::runtime_error("Cannot open input: " + args.input);
        const std::int64_t sample_count =
            static_cast<std::int64_t>(metadata.tellg()) / sizeof(int16_t);
        const int global_frames =
            1 + static_cast<int>((sample_count - args.n_fft) /
                                 args.hop_length);
        if (global_frames <= 0)
            throw std::runtime_error("Audio too short for given n_fft");

        const int frames_per_rank = global_frames / rank_count;
        const int extra_frames = global_frames % rank_count;
        const int local_frames =
            frames_per_rank + (rank < extra_frames ? 1 : 0);
        const int local_frame_offset =
            rank * frames_per_rank + std::min(rank, extra_frames);
        const int local_samples =
            (local_frames - 1) * args.hop_length + args.n_fft;

        const bool profile = std::getenv("MELSPEC_PROFILE_CPU") != nullptr;
        const double run_start = profile_now_008();
        std::vector<double> signal;
        double read_io_seconds = 0.0;
        double normalize_seconds = 0.0;
        read_local_pcm_009(
            args.input,
            static_cast<std::int64_t>(local_frame_offset) *
                args.hop_length,
            local_samples, signal, read_io_seconds, normalize_seconds);
        const double read_end = profile_now_008();

        std::vector<float> output;
        compute_melspectrogram_float_008(
            signal, args.sr, args.n_fft, args.hop_length, args.n_mels,
            args.f_min, args.f_max, output);
        const double compute_end = profile_now_008();

        write_distributed_output_009(
            args.output, output, args.n_mels, global_frames,
            local_frame_offset, local_frames, rank);
        const double write_end = profile_now_008();

        if (profile) {
            std::fprintf(
                stderr,
                "exp009_profile rank=%d read_io=%.6f normalize=%.6f "
                "compute_float=%.6f distributed_write=%.6f total=%.6f "
                "local_frames=%d\n",
                rank, read_io_seconds, normalize_seconds,
                compute_end - read_end, write_end - compute_end,
                write_end - run_start, local_frames);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
