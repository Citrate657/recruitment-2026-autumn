// Experiment 006
// Two Slurm nodes, one rank and two L40 GPUs per node.
// CPU roles are limited to I/O, argument parsing and GPU orchestration.
// Per-frame windowing, FFT, power and Mel projection run only on GPUs.

#include "melspec.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kGpuTile = 1024;
constexpr int kThreads = 256;

struct SparseMelFilter {
    int begin;
    int count;
    int value_offset;
};

struct SparseMelFilterbank {
    std::vector<SparseMelFilter> filters;
    std::vector<double> values;
    int last_bin = -1;
};

struct DriverArgs {
    std::string input;
    std::string output;
    int sr = 44100;
    int n_fft = 2048;
    int hop_length = 512;
    int n_mels = 128;
    double f_min = 0.0;
    double f_max = 22050.0;
};

double profile_now() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void validate_fft_size(int n_fft) {
    if (n_fft < 2 || (n_fft & (n_fft - 1)) != 0)
        throw std::runtime_error("n_fft must be a power of two");
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
        result.filters.push_back({begin, end - begin, value_offset});
        if (end > begin) result.last_bin = std::max(result.last_bin, end - 1);
    }
    return result;
}

void read_local_pcm(const std::string& path, std::int64_t first_sample,
                    int sample_count, std::vector<double>& signal) {
    if (sample_count <= 0)
        throw std::runtime_error("local PCM range is empty");
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
    signal.resize(pcm.size());
    for (std::size_t i = 0; i < pcm.size(); ++i)
        signal[i] = static_cast<double>(pcm[i]) / 32768.0;
}

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
    framed[index] =
        frame < frame_count
            ? signal[(frame_start + frame) * hop_length + sample] *
                  window[sample]
            : 0.0;
}

__global__ void power_transpose_kernel(const cufftDoubleComplex* spectrum,
                                       int frame_count, int power_bins,
                                       int spectrum_bins,
                                       double* power_transposed) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = frame_count * power_bins;
    if (index >= total) return;

    const int frame = index / power_bins;
    const int bin = index - frame * power_bins;
    const cufftDoubleComplex value = spectrum[frame * spectrum_bins + bin];
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
    const int count = counts[mel_index];
    const int coefficient_offset = offsets[mel_index];
    const int bin_start = starts[mel_index];
    double accumulator = 0.0;
    for (int k = 0; k < count; ++k) {
        accumulator +=
            coefficients[coefficient_offset + k] *
            power_transposed[(bin_start + k) * frame_count + frame];
    }
    mel[index] = accumulator;
}

__global__ void mel_to_float_kernel(const double* input, float* output,
                                    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = static_cast<float>(input[index]);
}

void run_gpu_tile(const std::vector<double>& signal,
                  const std::vector<double>& window, int frame_begin,
                  int frame_count, int hop_length, int n_fft, int n_mels,
                  const SparseMelFilterbank& mel_fb, int device,
                  int output_stride, std::vector<float>& output,
                  std::exception_ptr& error) {
    try {
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties");

        const int spectrum_bins = n_fft / 2 + 1;
        const int power_bins = mel_fb.last_bin + 1;
        const int signal_sample_count =
            (frame_count - 1) * hop_length + n_fft;
        const std::size_t framed_elements =
            static_cast<std::size_t>(kGpuTile) * n_fft;
        const std::size_t spectrum_elements =
            static_cast<std::size_t>(kGpuTile) * spectrum_bins;
        const std::size_t power_elements =
            static_cast<std::size_t>(kGpuTile) * power_bins;
        const std::size_t mel_elements =
            static_cast<std::size_t>(kGpuTile) * n_mels;

        double* device_signal = nullptr;
        double* device_window = nullptr;
        double* device_framed = nullptr;
        cufftDoubleComplex* device_spectrum = nullptr;
        double* device_power = nullptr;
        double* device_mel = nullptr;
        float* device_mel_float = nullptr;
        double* device_coefficients = nullptr;
        int* device_starts = nullptr;
        int* device_counts = nullptr;
        int* device_offsets = nullptr;
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
                              power_elements * sizeof(double)),
                   "cudaMalloc power");
        check_cuda(cudaMalloc(&device_mel, mel_elements * sizeof(double)),
                   "cudaMalloc mel");
        check_cuda(cudaMalloc(&device_mel_float,
                              mel_elements * sizeof(float)),
                   "cudaMalloc float mel");
        check_cuda(cudaMalloc(&device_coefficients,
                              mel_fb.values.size() * sizeof(double)),
                   "cudaMalloc coefficients");
        check_cuda(cudaMalloc(&device_starts, n_mels * sizeof(int)),
                   "cudaMalloc starts");
        check_cuda(cudaMalloc(&device_counts, n_mels * sizeof(int)),
                   "cudaMalloc counts");
        check_cuda(cudaMalloc(&device_offsets, n_mels * sizeof(int)),
                   "cudaMalloc offsets");

        std::vector<int> starts(static_cast<std::size_t>(n_mels));
        std::vector<int> counts(static_cast<std::size_t>(n_mels));
        std::vector<int> offsets(static_cast<std::size_t>(n_mels));
        for (int m = 0; m < n_mels; ++m) {
            starts[m] = mel_fb.filters[m].begin;
            counts[m] = mel_fb.filters[m].count;
            offsets[m] = mel_fb.filters[m].value_offset;
        }

        check_cuda(cudaMemcpy(device_signal,
                              signal.data() + frame_begin * hop_length,
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
                                  nullptr, 1, spectrum_bins, CUFFT_D2Z,
                                  kGpuTile),
                    "cufftPlanMany");
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        for (int tile_begin = 0; tile_begin < frame_count;
             tile_begin += kGpuTile) {
            const int tile_frames =
                std::min(kGpuTile, frame_count - tile_begin);
            const int framed_blocks =
                (static_cast<int>(framed_elements) + kThreads - 1) / kThreads;
            window_frames_kernel<<<framed_blocks, kThreads, 0, stream>>>(
                device_signal, device_window, tile_begin, tile_frames,
                kGpuTile, n_fft, hop_length, device_framed);
            check_cuda(cudaGetLastError(), "window_frames_kernel");

            check_cufft(cufftExecD2Z(plan, device_framed, device_spectrum),
                        "cufftExecD2Z");

            const int power_total = tile_frames * power_bins;
            const int power_blocks =
                (power_total + kThreads - 1) / kThreads;
            power_transpose_kernel<<<power_blocks, kThreads, 0, stream>>>(
                device_spectrum, tile_frames, power_bins, spectrum_bins,
                device_power);
            check_cuda(cudaGetLastError(), "power_transpose_kernel");

            const int mel_total = n_mels * tile_frames;
            const int mel_blocks = (mel_total + kThreads - 1) / kThreads;
            sparse_mel_kernel<<<mel_blocks, kThreads, 0, stream>>>(
                device_power, device_coefficients, device_starts,
                device_counts, device_offsets, tile_frames, n_mels,
                device_mel);
            check_cuda(cudaGetLastError(), "sparse_mel_kernel");
            mel_to_float_kernel<<<mel_blocks, kThreads, 0, stream>>>(
                device_mel, device_mel_float, mel_total);
            check_cuda(cudaGetLastError(), "mel_to_float_kernel");

            const std::size_t tile_bytes =
                static_cast<std::size_t>(tile_frames) * sizeof(float);
            check_cuda(
                cudaMemcpy2DAsync(
                    output.data() + frame_begin + tile_begin,
                    static_cast<std::size_t>(output_stride) * sizeof(float),
                    device_mel_float, tile_bytes, tile_bytes,
                    static_cast<std::size_t>(n_mels),
                    cudaMemcpyDeviceToHost, stream),
                "copy float mel tile");
            check_cuda(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize");
        }

        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        check_cufft(cufftDestroy(plan), "cufftDestroy");
        check_cuda(cudaFree(device_signal), "cudaFree signal");
        check_cuda(cudaFree(device_window), "cudaFree window");
        check_cuda(cudaFree(device_framed), "cudaFree framed");
        check_cuda(cudaFree(device_spectrum), "cudaFree spectrum");
        check_cuda(cudaFree(device_power), "cudaFree power");
        check_cuda(cudaFree(device_mel), "cudaFree mel");
        check_cuda(cudaFree(device_mel_float), "cudaFree float mel");
        check_cuda(cudaFree(device_coefficients), "cudaFree coefficients");
        check_cuda(cudaFree(device_starts), "cudaFree starts");
        check_cuda(cudaFree(device_counts), "cudaFree counts");
        check_cuda(cudaFree(device_offsets), "cudaFree offsets");
    } catch (...) {
        error = std::current_exception();
    }
}

void pwrite_exact(int fd, const void* data, std::size_t size, off_t offset) {
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

std::uint64_t run_token() {
    const char* value = std::getenv("MELSPEC_RUN_TOKEN");
    return value ? std::strtoull(value, nullptr, 10)
                 : static_cast<std::uint64_t>(1);
}

void write_distributed_output(const std::string& path,
                              const std::vector<float>& local,
                              int n_mels, int n_frames,
                              int local_frame_offset, int local_frames,
                              int rank) {
    const std::size_t total_bytes =
        static_cast<std::size_t>(n_mels) * n_frames * sizeof(float);
    const std::string ready_path = path + ".ready";
    const std::uint64_t token = run_token();

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
        pwrite_exact(ready_fd, &token, sizeof(token), 0);
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
            (static_cast<std::size_t>(m) * n_frames + local_frame_offset) *
            sizeof(float);
        pwrite_exact(fd,
                     local.data() + static_cast<std::size_t>(m) * local_frames,
                     static_cast<std::size_t>(local_frames) * sizeof(float),
                     static_cast<off_t>(byte_offset));
    }
    close(fd);
}

DriverArgs parse_args(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("missing input or output path");
    DriverArgs args;
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

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
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

        const DriverArgs args = parse_args(argc, argv);
        validate_fft_size(args.n_fft);
        if (args.hop_length <= 0 || args.n_mels <= 0 || args.sr <= 0 ||
            args.f_max <= args.f_min) {
            throw std::runtime_error("invalid mel-spectrogram parameters");
        }

        const int rank = env_int("SLURM_PROCID", 0);
        const int rank_count = env_int("SLURM_NTASKS", 1);
        if (rank_count != 2)
            throw std::runtime_error("exp006 requires exactly two Slurm ranks");
        const char* cpu_percent = std::getenv("MELSPEC_CPU_PERCENT");
        if (cpu_percent && std::atoi(cpu_percent) != 0)
            throw std::runtime_error("CPU frame work is forbidden in exp006");
        if (std::getenv("OMP_NUM_THREADS"))
            throw std::runtime_error("OpenMP frame workers are forbidden in exp006");

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

        const bool profile = std::getenv("MELSPEC_PROFILE_EXP006") != nullptr;
        const double read_start = profile_now();
        std::vector<double> signal;
        read_local_pcm(args.input,
                       static_cast<std::int64_t>(local_frame_offset) *
                           args.hop_length,
                       (local_frames - 1) * args.hop_length + args.n_fft,
                       signal);
        const double read_end = profile_now();

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count != 2)
            throw std::runtime_error("exp006 requires exactly two visible GPUs");
        for (int device = 0; device < device_count; ++device) {
            cudaDeviceProp properties{};
            check_cuda(cudaGetDeviceProperties(&properties, device),
                       "cudaGetDeviceProperties");
            if (std::strstr(properties.name, "L40") == nullptr)
                throw std::runtime_error("exp006 requires NVIDIA L40 GPUs");
            std::fprintf(stderr,
                         "exp006_resource rank=%d device=%d gpu=%s\n", rank,
                         device, properties.name);
        }
        check_cuda(cudaSetDevice(0), "cudaSetDevice host");

        const double setup_start = profile_now();
        const SparseMelFilterbank mel_fb =
            build_sparse_mel_filterbank(args.sr, args.n_fft, args.n_mels,
                                        args.f_min, args.f_max);
        if (mel_fb.last_bin < 0)
            throw std::runtime_error("mel filterbank has no nonzero bins");
        std::vector<double> window(static_cast<std::size_t>(args.n_fft));
        for (int i = 0; i < args.n_fft; ++i) {
            window[i] =
                0.5 *
                (1.0 -
                 std::cos(2.0 * kPi * i / static_cast<double>(args.n_fft)));
        }
        std::vector<float> output(
            static_cast<std::size_t>(args.n_mels) * local_frames);
        const double setup_end = profile_now();

        std::vector<int> frame_begins(static_cast<std::size_t>(device_count));
        std::vector<int> frame_counts(static_cast<std::size_t>(device_count));
        const int frames_per_device = local_frames / device_count;
        const int extra_device_frames = local_frames % device_count;
        int frame_cursor = 0;
        for (int device = 0; device < device_count; ++device) {
            frame_begins[device] = frame_cursor;
            frame_counts[device] =
                frames_per_device + (device < extra_device_frames ? 1 : 0);
            frame_cursor += frame_counts[device];
        }

        std::vector<std::exception_ptr> errors(
            static_cast<std::size_t>(device_count));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(device_count));
        const double compute_start = profile_now();
        for (int device = 0; device < device_count; ++device) {
            workers.emplace_back([&, device]() {
                run_gpu_tile(signal, window, frame_begins[device],
                             frame_counts[device], args.hop_length,
                             args.n_fft, args.n_mels, mel_fb, device,
                             local_frames, output, errors[device]);
            });
        }
        for (std::thread& worker : workers) worker.join();
        for (const std::exception_ptr& error : errors) {
            if (error) std::rethrow_exception(error);
        }
        const double compute_end = profile_now();

        const double write_start = profile_now();
        write_distributed_output(args.output, output, args.n_mels,
                                 global_frames, local_frame_offset,
                                 local_frames, rank);
        const double write_end = profile_now();
        if (profile) {
            std::fprintf(
                stderr,
                "exp006_profile rank=%d driver_read=%.6f setup=%.6f "
                "gpu_compute=%.6f distributed_write=%.6f local_frames=%d\n",
                rank, read_end - read_start, setup_end - setup_start,
                compute_end - compute_start, write_end - write_start,
                local_frames);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
