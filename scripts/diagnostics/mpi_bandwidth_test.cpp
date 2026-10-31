#include <mpi.h>

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0) std::fprintf(stderr, "run with exactly 2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    const std::size_t bytes = 200ULL * 1024 * 1024;
    std::vector<char> send_buffer(bytes, static_cast<char>(rank));
    std::vector<char> receive_buffer(bytes);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    MPI_Status status;
    MPI_Sendrecv(send_buffer.data(), static_cast<int>(bytes), MPI_BYTE,
                1 - rank, 0, receive_buffer.data(),
                 static_cast<int>(bytes), MPI_BYTE, 1 - rank, 0,
                 MPI_COMM_WORLD, &status);
    const double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0);
        std::printf("payload_mib=%.1f elapsed=%.6f effective_MiB_s=%.3f\n",
                    gib, elapsed, gib / elapsed);
    }

    MPI_Finalize();
    return 0;
}
