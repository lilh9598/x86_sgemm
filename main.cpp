#include "sgemm.h"

int main(int argc, char *argv[]) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    int ldc = ROUND_UP(M, MR);
    int ldc_ref = M;
    int lda = M;
    int ldb = K;
    float *c = new float[ROUND_UP(M, MR) * ROUND_UP(N, NR)];
    float *ref_c = new float[M * N];
    float *a = new float[M * K];
    float *b = new float[K * N];

    std::generate(ref_c, ref_c + M * N, []() { return 0; });
    std::generate(c, c + M * ldc, []() { return 0; });
    std::generate(a, a + M * K, []() { return (rand() % 1000) / 1000.f - 0.5f; });
    std::generate(b, b + K * N, []() { return (rand() % 1000) / 1000.f - 0.5f; });

    reference_sgemm(M, N, K, a, lda, b, ldb, ref_c, ldc_ref);
    my_sgemm(M, N, K, a, lda, b, ldb, c, ldc);

    if (!consistency_check(c, ref_c, M, N, ldc, N)) {
        printf("check fail\n");
        delete[] a;
        delete[] b;
        delete[] c;
        delete[] ref_c;
        return -1;
    }
    printf("check pass\n");

    const long long FLOPS = 2.f * M * N * K;

    {
        const int iteration = 2;
        auto begin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iteration; i++) {
            reference_sgemm(M, N, K, a, K, b, N, ref_c, N);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        double perf = FLOPS * iteration / (elapsed + 0.0001) / 1000.f / 1000.f;
        printf("reference_sgemm perf        : %.5lf gflop/s\n", perf);
        printf("reference_sgemm duration    : %d ms\n", elapsed / iteration);
    }

    {
        const int iteration = 50;
        float *n_c = new float[iteration * ROUND_UP(M, MR) * ROUND_UP(N, NR)];
        float *n_a = new float[iteration * M * K];
        float *n_b = new float[iteration * K * N];
        std::generate(n_b, n_b + iteration * K * N, []() { return (rand() % 1000) / 1000.f - 0.5f; });
        std::generate(n_a, n_a + iteration * M * K, []() { return (rand() % 1000) / 1000.f - 0.5f; });

        auto begin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iteration; i++) {
            my_sgemm(M, N, K, n_a + i * M * K, lda, n_b + i * K * N, ldb, n_c + i * ROUND_UP(M, MR) * ROUND_UP(N, NR), ldc);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        double perf = FLOPS * iteration / (elapsed + 0.0001) / 1000.f / 1000.f;
        printf("my_sgemm perf               : %.5lf gflop/s\n", perf);
        printf("my_sgemm duration           : %d ms\n", elapsed / iteration);
        delete []n_c;
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] ref_c;

    return 0;
}