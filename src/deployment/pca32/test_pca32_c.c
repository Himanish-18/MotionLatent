/**
 * @file test_pca32_c.c
 * @brief Standalone C test harness verifying numerical parity of PCA-32 C inference
 * against Python PCA reference outputs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "pca32_infer.h"

int main(int argc, char **argv) {
    printf("=================================================================\n");
    printf("MotionLatent Phase 4: Standalone C PCA-32 Inference Harness\n");
    printf("=================================================================\n");

    const char *mean_file = "src/deployment/pca32/pca32_mean.bin";
    const char *comp_file = "src/deployment/pca32/pca32_components.bin";
    const char *in_file   = "src/deployment/pca32/test_in.bin";
    const char *out_file  = "src/deployment/pca32/test_expected_out.bin";

    if (argc >= 5) {
        mean_file = argv[1];
        comp_file = argv[2];
        in_file   = argv[3];
        out_file  = argv[4];
    }

    FILE *f_mean = fopen(mean_file, "rb");
    FILE *f_comp = fopen(comp_file, "rb");
    FILE *f_in   = fopen(in_file, "rb");
    FILE *f_out  = fopen(out_file, "rb");

    if (!f_mean || !f_comp || !f_in || !f_out) {
        printf("Error: Could not open one or more binary parameter/data files.\n");
        printf("Paths: mean='%s', comp='%s', in='%s', out='%s'\n", mean_file, comp_file, in_file, out_file);
        if (f_mean) fclose(f_mean);
        if (f_comp) fclose(f_comp);
        if (f_in) fclose(f_in);
        if (f_out) fclose(f_out);
        return 1;
    }

    float mean[PCA32_INPUT_DIM];
    float components[PCA32_LATENT_DIM * PCA32_INPUT_DIM];

    if (fread(mean, sizeof(float), PCA32_INPUT_DIM, f_mean) != PCA32_INPUT_DIM) {
        printf("Error reading mean vector.\n");
        return 2;
    }
    if (fread(components, sizeof(float), PCA32_LATENT_DIM * PCA32_INPUT_DIM, f_comp) != PCA32_LATENT_DIM * PCA32_INPUT_DIM) {
        printf("Error reading components matrix.\n");
        return 3;
    }

    fclose(f_mean);
    fclose(f_comp);

    /* Read test samples */
    float input_window[PCA32_INPUT_DIM];
    float expected_latent[PCA32_LATENT_DIM];
    float actual_latent[PCA32_LATENT_DIM];

    int sample_idx = 0;
    double max_abs_diff = 0.0;
    double total_mse = 0.0;
    int total_values = 0;

    clock_t t_start = clock();
    int n_runs = 0;

    while (fread(input_window, sizeof(float), PCA32_INPUT_DIM, f_in) == PCA32_INPUT_DIM &&
           fread(expected_latent, sizeof(float), PCA32_LATENT_DIM, f_out) == PCA32_LATENT_DIM) {
        
        int status = pca32_transform(input_window, actual_latent, mean, components);
        if (status != 0) {
            printf("Error in pca32_transform at sample %d (status=%d)\n", sample_idx, status);
            return 4;
        }

        /* Benchmark repeats */
        for (int r = 0; r < 100; ++r) {
            pca32_transform(input_window, actual_latent, mean, components);
            n_runs++;
        }

        for (int d = 0; d < PCA32_LATENT_DIM; ++d) {
            double diff = fabs((double)actual_latent[d] - (double)expected_latent[d]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
            total_mse += diff * diff;
            total_values++;
        }
        sample_idx++;
    }

    clock_t t_end = clock();
    double total_time_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    double latency_ms_per_window = (total_time_sec / (double)n_runs) * 1000.0;

    fclose(f_in);
    fclose(f_out);

    double rmse = sqrt(total_mse / (total_values > 0 ? total_values : 1));

    printf("Parity Test Results:\n");
    printf("  Evaluated Samples:     %d\n", sample_idx);
    printf("  Max Absolute Diff:     %.8e\n", max_abs_diff);
    printf("  Parity RMSE:           %.8e\n", rmse);
    printf("  Execution Latency:     %.4f ms / window (%d runs)\n", latency_ms_per_window, n_runs);

    /* Tolerance: float32 numerical precision difference < 1e-4 */
    if (max_abs_diff < 1e-4) {
        printf(">>> VERDICT: PASS (Exact Numerical Parity Confirmed) <<<\n");
        return 0;
    } else {
        printf(">>> VERDICT: FAIL (Discrepancy exceeds tolerance 1e-4) <<<\n");
        return 5;
    }
}
