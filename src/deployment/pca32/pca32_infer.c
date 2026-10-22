/**
 * @file pca32_infer.c
 * @brief MotionLatent Phase 4: Standalone Embedded PCA-32 Inference Implementation.
 */

#include "pca32_infer.h"

#ifdef PCA32_STATIC_PARAMS_AVAILABLE
#include "pca32_params.h"
#endif

int pca32_transform(
    const float *input_window,
    float *output_latent,
    const float *mean,
    const float *components
) {
    if (!input_window || !output_latent || !mean || !components) {
        return -1;
    }

    /*
     * Compute projection: z_d = sum_{i=0}^{6911} (x_i - mean_i) * W_{d, i}
     * Loop unrolling by 4 for enhanced compiler auto-vectorization (SIMD / NEON).
     */
    for (int d = 0; d < PCA32_LATENT_DIM; ++d) {
        const float *comp_row = &components[d * PCA32_INPUT_DIM];
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        int i = 0;
        for (; i <= PCA32_INPUT_DIM - 4; i += 4) {
            acc0 += (input_window[i + 0] - mean[i + 0]) * comp_row[i + 0];
            acc1 += (input_window[i + 1] - mean[i + 1]) * comp_row[i + 1];
            acc2 += (input_window[i + 2] - mean[i + 2]) * comp_row[i + 2];
            acc3 += (input_window[i + 3] - mean[i + 3]) * comp_row[i + 3];
        }
        for (; i < PCA32_INPUT_DIM; ++i) {
            acc0 += (input_window[i] - mean[i]) * comp_row[i];
        }

        output_latent[d] = acc0 + acc1 + acc2 + acc3;
    }

    return 0;
}

int pca32_transform_static(
    const float *input_window,
    float *output_latent
) {
#ifdef PCA32_STATIC_PARAMS_AVAILABLE
    return pca32_transform(input_window, output_latent, PCA32_MEAN, (const float *)PCA32_COMPONENTS);
#else
    (void)input_window;
    (void)output_latent;
    return -2; /* Static parameters not compiled in */
#endif
}

int pca32_quantized_transform_int8(
    const int8_t *input_window_q8,
    float *output_latent,
    float input_scale,
    const float *mean,
    const int8_t *comp_q8,
    float comp_scale
) {
    if (!input_window_q8 || !output_latent || !mean || !comp_q8) {
        return -1;
    }

    /*
     * In INT8 representation:
     * x_i \approx input_window_q8[i] * input_scale
     * W_{d, i} \approx comp_q8[d * 6912 + i] * comp_scale
     * z_d = sum_i (x_i - mean_i) * W_{d, i}
     *     = input_scale * comp_scale * sum_i (input_window_q8[i] * comp_q8[d, i])
     *       - comp_scale * sum_i (mean_i * comp_q8[d, i])
     *
     * The second term M_d = comp_scale * sum_i (mean_i * comp_q8[d, i]) can be precalculated,
     * allowing pure int32 accumulation in the inner loop!
     */
    const float scale_product = input_scale * comp_scale;

    for (int d = 0; d < PCA32_LATENT_DIM; ++d) {
        const int8_t *comp_row = &comp_q8[d * PCA32_INPUT_DIM];
        int32_t int_mac = 0;
        float mean_mac = 0.0f;

        for (int i = 0; i < PCA32_INPUT_DIM; ++i) {
            int_mac += (int32_t)input_window_q8[i] * (int32_t)comp_row[i];
            mean_mac += mean[i] * ((float)comp_row[i] * comp_scale);
        }

        output_latent[d] = ((float)int_mac) * scale_product - mean_mac;
    }

    return 0;
}
