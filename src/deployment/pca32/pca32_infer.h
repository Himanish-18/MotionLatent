/**
 * @file pca32_infer.h
 * @brief MotionLatent Phase 4: Standalone Embedded PCA-32 Inference Engine.
 *
 * Provides a portable, zero-dependency C99 implementation of the frozen
 * PCA-32 dimensionality reduction transform:
 *     z = (x - mean) * W^T
 * 
 * Target environments: ARM Cortex-M4/M7/M33, RISC-V, Embedded Linux, x86.
 * Zero dynamic memory allocation (no malloc/free).
 * Deterministic execution time and O(1) stack memory footprint.
 */

#ifndef MOTIONLATENT_PCA32_INFER_H
#define MOTIONLATENT_PCA32_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define PCA32_INPUT_DIM   6912
#define PCA32_LATENT_DIM  32

/**
 * @brief Computes PCA-32 projection using external mean and components buffers.
 *
 * @param[in]  input_window   Pointer to normalized input array of 6,912 float32 values.
 * @param[out] output_latent  Pointer to output buffer of 32 float32 values.
 * @param[in]  mean           Pointer to centering mean vector (6,912 float32 values).
 * @param[in]  components     Pointer to flattened components matrix (32 * 6,912 float32 values).
 *                            Ordered row-major: component 0 (6912 floats), component 1, etc.
 * @return 0 on success, negative error code on invalid pointer.
 */
int pca32_transform(
    const float *input_window,
    float *output_latent,
    const float *mean,
    const float *components
);

/**
 * @brief Computes PCA-32 projection using statically compiled-in parameters.
 *
 * Requires PCA32_STATIC_PARAMS_AVAILABLE to be defined and pca32_params.h linked.
 *
 * @param[in]  input_window   Pointer to normalized input array of 6,912 float32 values.
 * @param[out] output_latent  Pointer to output buffer of 32 float32 values.
 * @return 0 on success, negative error code on invalid pointer.
 */
int pca32_transform_static(
    const float *input_window,
    float *output_latent
);

/**
 * @brief Computes fixed-point / INT8 quantized PCA-32 projection for ultra-constrained targets.
 *
 * Uses scaled integer MAC operations to evaluate memory/accuracy trade-offs.
 *
 * @param[in]  input_window_q8  Pointer to INT8 quantized input array (6,912 values).
 * @param[out] output_latent    Pointer to output float32 buffer (32 values).
 * @param[in]  input_scale      Quantization scale factor for input (e.g. S_x).
 * @param[in]  mean             Pointer to float32 mean vector (6,912 values).
 * @param[in]  comp_q8          Pointer to INT8 quantized components matrix (32 * 6,912).
 * @param[in]  comp_scale       Quantization scale factor for components.
 * @return 0 on success, negative error code on invalid pointer.
 */
int pca32_quantized_transform_int8(
    const int8_t *input_window_q8,
    float *output_latent,
    float input_scale,
    const float *mean,
    const int8_t *comp_q8,
    float comp_scale
);

#ifdef __cplusplus
}
#endif

#endif /* MOTIONLATENT_PCA32_INFER_H */
