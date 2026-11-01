/* Kernels NEON (ARM64); Q8xQ4 usa dotprod quando compilado com essa extensao. */
#include <arm_neon.h>
#include "kernels.h"

static void __attribute__((noinline)) gemm_q8_q4_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
#if defined(__ARM_FEATURE_DOTPROD)
    // Mask to keep the first 4 bits of each byte
    int8x16_t mask_first_4bits = vdupq_n_u8(0x0f);
    //Subtract 8 from each byte to get signed values
    int8x16_t eight = vdupq_n_s8(0x8);
#endif
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        float32x4_t sums[RM][RN];

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 4) { //128bits == 4floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    for(int j = 0; j < 4 && (i + j) < numBlocks; j++, ao += 32, bo += 16) {
#if defined(__ARM_FEATURE_DOTPROD)
                        // Load 4 bytes into a 128-bit integer register
                        int8x16_t int_va0 = vld1q_s8((const signed char *)(params.a + params.lda * (ii + mi) + ao));
                        int8x16_t int_va1 = vld1q_s8((const signed char *)(params.a + params.lda * (ii + mi) + ao + 16));

                        // Load 8 bytes into a 128-bit integer register
                        int8x16_t int_vb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)),
                                            mask_first_4bits)), eight);

                        int8x16_t int_vb1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)), 4)), eight);
#endif

#if defined(__ARM_FEATURE_DOTPROD)
                        int32x4_t dot = vdotq_s32(vdotq_s32(vdupq_n_s32(0), int_va0, int_vb0), int_va1, int_vb1);
                        sums[mi][ni] = vmlaq_n_f32(sums[mi][ni], vcvtq_f32_s32(dot), params.af[params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)] * params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)]);
#else
                        int dot = 0;
                        const signed char *qa = (const signed char *)(params.a + params.lda * (ii + mi) + ao);
                        const unsigned char *qb = (const unsigned char *)params.b + params.ldb * (jj + ni) + bo;
                        for (int q = 0; q < 16; ++q) {
                            dot += qa[q] * ((qb[q] & 0x0f) - 8);
                            dot += qa[q + 16] * ((qb[q] >> 4) - 8);
                        }
                        float32x4_t dotVector = vsetq_lane_f32((float)dot, vdupq_n_f32(0.0f), 0);
                        sums[mi][ni] = vmlaq_n_f32(sums[mi][ni], dotVector, params.af[params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)] * params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)]);
#endif
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}

static void gemm_f32_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    float32x4_t sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 4, ao += 4, bo += 4) { // 128bits == 4floats
                // Load float32
                float32x4_t vb = vld1q_f32(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    float32x4_t va = vld1q_f32(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va, vb);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}

static void __attribute__((noinline)) gemm_f32_q4_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    int8x16_t mask_first_4bits = vdupq_n_u8(0x0f);
    //Subtract 8 from each byte to get signed values
    int8x16_t eight = vdupq_n_s8(0x8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        float32x4_t sums[RM][RN];

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 4) { //128bits == 4floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32

                    for(int j = 0; j < 4 && (i + j) < numBlocks; j++, ao += 32, bo += 16) {
                        float32x4_t vb_f32 = vdupq_n_f32(params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)]);

                        // Load 4 bytes into a 128-bit integer register
                        float32x4_t f_va0 = vld1q_f32(params.af + params.lda * (ii + mi) + ao);
                        float32x4_t f_va1 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 4);
                        float32x4_t f_va2 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 8);
                        float32x4_t f_va3 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 12);

                        float32x4_t f_va4 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 16);
                        float32x4_t f_va5 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 20);
                        float32x4_t f_va6 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 24);
                        float32x4_t f_va7 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 28);

                        // Load 8 bytes into a 128-bit integer register
                        int8x16_t int_vb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)),
                                            mask_first_4bits)), eight);

                        // Convert int_vb0 into two float32x4_t registers
                        int16x8_t int_vb0_low = vmovl_s8(vget_low_s8(int_vb0));
                        int16x8_t int_vb0_high = vmovl_s8(vget_high_s8(int_vb0));
                        float32x4_t f_vb0_0 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb0_low))));
                        float32x4_t f_vb0_1 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb0_low))));
                        float32x4_t f_vb0_2 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb0_high))));
                        float32x4_t f_vb0_3 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb0_high))));

                        int8x16_t int_vb1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)), 4)), eight);

                        // Convert int_vb0 into two float32x4_t registers
                        int16x8_t int_vb1_low = vmovl_s8(vget_low_s8(int_vb1));
                        int16x8_t int_vb1_high = vmovl_s8(vget_high_s8(int_vb1));
                        float32x4_t f_vb1_0 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb1_low))));
                        float32x4_t f_vb1_1 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb1_low))));
                        float32x4_t f_vb1_2 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_low_s16(int_vb1_high))));
                        float32x4_t f_vb1_3 = vmulq_f32(vb_f32, vcvtq_f32_s32(vmovl_s16(vget_high_s16(int_vb1_high))));

                        // FMA operations for sums[mi][ni] with each of the 8 pairs of va and vb
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va0, f_vb0_0);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va1, f_vb0_1);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va2, f_vb0_2);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va3, f_vb0_3);

                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va4, f_vb1_0);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va5, f_vb1_1);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va6, f_vb1_2);
                        sums[mi][ni] = vmlaq_f32(sums[mi][ni], f_va7, f_vb1_3);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}

static void gemm_bf16_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    float32x4_t sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 8, ao += 8, bo += 8) { // 128bits == 8bfloats
                // Load shorts
                uint16x8_t vb = vld1q_u16((const uint16_t*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract and convert to float
                uint32x4_t vb0i = vmovl_u16(vget_low_u16(vb));
                uint32x4_t vb1i = vmovl_u16(vget_high_u16(vb));
                float32x4_t vb0 = vreinterpretq_f32_u32(vshlq_n_u32(vb0i, 16));
                float32x4_t vb1 = vreinterpretq_f32_u32(vshlq_n_u32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    uint16x8_t va = vld1q_u16((const uint16_t*)(params.as + params.lda * (ii + mi) + ao));

                    // Extract and convert to float
                    uint32x4_t va0i = vmovl_u16(vget_low_u16(va));
                    uint32x4_t va1i = vmovl_u16(vget_high_u16(va));
                    float32x4_t va0 = vreinterpretq_f32_u32(vshlq_n_u32(va0i, 16));
                    float32x4_t va1 = vreinterpretq_f32_u32(vshlq_n_u32(va1i, 16));

                    // Multiply and accumulate
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va0, vb0);
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va1, vb1);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float dot = vaddvq_f32(sums[mi][ni]);

                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(dot);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

static void gemm_f32_bf16_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    float32x4_t sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 8, ao += 8, bo += 8) { // 128bits == 8bfloats
                // Load shorts
                uint16x8_t vb = vld1q_u16((const uint16_t*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract and convert to float
                uint32x4_t vb0i = vmovl_u16(vget_low_u16(vb));
                uint32x4_t vb1i = vmovl_u16(vget_high_u16(vb));
                float32x4_t vb0 = vreinterpretq_f32_u32(vshlq_n_u32(vb0i, 16));
                float32x4_t vb1 = vreinterpretq_f32_u32(vshlq_n_u32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    float32x4_t va0 = vld1q_f32(params.af + params.lda * (ii + mi) + ao);
                    float32x4_t va1 = vld1q_f32(params.af + params.lda * (ii + mi) + ao + 4);

                    // Multiply and accumulate
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va0, vb0);
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va1, vb1);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product

                float dot = vaddvq_f32(sums[mi][ni]);
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(dot);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

#ifndef BI4J_NEON_TABLE
#define BI4J_NEON_TABLE bi4j_kernels_neon
#endif

const struct bi4j_kernels BI4J_NEON_TABLE = {
    "neon", gemm_q8_q4_128_arm, gemm_f32_128_arm, gemm_f32_q4_128_arm, gemm_bf16_128_arm, gemm_f32_bf16_128_arm
};
