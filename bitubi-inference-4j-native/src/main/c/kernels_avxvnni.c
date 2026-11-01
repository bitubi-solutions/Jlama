/* Unidade AVX2 + FMA + AVX-VNNI (codificacao VEX, sem AVX-512). Flags: -mavx2 -mfma -mavxvnni (ver Makefile). */
#include "kernels.h"

#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__AVXVNNI__)
#error "kernels_avxvnni.c exige -mavx2 -mfma -mavxvnni"
#endif

#include "kernels_x86_256.inc"

const struct bi4j_kernels bi4j_kernels_avxvnni = {
    "avxvnni", gemm_q8_q4_256, gemm_f32_256, gemm_f32_q4_256, gemm_bf16_256, gemm_f32_bf16_256
};
