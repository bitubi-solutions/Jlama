/* Unidade AVX-512 (F, BW, DQ, VL, VNNI). Flags: ver Makefile. Nao pode correr em CPUs sem AVX-512. */
#include "kernels.h"

#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX512DQ__) || !defined(__AVX512VL__) || !defined(__AVX512VNNI__)
#error "kernels_avx512.c exige -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni"
#endif

#include "kernels_x86_512.inc"

const struct bi4j_kernels bi4j_kernels_avx512 = {
    "avx512", gemm_q8_q4_512, gemm_f32_512, gemm_f32_q4_512, gemm_bf16_512, gemm_f32_bf16_512
};
