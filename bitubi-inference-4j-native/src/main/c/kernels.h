/**
 * @file kernels.h
 * @brief Interface privada entre a unidade base (vector_simd.c) e as unidades por ISA (kernels_*.c).
 *
 * Cada unidade por ISA e compilada com as suas proprias flags e exporta uma tabela de funcoes. A unidade base
 * (driver de blocos, API publica e deteccao da CPU) e compilada sem nenhuma instrucao acima do x86-64 generico.
 */
#ifndef BI4J_KERNELS_H
#define BI4J_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include "vector_simd.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static inline short fp32_to_bf16(float s) {
    char bf;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        bf = (u.i >> 16) | 64; /* force to quiet */
        return bf;
    }
    if (!(u.i & 0x7f800000)) { /* subnormal */
        bf = (u.i & 0x80000000) >> 16; /* flush to zero */
        return bf;
    }
    bf = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return bf;
}

//All params
struct gemm_params {
    int flags;
    const float* restrict af;
    const char* restrict a;
    const short* restrict as;
    int aoffset;
    const float* restrict bf;
    const char* restrict b;
    const short* restrict bs;
    int boffset;
    float * restrict r;
    short * restrict rs;
    int roffset;
    int m;
    int n;
    int k;
    int lda;
    int ldaf;
    int ldb;
    int ldbf;
    int ldc;
};

/* Calcula um bloco RM x RN da saida (ver gemm() em vector_simd.c). */
typedef void (*bi4j_tile_fn)(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params);

/* Tabela de funcoes de uma ISA: um kernel por combinacao de tipos. */
struct bi4j_kernels {
    const char *name;
    bi4j_tile_fn q8_q4;
    bi4j_tile_fn f32;
    bi4j_tile_fn f32_q4;
    bi4j_tile_fn bf16;
    bi4j_tile_fn f32_bf16;
};

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define BI4J_ARM 1
extern const struct bi4j_kernels bi4j_kernels_neon;
#else
#define BI4J_ARM 0
extern const struct bi4j_kernels bi4j_kernels_avx2;
extern const struct bi4j_kernels bi4j_kernels_avxvnni;
extern const struct bi4j_kernels bi4j_kernels_avx512;
#endif

#endif
