/**
 * @file vector_simd.h
 * @brief SIMD accelerated matrix multiplication
 *
 * SIMD accelerated matrix multiplication.  Derived from the work of
 *  J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
 *  Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].
 */
#ifndef DOT_H
#define DOT_H

//Flags passes in at runtime
#define HAS_F16C 2
#define HAS_AVX2 4 // legado: ja nao seleciona nada, a ISA e escolhida pela CPU (ver BI4J_ISA_CAP_SHIFT)
#define IS_M_SERIES_MAC 8

// Nivel de ISA: 1 = AVX2+FMA (NEON no ARM64), 2 = AVX-VNNI, 3 = AVX-512 (F, BW, DQ, VL, VNNI).
// Os bits 8..11 de "flags" sao um teto (0 = sem teto): a CPU escolhe o nivel mais alto suportado ate ao teto.
#define BI4J_ISA_AVX2 1
#define BI4J_ISA_NEON 1
#define BI4J_ISA_NEON_DOTPROD 2
#define BI4J_ISA_AVXVNNI 2
#define BI4J_ISA_AVX512 3
#define BI4J_ISA_CAP_SHIFT 8
#define BI4J_ISA_CAP_MASK 15

// Mascara dos niveis que esta CPU (e este SO) suportam: o bit "1 << nivel" esta ligado se o nivel esta disponivel.
int bi4j_supported_isas(void);
// Nivel que seria usado com estas flags (0 = a CPU nao tem o minimo e o nativo nao pode ser usado).
int bi4j_selected_isa(int flags);

// Info for quantization
#define Q8_BLOCK_SIZE 32
#define Q4_BLOCK_SIZE 32

//GEMM I8 Q4
void gemm_q8_q4(int flags, const float * restrict af, const char* restrict a, int aoffset, const float * restrict bf, const char* restrict b, int boffset, float * restrict r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc);
void gemm_q8_q4_batch(int flags, int batch_num, const float * restrict af, const char * restrict a, int aoffset, const float ** restrict bf, const char ** restrict b, int boffset, float ** restrict r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc);

//GEMM F32
void gemm_f32(int flags, const float *a, int aoffset, const float *b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
void gemm_f32_batch(int flags, int batch_num, const float *a, int aoffset, const float **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);

//GEMM F32 Q4
void gemm_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc);
void gemm_f32_q4_batch(int flags, int batch_num, const float *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc);

//GEMM BF16
void gemm_bf16(int flags, const short *a, int aoffset, const short *b, int boffset, short *cr, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
void gemm_bf16_batch(int flags, int batch_num, const short *a, int aoffset, const short **b, int boffset, short **cr, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);

//GEMM F32 BF16
void gemm_f32_bf16(int flags, const float *a, int aoffset, const short *b, int boffset, short *cr, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
void gemm_f32_bf16_batch(int flags, int batch_num, const float *a, int aoffset, const short **b, int boffset, short **cr, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);

#endif