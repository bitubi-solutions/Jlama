/**
 * @file vector_simd.c
 * @brief SIMD accelerated matrix multiplication (unidade base: driver, API publica e despacho por ISA)
 *
 * SIMD accelerated matrix multiplication.  Derived from the work of
 *  J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
 *  Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].
 *
 * Esta unidade NAO usa instrucoes acima do x86-64 generico (verificado por check-isa.sh): os kernels vivem em
 * kernels_<isa>.c, cada um com as suas flags, e a CPU decide qual usar em tempo de execucao.
 */
#include "kernels.h"
#include <stdatomic.h>
#if !BI4J_ARM
#include <cpuid.h>
#endif

void __attribute__((noinline)) gemm(int m0, int m, int n0, int n, void (*gemmPtr)(int, int, int, int, int, int, struct gemm_params), struct gemm_params params) {
    int mc, nc, mp, np;
    switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
            case 0x55:
                mc = 5;
                nc = 5;
                break;
            case 0x45:
                mc = 4;
                nc = 5;
                break;
            case 0x54:
                mc = 5;
                nc = 4;
                break;
            case 0x44:
                mc = 4;
                nc = 4;
                break;
            case 0x53:
                mc = 5;
                nc = 3;
                break;
            case 0x35:
                mc = 3;
                nc = 5;
                break;
            case 0x43:
                mc = 4;
                nc = 3;
                break;
            case 0x34:
                mc = 3;
                nc = 4;
                break;
            case 0x52:
                mc = 5;
                nc = 2;
                break;
            case 0x33:
                mc = 3;
                nc = 3;
                break;
            case 0x25:
                mc = 2;
                nc = 5;
                break;
            case 0x42:
                mc = 4;
                nc = 2;
                break;
            case 0x24:
                mc = 2;
                nc = 4;
                break;
            case 0x32:
                mc = 3;
                nc = 2;
                break;
            case 0x23:
                mc = 2;
                nc = 3;
                break;
            case 0x51:
                mc = 5;
                nc = 1;
                break;
            case 0x41:
                mc = 4;
                nc = 1;
                break;
            case 0x22:
                mc = 2;
                nc = 2;
                break;
            case 0x15:
                mc = 1;
                nc = 5;
                break;
            case 0x14:
                mc = 1;
                nc = 4;
                break;
            case 0x31:
                mc = 3;
                nc = 1;
                break;
            case 0x13:
                mc = 1;
                nc = 3;
                break;
            case 0x21:
                mc = 2;
                nc = 1;
                break;
            case 0x12:
                mc = 1;
                nc = 2;
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                break;
            default:
                return;
    }

    // If AVX512 is not supported, we can't use > 4x4 blocks
    /*if (((params.flags & HAS_AVX2) == 0 || (params.flags & IS_M_SERIES_MAC) == 0) && mc >= 4 && nc >= 4) {
        mc = 4;
        nc = 4;
    }*/

    gemmPtr(m0, m, n0, n, mc, nc, params);

    mp = m0 + (m - m0) / mc * mc;
    np = n0 + (n - n0) / nc * nc;
    gemm(mp, m, n0, np, gemmPtr, params);
    gemm(m0, mp, np, n, gemmPtr, params);
    // The corner where the row remainder and the column remainder meet must also be computed
    gemm(mp, m, np, n, gemmPtr, params);
}

///// Deteccao da CPU e despacho

#if !BI4J_ARM
static int detect_isas(void) {
    unsigned a, b, c, d;
    int mask = 0;

    if (!__get_cpuid(0, &a, &b, &c, &d) || a < 7) return 0;
    unsigned max_leaf = a;
    __cpuid(1, a, b, c, d);
    // FMA, OSXSAVE e AVX: o SO tem de ter ativado o estado AVX (xgetbv) alem de a CPU o ter
    if (!((c >> 12) & 1) || !((c >> 27) & 1) || !((c >> 28) & 1)) return 0;
    unsigned xlo, xhi;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    if ((xlo & 0x6) != 0x6) return 0;

    __cpuid_count(7, 0, a, b, c, d);
    unsigned max_sub = a;
    if (!((b >> 5) & 1)) return 0; // AVX2
    mask |= 1 << BI4J_ISA_AVX2;

    if (max_leaf >= 7 && max_sub >= 1) {
        unsigned a1, b1, c1, d1;
        __cpuid_count(7, 1, a1, b1, c1, d1);
        if ((a1 >> 4) & 1) mask |= 1 << BI4J_ISA_AVXVNNI;
    }

    // AVX-512 F (16), DQ (17), BW (30), VL (31) e VNNI (ecx 11); o SO tem de ter ativado opmask e zmm (xcr0 5..7)
    const unsigned need = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    if ((b & need) == need && ((c >> 11) & 1) && (xlo & 0xE0) == 0xE0) mask |= 1 << BI4J_ISA_AVX512;
    return mask;
}

static const struct bi4j_kernels *const tables[] = { NULL, &bi4j_kernels_avx2, &bi4j_kernels_avxvnni, &bi4j_kernels_avx512 };
#define BI4J_MAX_ISA BI4J_ISA_AVX512
#else
static int detect_isas(void) { return 1 << BI4J_ISA_NEON; }
static const struct bi4j_kernels *const tables[] = { NULL, &bi4j_kernels_neon };
#define BI4J_MAX_ISA BI4J_ISA_NEON
#endif

int bi4j_supported_isas(void) {
    static atomic_int cached = ATOMIC_VAR_INIT(0); // zero significa ainda nao calculado
    int value = atomic_load_explicit(&cached, memory_order_relaxed);
    if (value == 0) {
        int detected = detect_isas() + 1;
        int expected = 0;
        atomic_compare_exchange_strong_explicit(&cached, &expected, detected, memory_order_relaxed, memory_order_relaxed);
        value = atomic_load_explicit(&cached, memory_order_relaxed);
    }
    return value - 1;
}

int bi4j_selected_isa(int flags) {
    int cap = (flags >> BI4J_ISA_CAP_SHIFT) & BI4J_ISA_CAP_MASK;
    if (cap == 0 || cap > BI4J_MAX_ISA) cap = BI4J_MAX_ISA;
    int supported = bi4j_supported_isas();
    for (int level = cap; level >= 1; level--)
        if (supported & (1 << level)) return level;
    return 0;
}

static const struct bi4j_kernels *kernels_for(int flags) {
    int level = bi4j_selected_isa(flags);
    if (level == 0) {
        // O Java verifica isto antes de usar o nativo; chegar aqui e um erro de uso. Sem libc: no Windows a DLL e
        // ligada sem bibliotecas, por isso nada de fprintf/abort.
        __builtin_trap();
    }
    return tables[level];
}

///// GEMM I8 Q4
void gemm_q8_q4(int flags, const float * restrict af, const char * restrict a, int aoffset, const float * restrict bf, const char* restrict b, int boffset, float * restrict r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc) {

    struct gemm_params p = {
                        .flags = flags,
                        .af = af,
                        .a = a,
                        .aoffset = aoffset,
                        .bf = bf,
                        .b = b,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = ldaf,
                        .ldbf = ldbf,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    gemm(0, m, n0, n0 + n, kernels_for(flags)->q8_q4, p);
}

void gemm_q8_q4_batch(int flags, int batch_num, const float *af, const char *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_q8_q4(flags, af, a, aoffset, bf[i], b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldaf, ldb, ldbf, ldc);
    }
}

///// GEMM F32
void gemm_f32(int flags, const float *a, int aoffset, const float *b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .a = NULL,
                        .aoffset = aoffset,
                        .bf = b,
                        .b = NULL,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = 0,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    gemm(0, m, n0, n0 + n, kernels_for(flags)->f32, p);
}

void gemm_f32_batch(int flags, int batch_num, const float *a, int aoffset, const float **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32(flags, a, aoffset, b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldb, ldc);
    }
}

///// GEMM F32 Q4
void gemm_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .a = NULL,
                        .aoffset = aoffset,
                        .bf = bf,
                        .b = b,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = ldbf,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    gemm(0, m, n0, n0 + n, kernels_for(flags)->f32_q4, p);
}

void gemm_f32_q4_batch(int flags, int batch_num, const float *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32_q4(flags, a, aoffset, bf[i], b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
    }
}

///// GEMM BF16
void gemm_bf16(int flags, const short *a, int aoffset, const short *b, int boffset, short *rs, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .as = a,
                        .aoffset = aoffset,
                        .bs = b,
                        .boffset = boffset,
                        .rs = rs,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = 0,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    gemm(0, m, n0, n0 + n, kernels_for(flags)->bf16, p);
}

void gemm_bf16_batch(int flags, int batch_num, const short *a, int aoffset, const short **b, int boffset, short **rs, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_bf16(flags, a, aoffset, b[i], boffset, rs != NULL ? rs[i] : NULL, r != NULL ? r[i] : NULL, roffset, m, n0, n, k, lda, ldb, ldc);
    }
}

///// GEMM F32 BF16
void gemm_f32_bf16(int flags, const float *a, int aoffset, const short *b, int boffset, short *rs, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .aoffset = aoffset,
                        .bs = b,
                        .boffset = boffset,
                        .rs = rs,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = 0,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    gemm(0, m, n0, n0 + n, kernels_for(flags)->f32_bf16, p);
}

void gemm_f32_bf16_batch(int flags, int batch_num, const float *a, int aoffset, const short **b, int boffset, short **rs, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32_bf16(flags, a, aoffset, b[i], boffset, rs != NULL ? rs[i] : NULL, r != NULL ? r[i] : NULL, roffset, m, n0, n, k, lda, ldb, ldc);
    }
}
