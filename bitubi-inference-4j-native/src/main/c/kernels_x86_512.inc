/* Kernels AVX-512 (F, BW, DQ, VL, VNNI). Incluido por kernels_avx512.c; nao compilar sozinho. */
#include <immintrin.h>

static void __attribute__((noinline)) gemm_q8_q4_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m256i mask_first_4bits = _mm256_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m256i eight = _mm256_set1_epi8(8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __attribute__((aligned(16))) float scalef[8];
    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        __m256 sums[RM][RN];

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 8) { //256bits == 8floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                     __m256 ablock = _mm256_loadu_ps(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                     __m256 bblock = _mm256_loadu_ps(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                     __m256 scaled = _mm256_mul_ps(ablock, bblock);
                     _mm256_store_ps(scalef, scaled);

                    for(int j = 0; j < 8; j++, ao += 32, bo += 16) {
                        // Load 16 bytes into 2 128-bit integer registers
                        __m256i int_va1 = _mm256_loadu_si256((__m256i const*)(params.a + params.lda * (ii + mi) + ao));
                        __m256i int_va0 = _mm256_sign_epi8(int_va1, int_va1);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

                        __m256i vb0 = _mm256_and_si256(mask_first_4bits,
                                                       _mm256_insertf128_si256(_mm256_castsi128_si256(int_vb0),
                                                                               _mm_srli_epi16(int_vb0, 4), 1));

                        vb0 = _mm256_sign_epi8(_mm256_sub_epi8(vb0, eight), int_va1);

                        __m256i res;
                        #if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                                res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), int_va0, vb0);
                        #else
                                res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(int_va0, vb0));
                        #endif

                        __m256 resf = _mm256_cvtepi32_ps(res);

                        // broadcast the float32 version of 'factor' to all elements
                        __m256 scale_f32 = _mm256_set1_ps(scalef[j]);

                        sums[mi][ni] = _mm256_fmadd_ps(scale_f32, resf, sums[mi][ni]);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float dot = _mm512_reduce_add_ps(_mm512_zextps256_ps512(sums[mi][ni]));
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

static void gemm_f32_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 16, ao += 16, bo += 16) { // 512bits == 16floats
                // Load float32
                __m512 vb = _mm512_loadu_ps(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    __m512 va = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[mi][ni] = _mm512_fmadd_ps(va, vb, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
    }
}

static void gemm_f32_q4_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);
    //int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for(int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 16) {
                for (int mi = 0; mi < RM; ++mi) {
                        // Load float32
                        __m512 va0 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                        __m512 va1 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao + 16);

                        // Load float32
                        float bfactor = params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)];

                        // broadcast the float32 version of 'factor' to all elements
                        __m512 vb_f32 = _mm512_set1_ps(bfactor);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

                        // Masked values
                        __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);

                        // Shift first 4 bits to rightmost positions
                        __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
                        last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);

                        //Subtract 8 from each int
                        first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
                        last_4bits0 = _mm_sub_epi8(last_4bits0, eight);

                        // Extend these bytes to 32-bit integers (low and high)
                        __m512i int_vb_ext_lo0 = _mm512_cvtepi8_epi32(first_4bits0);
                        __m512i int_vb_ext_hi0 = _mm512_cvtepi8_epi32(last_4bits0);

                        __m512 float_vb_lo0 = _mm512_cvtepi32_ps(int_vb_ext_lo0);
                        __m512 float_vb_hi0 = _mm512_cvtepi32_ps(int_vb_ext_hi0);

                        // Perform the scaling
                        __m512 vb_scaled_lo0 = _mm512_mul_ps(vb_f32, float_vb_lo0);
                        __m512 vb_scaled_hi0 = _mm512_mul_ps(vb_f32, float_vb_hi0);

                        // Multiply and accumulate
                        sums[mi][ni] = _mm512_fmadd_ps(va0, vb_scaled_lo0, sums[mi][ni]);
                        sums[mi][ni] = _mm512_fmadd_ps(va1, vb_scaled_hi0, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
   }

}

static void gemm_bf16_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 32) { // 512bits == 32bfloats
                // Load shorts
                __m512i vb = _mm512_loadu_si512((__m512i*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract lower 8 shorts and convert to int (lower 128 bits)
                __m512i vb0i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 0));
                // Shift left 16 bits and convert to float
                __m512 vb0 = _mm512_castsi512_ps(_mm512_slli_epi32(vb0i, 16));

                // Extract lower 8 shorts and convert to int (upper 128 bits)
                __m512i vb1i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 1));
                // Shift left 16 bits and convert to float
                __m512 vb1 = _mm512_castsi512_ps(_mm512_slli_epi32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    // Load shorts
                    __m512i va = _mm512_loadu_si512((__m512i*)(params.as + params.lda * (ii + mi) + ao));

                    // Extract lower 8 shorts and convert to int (lower 128 bits)
                    __m512i va0i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 0));
                    // Shift left 16 bits and convert to float
                    __m512 va0 = _mm512_castsi512_ps(_mm512_slli_epi32(va0i, 16));

                    // Extract lower 8 shorts and convert to int (upper 128 bits)
                    __m512i va1i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 1));
                    // Shift left 16 bits and convert to float
                    __m512 va1 = _mm512_castsi512_ps(_mm512_slli_epi32(va1i, 16));


                    // Multiply and accumulate
                    sums[mi][ni] = _mm512_fmadd_ps(va0, vb0, sums[mi][ni]);
                    sums[mi][ni] = _mm512_fmadd_ps(va1, vb1, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(r);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
    }
}

static void gemm_f32_bf16_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 32) { // 512bits == 32bfloats
                // Load shorts
                __m512i vb = _mm512_loadu_si512((__m512i*)(params.bs + params.ldb * (jj + ni) + bo));

                // Extract lower 8 shorts and convert to int (lower 128 bits)
                __m512i vb0i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 0));
                // Shift left 16 bits and convert to float
                __m512 vb0 = _mm512_castsi512_ps(_mm512_slli_epi32(vb0i, 16));

                // Extract lower 8 shorts and convert to int (upper 128 bits)
                __m512i vb1i = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(vb, 1));
                // Shift left 16 bits and convert to float
                __m512 vb1 = _mm512_castsi512_ps(_mm512_slli_epi32(vb1i, 16));

                for (int mi = 0; mi < RM; ++mi) {
                    __m512 va0 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                    __m512 va1 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao + 16);

                    // Multiply and accumulate
                    sums[mi][ni] = _mm512_fmadd_ps(va0, vb0, sums[mi][ni]);
                    sums[mi][ni] = _mm512_fmadd_ps(va1, vb1, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                if (params.rs != NULL)
                    params.rs[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = fp32_to_bf16(r);
                else
                    params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
    }
}
