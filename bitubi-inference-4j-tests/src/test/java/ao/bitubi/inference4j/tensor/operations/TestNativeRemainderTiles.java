/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * Modifications Copyright 2026 Bitubi Solutions
 */
package ao.bitubi.inference4j.tensor.operations;

import ao.bitubi.inference4j.tensor.AbstractTensor;
import ao.bitubi.inference4j.tensor.BFloat16BufferTensor;
import ao.bitubi.inference4j.tensor.FloatBufferTensor;
import ao.bitubi.inference4j.tensor.Q4ByteBufferTensor;
import ao.bitubi.inference4j.tensor.Q8ByteBufferTensor;
import ao.bitubi.inference4j.util.MachineSpec;
import java.util.Random;
import java.util.function.Function;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

/**
 * The native GEMM tiles the output in blocks of up to 5x5. When both the row count (M) and the column count (N)
 * leave a remainder, the corner cell block must still be computed. Every cell must match the Panama backend.
 */
public class TestNativeRemainderTiles {

    private static final float SENTINEL = 12345.678f;
    private static final int K = 256; // multiple of the Q4/Q8 block size and of the vector widths (real models use 896, 4864)

    private static FloatBufferTensor random(Random r, int rows, int cols) {
        FloatBufferTensor t = new FloatBufferTensor(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t.set(r.nextFloat() * 2 - 1, i, j);
        return t;
    }

    private static FloatBufferTensor filled(int rows, int cols) {
        FloatBufferTensor t = new FloatBufferTensor(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t.set(SENTINEL, i, j);
        return t;
    }

    /** Independent oracle: plain loops over the (de)quantized values, no SIMD, no tiling. */
    private static float oracle(AbstractTensor a, AbstractTensor b, int i, int j) {
        double s = 0;
        for (int k = 0; k < K; k++)
            s += (double) a.get(i, k) * b.get(j, k);
        return (float) s;
    }

    /** Returns a description of the first wrong cell, or null when every cell matches the oracle. */
    private static String firstBadCell(TensorOperations ops, AbstractTensor a, AbstractTensor b, int m, int n) {
        FloatBufferTensor r = filled(m, n);
        ops.batchDotProduct(r, a, b, 0, 0, K);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                float e = oracle(a, b, i, j);
                float g = r.get(i, j);
                if (Math.abs(e - g) > 1e-2f * Math.max(1f, Math.abs(e))) return "[" + i + "," + j + "] expected=" + e + " got=" + g;
            }
        return null;
    }

    @Test
    public void everyCellIsComputedForShapesWithRowAndColumnRemainders() {
        // One native operations object per ISA this CPU supports (avx2, avxvnni, avx512...), so every variant is checked
        java.util.Map<String, TensorOperations> natives = new java.util.LinkedHashMap<>();
        try {
            for (String isa : NativeSimdTensorOperations.supportedIsas())
                natives.put("native-" + isa, new NativeSimdTensorOperations(NativeSimdTensorOperations.capFlags(isa)));
        } catch (Throwable t) {
            Assume.assumeNoException("native library not loadable here", t);
        }
        Assert.assertFalse("no native ISA supported", natives.isEmpty());
        TensorOperations panama = new PanamaTensorOperations(MachineSpec.Type.AVX_256);

        String[] names = { "F32xF32", "F32xQ4", "F32xBF16", "I8xQ4" };
        java.util.List<Function<AbstractTensor, AbstractTensor>> as = java.util.List.of(
            x -> x,
            x -> x,
            x -> x,
            Q8ByteBufferTensor::new
        );
        java.util.List<Function<AbstractTensor, AbstractTensor>> bs = java.util.List.of(
            x -> x,
            Q4ByteBufferTensor::new,
            BFloat16BufferTensor::new,
            Q4ByteBufferTensor::new
        );

        int[] ms = { 1, 2, 3, 5, 6, 7, 37 };
        int[] ns = { 1, 2, 3, 4, 5, 6, 7, 16, 111, 112 };
        java.util.Map<String, Integer> badCount = new java.util.TreeMap<>();
        java.util.Map<String, String> firstBad = new java.util.TreeMap<>();
        for (int m : ms) {
            for (int n : ns) {
                for (int d = 0; d < names.length; d++) {
                    Random r = new Random(7L * m + n);
                    AbstractTensor a = as.get(d).apply(random(r, m, K));
                    AbstractTensor b = bs.get(d).apply(random(r, n, K));
                    java.util.Map<String, TensorOperations> backends = new java.util.LinkedHashMap<>(natives);
                    backends.put("panama", panama);
                    for (java.util.Map.Entry<String, TensorOperations> be : backends.entrySet()) {
                        String bad = firstBadCell(be.getValue(), a, b, m, n);
                        if (bad != null) {
                            String key = be.getKey() + " " + names[d];
                            badCount.merge(key, 1, Integer::sum);
                            firstBad.putIfAbsent(key, "M=" + m + " N=" + n + " " + bad);
                        }
                    }
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        badCount.forEach((k, v) -> sb.append(k).append(": ").append(v).append(" wrong shapes, first ").append(firstBad.get(k)).append('\n'));
        Assert.assertTrue("cells not computed or wrong:\n" + sb, badCount.isEmpty());
    }
}
