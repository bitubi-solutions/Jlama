/*
 * Copyright 2026 Bitubi Solutions
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the License.
 */
package ao.bitubi.inference4j.tensor.operations;

import ao.bitubi.inference4j.tensor.AbstractTensor;
import ao.bitubi.inference4j.tensor.BFloat16BufferTensor;
import ao.bitubi.inference4j.tensor.FloatBufferTensor;
import ao.bitubi.inference4j.tensor.Q4ByteBufferTensor;
import ao.bitubi.inference4j.tensor.Q8ByteBufferTensor;
import ao.bitubi.inference4j.util.MachineSpec;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

/**
 * Extends {@link TestNativeRemainderTiles} along K: real models use K = 896 and 4864, which are not multiples of the
 * 8-block (256 values) step of the I8xQ4 kernels, so the tail of every row must be handled without reading the next
 * row. Every cell of every backend is compared with a scalar oracle, for each native ISA this CPU can run and for
 * Panama. A second test checks the BF16 output path (result tensor in BF16), which goes through {@code fp32_to_bf16}.
 */
public class TestNativeRemainderTilesExtended {

    private static final float SENTINEL = 12345.678f;
    private static final int[] KS = { 32, 96, 256, 896, 4864 };
    private static final int[] MS = { 1, 2, 5, 7 };
    private static final int[] NS = { 1, 3, 5, 16, 111 };

    private static FloatBufferTensor random(Random r, int rows, int cols) {
        FloatBufferTensor t = new FloatBufferTensor(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t.set(r.nextFloat() * 2 - 1, i, j);
        return t;
    }

    /** Plain loops over the (de)quantized values: no SIMD, no tiling. */
    private static float oracle(AbstractTensor a, AbstractTensor b, int i, int j, int k) {
        double s = 0;
        for (int c = 0; c < k; c++)
            s += (double) a.get(i, c) * b.get(j, c);
        return (float) s;
    }

    /**
     * Float accumulation error grows with the size of the terms, not of the sum (sums of random signs can be close to
     * zero); a BF16 result adds its own rounding, relative to the value.
     */
    private static float tolerance(AbstractTensor a, AbstractTensor b, int i, int j, int k, float expected, float outputRelative) {
        double s = 0;
        for (int c = 0; c < k; c++)
            s += Math.abs((double) a.get(i, c) * b.get(j, c));
        return (float) (1e-3 + 1e-4 * s + outputRelative * Math.abs(expected));
    }

    /** Returns a description of the first wrong cell, or null when every cell matches the oracle. */
    private static String firstBadCell(
        TensorOperations ops,
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor r,
        int m,
        int n,
        int k,
        float outputRelative
    ) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                r.set(SENTINEL, i, j);
        try {
            ops.batchDotProduct(r, a, b, 0, 0, k);
        } catch (RuntimeException e) {
            return "threw " + e;
        }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                float e = oracle(a, b, i, j, k);
                float g = r.get(i, j);
                if (!(Math.abs(e - g) <= tolerance(a, b, i, j, k, e, outputRelative))) return "[" + i + "," + j + "] expected=" + e + " got=" + g;
            }
        return null;
    }

    /** One native operations object per ISA this CPU runs (avx2, avxvnni, ...), plus Panama. */
    private static Map<String, TensorOperations> backends() {
        Map<String, TensorOperations> out = new LinkedHashMap<>();
        try {
            for (String isa : NativeSimdTensorOperations.supportedIsas())
                out.put("native-" + isa, new NativeSimdTensorOperations(NativeSimdTensorOperations.capFlags(isa)));
        } catch (Throwable t) {
            Assume.assumeNoException("native library not loadable here", t);
        }
        Assert.assertFalse("no native ISA supported", out.isEmpty());
        // The Vector API runs every species on any CPU (in software when not accelerated)
        for (MachineSpec.Type type : new MachineSpec.Type[] { MachineSpec.Type.AVX_256, MachineSpec.Type.AVX_512, MachineSpec.Type.ARM_128 })
            out.put("panama-" + type, new PanamaTensorOperations(type));
        return out;
    }

    private record Case(String name, Function<AbstractTensor, AbstractTensor> a, Function<AbstractTensor, AbstractTensor> b, float outputRelative) {}

    private static void run(List<Case> cases, Supplier<Function<int[], AbstractTensor>> result) {
        Map<String, TensorOperations> backends = backends();
        Map<String, Integer> badCount = new TreeMap<>();
        Map<String, String> firstBad = new TreeMap<>();
        int checked = 0;
        for (int k : KS)
            for (int m : MS)
                for (int n : NS)
                    for (Case c : cases) {
                        Random r = new Random(31L * k + 7L * m + n);
                        AbstractTensor a = c.a().apply(random(r, m, k));
                        AbstractTensor b = c.b().apply(random(r, n, k));
                        for (Map.Entry<String, TensorOperations> be : backends.entrySet()) {
                            AbstractTensor out = result.get().apply(new int[] { m, n });
                            String bad = firstBadCell(be.getValue(), a, b, out, m, n, k, c.outputRelative());
                            checked++;
                            if (bad != null) {
                                String key = be.getKey() + " " + c.name();
                                badCount.merge(key, 1, Integer::sum);
                                firstBad.putIfAbsent(key, "K=" + k + " M=" + m + " N=" + n + " " + bad);
                            }
                        }
                    }
        StringBuilder sb = new StringBuilder();
        badCount.forEach((key, v) -> sb.append(key).append(": ").append(v).append(" wrong shapes, first ").append(firstBad.get(key)).append('\n'));
        Assert.assertTrue("checked " + checked + " (backend, type, K, M, N) combinations; wrong:\n" + sb, badCount.isEmpty());
    }

    @Test
    public void everyCellMatchesTheOracleForRealModelK() {
        run(
            List.of(
                new Case("F32xF32", x -> x, x -> x, 0f),
                new Case("F32xQ4", x -> x, Q4ByteBufferTensor::new, 0f),
                new Case("F32xBF16", x -> x, BFloat16BufferTensor::new, 0f),
                // A is quantized to I8 inside the kernel's own representation; the oracle reads the dequantized I8 values
                new Case("I8xQ4", Q8ByteBufferTensor::new, Q4ByteBufferTensor::new, 0f)
            ),
            () -> shape -> new FloatBufferTensor(shape[0], shape[1])
        );
    }

    @Test
    public void bf16OutputMatchesTheOracle() {
        // BF16 keeps 8 bits of mantissa: round-to-nearest is within 2^-9 of the value, allow 2^-7
        run(
            List.of(
                new Case("BF16xBF16->BF16", BFloat16BufferTensor::new, BFloat16BufferTensor::new, 1f / 128),
                new Case("F32xBF16->BF16", x -> x, BFloat16BufferTensor::new, 1f / 128)
            ),
            () -> shape -> new BFloat16BufferTensor(shape[0], shape[1])
        );
    }
}
