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

import ao.bitubi.inference4j.model.AbstractModel;
import ao.bitubi.inference4j.model.ModelSupport;
import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.safetensors.SafeTensorSupport;
import ao.bitubi.inference4j.safetensors.prompt.PromptContext;
import ao.bitubi.inference4j.tensor.AbstractTensor;
import ao.bitubi.inference4j.util.MachineSpec;
import java.io.File;
import java.lang.reflect.Field;
import java.util.Comparator;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.junit.Assume;
import org.junit.Test;

/**
 * Investigation aid: runs a real model on the Panama backend and, for every tensor operation, repeats the
 * operation with the native backend on copies of the data, recording where the two disagree.
 * Opt-in with SHADOW=1.
 */
public class ShadowNativeInvestigation {

    static class Stat {
        long calls;
        long badCalls;
        double maxRel;
        String first;
    }

    static final Map<String, Stat> stats = new ConcurrentHashMap<>();

    static AbstractTensor copyOf(AbstractTensor t) {
        AbstractTensor c = t.copyShape();
        c.copyFrom(t, 0, 0, (int) t.size());
        return c;
    }

    static final float SENTINEL = 12345.678f;

    /** A private tensor with the same shape, filled with a sentinel so we can tell which cells an operation wrote. */
    static AbstractTensor sentinelLike(AbstractTensor t) {
        AbstractTensor c = t.copyShape();
        int[] cur = new int[c.dims()];
        do {
            c.set(SENTINEL, cur);
        } while (c.iterate(cur));
        return c;
    }

    static float flat(AbstractTensor t, int i) {
        return t.dims() == 2 ? t.get(0, i) : t.get(i);
    }

    /** Compares only [from, to) of the last dimension (row 0), recording the worst relative error. */
    static void compareRange(String sig, AbstractTensor expected, AbstractTensor actual, int from, int to) {
        Stat s = stats.computeIfAbsent(sig, k -> new Stat());
        double maxRel = 0;
        String first = null;
        for (int i = from; i < to; i++) {
            float e = flat(expected, i);
            float a = flat(actual, i);
            double rel = Math.abs(e - a) / Math.max(1.0, Math.abs(e));
            if (Float.isNaN(a) && !Float.isNaN(e)) rel = Double.POSITIVE_INFINITY;
            if (rel > maxRel) maxRel = rel;
            if (rel > 1e-2 && first == null) first = "[" + i + "] expected=" + e + " native=" + a;
        }
        synchronized (s) {
            s.calls++;
            if (maxRel > s.maxRel) s.maxRel = maxRel;
            if (first != null) {
                s.badCalls++;
                if (s.first == null) s.first = first;
            }
        }
    }

    /** Compares all elements; records the worst relative error and the first element that is off. */
    static void compare(String sig, AbstractTensor expected, AbstractTensor actual) {
        Stat s = stats.computeIfAbsent(sig, k -> new Stat());
        double maxRel = 0;
        String first = null;
        int[] cur = new int[expected.dims()];
        do {
            float e = expected.get(cur);
            float a = actual.get(cur);
            double rel = Math.abs(e - a) / Math.max(1.0, Math.abs(e));
            if (Float.isNaN(a) && !Float.isNaN(e)) rel = Double.POSITIVE_INFINITY;
            if (rel > maxRel) maxRel = rel;
            if (rel > 1e-2 && first == null) first = java.util.Arrays.toString(cur) + " expected=" + e + " native=" + a;
        } while (expected.iterate(cur));
        synchronized (s) {
            s.calls++;
            if (maxRel > s.maxRel) s.maxRel = maxRel;
            if (first != null) {
                s.badCalls++;
                if (s.first == null) s.first = first;
            }
        }
    }

    static class Shadow implements TensorOperations {
        final TensorOperations ref = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
        final TensorOperations nat = new NativeSimdTensorOperations();

        @Override
        public String name() {
            return "Shadow(panama,native)";
        }

        @Override
        public int parallelSplitSize() {
            return ref.parallelSplitSize();
        }

        @Override
        public void batchDotProduct(
            AbstractTensor result,
            AbstractTensor a,
            AbstractTensor b,
            int aColumnOffset,
            int bColumnOffset,
            int columnLimit,
            int rRowOffset,
            int bRowOffset,
            int rowChunkSize
        ) {
            AbstractTensor refC = sentinelLike(result);
            AbstractTensor c = sentinelLike(result);
            ref.batchDotProduct(result, a, b, aColumnOffset, bColumnOffset, columnLimit, rRowOffset, bRowOffset, rowChunkSize);
            ref.batchDotProduct(refC, a, b, aColumnOffset, bColumnOffset, columnLimit, rRowOffset, bRowOffset, rowChunkSize);
            String sig = "batchDotProduct " + a.dType() + "x" + b.dType() + " M=" + a.shape().dim(0) + " K=" + columnLimit
                + " N=" + rowChunkSize;
            try {
                nat.batchDotProduct(c, a, b, aColumnOffset, bColumnOffset, columnLimit, rRowOffset, bRowOffset, rowChunkSize);
                compare(sig, refC, c);
            } catch (Throwable t) {
                stats.computeIfAbsent(sig + " (exception)", k -> new Stat()).first = t.toString();
            }
        }

        @Override
        public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
            AbstractTensor c = copyOf(a);
            ref.accumulate(a, b, offset, length);
            nat.accumulate(c, b, offset, length);
            compareRange("accumulate " + a.dType() + "+" + b.dType() + " len=" + length, a, c, offset, offset + length);
        }

        @Override
        public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
            AbstractTensor c = copyOf(a);
            ref.maccumulate(a, b, offset, length);
            nat.maccumulate(c, b, offset, length);
            compareRange("maccumulate " + a.dType() + "*" + b.dType() + " len=" + length, a, c, offset, offset + length);
        }

        @Override
        public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
            AbstractTensor c = copyOf(y);
            ref.saxpy(alpha, x, y, xoffset, yoffset, limit);
            nat.saxpy(alpha, x, c, xoffset, yoffset, limit);
            compareRange("saxpy " + x.dType() + "->" + y.dType() + " limit=" + limit, y, c, yoffset, yoffset + limit);
        }

        @Override
        public void scale(float factor, AbstractTensor x, int offset, int length) {
            AbstractTensor c = copyOf(x);
            ref.scale(factor, x, offset, length);
            nat.scale(factor, c, offset, length);
            compareRange("scale " + x.dType() + " len=" + length, x, c, offset, offset + length);
        }

        @Override
        public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
            AbstractTensor r = ref.quantize(t, qtype, offset, length);
            AbstractTensor n = nat.quantize(t, qtype, offset, length);
            String sig = "quantize " + t.dType() + "->" + qtype + " off=" + offset + " len=" + length;
            Stat s = stats.computeIfAbsent(sig, k -> new Stat());
            double maxRel = 0;
            String first = null;
            if (r.dims() == 2) {
                for (int i = 0; i < r.shape().dim(0); i++)
                    for (int j = offset; j < offset + length; j++) {
                        float e = r.get(i, j);
                        float a = n.get(i, j);
                        double rel = Math.abs(e - a) / Math.max(1.0, Math.abs(e));
                        if (rel > maxRel) maxRel = rel;
                        if (rel > 1e-2 && first == null) first = "[" + i + "," + j + "] expected=" + e + " native=" + a;
                    }
            }
            synchronized (s) {
                s.calls++;
                if (maxRel > s.maxRel) s.maxRel = maxRel;
                if (first != null) {
                    s.badCalls++;
                    if (s.first == null) s.first = first;
                }
            }
            return r;
        }
    }

    @Test
    public void nativeVersusPanamaOnARealModel() throws Exception {
        Assume.assumeTrue("SHADOW=1 required", "1".equals(System.getenv("SHADOW")));

        TensorOperationsProvider.get(); // make sure the singleton exists
        Field instanceF = TensorOperationsProvider.class.getDeclaredField("instance");
        instanceF.setAccessible(true);
        Field providerF = TensorOperationsProvider.class.getDeclaredField("provider");
        providerF.setAccessible(true);
        providerF.set(instanceF.get(null), new Shadow());

        File dir = SafeTensorSupport.maybeDownloadModel("./models", "tjake/Qwen2.5-0.5B-Instruct-JQ4");
        AbstractModel m = ModelSupport.loadModel(dir, DType.F32, DType.I8);
        PromptContext ctx = m.promptSupport().get().builder().addUserMessage("What is the capital of France?").build();
        String text = m.generate(UUID.randomUUID(), ctx, 0.0f, 96, (s, f) -> {}).responseText;
        System.out.println("SHADOW response (panama drives the model): " + text);

        stats.entrySet()
            .stream()
            .sorted(Comparator.comparingDouble((Map.Entry<String, Stat> e) -> -e.getValue().maxRel))
            .forEach(e -> {
                Stat s = e.getValue();
                System.out.printf(
                    "SHADOW %-62s calls=%d bad=%d maxRel=%.3g %s%n",
                    e.getKey(),
                    s.calls,
                    s.badCalls,
                    s.maxRel,
                    s.first == null ? "" : "first: " + s.first
                );
            });
    }
}
