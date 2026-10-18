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
 */
package ao.bitubi.inference4j.model;

import static org.junit.Assert.assertArrayEquals;

import ao.bitubi.inference4j.tensor.AbstractTensor;
import ao.bitubi.inference4j.tensor.FloatBufferTensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.Test;

/**
 * Calls the forward of a non-GQA {@link CausalSelfAttention} (the path that uses the fused QKV batch
 * projection) concurrently on one instance, with different batch sizes per thread, and compares each
 * result with the sequential one. Needs no model files.
 */
public class AttentionConcurrencyTest {

    private static final int EMBEDDING = 64;
    private static final int HEADS = 4;
    private static final int[] BATCH_SIZES = { 1, 3, 5, 8 };
    private static final int ROUNDS = 200;
    // Sequential reruns of the same input already differ by ~1 ulp for some batch sizes (not bit-reproducible),
    // so compare with a tolerance far below what a corrupted batch produces.
    private static final float TOLERANCE = 1e-5f;

    private static FloatBufferTensor randomTensor(Random rnd, int rows, int cols) {
        FloatBufferTensor t = new FloatBufferTensor(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t.set((rnd.nextFloat() - 0.5f) * 0.2f, r, c);
        return t;
    }

    private static float[] run(AbstractModel model, CausalSelfAttention attention, AbstractTensor input) {
        try (var kv = model.kvBufferCache.getKvBuffer(UUID.randomUUID())) {
            AbstractTensor out = attention.forward(input, 0, kv, Optional.empty());
            int n = input.shape().first();
            float[] r = new float[n * EMBEDDING];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < EMBEDDING; j++)
                    r[i * EMBEDDING + j] = out.get(i, j);
            return r;
        }
    }

    @Test
    public void concurrentForwardWithDifferentBatchSizesMatchesSequential() throws Exception {
        AbstractModel model = Mocks.makeModel(64, EMBEDDING, EMBEDDING * 2, HEADS, 1);
        Random rnd = new Random(42);
        CausalSelfAttention attention = new CausalSelfAttention(
            model,
            0,
            randomTensor(rnd, EMBEDDING, EMBEDDING),
            randomTensor(rnd, EMBEDDING, EMBEDDING),
            randomTensor(rnd, EMBEDDING, EMBEDDING),
            randomTensor(rnd, EMBEDDING, EMBEDDING)
        );

        AbstractTensor[] inputs = new AbstractTensor[BATCH_SIZES.length];
        float[][] expected = new float[BATCH_SIZES.length][];
        for (int i = 0; i < BATCH_SIZES.length; i++) {
            inputs[i] = randomTensor(rnd, BATCH_SIZES[i], EMBEDDING);
            expected[i] = run(model, attention, inputs[i]);
        }

        ExecutorService executor = Executors.newFixedThreadPool(BATCH_SIZES.length);
        try {
            for (int round = 0; round < ROUNDS; round++) {
                CountDownLatch start = new CountDownLatch(1);
                List<Future<float[]>> futures = new ArrayList<>();
                for (int i = 0; i < BATCH_SIZES.length; i++) {
                    final int idx = i;
                    futures.add(executor.submit(() -> {
                        start.await();
                        return run(model, attention, inputs[idx]);
                    }));
                }
                start.countDown();
                for (int i = 0; i < BATCH_SIZES.length; i++) {
                    assertArrayEquals("round " + round + ", batch size " + BATCH_SIZES[i], expected[i], futures.get(i).get(60, TimeUnit.SECONDS), TOLERANCE);
                }
            }
        } finally {
            executor.shutdownNow();
        }
    }
}
