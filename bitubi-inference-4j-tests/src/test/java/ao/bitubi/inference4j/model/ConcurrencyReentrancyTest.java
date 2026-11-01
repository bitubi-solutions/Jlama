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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.safetensors.SafeTensorSupport;
import ao.bitubi.inference4j.safetensors.prompt.PromptContext;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.Assume;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tests concurrency and reentrancy on a single model instance.
 * Opt-in with CONCURRENCY_TEST=1.
 */
public class ConcurrencyReentrancyTest {

    private static final Logger logger = LoggerFactory.getLogger(ConcurrencyReentrancyTest.class);

    private static final String[] PROMPTS = new String[] {
        "What is the capital of France?",
        "What is 2 + 2?",
        "Name three primary colors.",
        "Say hello in Portuguese."
    };

    private static final int MAX_TOKENS = 48;
    private static AbstractModel model;
    private static final String[] seqOutputs = new String[PROMPTS.length];

    @BeforeClass
    public static void setUp() throws Exception {
        Assume.assumeTrue("CONCURRENCY_TEST=1 required", "1".equals(System.getenv("CONCURRENCY_TEST")));
        File dir = SafeTensorSupport.maybeDownloadModel("./models", "tjake/Qwen2.5-0.5B-Instruct-JQ4");
        model = ModelSupport.loadModel(dir, DType.F32, DType.I8);
        for (int i = 0; i < PROMPTS.length; i++) {
            seqOutputs[i] = generate(UUID.randomUUID(), i);
            assertFalse("Sequential output was blank for prompt: " + PROMPTS[i], seqOutputs[i].isBlank());
            logger.info("Sequential output for prompt [{}]: {}", i, seqOutputs[i].trim());
            logger.info("Code points for prompt [{}]: {}", i, seqOutputs[i].trim().codePoints().mapToObj(c -> String.format("U+%04X", c)).collect(java.util.stream.Collectors.joining(" ")));
            // UTF-8 bytes shown as Latin-1 ('Ã' followed by a character in U+0080..U+00BF) mean a decoding bug
            assertFalse("Mojibake in response text: " + seqOutputs[i], seqOutputs[i].matches("(?s).*\\u00C3[\\u0080-\\u00BF].*"));
        }
    }

    private static String generate(UUID session, int promptIdx) {
        PromptContext ctx = model.promptSupport().get().builder().addUserMessage(PROMPTS[promptIdx]).build();
        return model.generate(session, ctx, 0.0f, MAX_TOKENS, (s, f) -> {}).responseText;
    }

    @Before
    public void checkEnabled() {
        Assume.assumeTrue("CONCURRENCY_TEST=1 required", "1".equals(System.getenv("CONCURRENCY_TEST")));
    }

    @Test
    public void parallelGenerationsWithDistinctSessionsMatchSequential() throws Exception {
        int parallelRuns = 5;
        List<String> divergences = new ArrayList<>();
        ExecutorService executor = Executors.newFixedThreadPool(PROMPTS.length);
        try {
            for (int iter = 0; iter < parallelRuns; iter++) {
                CountDownLatch readyLatch = new CountDownLatch(PROMPTS.length);
                CountDownLatch startLatch = new CountDownLatch(1);
                List<Future<String>> futures = new ArrayList<>();

                for (int i = 0; i < PROMPTS.length; i++) {
                    final int promptIdx = i;
                    futures.add(executor.submit(() -> {
                        readyLatch.countDown();
                        startLatch.await();
                        return generate(UUID.randomUUID(), promptIdx);
                    }));
                }

                assertTrue("Timeout waiting for threads to be ready", readyLatch.await(30, TimeUnit.SECONDS));
                startLatch.countDown();

                for (int i = 0; i < PROMPTS.length; i++) {
                    String parOutput = futures.get(i).get(120, TimeUnit.SECONDS);
                    if (!seqOutputs[i].equals(parOutput)) {
                        String msg = String.format("Iteration %d, prompt %d (\"%s\"): expected=[%s], actual=[%s]",
                            iter, i, PROMPTS[i], seqOutputs[i].trim(), parOutput.trim());
                        logger.error("DIVERGENCE: {}", msg);
                        divergences.add(msg);
                    }
                }
            }
        } finally {
            executor.shutdownNow();
        }

        assertTrue("Found " + divergences.size() + " divergence(s) between sequential and parallel executions:\n"
            + String.join("\n", divergences), divergences.isEmpty());
    }

    /**
     * Two requests released together on the same session: exactly one must finish with the correct output and
     * the other must be rejected with {@link SessionBusyException}. The session must stay usable afterwards.
     */
    @Test
    public void concurrentGenerationsWithSameSessionAreRejected() throws Exception {
        UUID sharedSession = UUID.randomUUID();
        ExecutorService executor = Executors.newFixedThreadPool(2);
        int winner;
        try {
            CountDownLatch readyLatch = new CountDownLatch(2);
            CountDownLatch startLatch = new CountDownLatch(1);
            List<Future<String>> futures = new ArrayList<>();
            for (int p = 0; p < 2; p++) {
                final int promptIdx = p;
                futures.add(executor.submit(() -> {
                    readyLatch.countDown();
                    startLatch.await();
                    return generate(sharedSession, promptIdx);
                }));
            }
            assertTrue("Timeout waiting for threads to be ready", readyLatch.await(30, TimeUnit.SECONDS));
            startLatch.countDown();

            int succeeded = 0;
            int rejected = 0;
            winner = -1;
            for (int p = 0; p < 2; p++) {
                try {
                    String out = futures.get(p).get(120, TimeUnit.SECONDS);
                    succeeded++;
                    winner = p;
                    assertEquals("Winning request must produce the sequential output", seqOutputs[p], out);
                } catch (ExecutionException e) {
                    assertTrue("Unexpected failure: " + e.getCause(), e.getCause() instanceof SessionBusyException);
                    assertTrue(e.getCause().getMessage().contains(sharedSession.toString()));
                    rejected++;
                }
            }
            logger.info("Same session: succeeded={}, rejected={}, winner prompt={}", succeeded, rejected, winner);
            assertEquals("Exactly one request must succeed", 1, succeeded);
            assertEquals("Exactly one request must be rejected", 1, rejected);
        } finally {
            executor.shutdownNow();
        }

        // The rejection must not have corrupted the session: a follow-up turn must equal the same two turns on a
        // fresh session that never saw a rejection.
        int followUp = 1 - winner;
        UUID reference = UUID.randomUUID();
        generate(reference, winner);
        String expectedFollowUp = generate(reference, followUp);
        assertEquals("Follow-up on the rejected session must match a clean session", expectedFollowUp, generate(sharedSession, followUp));
    }
}
