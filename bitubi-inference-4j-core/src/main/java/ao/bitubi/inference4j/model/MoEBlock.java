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

import ao.bitubi.inference4j.math.ActivationFunction;
import ao.bitubi.inference4j.math.VectorMath;
import ao.bitubi.inference4j.model.functions.FeedForward;
import ao.bitubi.inference4j.tensor.AbstractTensor;
import ao.bitubi.inference4j.tensor.FloatBufferTensor;
import ao.bitubi.inference4j.tensor.operations.TensorOperationsProvider;
import java.util.*;
import java.util.function.Consumer;

/**
 * A Mixture of Experts block. See https://huggingface.co/blog/moe for more details
 */
public class MoEBlock implements FeedForward {

    private final AbstractModel model;
    private final DistributedContext dctx;
    private final AbstractTensor moeGateWeight;
    private final int numberOfExperts;
    private final int numberOfExpertsPerToken;
    private final AbstractTensor fullyConnectedWeights[];
    private final AbstractTensor projectionWeights[];
    private final AbstractTensor upProjectionWeights[];
    private final ActivationFunction.Type activationFunction;

    public MoEBlock(
        AbstractModel model,
        int numberOfExperts,
        int numberOfExpertsPerToken,
        ActivationFunction.Type activationFunction,
        AbstractTensor moeGateWeight,
        AbstractTensor[] fullyConnectedWeights,
        AbstractTensor[] projectionWeights,
        AbstractTensor[] upProjectionWeights
    ) {
        this.model = model;
        this.dctx = model.c.dctx();
        this.numberOfExperts = numberOfExperts;
        this.numberOfExpertsPerToken = numberOfExpertsPerToken;
        this.moeGateWeight = moeGateWeight;
        this.activationFunction = activationFunction;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = upProjectionWeights;
    }

    @Override
    public AbstractTensor forward(AbstractTensor lnemb, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int batchSize = lnemb.shape().first();

        int hiddenLength = model.c.hiddenLength;
        AbstractTensor result = model.makeTensor(batchSize, model.c.embeddingLength);
        // Per-call scratch state: shared fields would be overwritten by concurrent requests on the same model
        FloatBufferTensor expertResults = new FloatBufferTensor(numberOfExperts);
        int[] selectedExperts = new int[numberOfExpertsPerToken];

        try (
            AbstractTensor buf = model.makeTensor(1, hiddenLength);
            AbstractTensor buf2 = model.makeTensor(1, hiddenLength);
            AbstractTensor moeResult = model.makeTensor(1, model.c.embeddingLength)
        ) {

            for (int b = 0; b < batchSize; b++) {
                AbstractTensor lnembSlice = lnemb.slice(true, b);
                // Apply each experts gate to the input
                VectorMath.pfor(0, numberOfExperts, i -> {
                    expertResults.set(
                        TensorOperationsProvider.get().dotProduct(lnembSlice, moeGateWeight.slice(true, i), 0, 0, model.c.embeddingLength),
                        0,
                        i
                    );
                });

                // Pick the top experts for this token
                VectorMath.softMax(expertResults, 0, numberOfExperts);
                topk(expertResults, selectedExperts);

                // Apply the selected experts to the input
                for (int i = 0; i < numberOfExpertsPerToken; i++) {
                    AbstractTensor[] batchWeights = new AbstractTensor[] {
                        fullyConnectedWeights[selectedExperts[i]],
                        upProjectionWeights[selectedExperts[i]] };
                    AbstractTensor projectionWeight = projectionWeights[selectedExperts[i]];
                    AbstractTensor[] batchResults = new AbstractTensor[] { buf, buf2 };

                    VectorMath.pchunk(dctx.hiddenSegmentStart, dctx.hiddenSegmentLength, (chunkStart, chunkSize) -> {
                        TensorOperationsProvider.get()
                            .dotProductBatchChunk(
                                batchResults,
                                lnembSlice,
                                batchWeights,
                                0,
                                model.c.embeddingLength,
                                chunkStart,
                                chunkSize
                            );
                    });

                    VectorMath.pfor(dctx.hiddenSegmentStart, dctx.hiddenSegmentEnd, iv -> {
                        float w1 = buf.get(0, iv);
                        float w1a = ActivationFunction.eval(activationFunction, w1);
                        buf.set(w1a, 0, iv);
                    });

                    TensorOperationsProvider.get().maccumulate(buf, buf2, dctx.hiddenSegmentStart, dctx.hiddenSegmentLength);

                    tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(buf)));

                    // matmul the projection and sum into result
                    try (AbstractTensor bufq = model.maybeQuantize(buf)) {
                        VectorMath.pchunk(0, model.c.embeddingLength, (chunkStart, chunkSize) -> {
                            TensorOperationsProvider.get()
                                .dotProductChunk(moeResult, bufq, projectionWeight, 0, hiddenLength, chunkStart, chunkSize);
                        });
                    }

                    if (i == 0) {
                        result.copyFrom(moeResult, 0, 0, model.c.embeddingLength);
                    } else {
                        TensorOperationsProvider.get().accumulate(result.slice(b), moeResult, 0, model.c.embeddingLength);
                    }
                }
            }

            return result;
        }
    }

    private int[] topk(FloatBufferTensor probs, int[] selectedExperts) {
        long length = probs.size();
        for (int i = 0; i < numberOfExpertsPerToken; i++) {
            selectedExperts[i] = i;
        }
        for (int i = numberOfExpertsPerToken; i < length; i++) {
            int min = 0;
            for (int j = 1; j < numberOfExpertsPerToken; j++) {
                if (probs.get(0, selectedExperts[j]) < probs.get(0, selectedExperts[min])) {
                    min = j;
                }
            }
            if (probs.get(0, i) > probs.get(0, selectedExperts[min])) {
                selectedExperts[min] = i;
            }
        }
        return selectedExperts;
    }
}
