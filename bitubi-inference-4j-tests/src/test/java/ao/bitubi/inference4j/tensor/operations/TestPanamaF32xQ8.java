package ao.bitubi.inference4j.tensor.operations;

import ao.bitubi.inference4j.tensor.FloatBufferTensor;
import ao.bitubi.inference4j.tensor.Q8ByteBufferTensor;
import ao.bitubi.inference4j.util.MachineSpec;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public class TestPanamaF32xQ8 {
    private static final int K = 64;

    @Test
    public void matchesScalarReferenceForTileRemaindersAndVectorWidths() {
        TensorOperations oracle = new ScalarReferenceOperations();
        for (MachineSpec.Type vectorType : new MachineSpec.Type[] {
            MachineSpec.Type.AVX_512, MachineSpec.Type.AVX_256, MachineSpec.Type.ARM_128
        }) {
            TensorOperations panama = new PanamaTensorOperations(vectorType);
            for (int m : new int[] { 1, 2, 5, 7 }) {
                for (int n : new int[] { 1, 3, 5, 9 }) {
                    Random random = new Random(31L * m + n);
                    FloatBufferTensor a = randomTensor(random, m, K);
                    Q8ByteBufferTensor b = new Q8ByteBufferTensor(randomTensor(random, n, K));
                    FloatBufferTensor expected = new FloatBufferTensor(m, n);
                    FloatBufferTensor actual = new FloatBufferTensor(m, n);
                    oracle.batchDotProduct(expected, a, b, 0, 0, K);
                    panama.batchDotProduct(actual, a, b, 0, 0, K);
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            Assert.assertEquals("vector=" + vectorType + " M=" + m + " N=" + n + " cell=" + i + "," + j,
                                expected.get(i, j), actual.get(i, j), 2e-4f * Math.max(1f, Math.abs(expected.get(i, j))));
                        }
                    }
                }
            }
        }
    }

    @Test
    public void handlesUnalignedColumnOffsetsAndPartialBlocks() {
        Random random = new Random(901);
        int m = 3, n = 5, aOffset = 3, bOffset = 5, k = 61;
        FloatBufferTensor a = randomTensor(random, m, 96);
        Q8ByteBufferTensor b = new Q8ByteBufferTensor(randomTensor(random, n, 96));
        FloatBufferTensor expected = new FloatBufferTensor(m, n);
        FloatBufferTensor actual = new FloatBufferTensor(m, n);
        new ScalarReferenceOperations().batchDotProduct(expected, a, b, aOffset, bOffset, k);
        for (MachineSpec.Type vectorType : new MachineSpec.Type[] {
            MachineSpec.Type.AVX_512, MachineSpec.Type.AVX_256, MachineSpec.Type.ARM_128
        }) {
            new PanamaTensorOperations(vectorType).batchDotProduct(actual, a, b, aOffset, bOffset, k);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    Assert.assertEquals("vector=" + vectorType + " cell=" + i + "," + j,
                        expected.get(i, j), actual.get(i, j), 2e-4f * Math.max(1f, Math.abs(expected.get(i, j))));
                }
            }
        }
    }

    private static FloatBufferTensor randomTensor(Random random, int rows, int columns) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) tensor.set(random.nextFloat() * 2f - 1f, i, j);
        }
        return tensor;
    }
}
