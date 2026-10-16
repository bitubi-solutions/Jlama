package ao.bitubi.inference4j.gguf;

import ao.bitubi.inference4j.safetensors.TensorInfo;
import ao.bitubi.inference4j.tensor.AbstractTensor;
import ao.bitubi.inference4j.tensor.Q4ByteBufferTensor;
import ao.bitubi.inference4j.tensor.Q8ByteBufferTensor;
import org.junit.Assume;
import org.junit.Test;
import java.nio.file.Path;
import java.util.Map;
import static org.junit.Assert.*;

public class GgufWeightParityTest {
    @Test public void quantizedWeightsStayWithinHalfStepOfF16() throws Exception {
        String quantized=System.getenv("GGUF_QUANTIZED_MODEL_PATH"), reference=System.getenv("GGUF_FP16_MODEL_PATH");
        Assume.assumeTrue(quantized != null && reference != null);
        GgufWeightLoader q=new GgufWeightLoader(new GgufFile(Path.of(quantized)));
        GgufWeightLoader f=new GgufWeightLoader(new GgufFile(Path.of(reference)));
        for (Map.Entry<String,TensorInfo> entry : q.tensorInfoMap().entrySet()) {
            String name=entry.getKey();
            if (!f.isWeightPresent(name)) continue;
            AbstractTensor actual=q.load(name), expected=f.load(name);
            assertEquals(name,expected.shape(),actual.shape());
            int rows=actual.shape().first(), cols=actual.shape().last();
            for(int r=0;r<rows;r++) for(int c=0;c<cols;c++) {
                float step=actual instanceof Q4ByteBufferTensor q4 ? Math.abs(q4.getFactorForIndex(r,c))
                    : actual instanceof Q8ByteBufferTensor q8 ? Math.abs(q8.getFactorForIndex(r,c)) : 0f;
                float tolerance=step == 0f ? Math.abs(expected.get(r,c))*0.001f + 1e-6f : step*0.5001f + 1e-6f;
                assertTrue(name+"["+r+","+c+"]",Math.abs(actual.get(r,c)-expected.get(r,c)) <= tolerance);
            }
        }
    }
}
