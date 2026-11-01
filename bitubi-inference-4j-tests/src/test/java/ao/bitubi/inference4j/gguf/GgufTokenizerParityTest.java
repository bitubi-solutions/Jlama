package ao.bitubi.inference4j.gguf;

import ao.bitubi.inference4j.model.gpt2.GPT2Tokenizer;
import ao.bitubi.inference4j.safetensors.SafeTensorSupport;
import org.junit.Assume;
import org.junit.Test;
import java.nio.file.Path;
import static org.junit.Assert.assertArrayEquals;

public class GgufTokenizerParityTest {
    @Test public void matchesTokenizerJson() throws Exception {
        String ggufPath=System.getenv("GGUF_MODEL_PATH"), tokenizerPath=System.getenv("GGUF_TOKENIZER_DIR");
        Assume.assumeTrue(ggufPath != null && tokenizerPath != null);
        var gguf=new GgufFile(Path.of(ggufPath));
        var actual=new GPT2Tokenizer(GgufTokenizerFactory.create(gguf.getMetadata()));
        var expected=new GPT2Tokenizer(Path.of(tokenizerPath));
        for (String text : new String[]{"Hello, world!", "capital of France?", " Olá, Luanda!", "emoji 😀 中文", "x = 12345;\nnext line", "<|im_start|>assistant\n"}) {
            long[] expectedIds=expected.encode(text), actualIds=actual.encode(text);
            assertArrayEquals(text+" expected="+java.util.Arrays.toString(expectedIds)+" actual="+java.util.Arrays.toString(actualIds),expectedIds,actualIds);
        }
    }
}
