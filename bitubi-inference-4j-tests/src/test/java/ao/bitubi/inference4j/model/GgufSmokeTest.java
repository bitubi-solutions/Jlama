package ao.bitubi.inference4j.model;

import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.safetensors.prompt.PromptContext;
import ao.bitubi.inference4j.model.functions.Generator;
import org.junit.Assume;
import org.junit.Test;
import java.io.File;
import java.util.UUID;
import static org.junit.Assert.assertTrue;

public class GgufSmokeTest {
    @Test public void generatesExpectedAnswerFromQwenGguf() {
        String path=System.getenv("GGUF_MODEL_PATH");
        Assume.assumeTrue(path != null && "1".equals(System.getenv("GGUF_SMOKE")));
        AbstractModel model=ModelSupport.loadModel(new File(path),DType.F32,DType.I8);
        PromptContext prompt=model.promptSupport().orElseThrow().builder().addUserMessage("What is the capital of France?").build();
        Generator.Response response=model.generate(UUID.randomUUID(),prompt,0f,96,(s,f)->{});
        assertTrue(response.responseText,response.responseText.toLowerCase().contains("paris"));
    }
}
