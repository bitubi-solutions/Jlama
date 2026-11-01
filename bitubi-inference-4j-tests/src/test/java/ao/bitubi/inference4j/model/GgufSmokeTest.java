package ao.bitubi.inference4j.model;

import ao.bitubi.inference4j.model.AbstractModel;
import ao.bitubi.inference4j.model.ModelSupport;
import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.safetensors.prompt.PromptContext;
import ao.bitubi.inference4j.model.functions.Generator;
import org.junit.Assume;
import org.junit.Test;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class GgufSmokeTest {
    @Test public void generatesExpectedAnswerFromQwenGguf() {
        String path=System.getenv("GGUF_MODEL_PATH");
        Assume.assumeTrue(path != null && "1".equals(System.getenv("GGUF_SMOKE")));
        String variant=System.getenv("GGUF_VARIANT");
        assertTrue("GGUF_VARIANT must be f16, q8_0 or q4_0",variant != null && java.util.Set.of("f16","q8_0","q4_0").contains(variant));
        AbstractModel model=ModelSupport.loadModel(new File(path),DType.F32,DType.I8);
        PromptContext prompt=model.promptSupport().orElseThrow().builder().addUserMessage("What is the capital of France?").build();
        Generator.Response response=model.generate(UUID.randomUUID(),prompt,0f,96,(s,f)->{});
        assertTrue(response.responseText,response.responseText.toLowerCase().contains("paris"));
        Path golden=Path.of("src/test/resources/smoke-golden-gguf-"+variant+".txt");
        if ("1".equals(System.getenv("GGUF_SMOKE_RECORD"))) {
            try { Files.writeString(golden,response.responseText,StandardCharsets.UTF_8); }
            catch (java.io.IOException e) { throw new IllegalStateException("Falha ao gravar golden GGUF " + golden,e); }
        } else {
            try {
                assertEquals("resposta diferente do golden " + golden,Files.readString(golden,StandardCharsets.UTF_8),response.responseText);
            } catch (java.io.IOException e) { throw new IllegalStateException("Falha ao ler golden GGUF " + golden,e); }
        }
    }
}
