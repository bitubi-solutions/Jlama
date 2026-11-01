package ao.bitubi.inference4j.model;

import org.junit.Test;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.Assert.assertTrue;

public class GgufSmokeGoldenParityTest {
    @Test public void quantizedResponsesMatchF16() throws Exception {
        String f16=Files.readString(Path.of("src/test/resources/smoke-golden-gguf-f16.txt"),StandardCharsets.UTF_8);
        String q8=Files.readString(Path.of("src/test/resources/smoke-golden-gguf-q8_0.txt"),StandardCharsets.UTF_8);
        String q4=Files.readString(Path.of("src/test/resources/smoke-golden-gguf-q4_0.txt"),StandardCharsets.UTF_8);
        int q8Prefix=commonPrefix(f16,q8), q4Prefix=commonPrefix(f16,q4);
        System.out.printf("GGUF goldens: F16/Q8_0 prefix=%d chars; F16/Q4_0 prefix=%d chars; lengths=%d/%d/%d%n",
            q8Prefix,q4Prefix,f16.length(),q8.length(),q4.length());
        assertTrue("Q8_0 diverge de F16: prefixo="+q8Prefix, f16.equals(q8));
    }

    private static int commonPrefix(String a, String b) {
        int i=0, limit=Math.min(a.length(),b.length());
        while (i<limit && a.charAt(i)==b.charAt(i)) i++;
        return i;
    }
}
