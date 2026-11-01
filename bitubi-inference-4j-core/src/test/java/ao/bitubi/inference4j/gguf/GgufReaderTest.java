package ao.bitubi.inference4j.gguf;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.*;

public class GgufReaderTest {
    @Rule public final TemporaryFolder temporary = new TemporaryFolder();

    @Test
    public void readsMetadataTensorDescriptorsAndMapsIndividualTensors() throws Exception {
        Path path = temporary.newFile("model.gguf").toPath();
        ByteBuffer f32 = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 1; i <= 6; i++) f32.putFloat(i);
        ByteBuffer q8 = ByteBuffer.allocate(34).order(ByteOrder.LITTLE_ENDIAN);
        q8.putShort((short) 0x3c00);
        for (int i = 0; i < 32; i++) q8.put((byte) i);

        new GgufTestWriter(64)
            .addMetadata("general.architecture", 8, "qwen2")
            .addMetadata("general.alignment", 4, 64)
            .addMetadata("test.signed", 5, -7)
            .addMetadata("test.float", 12, 1.25d)
            .addMetadata("tokenizer.ggml.tokens", 9, GgufTestWriter.array(8, "a", "b", "c"))
            .addMetadata("test.ids", 9, GgufTestWriter.array(4, 0, 1, 2))
            .addTensor("matrix", GgufType.F32, new long[] { 2, 3 }, f32.flip())
            .addTensor("quant", GgufType.Q8_0, new long[] { 32 }, q8.flip())
            .write(path);

        GgufFile file = new GgufFile(path);
        assertEquals(64, file.getAlignment());
        assertEquals("qwen2", file.getMetadata().get("general.architecture"));
        assertEquals(-7, file.getMetadata().get("test.signed"));
        assertEquals(1.25d, (Double) file.getMetadata().get("test.float"), 0d);
        assertEquals(List.of("a", "b", "c"), file.getMetadata().get("tokenizer.ggml.tokens"));
        assertEquals(List.of(0L, 1L, 2L), file.getMetadata().get("test.ids"));
        assertEquals(2, file.getTensors().size());

        GgufTensorInfo matrix = file.getTensor("matrix");
        assertArrayEquals(new long[] { 2, 3 }, matrix.getDims());
        assertEquals(0, matrix.getRelativeOffset());
        assertEquals(file.getDataOffset(), matrix.getOffset());
        assertEquals(24, matrix.getByteSize());
        ByteBuffer mapped = file.mapTensor(matrix);
        for (int i = 0; i < 6; i++) assertEquals(i + 1, mapped.getFloat(), 0f);

        GgufTensorInfo quant = file.getTensor("quant");
        assertArrayEquals(new long[] { 32 }, quant.getDims());
        assertEquals(64, quant.getRelativeOffset());
        assertEquals(34, quant.getByteSize());
        assertEquals(0x3c00, Short.toUnsignedInt(file.mapTensor(quant).getShort()));
    }

    @Test
    public void usesDefaultAlignmentAndRejectsBadHeaders() throws Exception {
        Path valid = temporary.newFile("empty.gguf").toPath();
        new GgufTestWriter(32).write(valid);
        assertEquals(32, new GgufFile(valid).getAlignment());

        Path badMagic = temporary.newFile("bad-magic.gguf").toPath();
        Files.write(badMagic, new byte[] { 0, 0, 0, 0 });
        assertIOException(badMagic, "Magic");

        Path badVersion = temporary.newFile("bad-version.gguf").toPath();
        ByteBuffer header = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(0x46554747).putInt(1).putLong(0).putLong(0);
        Files.write(badVersion, header.array());
        assertIOException(badVersion, "Versao");

        Path truncated = temporary.newFile("truncated.gguf").toPath();
        Files.write(truncated, new byte[] { 0x47, 0x47, 0x55, 0x46 });
        assertIOException(truncated, "prematuro");
    }

    private static void assertIOException(Path path, String messagePart) throws Exception {
        try { new GgufFile(path); fail("Esperava IOException"); }
        catch (IOException e) { assertTrue(e.getMessage(), e.getMessage().contains(messagePart)); }
    }
}
