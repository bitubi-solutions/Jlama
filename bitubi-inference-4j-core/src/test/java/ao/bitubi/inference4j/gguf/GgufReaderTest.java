/*
 * Copyright 2026 Bitubi Solutions
 *
 * Este ficheiro é parte do projeto bitubi-inference-4j.
 * Licenciado sob Apache License 2.0.
 */
package ao.bitubi.inference4j.gguf;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Teste para validar o leitor GGUF com um arquivo mínimo conhecido.
 */
public class GgufReaderTest {

    @Test
    public void testMinimalGgufReader() throws IOException {
        Path tempFile = Path.of("/tmp/test.gguf");
        try (FileChannel channel = FileChannel.open(tempFile, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(1024).order(ByteOrder.LITTLE_ENDIAN);

            // Cabeçalho GGUF
            buffer.putInt(0x47475546); // GGUF magic
            buffer.putInt(3);           // version
            buffer.putLong(1);          // tensorCount
            buffer.putLong(0);          // metadataKvCount (sem metadados)
            buffer.putInt(32);          // alignment

            // Informações do tensor (após o cabeçalho, sem alinhamento)
            // Ordem GGUF: name, n_dimensions, dims[], type, offset
            byte[] nameBytes = "test_tensor".getBytes(StandardCharsets.UTF_8);
            buffer.putLong(nameBytes.length);   // nome length
            buffer.put(nameBytes);              // nome
            buffer.putInt(1);                   // nDimensions
            buffer.putLong(1);                  // dims[0] = 1
            buffer.putInt(0);                   // tipo F32 (id 0)
            buffer.putLong(0);                  // offset relativo ao início dos dados (0 = primeiro tensor)

            // Alinhar até o início dos dados (32-byte alignment)
            while (buffer.position() % 32 != 0) {
                buffer.put((byte) 0);
            }

            // Dados do tensor: 1 float = 1.0f
            buffer.putFloat(1.0f);

            // Escrever no ficheiro
            buffer.flip();
            channel.write(buffer);
            channel.force(true);
        }

        // Ler com o leitor real
        GgufFile file = new GgufFile(tempFile);
        Assert.assertEquals(1, file.getTensors().size());
        GgufTensorInfo tensorInfo = file.getTensors().get(0);
        Assert.assertEquals("test_tensor", tensorInfo.getName());
        Assert.assertEquals(1, tensorInfo.getDimensions());
        Assert.assertEquals(1.0f, tensorInfo.getType().getElementsPerBlock(), 0.001f);
        Assert.assertEquals(0L, tensorInfo.getOffset());
        Assert.assertEquals(96L, file.getDataOffset());
    }

    @Test
    public void testInvalidMagicThrows() throws IOException {
        Path tempFile = Path.of("/tmp/test_invalid.gguf");
        try (FileChannel channel = FileChannel.open(tempFile, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(32).order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(0x12345678); // magic inválido
            buffer.flip();
            channel.write(buffer);
        }

        try {
            new GgufFile(tempFile);
            Assert.fail("Deveria ter lançado IOException para magic inválido");
        } catch (IOException e) {
            Assert.assertTrue(e.getMessage().contains("inválido"));
        }
    }

    @Test
    public void testInvalidVersionThrows() throws IOException {
        Path tempFile = Path.of("/tmp/test_invalid_version.gguf");
        try (FileChannel channel = FileChannel.open(tempFile, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(32).order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(0x47475546); // GGUF magic
            buffer.putInt(1);           // versão inválida (1 em vez de 3)
            buffer.flip();
            channel.write(buffer);
        }

        try {
            new GgufFile(tempFile);
            Assert.fail("Deveria ter lançado IOException para versão inválida");
        } catch (IOException e) {
            Assert.assertTrue(e.getMessage().contains("inválida"));
        }
    }

    @Test
    public void testMetadataReading() throws IOException {
        Path tempFile = Path.of("/tmp/test_metadata.gguf");
        try (FileChannel channel = FileChannel.open(tempFile, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(1024).order(ByteOrder.LITTLE_ENDIAN);

            // Cabeçalho GGUF
            buffer.putInt(0x47475546); // magic
            buffer.putInt(3);          // version
            buffer.putLong(0);         // tensorCount
            buffer.putLong(2);         // metadataKvCount = 2
            buffer.putInt(32);         // alignment

            // Metadado 1: chave string, valor string
            byte[] key1 = "general.architecture".getBytes(StandardCharsets.UTF_8);
            buffer.putLong(key1.length);
            buffer.put(key1);
            buffer.putInt(8); // STRING
            byte[] val1 = "qwen2".getBytes(StandardCharsets.UTF_8);
            buffer.putLong(val1.length);
            buffer.put(val1);

            // Metadado 2: chave string, valor int32
            byte[] key2 = "general.alignment".getBytes(StandardCharsets.UTF_8);
            buffer.putLong(key2.length);
            buffer.put(key2);
            buffer.putInt(5); // INT32
            buffer.putInt(64);

            buffer.flip();
            channel.write(buffer);
        }

        GgufFile file = new GgufFile(tempFile);
        Assert.assertEquals("qwen2", file.getMetadata().get("general.architecture"));
        Assert.assertEquals("64", file.getMetadata().get("general.alignment"));
    }
}