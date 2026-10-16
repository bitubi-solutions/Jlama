/*
 * Copyright 2026 Bitubi Solutions
 *
 * Este ficheiro é parte do projeto bitubi-inference-4j.
 * Licenciado sob Apache License 2.0.
 */
package ao.bitubi.inference4j.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utilitário de teste para gerar arquivos GGUF mínimos para validação do leitor.
 */
public final class GgufTestWriter {
    private static final int GGUF_MAGIC = 0x47475546; // 'GGUF' little-endian
    private static final int VERSION = 3;
    private static final int ALIGNMENT = 32;

    private final ByteBuffer buffer;
    private final List<GgufTensorInfo> tensors = new ArrayList<>();
    private final Map<String, Object> metadata = new HashMap<>();
    private long dataOffset;

    public GgufTestWriter(int capacity) {
        this.buffer = ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Escreve o cabeçalho GGUF.
     */
    private void writeHeader() {
        buffer.putInt(GGUF_MAGIC);
        buffer.putInt(VERSION);
        buffer.putLong(tensors.size());
        buffer.putLong(metadata.size());
        buffer.putInt(ALIGNMENT);
    }

    /**
     * Adiciona um metadado de tipo inteiro.
     */
    public void addMetadata(String key, int value) {
        metadata.put(key, value);
    }

    /**
     * Adiciona um metadado de tipo string.
     */
    public void addMetadata(String key, String value) {
        metadata.put(key, value);
    }

    /**
     * Adiciona um metadado de tipo float.
     */
    public void addMetadata(String key, float value) {
        metadata.put(key, value);
    }

    /**
     * Adiciona um metadado de tipo booleano.
     */
    public void addMetadata(String key, boolean value) {
        metadata.put(key, value);
    }

    /**
     * Adiciona um metadado de tipo array de inteiros.
     */
    public void addMetadata(String key, int[] value) {
        metadata.put(key, value);
    }

    /**
     * Escreve um metadado no buffer.
     */
    private void writeMetadata(String key, Object value) {
        // Chave
        byte[] keyBytes = key.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        buffer.putLong(keyBytes.length);
        buffer.put(keyBytes);

        if (value instanceof String) {
            buffer.putInt(8); // STRING
            byte[] valBytes = ((String) value).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            buffer.putLong(valBytes.length);
            buffer.put(valBytes);
        } else if (value instanceof Integer) {
            buffer.putInt(5); // INT32
            buffer.putInt((Integer) value);
        } else if (value instanceof Float) {
            buffer.putInt(6); // FLOAT32
            buffer.putFloat((Float) value);
        } else if (value instanceof Boolean) {
            buffer.putInt(7); // BOOL
            buffer.put((byte) ((Boolean) value ? 1 : 0));
        } else if (value instanceof int[]) {
            buffer.putInt(9); // ARRAY
            buffer.putInt(5); // INT32
            int[] arr = (int[]) value;
            buffer.putLong(arr.length);
            for (int v : arr) buffer.putInt(v);
        } else {
            throw new UnsupportedOperationException("Tipo de metadado não suportado: " + value.getClass());
        }
    }

    /**
     * Adiciona um tensor ao arquivo.
     */
    public void addTensor(String name, GgufType type, long[] dims, ByteBuffer data) {
        long offset = dataOffset;
        // Alinhar offset
        int tensorSize = calculateTensorSize(type, dims);
        long alignedOffset = alignUp(offset, ALIGNMENT);
        dataOffset = alignedOffset + tensorSize;

        tensors.add(new GgufTensorInfo(name, type, dims, alignedOffset));
    }

    /**
     * Calcula o tamanho em bytes de um tensor.
     */
    private int calculateTensorSize(GgufType type, long[] dims) {
        long elements = 1;
        for (long d : dims) elements *= d;
        int blocks = (int) ((elements + type.getElementsPerBlock() - 1) / type.getElementsPerBlock());
        return blocks * type.getBytesPerBlock();
    }

    /**
     * Escreve o arquivo GGUF completo.
     */
    public ByteBuffer write() {
        // Posição inicial
        buffer.clear();

        // Cabeçalho
        writeHeader();

        // Metadados
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            writeMetadata(entry.getKey(), entry.getValue());
        }

        // Alinhamento até o início dos dados
        dataOffset = alignUp(buffer.position(), ALIGNMENT);
        // Preencher com zeros
        while (buffer.position() < dataOffset) {
            buffer.put((byte) 0);
        }

        // Dados dos tensores
        for (GgufTensorInfo tensor : tensors) {
            // Preencher com zeros até o offset absoluto
            while (buffer.position() < dataOffset + tensor.getOffset()) {
                buffer.put((byte) 0);
            }
            // Dados do tensor (já devem ter sido escritos anteriormente)
            // Aqui apenas marcamos a posição
        }

        buffer.flip();
        return buffer;
    }

    /**
     * Alinha um valor para cima até o múltiplo mais próximo do alinhamento.
     */
    private static long alignUp(long value, int alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    /**
     * Escreve o arquivo para um ficheiro no disco.
     */
    public void writeToFile(Path path) throws IOException {
        ByteBuffer buf = write();
        java.io.FileOutputStream fos = new java.io.FileOutputStream(path.toFile());
        fos.getChannel().write(buf);
        fos.close();
    }
}