/*
 * Copyright 2026 Bitubi Solutions
 *
 * Este ficheiro é parte do projeto bitubi-inference-4j.
 * Licenciado sob Apache License 2.0.
 */
package ao.bitubi.inference4j.gguf;

import ao.bitubi.inference4j.util.UnsafeDirectByteBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Representa um arquivo GGUF aberto com mapeamento de memória.
 * Valida magic e versão, lê metadados e informações dos tensores.
 */
public final class GgufFile {
    private static final int GGUF_MAGIC = 0x47475546; // 'GGUF' in ASCII, little-endian
    private static final int VERSION = 3;

    private final MappedByteBuffer buffer;
    private final long dataOffset;
    private final int alignment;
    private final long metadataKvCount;
    private final Map<String, String> metadata;
    private final List<GgufTensorInfo> tensors;

    /**
     * Abre um arquivo GGUF e lê seu cabeçalho e informações dos tensores.
     *
     * @param path caminho para o arquivo GGUF
     * @throws IOException se ocorrer erro ao abrir ou ler o arquivo
     */
    public GgufFile(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path)) {
            this.buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            this.buffer.order(ByteOrder.LITTLE_ENDIAN);

            // Validar magic
            int magic = buffer.getInt();
            if (magic != GGUF_MAGIC) {
                throw new IOException("Magic GGUF inválido: " + Integer.toHexString(magic) + ", esperado: " + Integer.toHexString(GGUF_MAGIC));
            }

            // Validar versão
            int version = buffer.getInt();
            if (version != VERSION) {
                throw new IOException("Versão GGUF inválida: " + version + ", esperada: " + VERSION);
            }

            // Ler contagem de tensores e pares chave-valor de metadados
            long tensorCount = buffer.getLong();
            long metadataKvCount = buffer.getLong();
            this.metadataKvCount = metadataKvCount;

            // Ler alinhamento (general.alignment)
            int alignment = buffer.getInt();
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
                throw new IOException("Alinhamento inválido: " + alignment + ". Deve ser potência de 2 positiva.");
            }
            this.alignment = alignment;

            // Ler metadados para determinar a posição atual do buffer
            this.metadata = new HashMap<>();
            this.tensors = new ArrayList<>();

            long metadataStart = buffer.position();
            long metadataEnd = skipMetadata(buffer, metadataKvCount);

            // Retornar ao início dos metadados para lê-los
            if (metadataStart > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Posição de metadados GGUF muito grande: " + metadataStart);
            }
            buffer.position((int) metadataStart);
            readMetadata(metadataKvCount);

            // Ler informações dos tensores
            for (long i = 0; i < tensorCount; i++) {
                tensors.add(readTensorInfo());
            }

            // Calcular o início dos dados (depois das informações dos tensores, alinhado)
            this.dataOffset = alignUp(buffer.position(), alignment);
        }
    }

    /**
     * Pula os metadados no buffer e retorna a posição final.
     */
    private long skipMetadata(ByteBuffer buffer, long count) {
        long pos = buffer.position();
        for (long i = 0; i < count; i++) {
            // Chave: len (uint64) + bytes UTF-8
            long keyLen = buffer.getLong();
            if (keyLen > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Comprimento de chave GGUF muito grande: " + keyLen);
            }
            buffer.position(buffer.position() + (int) keyLen);

            // Valor: primeiro o tipo (uint32)
            int type = buffer.getInt();
            switch (type) {
                case 0: // UINT8
                case 1: // INT8
                case 2: // UINT16
                case 3: // INT16
                case 4: // UINT32
                case 5: // INT32
                    buffer.position(buffer.position() + 4);
                    break;
                case 6: // FLOAT32
                    buffer.position(buffer.position() + 4);
                    break;
                case 7: // BOOL
                    buffer.position(buffer.position() + 1);
                    break;
                case 8: // STRING
                    long stringLen = buffer.getLong();
                    if (stringLen > Integer.MAX_VALUE) {
                        throw new IllegalArgumentException("Comprimento de string GGUF muito grande: " + stringLen);
                    }
                    buffer.position(buffer.position() + (int) stringLen);
                    break;
                case 9: // ARRAY
                    int arrayType = buffer.getInt();
                    long arrayLen = buffer.getLong();
                    int elementSize;
                    switch (arrayType) {
                        case 0: // UINT8
                        case 1: // INT8
                        case 2: // UINT16
                        case 3: // INT16
                        case 4: // UINT32
                        case 5: // INT32
                            elementSize = 4;
                            break;
                        case 6: // FLOAT32
                            elementSize = 4;
                            break;
                        case 7: // BOOL
                            elementSize = 1;
                            break;
                        case 8: // STRING
                            // Para arrays de string, precisamos ler cada string
                            for (long j = 0; j < arrayLen; j++) {
                                long strLen = buffer.getLong();
                                if (strLen > Integer.MAX_VALUE) {
                                    throw new IllegalArgumentException("Comprimento de string GGUF muito grande: " + strLen);
                                }
                                buffer.position(buffer.position() + (int) strLen);
                            }
                            elementSize = 0; // já avançamos
                            break;
                        case 9: // ARRAY (aninhado) - não suportado neste parser simples
                        case 10: // UINT64
                        case 11: // INT64
                        case 12: // FLOAT64
                            elementSize = 8;
                            break;
                        default:
                            throw new IllegalArgumentException("Tipo de array GGUF desconhecido: " + arrayType);
                    }
                    if (elementSize > 0) {
                        buffer.position(buffer.position() + (int) (arrayLen * elementSize));
                    }
                    break;
                default:
                    throw new IllegalArgumentException("Tipo de valor GGUF desconhecido: " + type);
            }
        }
        return buffer.position();
    }

    /**
     * Lê os metadados chave-valor do buffer.
     */
    private void readMetadata(long count) {
        for (long i = 0; i < count; i++) {
            long keyLen = buffer.getLong();
            byte[] keyBytes = new byte[(int) keyLen];
            buffer.get(keyBytes);
            String key = new String(keyBytes, java.nio.charset.StandardCharsets.UTF_8);

            int type = buffer.getInt();
            String value;
            switch (type) {
                case 0: // UINT8
                    value = String.valueOf(buffer.get() & 0xFF);
                    break;
                case 1: // INT8
                    value = String.valueOf(buffer.get());
                    break;
                case 2: // UINT16
                    value = String.valueOf(buffer.getShort() & 0xFFFF);
                    break;
                case 3: // INT16
                    value = String.valueOf(buffer.getShort());
                    break;
                case 4: // UINT32
                    value = String.valueOf(buffer.getInt() & 0xFFFFFFFFL);
                    break;
                case 5: // INT32
                    value = String.valueOf(buffer.getInt());
                    break;
                case 6: // FLOAT32
                    value = String.valueOf(buffer.getFloat());
                    break;
                case 7: // BOOL
                    value = String.valueOf(buffer.get() != 0);
                    break;
                case 8: // STRING
                    long stringLen = buffer.getLong();
                    if (stringLen > Integer.MAX_VALUE) {
                        throw new IllegalArgumentException("Comprimento de string GGUF muito grande: " + stringLen);
                    }
                    byte[] valBytes = new byte[(int) stringLen];
                    buffer.get(valBytes);
                    value = new String(valBytes, java.nio.charset.StandardCharsets.UTF_8);
                    break;
                case 9: // ARRAY
                    // Para simplificação, representamos arrays como strings JSON
                    // Implementação completa deveria parsear o array corretamente
                    int arrayType = buffer.getInt();
                    long arrayLen = buffer.getLong();
                    value = parseArray(buffer, arrayType, arrayLen);
                    break;
                default:
                    throw new IllegalArgumentException("Tipo de valor GGUF desconhecido: " + type);
            }
            metadata.put(key, value);
        }
    }

    /**
     * Parseia um array GGUF para representação string (simplificado).
     */
    private String parseArray(ByteBuffer buffer, int arrayType, long arrayLen) {
        StringBuilder sb = new StringBuilder("[");
        switch (arrayType) {
            case 0: // UINT8
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.get() & 0xFF);
                }
                break;
            case 1: // INT8
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.get());
                }
                break;
            case 2: // UINT16
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getShort() & 0xFFFF);
                }
                break;
            case 3: // INT16
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getShort());
                }
                break;
            case 4: // UINT32
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getInt() & 0xFFFFFFFFL);
                }
                break;
            case 5: // INT32
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getInt());
                }
                break;
            case 6: // FLOAT32
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getFloat());
                }
                break;
            case 7: // BOOL
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.get() != 0);
                }
                break;
            case 9: // ARRAY (aninhado)
                // Não implementado totalmente
                sb.append("<nested array>");
                buffer.position((int) (buffer.position() + arrayLen * 4)); // aproximação
                break;
            case 10: // UINT64
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getLong() & 0xFFFFFFFFFFFFFFFFL);
                }
                break;
            case 11: // INT64
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getLong());
                }
                break;
            case 12: // FLOAT64
                for (long i = 0; i < arrayLen; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(buffer.getDouble());
                }
                break;
            default:
                throw new IllegalArgumentException("Tipo de array GGUF desconhecido: " + arrayType);
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Lê as informações de um tensor do buffer.
     */
    private GgufTensorInfo readTensorInfo() {
        // Nome: string (len + bytes)
        long nameLen = buffer.getLong();
        byte[] nameBytes = new byte[(int) nameLen];
        buffer.get(nameBytes);
        String name = new String(nameBytes, java.nio.charset.StandardCharsets.UTF_8);

        // Número de dimensões
        int nDimensions = buffer.getInt();

        // Dimensões
        long[] dims = new long[nDimensions];
        for (int i = 0; i < nDimensions; i++) {
            dims[i] = buffer.getLong();
        }

        // Tipo
        int typeId = buffer.getInt();
        GgufType type = GgufType.fromId(typeId);

        // Offset relativo ao início dos dados
        long offset = buffer.getLong();

        return new GgufTensorInfo(name, type, dims, offset);
    }

    /**
     * Alinha um valor para cima até o múltiplo mais próximo do alinhamento.
     */
    private static long alignUp(long value, int alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    /**
     * Retorna o mapeamento de memória do arquivo.
     */
    public MappedByteBuffer getBuffer() {
        return buffer;
    }

    /**
     * Retorna o offset onde os dados dos tensores começam.
     */
    public long getDataOffset() {
        return dataOffset;
    }

    /**
     * Retorna o alinhamento usado no arquivo.
     */
    public int getAlignment() {
        return alignment;
    }

    /**
     * Retorna os metadados chave-valor lidos do arquivo.
     */
    public Map<String, String> getMetadata() {
        return metadata;
    }

    /**
     * Retorna a lista de informações dos tensores.
     */
    public List<GgufTensorInfo> getTensors() {
        return tensors;
    }

    /**
     * Fecha o mapeamento de memória.
     * Nota: Em Java, o mapeamento é fechado quando o objeto é coletado pelo GC,
     * mas podemos tentar liberar explicitamente se necessário.
     */
    public void close() {
        // Em Java, MappedByteBuffer não tem método close explícito
        // O buffer será liberado quando o objeto for coletado pelo GC
        // Para forçar a liberação, precisaríamos usar Cleaner ou técnicas similares
        // Por simplicidade, deixamos o GC cuidar disso
    }
}
