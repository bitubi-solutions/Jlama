/*
 * Copyright 2026 Bitubi Solutions
 * Licenciado sob Apache License 2.0.
 */
package ao.bitubi.inference4j.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Leitor GGUF v3 little-endian. Os metadados sao lidos por canal; cada tensor e mapeado separadamente. */
public final class GgufFile {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int VERSION = 3;
    private final Path path;
    private final long fileSize;
    private final long dataOffset;
    private final int alignment;
    private final Map<String, Object> metadata;
    private final List<GgufTensorInfo> tensors;
    private final Map<String, GgufTensorInfo> tensorsByName;

    public GgufFile(Path path) throws IOException {
        this.path = path;
        try (FileChannel channel = FileChannel.open(path)) {
            this.fileSize = channel.size();
            Reader reader = new Reader(channel);
            int magic = reader.readInt();
            if (magic != GGUF_MAGIC) throw new IOException("Magic GGUF invalido: " + Integer.toHexString(magic));
            int version = reader.readInt();
            if (version != VERSION) throw new IOException("Versao GGUF nao suportada: " + version + "; esperada: " + VERSION);
            long tensorCount = checkedCount(reader.readLong(), "tensores");
            long metadataCount = checkedCount(reader.readLong(), "metadados");
            Map<String, Object> parsedMetadata = new HashMap<>();
            for (long i = 0; i < metadataCount; i++) {
                String key = reader.readString();
                Object value = reader.readValue(reader.readInt(), 0);
                if (parsedMetadata.putIfAbsent(key, value) != null) throw new IOException("Chave GGUF duplicada: " + key);
            }
            Object alignmentValue = parsedMetadata.get("general.alignment");
            this.alignment = alignmentValue == null ? 32 : asInt(alignmentValue, "general.alignment");
            if (alignment < 8 || (alignment & (alignment - 1)) != 0) throw new IOException("Alinhamento GGUF invalido: " + alignment);
            this.metadata = Collections.unmodifiableMap(parsedMetadata);
            if (tensorCount > (fileSize - reader.position()) / 24) throw new IOException("Contagem de tensores GGUF inconsistente: " + tensorCount);
            List<RawTensor> rawTensors = new ArrayList<>((int) tensorCount);
            for (int i = 0; i < tensorCount; i++) {
                String name = reader.readString();
                int dimensionCount = reader.readInt();
                if (dimensionCount < 1 || dimensionCount > 4) throw new IOException("Numero de dimensoes GGUF invalido em " + name + ": " + dimensionCount);
                long[] dims = new long[dimensionCount];
                for (int d = 0; d < dimensionCount; d++) dims[d] = reader.readLong();
                int typeId = reader.readInt();
                long relativeOffset = reader.readLong();
                if (relativeOffset < 0 || relativeOffset % alignment != 0) throw new IOException("Offset/alinhamento invalido em " + name + ": " + relativeOffset);
                try { rawTensors.add(new RawTensor(name, GgufType.fromId(typeId), dims, relativeOffset)); }
                catch (IllegalArgumentException e) { throw new IOException("Tensor " + name + ": " + e.getMessage(), e); }
            }
            this.dataOffset = alignUp(reader.position(), alignment);
            if (dataOffset > fileSize) throw new IOException("Cabecalho GGUF truncado antes dos dados dos tensores");
            List<GgufTensorInfo> parsedTensors = new ArrayList<>((int) tensorCount);
            Map<String, GgufTensorInfo> byName = new HashMap<>();
            for (RawTensor raw : rawTensors) {
                GgufTensorInfo info;
                try { info = new GgufTensorInfo(raw.name, raw.type, raw.dims, raw.offset, dataOffset); }
                catch (ArithmeticException | IllegalArgumentException e) { throw new IOException("Descritor de tensor GGUF invalido: " + raw.name, e); }
                if (info.getOffset() > fileSize || info.getByteSize() > fileSize - info.getOffset()) {
                    throw new IOException("Dados truncados para tensor GGUF " + raw.name);
                }
                if (byName.putIfAbsent(raw.name, info) != null) throw new IOException("Nome de tensor GGUF duplicado: " + raw.name);
                parsedTensors.add(info);
            }
            this.tensors = Collections.unmodifiableList(parsedTensors);
            this.tensorsByName = Collections.unmodifiableMap(byName);
        } catch (java.nio.BufferUnderflowException | IndexOutOfBoundsException e) {
            throw new IOException("Cabecalho GGUF truncado ou malformado", e);
        } catch (ArithmeticException e) {
            throw new IOException("Tamanho/offset GGUF excede os limites suportados", e);
        }
    }

    private static long checkedCount(long value, String what) throws IOException {
        if (value < 0 || value > Integer.MAX_VALUE) throw new IOException("Contagem de " + what + " GGUF invalida: " + value);
        return value;
    }

    private static int asInt(Object value, String key) throws IOException {
        if (!(value instanceof Number number) || number.longValue() < 0 || number.longValue() > Integer.MAX_VALUE) {
            throw new IOException("Valor " + key + " invalido: " + value);
        }
        return number.intValue();
    }

    private static long alignUp(long value, int alignment) {
        return Math.addExact(value, alignment - 1) & -((long) alignment);
    }

    public int getAlignment() { return alignment; }
    public long getDataOffset() { return dataOffset; }
    public Map<String, Object> getMetadata() { return metadata; }
    public List<GgufTensorInfo> getTensors() { return tensors; }
    public GgufTensorInfo getTensor(String name) { return tensorsByName.get(name); }

    /** Mapeia apenas o intervalo deste tensor; o chamador deve terminar de consumir o buffer rapidamente. */
    public MappedByteBuffer mapTensor(GgufTensorInfo tensor) throws IOException {
        Objects.requireNonNull(tensor, "tensor");
        if (tensorsByName.get(tensor.getName()) != tensor) throw new IllegalArgumentException("Tensor nao pertence a este ficheiro: " + tensor.getName());
        long size = tensor.getByteSize();
        if (size > Integer.MAX_VALUE) throw new IOException("Tensor excede o limite de um mapeamento Java: " + tensor.getName() + " (" + size + " bytes)");
        try (FileChannel channel = FileChannel.open(path)) {
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, tensor.getOffset(), size);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            return mapped;
        }
    }

    private record RawTensor(String name, GgufType type, long[] dims, long offset) {}

    private static final class Reader {
        private final FileChannel channel;
        private final ByteBuffer buffer = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN);
        private long position;
        Reader(FileChannel channel) { this.channel = channel; buffer.limit(0); }
        long position() { return position; }
        private void require(int count) throws IOException {
            while (buffer.remaining() < count) {
                buffer.compact();
                int read = channel.read(buffer);
                if (read < 0) throw new IOException("Fim prematuro do ficheiro GGUF na posicao " + position);
                if (read == 0) continue;
                buffer.flip();
            }
        }
        int readInt() throws IOException { require(4); position += 4; return buffer.getInt(); }
        long readLong() throws IOException { require(8); position += 8; return buffer.getLong(); }
        short readShort() throws IOException { require(2); position += 2; return buffer.getShort(); }
        int readByte() throws IOException { require(1); position++; return buffer.get() & 0xff; }
        int readSignedByte() throws IOException { require(1); position++; return buffer.get(); }
        byte[] readBytes(int length) throws IOException {
            if (length < 0) throw new IOException("Comprimento GGUF negativo: " + length);
            byte[] bytes = new byte[length];
            int offset = 0;
            while (offset < length) {
                if (!buffer.hasRemaining()) { buffer.clear(); int read = channel.read(buffer); if (read < 0) throw new IOException("Fim prematuro de string GGUF"); buffer.flip(); }
                int count = Math.min(buffer.remaining(), length - offset);
                buffer.get(bytes, offset, count); offset += count; position += count;
            }
            return bytes;
        }
        String readString() throws IOException {
            long length = readLong();
            if (length < 0 || length > Integer.MAX_VALUE) throw new IOException("String GGUF demasiado longa: " + length);
            return new String(readBytes((int) length), StandardCharsets.UTF_8);
        }
        Object readValue(int type, int depth) throws IOException {
            if (depth > 8) throw new IOException("Arrays GGUF aninhados em excesso");
            return switch (type) {
                case 0 -> readByte();
                case 1 -> readSignedByte();
                case 2 -> Short.toUnsignedInt(readShort());
                case 3 -> readShort();
                case 4 -> Integer.toUnsignedLong(readInt());
                case 5 -> readInt();
                case 6 -> Float.intBitsToFloat(readInt());
                case 7 -> { int v = readByte(); if (v > 1) throw new IOException("Booleano GGUF invalido: " + v); yield v != 0; }
                case 8 -> readString();
                case 9 -> {
                    int elementType = readInt(); long count = readLong();
                    if (count < 0 || count > Integer.MAX_VALUE) throw new IOException("Array GGUF demasiado grande: " + count);
                    List<Object> values = new ArrayList<>((int) count);
                    for (int i = 0; i < count; i++) values.add(readValue(elementType, depth + 1));
                    yield Collections.unmodifiableList(values);
                }
                case 10, 11 -> readLong();
                case 12 -> Double.longBitsToDouble(readLong());
                default -> throw new IOException("Tipo de metadado GGUF desconhecido: " + type);
            };
        }
    }
}
