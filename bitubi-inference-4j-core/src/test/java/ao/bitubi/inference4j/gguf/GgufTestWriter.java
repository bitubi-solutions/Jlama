package ao.bitubi.inference4j.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class GgufTestWriter {
    private final int alignment;
    private final Map<String, Metadata> metadata = new LinkedHashMap<>();
    private final List<Tensor> tensors = new ArrayList<>();

    GgufTestWriter(int alignment) { this.alignment = alignment; }

    GgufTestWriter addMetadata(String key, int type, Object value) { metadata.put(key, new Metadata(type, value)); return this; }

    GgufTestWriter addTensor(String name, GgufType type, long[] dims, ByteBuffer data) {
        byte[] bytes = new byte[data.remaining()];
        data.duplicate().get(bytes);
        tensors.add(new Tensor(name, type, dims.clone(), bytes));
        return this;
    }

    Path write(Path path) throws IOException {
        ByteBuffer out = ByteBuffer.allocate(1024 * 1024).order(ByteOrder.LITTLE_ENDIAN);
        out.putInt(0x46554747).putInt(3).putLong(tensors.size()).putLong(metadata.size());
        for (Map.Entry<String, Metadata> entry : metadata.entrySet()) {
            putString(out, entry.getKey());
            out.putInt(entry.getValue().type);
            putValue(out, entry.getValue().type, entry.getValue().value);
        }
        long[] offsets = new long[tensors.size()];
        long next = 0;
        for (int i = 0; i < tensors.size(); i++) {
            Tensor tensor = tensors.get(i);
            offsets[i] = alignUp(next, alignment);
            next = offsets[i] + tensor.data.length;
        }
        for (int i = 0; i < tensors.size(); i++) {
            Tensor tensor = tensors.get(i);
            putString(out, tensor.name);
            out.putInt(tensor.dims.length);
            for (long dim : tensor.dims) out.putLong(dim);
            out.putInt(tensor.type.getId()).putLong(offsets[i]);
        }
        long dataStart = alignUp(out.position(), alignment);
        while (out.position() < dataStart) out.put((byte) 0);
        for (int i = 0; i < tensors.size(); i++) {
            long target = dataStart + offsets[i];
            while (out.position() < target) out.put((byte) 0);
            out.put(tensors.get(i).data);
        }
        out.flip();
        byte[] bytes = new byte[out.remaining()];
        out.get(bytes);
        Files.write(path, bytes);
        return path;
    }

    private static void putValue(ByteBuffer out, int type, Object value) {
        switch (type) {
            case 0, 1, 7 -> out.put(((Number) value).byteValue());
            case 2, 3 -> out.putShort(((Number) value).shortValue());
            case 4, 5 -> out.putInt(((Number) value).intValue());
            case 6 -> out.putFloat(((Number) value).floatValue());
            case 8 -> putString(out, (String) value);
            case 9 -> {
                ArrayValue array = (ArrayValue) value;
                out.putInt(array.elementType).putLong(array.values.size());
                for (Object element : array.values) putValue(out, array.elementType, element);
            }
            case 10, 11 -> out.putLong(((Number) value).longValue());
            case 12 -> out.putDouble(((Number) value).doubleValue());
            default -> throw new IllegalArgumentException("tipo de metadado " + type);
        }
    }

    static ArrayValue array(int elementType, Object... values) { return new ArrayValue(elementType, List.of(values)); }
    private static void putString(ByteBuffer out, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        out.putLong(bytes.length).put(bytes);
    }
    private static long alignUp(long value, int alignment) { return (value + alignment - 1) & -((long) alignment); }

    record ArrayValue(int elementType, List<Object> values) {}
    private record Metadata(int type, Object value) {}
    private record Tensor(String name, GgufType type, long[] dims, byte[] data) {}
}
