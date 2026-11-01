/*
 * Copyright 2026 Bitubi Solutions
 * Licenciado sob Apache License 2.0.
 */
package ao.bitubi.inference4j.gguf;

import java.util.Arrays;

/** Descritor de tensor; as dimensoes mantem a ordem GGML e offset e absoluto no ficheiro. */
public final class GgufTensorInfo {
    private final String name;
    private final GgufType type;
    private final long[] dims;
    private final long relativeOffset;
    private final long offset;
    private final long byteSize;

    public GgufTensorInfo(String name, GgufType type, long[] dims, long relativeOffset, long dataOffset) {
        if (name == null || type == null || dims == null || dims.length == 0 || dims.length > 4) {
            throw new IllegalArgumentException("Descritor de tensor GGUF invalido");
        }
        this.name = name;
        this.type = type;
        this.dims = dims.clone();
        this.relativeOffset = relativeOffset;
        this.offset = Math.addExact(dataOffset, relativeOffset);
        long elements = 1;
        for (long dim : dims) {
            if (dim <= 0) throw new IllegalArgumentException("Dimensao GGUF invalida em " + name + ": " + dim);
            elements = Math.multiplyExact(elements, dim);
        }
        if (elements % type.getElementsPerBlock() != 0) {
            throw new IllegalArgumentException("Numero de elementos nao e multiplo do bloco " + type + " em " + name);
        }
        this.byteSize = Math.multiplyExact(elements / type.getElementsPerBlock(), type.getBytesPerBlock());
    }

    public String getName() { return name; }
    public GgufType getType() { return type; }
    public long[] getDims() { return dims.clone(); }
    public int getDimensions() { return dims.length; }
    public long getRelativeOffset() { return relativeOffset; }
    public long getOffset() { return offset; }
    public long getByteSize() { return byteSize; }

    @Override
    public String toString() {
        return "GgufTensorInfo{name='" + name + "', type=" + type + ", dims=" + Arrays.toString(dims) + ", offset=" + offset + '}';
    }
}
