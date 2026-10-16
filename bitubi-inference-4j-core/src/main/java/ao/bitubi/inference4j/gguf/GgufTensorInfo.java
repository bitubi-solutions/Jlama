/*
 * Copyright 2026 Bitubi Solutions
 *
 * Este ficheiro é parte do projeto bitubi-inference-4j.
 * Licenciado sob Apache License 2.0.
 */
package ao.bitubi.inference4j.gguf;

import java.util.Arrays;

/**
 * Informação sobre um tensor armazenado num arquivo GGUF.
 */
public final class GgufTensorInfo {
    private final String name;
    private final GgufType type;
    private final long[] dims;
    private final long offset;

    public GgufTensorInfo(String name, GgufType type, long[] dims, long offset) {
        this.name = name;
        this.type = type;
        this.dims = dims;
        this.offset = offset;
    }

    public String getName() {
        return name;
    }

    public GgufType getType() {
        return type;
    }

    public long[] getDims() {
        return dims;
    }

    public int getDimensions() {
        return dims.length;
    }

    public long getOffset() {
        return offset;
    }

    @Override
    public String toString() {
        return "GgufTensorInfo{" +
            "name='" + name + '\'' +
            ", type=" + type +
            ", dims=" + Arrays.toString(dims) +
            ", offset=" + offset +
            '}';
    }
}
