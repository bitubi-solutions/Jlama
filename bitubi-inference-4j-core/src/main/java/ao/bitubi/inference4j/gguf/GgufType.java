/*
 * Copyright 2026 Bitubi Solutions
 *
 * Este ficheiro é parte do projeto bitubi-inference-4j.
 * Licenciado sob Apache License 2.0.
 */
package ao.bitubi.inference4j.gguf;

/**
 * Enumeração do tipo ggml usado no formato GGUF.
 * Mapeia o id do ggml_type para elementos e bytes por bloco.
 */
public enum GgufType {
    F32(0, 1, 4),
    F16(1, 1, 2),
    Q4_0(2, 32, 18),
    Q4_1(3, 32, 20),
    Q5_0(6, 32, 22),
    Q5_1(7, 32, 24),
    Q8_0(8, 32, 34),
    Q8_1(9, 32, 36),
    Q2_K(10, 256, 96),
    Q3_K(11, 256, 110),
    Q4_K(12, 256, 144),
    Q5_K(13, 256, 176),
    Q6_K(14, 256, 210),
    Q8_K(15, 256, 292),
    I8(24, 1, 1),
    I16(25, 1, 2),
    I32(26, 1, 4),
    I64(27, 1, 8),
    F64(28, 1, 8),
    BF16(30, 1, 2);

    private final int id;
    private final int elementsPerBlock;
    private final int bytesPerBlock;

    GgufType(int id, int elementsPerBlock, int bytesPerBlock) {
        this.id = id;
        this.elementsPerBlock = elementsPerBlock;
        this.bytesPerBlock = bytesPerBlock;
    }

    public int getId() {
        return id;
    }

    public int getElementsPerBlock() {
        return elementsPerBlock;
    }

    public int getBytesPerBlock() {
        return bytesPerBlock;
    }

    public static GgufType fromId(int id) {
        for (GgufType t : values()) {
            if (t.id == id) return t;
        }
        throw new IllegalArgumentException("Tipo GGUF desconhecido: " + id);
    }
}
