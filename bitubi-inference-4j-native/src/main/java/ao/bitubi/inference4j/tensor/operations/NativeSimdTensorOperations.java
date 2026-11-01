/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package ao.bitubi.inference4j.tensor.operations;

import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.tensor.AbstractTensor;
import ao.bitubi.inference4j.tensor.Q4ByteBufferTensor;
import ao.bitubi.inference4j.tensor.Q8ByteBufferTensor;
import ao.bitubi.inference4j.tensor.operations.cnative.NativeSimd;
import ao.bitubi.inference4j.tensor.operations.util.JarSupport;
import ao.bitubi.inference4j.tensor.operations.util.MemorySegmentSupport;
import ao.bitubi.inference4j.util.Bi4jProperties;
import ao.bitubi.inference4j.util.MachineSpec;
import ao.bitubi.inference4j.util.RuntimeSupport;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NativeSimdTensorOperations implements TensorOperations {
    private static final Logger logger = LoggerFactory.getLogger(NativeSimdTensorOperations.class);

    static {
        if (!JarSupport.maybeLoadLibrary("bi4j")) System.loadLibrary("bi4j");
    }

    public static final int HAS_F16C = NativeSimd.HAS_F16C();

    // Operations that are not GEMMs (accumulate, saxpy, scale, quantize) run on the Java Vector API
    private static final TensorOperations delegate = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);

    final int flags;

    /**
     * Picks the best ISA this CPU supports, capped by {@code -Dbi4j.native.isa=auto|avx2|avxvnni|avx512} (or
     * {@code neon|neon-dotprod} on ARM64). The cap exists to test every variant on a single machine.
     */
    public NativeSimdTensorOperations() {
        this(capFlags(Bi4jProperties.getString("native.isa")) | (RuntimeSupport.isLinux() ? HAS_F16C : 0));
    }

    NativeSimdTensorOperations(int flags) {
        this.flags = flags;
        int level = NativeSimd.bi4j_selected_isa(flags);
        if (level == 0) throw new UnsupportedOperationException(
            "bitubi-inference-4j-native needs AVX2 + FMA (x86-64) or NEON (ARM64); this CPU has neither"
        );
        int cap = (flags >> NativeSimd.BI4J_ISA_CAP_SHIFT()) & NativeSimd.BI4J_ISA_CAP_MASK();
        if (cap > level) logger.warn("bi4j.native.isa={} is not supported by this CPU; using {}", isaName(cap), isaName(level));
        logger.info("Native ISA: {} (this CPU supports {})", isaName(level), supportedIsas());
    }

    /** Name of an ISA level: 1 = avx2 (neon on ARM64), 2 = avxvnni (neon-dotprod on ARM64), 3 = avx512. */
    static String isaName(int level) {
        if (RuntimeSupport.isArm()) return switch (level) {
            case 1 -> "neon";
            case 2 -> "neon-dotprod";
            default -> "none";
        };
        return switch (level) {
            case 1 -> "avx2";
            case 2 -> "avxvnni";
            case 3 -> "avx512";
            default -> "none";
        };
    }

    /** The ISAs this CPU (and OS) can run, lowest first. Needs the native library. */
    public static List<String> supportedIsas() {
        int mask = NativeSimd.bi4j_supported_isas();
        List<String> out = new ArrayList<>();
        for (int level = 1; level <= NativeSimd.BI4J_ISA_AVX512(); level++)
            if ((mask & (1 << level)) != 0) out.add(isaName(level));
        return out;
    }

    /** The {@code flags} bits that cap the ISA at {@code isa}; "auto", empty or null means no cap. */
    static int capFlags(String isa) {
        if (isa == null || isa.isBlank() || isa.equalsIgnoreCase("auto")) return 0;
        for (int level = 1; level <= NativeSimd.BI4J_ISA_AVX512(); level++)
            if (isaName(level).equalsIgnoreCase(isa.trim())) return level << NativeSimd.BI4J_ISA_CAP_SHIFT();
        throw new IllegalArgumentException("Unknown bi4j.native.isa '" + isa + "'; valid: auto, avx2, avxvnni, avx512, neon, neon-dotprod");
    }

    @Override
    public String name() {
        return "Native SIMD Operations";
    }

    public int parallelSplitSize() {
        return 128;
    }

    @Override
    public void batchDotProduct(
        AbstractTensor result,
        AbstractTensor at,
        AbstractTensor bt,
        int aColumnOffset,
        int bColumnOffset,
        int columnLength,
        int rRowOffset,
        int bRowOffset,
        int rowChunkSize
    ) {

        int M = at.shape().dim(0);
        int N = rowChunkSize; // b.shape().dim(0);
        int K = columnLength; // a.shape().dim(1);

        int aOffset = at.getOffset(0, aColumnOffset);
        int bOffset = bt.getOffset(bt.shape().sparseRowOffset(), bColumnOffset);

        // Adjusts for both sparse columns and rows this goes negative because we subtract the row offset
        // And the row offsets need to add to the result offset
        int rOffset = result.shape().sparseColumnOffset() - bt.shape().sparseRowOffset() - rRowOffset;

        int adjBRowOffset = bRowOffset - bt.shape().sparseRowOffset();

        switch (at.dType()) {
            case BF16:
                switch (bt.dType()) {
                    case BF16:
                        NativeSimd.gemm_bf16(
                            flags,
                            at.getMemorySegment(),
                            aOffset,
                            bt.getMemorySegment(),
                            bOffset,
                            result.dType() == DType.BF16 ? result.getMemorySegment() : MemorySegment.NULL,
                            result.dType() == DType.F32 ? result.getMemorySegment() : MemorySegment.NULL,
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            at.getStride(),
                            bt.getStride(),
                            result.getStride()
                        );
                        break;
                    default:
                        throw new UnsupportedOperationException(at.dType().name() + " " + bt.dType().name());
                }
                break;
            case F32:
                switch (bt.dType()) {
                    case F32:
                        NativeSimd.gemm_f32(
                            flags,
                            at.getMemorySegment(),
                            aOffset,
                            bt.getMemorySegment(),
                            bOffset,
                            result.getMemorySegment(),
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            at.getStride(),
                            bt.getStride(),
                            result.getStride()
                        );
                        break;
                    case BF16:
                        NativeSimd.gemm_f32_bf16(
                            flags,
                            at.getMemorySegment(),
                            aOffset,
                            bt.getMemorySegment(),
                            bOffset,
                            result.dType() == DType.BF16 ? result.getMemorySegment() : MemorySegment.NULL,
                            result.dType() == DType.F32 ? result.getMemorySegment() : MemorySegment.NULL,
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            at.getStride(),
                            bt.getStride(),
                            result.getStride()
                        );
                        break;
                    case Q4:
                        Q4ByteBufferTensor b = (Q4ByteBufferTensor) bt;
                        NativeSimd.gemm_f32_q4(
                            flags,
                            at.getMemorySegment(),
                            aOffset,
                            b.getBlockF().getMemorySegment(),
                            b.getMemorySegment(),
                            b.getMemorySegmentOffset(bOffset),
                            result.getMemorySegment(),
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            at.getStride(),
                            b.getMemorySegmentOffset(b.getStride()),
                            b.getBlockF().getStride(),
                            result.getStride()
                        );
                        break;
                    default:
                        throw new UnsupportedOperationException(at.dType().name() + " " + bt.dType().name());
                }
                break;
            case I8:
                switch (bt.dType()) {
                    case Q4:
                        Q8ByteBufferTensor a = (Q8ByteBufferTensor) at;
                        Q4ByteBufferTensor b = (Q4ByteBufferTensor) bt;
                        NativeSimd.gemm_q8_q4(
                            flags,
                            a.getBlockF().getMemorySegment(),
                            a.getMemorySegment(),
                            aOffset,
                            b.getBlockF().getMemorySegment(),
                            b.getMemorySegment(),
                            b.getMemorySegmentOffset(bOffset),
                            result.getMemorySegment(),
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            a.getStride(),
                            a.getBlockF().getStride(),
                            b.getMemorySegmentOffset(b.getStride()),
                            b.getBlockF().getStride(),
                            result.getStride()
                        );
                        break;
                    default:
                        throw new UnsupportedOperationException(at.dType().name() + " " + bt.dType().name());
                }
                break;
            default:
                throw new UnsupportedOperationException(at.dType().name());
        }
    }

    @Override
    public void dotProductBatchChunk(
        AbstractTensor[] r,
        AbstractTensor a,
        AbstractTensor[] b,
        int columnOffset,
        int columnLength,
        int bRowOffset,
        int rowChunkSize
    ) {

        MemorySegment[] tmp = MemorySegmentSupport.setupBatch(
            i -> r[i].getMemorySegment(),
            i -> b[i].getMemorySegment(),
            i -> b[i] instanceof Q4ByteBufferTensor ? ((Q4ByteBufferTensor) b[i]).getBlockF().getMemorySegment() : MemorySegment.NULL,
            r.length
        );
        MemorySegment ra = tmp[0];
        MemorySegment rb = tmp[1];
        MemorySegment rc = tmp[2];

        int M = a.shape().dim(0);
        int N = rowChunkSize; // b.shape().dim(0);
        int K = columnLength; // a.shape().dim(1);

        int aOffset = a.getOffset(0, columnOffset);
        int bOffset = b[0].getOffset(b[0].shape().sparseRowOffset(), columnOffset);

        int adjBRowOffset = bRowOffset - b[0].shape().sparseRowOffset();

        // Adjusts for both sparse columns and rows
        int rOffset = r[0].shape().sparseColumnOffset() - b[0].shape().sparseRowOffset();

        switch (a.dType()) {
            case BF16:
                switch (b[0].dType()) {
                    case BF16:
                        NativeSimd.gemm_bf16_batch(
                            flags,
                            r.length,
                            a.getMemorySegment(),
                            aOffset,
                            rb,
                            bOffset,
                            r[0].dType() == DType.BF16 ? ra : MemorySegment.NULL,
                            r[0].dType() == DType.F32 ? ra : MemorySegment.NULL,
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            a.getStride(),
                            b[0].getStride(),
                            r[0].getStride()
                        );
                        break;
                    default:
                        throw new UnsupportedOperationException(a.dType().name() + " " + b[0].dType().name());
                }
                break;
            case F32:
                switch (b[0].dType()) {
                    case F32:
                        NativeSimd.gemm_f32_batch(
                            flags,
                            r.length,
                            a.getMemorySegment(),
                            aOffset,
                            rb,
                            bOffset,
                            ra,
                            rOffset,
                            M,
                            bRowOffset,
                            N,
                            K,
                            a.getStride(),
                            b[0].getStride(),
                            r[0].getStride()
                        );
                        break;
                    case BF16:
                        NativeSimd.gemm_f32_bf16_batch(
                            flags,
                            r.length,
                            a.getMemorySegment(),
                            aOffset,
                            rb,
                            bOffset,
                            r[0].dType() == DType.BF16 ? ra : MemorySegment.NULL,
                            r[0].dType() == DType.F32 ? ra : MemorySegment.NULL,
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            a.getStride(),
                            b[0].getStride(),
                            r[0].getStride()
                        );
                        break;
                    case Q4:
                        Q4ByteBufferTensor bt = (Q4ByteBufferTensor) b[0];
                        NativeSimd.gemm_f32_q4_batch(
                            flags,
                            r.length,
                            a.getMemorySegment(),
                            aOffset,
                            rc,
                            rb,
                            bt.getMemorySegmentOffset(bOffset),
                            ra,
                            rOffset,
                            M,
                            bRowOffset,
                            N,
                            K,
                            a.getStride(),
                            b[0].getMemorySegmentOffset(b[0].getStride()),
                            bt.getBlockF().getStride(),
                            r[0].getStride()
                        );
                        break;
                    default:
                        throw new UnsupportedOperationException(a.dType().name() + " " + b[0].dType().name());
                }
                break;
            case I8:
                switch (b[0].dType()) {
                    case Q4:
                        Q8ByteBufferTensor at = (Q8ByteBufferTensor) a;
                        Q4ByteBufferTensor bt = (Q4ByteBufferTensor) b[0];
                        NativeSimd.gemm_q8_q4_batch(
                            flags,
                            r.length,
                            at.getBlockF().getMemorySegment(),
                            a.getMemorySegment(),
                            aOffset,
                            rc,
                            rb,
                            bt.getMemorySegmentOffset(bOffset),
                            ra,
                            rOffset,
                            M,
                            adjBRowOffset,
                            N,
                            K,
                            a.getStride(),
                            at.getBlockF().getStride(),
                            bt.getMemorySegmentOffset(bt.getStride()),
                            bt.getBlockF().getStride(),
                            r[0].getStride()
                        );
                        break;
                    default:
                        throw new UnsupportedOperationException(a.dType().name() + " " + b[0].dType().name());
                }
                break;
            default:
                throw new UnsupportedOperationException(a.dType().name());
        }
    }

    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        delegate.accumulate(a, b, offset, length);
    }

    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        delegate.maccumulate(a, b, offset, length);
    }

    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit);
    }

    @Override
    public void saxpy(
        AbstractTensor alpha,
        AbstractTensor x,
        AbstractTensor y,
        int xoffset,
        int yoffset,
        int limit,
        int rOffset,
        int xOffset,
        int batchSize
    ) {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit, rOffset, xOffset, batchSize);
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length) {
        delegate.scale(factor, x, offset, length);
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        return delegate.quantize(t, qtype, offset, length);
    }
}
