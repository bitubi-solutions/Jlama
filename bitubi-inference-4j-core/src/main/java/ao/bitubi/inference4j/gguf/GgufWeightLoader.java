package ao.bitubi.inference4j.gguf;

import ao.bitubi.inference4j.model.DistributedContext;
import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.safetensors.TensorInfo;
import ao.bitubi.inference4j.safetensors.WeightLoader;
import ao.bitubi.inference4j.tensor.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;

/** WeightLoader for a single-file Qwen2 GGUF. */
public final class GgufWeightLoader implements WeightLoader {
    private final GgufFile file;
    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> infos;
    private final Map<String, GgufTensorInfo> source;

    public GgufWeightLoader(GgufFile file) {
        this.file = file;
        Map<String, String> m = new HashMap<>();
        file.getMetadata().forEach((k, v) -> m.put(k, String.valueOf(v)));
        metadata = Collections.unmodifiableMap(m);
        Map<String, TensorInfo> i = new HashMap<>();
        Map<String, GgufTensorInfo> s = new HashMap<>();
        for (GgufTensorInfo tensor : file.getTensors()) {
            String name = tensor.getName().equals("token_embd.weight") ? "model.embed_tokens.weight"
                : tensor.getName().equals("output_norm.weight") ? "model.norm.weight"
                : tensor.getName().equals("output.weight") ? "lm_head.weight" : layerName(tensor.getName());
            int[] shape = new int[tensor.getDims().length];
            for (int n = 0; n < shape.length; n++) shape[n] = Math.toIntExact(tensor.getDims()[shape.length - n - 1]);
            DType type = switch (tensor.getType()) {
                case F32 -> DType.F32; case F16 -> DType.F16; case BF16 -> DType.BF16;
                case Q4_0 -> DType.Q4; case Q8_0 -> DType.I8;
            };
            i.put(name, new TensorInfo(type, Arrays.stream(shape).asLongStream().toArray(), new long[] {0, tensor.getByteSize()}));
            s.put(name, tensor);
        }
        infos = Collections.unmodifiableMap(i);
        source = Collections.unmodifiableMap(s);
    }

    private static String layerName(String n) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("blk\\.(\\d+)\\.(.*)").matcher(n);
        if (!m.matches()) return n;
        String suffix = m.group(2).replace("attn_q.", "self_attn.q_proj.").replace("attn_k.", "self_attn.k_proj.")
            .replace("attn_v.", "self_attn.v_proj.").replace("attn_output.", "self_attn.o_proj.")
            .replace("attn_norm.", "input_layernorm.").replace("ffn_norm.", "post_attention_layernorm.")
            .replace("ffn_gate.", "mlp.gate_proj.").replace("ffn_up.", "mlp.up_proj.").replace("ffn_down.", "mlp.down_proj.");
        return "model.layers." + m.group(1) + "." + suffix;
    }

    @Override public Map<String, String> metadata() { return metadata; }
    @Override public Map<String, TensorInfo> tensorInfoMap() { return infos; }
    @Override public DType getModelDType() { return DType.F32; }
    @Override public AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns) {
        if (dctx != null && (dctx.hasModelShard() || dctx.numberOfLayers != ((Number) file.getMetadata().get("qwen2.block_count")).intValue()))
            throw new UnsupportedOperationException("GGUF nao suporta inferencia distribuida nesta fatia");
        GgufTensorInfo info = source.get(name);
        if (info == null) throw new NoSuchElementException(name + " nao encontrado no GGUF");
        try { return unpack(name, info, file.mapTensor(info), TensorShape.of(infos.get(name).shape)); }
        catch (IOException e) { throw new IllegalStateException("Falha ao mapear tensor GGUF " + name, e); }
    }

    private static AbstractTensor unpack(String name, GgufTensorInfo info, ByteBuffer b, TensorShape shape) {
        b.order(ByteOrder.LITTLE_ENDIAN);
        if (info.getType() == GgufType.F32) return new FloatBufferTensor(name, b.asFloatBuffer(), shape, true);
        if (info.getType() == GgufType.F16) {
            FloatBuffer out = ByteBuffer.allocateDirect(Math.toIntExact(shape.size() * 4)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            while (b.remaining() >= 2) out.put(Float.float16ToFloat(b.getShort()));
            out.flip(); return new FloatBufferTensor(name, out, shape, true);
        }
        if (info.getType() == GgufType.BF16) return new BFloat16BufferTensor(name, b.asShortBuffer(), shape, true);
        int blocks = Math.toIntExact(shape.size() / 32);
        ByteBuffer values = ByteBuffer.allocateDirect(blocks * (info.getType() == GgufType.Q8_0 ? 32 : 16)).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer scales = ByteBuffer.allocateDirect(blocks * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        for (int i = 0; i < blocks; i++) {
            float scale = Float.float16ToFloat(b.getShort()); scales.put(scale);
            if (info.getType() == GgufType.Q8_0) {
                for (int j = 0; j < 32; j++) values.put(b.get());
            } else for (int j = 0; j < 16; j++) values.put(b.get());
        }
        values.flip(); scales.flip();
        FloatBufferTensor scaleTensor = new FloatBufferTensor(scales, shape.scaleLastDim(1f / 32), true);
        return info.getType() == GgufType.Q8_0
            ? new Q8ByteBufferTensor(name, values, scaleTensor, shape, true)
            : new Q4ByteBufferTensor(name, values, scaleTensor, shape, true);
    }
    @Override public void close() {}
}
