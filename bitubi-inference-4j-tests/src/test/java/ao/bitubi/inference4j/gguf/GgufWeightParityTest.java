package ao.bitubi.inference4j.gguf;

import ao.bitubi.inference4j.tensor.AbstractTensor;
import org.junit.Assume;
import org.junit.Test;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import static org.junit.Assert.*;

public class GgufWeightParityTest {
    @Test public void rawBlocksMatchLoaderAndF16ComparisonIsInformative() throws Exception {
        String quantized=System.getenv("GGUF_QUANTIZED_MODEL_PATH");
        Assume.assumeTrue(quantized != null);
        GgufFile source=new GgufFile(Path.of(quantized));
        GgufWeightLoader loader=new GgufWeightLoader(source);
        String referencePath=System.getenv("GGUF_FP16_MODEL_PATH");
        GgufWeightLoader reference=referencePath == null ? null : new GgufWeightLoader(new GgufFile(Path.of(referencePath)));
        Stats q4=new Stats(), q8=new Stats();
        int q4Tensors=0, q8Tensors=0;
        for (GgufTensorInfo info : source.getTensors()) {
            if (info.getType() != GgufType.Q4_0 && info.getType() != GgufType.Q8_0) continue;
            String name=loaderName(info.getName());
            AbstractTensor loaded=loader.load(name);
            assertEquals(name,2,loaded.dims());
            ByteBuffer raw=source.mapTensor(info).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(name,info.getByteSize(),raw.capacity());
            AbstractTensor f16=reference == null ? null : reference.load(name);
            if (f16 != null) assertEquals(name,f16.shape(),loaded.shape());
            Stats stats=info.getType() == GgufType.Q4_0 ? q4 : q8;
            if (info.getType() == GgufType.Q4_0) q4Tensors++; else q8Tensors++;
            int elements=Math.toIntExact(java.util.Arrays.stream(info.getDims()).reduce(1L,Math::multiplyExact));
            assertEquals(name,elements % 32,0);
            for (int block=0; block<elements/32; block++) {
                float scale=Float.float16ToFloat(raw.getShort());
                if (info.getType() == GgufType.Q4_0) {
                    for (int j=0; j<16; j++) {
                        int packed=raw.get() & 0xff;
                        check(loaded,f16,block*32+j,(packed & 15)-8,scale,stats,true);
                        check(loaded,f16,block*32+j+16,(packed >>> 4)-8,scale,stats,true);
                    }
                } else {
                    for (int j=0; j<32; j++) check(loaded,f16,block*32+j,raw.get(),scale,stats,false);
                }
            }
            assertEquals(name,0,raw.remaining());
        }
        assertTrue("GGUF sem tensores quantizados suportados",q4Tensors + q8Tensors > 0);
        System.out.println("Repack Q4_0: " + q4Tensors + " tensores, " + q4.elements + " valores, comparação float exata com os bytes GGUF");
        System.out.println("Repack Q8_0: " + q8Tensors + " tensores, " + q8.elements + " valores, comparação float exata com os bytes GGUF");
        if (reference != null) {
            if (q4.elements > 0) q4.print("Q4_0 vs F16");
            if (q8.elements > 0) q8.print("Q8_0 vs F16");
        }
    }

    private static void check(AbstractTensor loaded, AbstractTensor reference, int index, int quantized,
                              float scale, Stats stats, boolean q4) {
        int columns=loaded.shape().last();
        int row=index/columns, column=index%columns;
        float decoded=quantized*scale;
        float actual=loaded.get(row,column);
        assertEquals("repack em ["+row+","+column+"]",Float.floatToIntBits(decoded),Float.floatToIntBits(actual));
        stats.elements++;
        if (reference == null) return;
        float expected=reference.get(row,column);
        float steps=scale == 0f ? (decoded == expected ? 0f : Float.POSITIVE_INFINITY)
            : Math.abs(decoded-expected)/Math.abs(scale);
        stats.sumSteps+=steps;
        stats.maxSteps=Math.max(stats.maxSteps,steps);
        if (steps <= 0.5001f) stats.withinHalf++;
        if (steps == 0f) stats.exactF16++;
        if (q4 && scale != 0f && quantized+8 == 15 && expected/scale+8.5f > 15f) {
            stats.clipped++;
            stats.maxClippedSteps=Math.max(stats.maxClippedSteps,steps);
        } else {
            stats.maxOtherSteps=Math.max(stats.maxOtherSteps,steps);
        }
    }

    private static String loaderName(String raw) {
        if (raw.equals("token_embd.weight")) return "model.embed_tokens.weight";
        if (raw.equals("output_norm.weight")) return "model.norm.weight";
        if (raw.equals("output.weight")) return "lm_head.weight";
        Matcher m=Pattern.compile("blk\\.(\\d+)\\.(.*)").matcher(raw);
        if (!m.matches()) return raw;
        String suffix=m.group(2).replace("attn_q.","self_attn.q_proj.").replace("attn_k.","self_attn.k_proj.")
            .replace("attn_v.","self_attn.v_proj.").replace("attn_output.","self_attn.o_proj.")
            .replace("attn_norm.","input_layernorm.").replace("ffn_norm.","post_attention_layernorm.")
            .replace("ffn_gate.","mlp.gate_proj.").replace("ffn_up.","mlp.up_proj.").replace("ffn_down.","mlp.down_proj.");
        return "model.layers."+m.group(1)+"."+suffix;
    }

    private static final class Stats {
        long elements, withinHalf, exactF16, clipped;
        double sumSteps;
        float maxSteps, maxClippedSteps, maxOtherSteps;
        void print(String label) {
            System.out.printf(Locale.ROOT,"%s: mean=%.5f steps, max=%.5f, <=0.5 step=%.4f%%, exact F16=%d/%d, Q4 clipped=%d (max error %.5f steps), other max=%.5f steps%n",
                label,sumSteps/elements,maxSteps,100d*withinHalf/elements,exactF16,elements,clipped,maxClippedSteps,maxOtherSteps);
        }
    }
}
