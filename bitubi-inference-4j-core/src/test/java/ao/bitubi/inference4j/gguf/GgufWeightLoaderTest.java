package ao.bitubi.inference4j.gguf;

import ao.bitubi.inference4j.tensor.FloatBufferTensor;
import ao.bitubi.inference4j.tensor.Q8ByteBufferTensor;
import ao.bitubi.inference4j.tensor.Q4ByteBufferTensor;
import ao.bitubi.inference4j.tensor.TensorShape;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import static org.junit.Assert.*;

public class GgufWeightLoaderTest {
    @Rule public final TemporaryFolder temporary = new TemporaryFolder();

    @Test public void reversesGgmlDimensionsAndUnpacksQ8Scale() throws Exception {
        ByteBuffer raw = ByteBuffer.allocate(68).order(ByteOrder.LITTLE_ENDIAN);
        raw.putShort((short) 0x3c00);
        for (int i=0;i<32;i++) raw.put((byte)(i-16));
        raw.putShort((short) 0x4000);
        for (int i=0;i<32;i++) raw.put((byte)1);
        Path path=temporary.newFile("weights.gguf").toPath();
        new GgufTestWriter(32).addTensor("blk.0.attn_q.weight",GgufType.Q8_0,new long[]{32,2},raw.flip())
            .addTensor("token_embd.weight",GgufType.Q8_0,new long[]{32,2},raw.rewind()).write(path);
        GgufWeightLoader loader=new GgufWeightLoader(new GgufFile(path));
        assertArrayEquals(new int[]{2,32},loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").shape);
        Q8ByteBufferTensor tensor=(Q8ByteBufferTensor)loader.load("model.layers.0.self_attn.q_proj.weight");
        assertEquals(-16f,tensor.get(0,0),0f);
        assertEquals(15f,tensor.get(0,31),0f);
        assertEquals(2f,tensor.get(1,0),0f);
        Q8ByteBufferTensor embedding=(Q8ByteBufferTensor)loader.load("model.embed_tokens.weight");
        assertEquals(-16f,embedding.get(0,0),0f);
    }

    @Test public void widensF16AndUnpacksQ4Scale() throws Exception {
        ByteBuffer half=ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putShort((short)0x3c00).putShort((short)0xc000);
        ByteBuffer q4=ByteBuffer.allocate(18).order(ByteOrder.LITTLE_ENDIAN).putShort((short)0x3800).put((byte)0x98);
        for(int i=1;i<16;i++)q4.put((byte)0x88);
        Path path=temporary.newFile("mixed.gguf").toPath();
        new GgufTestWriter(32).addTensor("norm",GgufType.F16,new long[]{2},half.flip())
            .addTensor("blk.0.ffn_gate.weight",GgufType.Q4_0,new long[]{32},q4.flip()).write(path);
        GgufWeightLoader loader=new GgufWeightLoader(new GgufFile(path));
        FloatBufferTensor f16=(FloatBufferTensor)loader.load("norm");
        assertEquals(1f,f16.get(0,0),0f); assertEquals(-2f,f16.get(0,1),0f);
        Q4ByteBufferTensor tensor=(Q4ByteBufferTensor)loader.load("model.layers.0.mlp.gate_proj.weight");
        assertEquals(0f,tensor.get(0,0),0f); assertEquals(.5f,tensor.get(0,16),0f);
    }
}
