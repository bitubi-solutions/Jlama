package ao.bitubi.inference4j.gguf;

import ao.bitubi.inference4j.safetensors.tokenizer.TokenizerModel;
import java.util.*;

public final class GgufTokenizerFactory {
    private GgufTokenizerFactory() {}
    public static TokenizerModel create(Map<String, Object> m) {
        if (!"gpt2".equals(m.get("tokenizer.ggml.model"))) throw new IllegalArgumentException("Tokenizer GGUF nao suportado: " + m.get("tokenizer.ggml.model"));
        List<String> tokens = (List<String>) m.get("tokenizer.ggml.tokens");
        List<String> merges = (List<String>) m.get("tokenizer.ggml.merges");
        Map<String, Long> vocab = new LinkedHashMap<>();
        for (int i=0;i<tokens.size();i++) vocab.put(tokens.get(i),(long)i);
        TokenizerModel model = new TokenizerModel("BPE", null, false, false, vocab, false, new ArrayList<>(merges));
        List<Integer> types = (List<Integer>) m.get("tokenizer.ggml.token_type");
        List<Map<String,Object>> added = new ArrayList<>();
        for (int i=0;i<tokens.size();i++) if (((Number)types.get(i)).intValue()==3 || ((Number)types.get(i)).intValue()==4) {
            added.add(Map.of("content",tokens.get(i),"id",i,"special",((Number)types.get(i)).intValue()==3));
        }
        model.setAddedTokens(added);
        var split = new TokenizerModel.PretokenizerItem("Split", new TokenizerModel.Pattern("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), "Isolated", true, false, false, true, true);
        var bytes = new TokenizerModel.PretokenizerItem("ByteLevel", null, null, null, null, true, true, true);
        model.setPreTokenizer(new TokenizerModel.PreTokenizer("Sequence", null, null, List.of(split,bytes)));
        String bos=tokens.get(i(m,"tokenizer.ggml.bos_token_id")), eos=tokens.get(i(m,"tokenizer.ggml.eos_token_id"));
        model.setBosToken(bos); model.setEosToken(eos);
        Object template=m.get("tokenizer.chat_template");
        if (template instanceof String s) model.setPromptTemplates(Map.of("default",s));
        return model;
    }
    private static int i(Map<String,Object> m,String k) { return ((Number)m.get(k)).intValue(); }
}
