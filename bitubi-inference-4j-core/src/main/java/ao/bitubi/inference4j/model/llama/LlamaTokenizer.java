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
package ao.bitubi.inference4j.model.llama;

import ao.bitubi.inference4j.safetensors.tokenizer.BPETokenizer;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Optional;

public class LlamaTokenizer extends BPETokenizer {
    static final String SPIECE_UNDERLINE = "▁";

    private final int byteFallbackEncodingOffset;

    public LlamaTokenizer(Path modelRoot) {
        super(modelRoot);
        this.byteFallbackEncodingOffset = this.getModel().vocabLookup.getOrDefault("<0x00>", -1L).intValue();
    }

    @Override
    protected long encodeCharacterAsToken(byte c) {
        return Byte.toUnsignedLong(c) + Math.max(byteFallbackEncodingOffset, 0);
    }

    @Override
    protected Optional<Character> maybeDecodeTokenAsCharacter(long id) {
        // Handle ascii codes (shifted by N in vocab)
        if (model.byteFallback
            && byteFallbackEncodingOffset > 0
            && id >= byteFallbackEncodingOffset
            && id < 256 + byteFallbackEncodingOffset) {
            char c = (char) (id - byteFallbackEncodingOffset);
            return Optional.of(c);
        }

        return Optional.empty();
    }

    @Override
    protected String preProcess(String sentence) {
        if (model.normalizer() != null) sentence = model.normalizer().normalize(sentence);

        if (isByteLevel()) {
            // Byte-level BPE works on the UTF-8 bytes of the text, each one shown as a printable character
            byte[] utf8 = sentence.getBytes(StandardCharsets.UTF_8);
            StringBuilder sb = new StringBuilder(utf8.length);
            for (byte b : utf8) {
                int unsigned = Byte.toUnsignedInt(b);
                sb.appendCodePoint(alteredBytes.getOrDefault(unsigned, unsigned));
            }
            sentence = sb.toString();
        }

        return sentence;
    }

    @Override
    protected String postProcess(String sentence) {
        return sentence.stripLeading();
    }

    @Override
    protected String postProcessToken(String decoded) {
        return postProcessToken(decoded, true);
    }

    private String postProcessToken(String decoded, boolean decodeUtf8) {
        if (decoded == null) decoded = model.unkToken;

        decoded = decoded.replaceAll("</?s>", "");
        decoded = decoded.replaceAll(SPIECE_UNDERLINE, " ");

        if (decodeUtf8 && isByteLevel()) {
            decoded = new String(toBytes(decoded), StandardCharsets.UTF_8);
        }

        return decoded;
    }

    @Override
    public Optional<byte[]> decodeBytes(long id) {
        if (!isByteLevel()) return Optional.empty();

        String s = model.vocabLookup.inverse().get(id);
        return Optional.of(toBytes(postProcessToken(s, false)));
    }

    @Override
    public String decode(long[] ids) {
        if (!isByteLevel()) return super.decode(ids);

        // Join the bytes first: a character can be split between tokens
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        for (long id : ids)
            out.writeBytes(decodeBytes(id).orElseThrow());
        return postProcess(new String(out.toByteArray(), StandardCharsets.UTF_8));
    }

    private boolean isByteLevel() {
        return model.isLegacy() && !model.byteFallback;
    }

    // Maps the printable characters of a byte-level token back to the bytes they stand for
    private static byte[] toBytes(String token) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(token.length());
        token.codePoints().forEach(c -> {
            int b = alteredBytes.inverse().getOrDefault(c, c);
            if (b < 256) out.write(b);
            else out.writeBytes(Character.toString(b).getBytes(StandardCharsets.UTF_8));
        });
        return out.toByteArray();
    }
}
