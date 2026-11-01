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
package ao.bitubi.inference4j.model;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import ao.bitubi.inference4j.model.llama.LlamaTokenizer;
import ao.bitubi.inference4j.safetensors.tokenizer.BPETokenizer;
import ao.bitubi.inference4j.safetensors.tokenizer.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Collectors;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Round trip of non-ASCII text through the Qwen2.5 tokenizer (byte-level BPE). Compares Java Strings code point
 * by code point, so console/log encoding plays no part. Skipped when the tokenizer is not in ./models.
 */
public class TokenizerRoundTripTest {

    private static final String TEXT = "Olá, você, ação, 日本語, emoji 🙂";
    private static Tokenizer tokenizer;

    @BeforeClass
    public static void setUp() {
        Path dir = Paths.get("./models/tjake_Qwen2.5-0.5B-Instruct-JQ4");
        Assume.assumeTrue("Qwen2.5 tokenizer not in ./models", Files.exists(dir.resolve("tokenizer.json")));
        tokenizer = new LlamaTokenizer(dir);
    }

    private static String codePoints(String s) {
        return s.codePoints().mapToObj(c -> String.format("U+%04X", c)).collect(Collectors.joining(" "));
    }

    @Test
    public void decodeOfEncodeIsIdentity() {
        assertEquals(codePoints(TEXT), codePoints(tokenizer.decode(tokenizer.encode(TEXT))));
    }

    @Test
    public void joinedTokenBytesDecodeToTheOriginalText() {
        // The generation loop joins the bytes of the tokens and decodes them as UTF-8, because a character can be split
        // between tokens (here the CJK and emoji ones).
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (long id : tokenizer.encode(TEXT))
            bytes.writeBytes(tokenizer.decodeBytes(id).orElseThrow());
        assertEquals(codePoints(TEXT), codePoints(new String(bytes.toByteArray(), StandardCharsets.UTF_8)));
    }

    @Test
    public void streamingDecodeHoldsBackSplitCharacters() {
        // Same path as the generation loop that builds Response.responseText. The model can emit a character as several
        // tokens, so the text is streamed both as encoded and as one token per UTF-8 byte.
        assertEquals(codePoints(TEXT), codePoints(streamDecode(tokenizer.encode(TEXT))));

        byte[] utf8 = TEXT.getBytes(StandardCharsets.UTF_8);
        long[] byteTokens = new long[utf8.length];
        for (int i = 0; i < utf8.length; i++) {
            int b = Byte.toUnsignedInt(utf8[i]);
            byteTokens[i] = tokenizer.getModel().vocabLookup.get(Character.toString(BPETokenizer.alteredBytes.getOrDefault(b, b)));
        }
        assertEquals(codePoints(TEXT), codePoints(streamDecode(byteTokens)));
    }

    private static String streamDecode(long[] ids) {
        ByteArrayOutputStream pending = new ByteArrayOutputStream();
        StringBuilder sb = new StringBuilder();
        for (long id : ids)
            sb.append(AbstractModel.decodeToken(tokenizer, id, pending));
        assertEquals(0, pending.size());
        return sb.toString();
    }

    @Test
    public void multiByteCharacterTokenMatchesTheVocabulary() {
        // " você" is a single vocabulary entry, stored as the printable form of its UTF-8 bytes (C3 AA shows as "Ãª")
        long id = tokenizer.getModel().vocabLookup.get("\u0120voc\u00C3\u00AA");
        assertEquals(codePoints(" você"), codePoints(tokenizer.decode(id)));
        assertArrayEquals(new long[] { id }, tokenizer.encode(" você"));
    }

    @Test
    public void tokenByTokenDecodeIsIdentityWhenCharactersAreNotSplit() {
        String text = "Olá, você, ação";
        StringBuilder sb = new StringBuilder();
        for (long id : tokenizer.encode(text))
            sb.append(tokenizer.decode(id));
        assertEquals(codePoints(text), codePoints(sb.toString()));
    }
}
