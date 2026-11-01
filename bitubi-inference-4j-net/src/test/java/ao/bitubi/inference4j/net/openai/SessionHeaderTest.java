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
package ao.bitubi.inference4j.net.openai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.Test;

public class SessionHeaderTest {

    @Test
    public void readsTheBi4jSessionHeader() {
        UUID id = UUID.randomUUID();
        assertEquals(id, OpenAIChatService.sessionId(Map.of("X-Bi4j-Session", id.toString())));
    }

    @Test
    public void withoutTheHeaderEachRequestGetsANewSession() {
        assertNotEquals(OpenAIChatService.sessionId(Map.of()), OpenAIChatService.sessionId(Map.of()));
    }

    @Test
    public void aSessionThatIsNotAUuidIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> OpenAIChatService.sessionId(Map.of("X-Bi4j-Session", "nope")));
    }

    @Test
    public void theRemovedJlamaHeaderIsIgnored() {
        UUID id = UUID.randomUUID();
        assertNotEquals(id, OpenAIChatService.sessionId(Map.of("X-Jlama-Session", id.toString())));
        // an invalid value of the removed header is not even parsed
        OpenAIChatService.sessionId(Map.of("X-Jlama-Session", "nope"));
    }
}
