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

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import ao.bitubi.inference4j.model.SessionBusyException;
import ao.bitubi.inference4j.model.functions.Generator;
import ao.bitubi.inference4j.safetensors.prompt.PromptSupport;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Answers;
import org.springframework.http.MediaType;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/**
 * A request on a session that is already generating is answered with 409 and an OpenAI-style error body,
 * before any stream is opened. Uses a stubbed {@link Generator}: no model files are needed.
 */
public class SessionConflictTest {

    private static final String SESSION_HEADER = "X-Bi4j-Session";

    private Generator model;
    private MockMvc mvc;

    @BeforeEach
    public void setUp() {
        model = mock(Generator.class);
        when(model.promptSupport()).thenReturn(Optional.of(mock(PromptSupport.class, Answers.RETURNS_DEEP_STUBS)));
        OpenAIChatService service = new OpenAIChatService();
        ReflectionTestUtils.setField(service, "model", model);
        mvc = MockMvcBuilders.standaloneSetup(service).build();
    }

    private static String body(boolean stream) {
        return "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"stream\":" + stream + "}";
    }

    private static Generator.Response response() {
        return new Generator.Response("ok", "ok", Generator.FinishReason.STOP_TOKEN, 1, 1, 1, 1);
    }

    @Test
    public void engineRejectionBecomes409WithOpenAIErrorBody() throws Exception {
        UUID session = UUID.randomUUID();
        when(model.generate(any(), any(), anyFloat(), anyInt(), any())).thenThrow(new SessionBusyException(session));

        mvc.perform(post("/chat/completions").header(SESSION_HEADER, session.toString()).contentType(MediaType.APPLICATION_JSON).content(body(false)))
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error.type").value("invalid_request_error"))
            .andExpect(jsonPath("$.error.code").value("session_busy"))
            .andExpect(jsonPath("$.error.message").value(new SessionBusyException(session).getMessage()));
    }

    @Test
    public void secondRequestOnBusySessionIsRejectedBeforeStreamOpens() throws Exception {
        UUID session = UUID.randomUUID();
        CountDownLatch inFlight = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(model.generate(any(), any(), anyFloat(), anyInt(), any())).thenAnswer(inv -> {
            inFlight.countDown();
            release.await(30, TimeUnit.SECONDS);
            return response();
        });

        // First request streams and stays in flight.
        mvc.perform(post("/chat/completions").header(SESSION_HEADER, session.toString()).contentType(MediaType.APPLICATION_JSON).content(body(true)))
            .andExpect(request().asyncStarted());
        assertTrue(inFlight.await(30, TimeUnit.SECONDS));

        // Same session, streaming or not: 409 straight away, no async stream.
        for (boolean stream : new boolean[] { true, false }) {
            mvc.perform(post("/chat/completions").header(SESSION_HEADER, session.toString()).contentType(MediaType.APPLICATION_JSON).content(body(stream)))
                .andExpect(request().asyncNotStarted())
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.error.code").value("session_busy"));
        }

        // A different session is not affected.
        when(model.generate(any(), any(), anyFloat(), anyInt(), any())).thenReturn(response());
        mvc.perform(post("/chat/completions").header(SESSION_HEADER, UUID.randomUUID().toString()).contentType(MediaType.APPLICATION_JSON).content(body(false)))
            .andExpect(status().isOk());

        // Once the first request finishes the session is usable again.
        release.countDown();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        int last = 0;
        while (System.nanoTime() < deadline) {
            last = mvc.perform(post("/chat/completions").header(SESSION_HEADER, session.toString()).contentType(MediaType.APPLICATION_JSON).content(body(false)))
                .andReturn().getResponse().getStatus();
            if (last == 200) break;
            Thread.sleep(50);
        }
        assertTrue(last == 200, "session still busy after the first request finished, last status " + last);
    }
}
