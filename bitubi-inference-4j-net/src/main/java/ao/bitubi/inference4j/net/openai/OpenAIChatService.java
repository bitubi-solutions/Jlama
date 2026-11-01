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

import ao.bitubi.inference4j.model.SessionBusyException;
import ao.bitubi.inference4j.model.functions.Generator;
import ao.bitubi.inference4j.net.openai.model.*;
import ao.bitubi.inference4j.safetensors.prompt.PromptSupport;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@Validated
public class OpenAIChatService {

    private static final Logger logger = LoggerFactory.getLogger(OpenAIChatService.class);
    private static final String SESSION_HEADER = "X-Bi4j-Session";

    @Autowired
    private Generator model;

    // Sessions with a request in flight, in either mode. Taken before any stream is opened so a busy session
    // can be answered with a plain 409; the engine rejects concurrent generations on a session as well.
    private final Set<UUID> activeSessions = ConcurrentHashMap.newKeySet();

    /** The session from the {@code X-Bi4j-Session} header, or a new one; throws if the header is not a UUID. */
    static UUID sessionId(Map<String, String> headers) {
        String value = headers.get(SESSION_HEADER);
        return value == null ? UUID.randomUUID() : UUID.fromString(value);
    }

    /**
     * POST /chat/completions : Creates a model response for the given chat conversation.
     *
     * @param request  (required)
     * @return OK (status code 200)
     */
    @RequestMapping(method = RequestMethod.POST, value = "/chat/completions", produces = { "application/json",
        "text/event-stream" }, consumes = { "application/json" })
    Object createChatCompletion(@RequestHeader Map<String, String> headers, @Valid @RequestBody CreateChatCompletionRequest request) {

        List<ChatCompletionRequestMessage> messages = request.getMessages();

        if (model.promptSupport().isEmpty()) {
            return new ResponseEntity<>(HttpStatus.BAD_GATEWAY);
        }

        UUID id;
        try {
            id = sessionId(headers);
        } catch (IllegalArgumentException e) {
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }

        UUID sessionId = id;

        PromptSupport.Builder builder = model.promptSupport().get().builder();

        for (ChatCompletionRequestMessage m : messages) {

            if (m.getActualInstance() instanceof ChatCompletionRequestUserMessage) {
                ChatCompletionRequestUserMessageContent content = m.getChatCompletionRequestUserMessage().getContent();

                if (content.getActualInstance() instanceof String) {
                    builder.addUserMessage(content.getString());
                } else {
                    for (ChatCompletionRequestMessageContentPart p : content.getListChatCompletionRequestMessageContentPart()) {
                        if (p.getActualInstance() instanceof ChatCompletionRequestMessageContentPartText) {
                            builder.addUserMessage(p.getChatCompletionRequestMessageContentPartText().getText());
                        } else {
                            // We don't support other types of content... yet...
                            return new ResponseEntity<>(HttpStatus.NOT_IMPLEMENTED);
                        }
                    }
                }
            } else if (m.getActualInstance() instanceof ChatCompletionRequestSystemMessage) {
                builder.addSystemMessage(m.getChatCompletionRequestSystemMessage().getContent());
            } else if (m.getActualInstance() instanceof ChatCompletionRequestAssistantMessage) {
                builder.addAssistantMessage(m.getChatCompletionRequestAssistantMessage().getContent());
            } else {
                return new ResponseEntity<>(HttpStatus.NOT_IMPLEMENTED);
            }
        }

        float temperature = 0.3f;
        int maxTokens = request.getMaxTokens() == null ? model.getConfig().contextLength : request.getMaxTokens();

        if (!activeSessions.add(sessionId)) {
            return sessionBusy(new SessionBusyException(sessionId));
        }

        logger.info("Generating completion for session {} with temperature {} and max tokens {}", sessionId, temperature, maxTokens);
        AtomicInteger index = new AtomicInteger(0);
        if (request.getStream() != null && request.getStream()) {
            SseEmitter emitter = new SseEmitter(-1L);
            CompletableFuture.supplyAsync(
                () -> model.    generate(sessionId, builder.build(), temperature, maxTokens, (t, f) -> CompletableFuture.supplyAsync(() -> {
                    try {
                        emitter.send(
                            new CreateChatCompletionStreamResponse().id(sessionId.toString())
                                .choices(
                                    List.of(
                                        new CreateChatCompletionStreamResponseChoicesInner().index(index.getAndIncrement())
                                            .delta(new ChatCompletionStreamResponseDelta().content(t))
                                    )
                                )
                        );
                    } catch (IOException e) {
                        emitter.completeWithError(e);
                    }
                    return null;
                }))
            ).handle((r, ex) -> {
                try {
                    activeSessions.remove(sessionId);
                    emitter.send(
                        new CreateChatCompletionStreamResponse().id(sessionId.toString())
                            .choices(
                                List.of(
                                    new CreateChatCompletionStreamResponseChoicesInner().finishReason(
                                        CreateChatCompletionStreamResponseChoicesInner.FinishReasonEnum.STOP
                                    ).delta(new ChatCompletionStreamResponseDelta().content(""))
                                )
                            )
                    );

                    emitter.complete();

                    logger.info(
                        "{} tokens/s (prompt), {} tokens/s (gen)",
                        Math.round(r.promptTokens / (double) (r.promptTimeMs / 1000f)),
                        Math.round(r.generatedTokens / (double) (r.generateTimeMs / 1000f))
                    );

                } catch (IOException e) {
                    emitter.completeWithError(e);
                }

                return null;
            });

            return emitter;
        } else {
            Generator.Response r;
            try {
                r = model.generate(sessionId, builder.build(), temperature, maxTokens, (s, f) -> {});
            } finally {
                activeSessions.remove(sessionId);
            }

            CreateChatCompletionResponse out = new CreateChatCompletionResponse().id(sessionId.toString())
                .choices(
                    List.of(
                        new CreateChatCompletionResponseChoicesInner().finishReason(
                            CreateChatCompletionResponseChoicesInner.FinishReasonEnum.STOP
                        ).message(new ChatCompletionResponseMessage().content(r.responseText))
                    )
                );

            return new ResponseEntity<>(out, HttpStatus.OK);
        }
    }

    @ExceptionHandler(SessionBusyException.class)
    ResponseEntity<Map<String, Object>> sessionBusy(SessionBusyException e) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", e.getMessage());
        error.put("type", "invalid_request_error");
        error.put("param", null);
        error.put("code", "session_busy");
        return new ResponseEntity<>(Map.of("error", error), HttpStatus.CONFLICT);
    }
}
