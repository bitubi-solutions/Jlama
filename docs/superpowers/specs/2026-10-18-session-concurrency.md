# Concurrent requests: shared state and same-session policy

Status: decided (2026-10-18). Policy for the same session: **reject**.

## 1. Requests on different sessions

The REST server may receive concurrent requests on one model instance. Per-instance scratch arrays in the layers were
overwritten by concurrent calls. Reproduced with `ConcurrencyReentrancyTest` (Qwen2.5-0.5B, 4 threads, distinct
session UUIDs): `IllegalArgumentException: BAD M` in `PanamaTensorOperations.batchDotProduct`, caused by
`MLPBlock.batchResults` being replaced by a buffer of another batch size between the write and the dot product.

Fixed by making the arrays local to the call (`MLPBlock`, `CausalSelfAttention.qkvResults`, and the scratch state of
`MoEBlock`: results, weights, `expertResults`, `selectedExperts`).

Evidence and limits:

- `ConcurrencyReentrancyTest` (opt-in, `CONCURRENCY_TEST=1`): 5 rounds of 4 parallel greedy generations match the sequential
  ones. This does not prove the absence of races: one model, one backend (Panama), 4 threads, 48 tokens.
- `AttentionConcurrencyTest` covers the non-GQA `CausalSelfAttention` path with no model files (it failed with the same
  `BAD M` before the fix). Qwen2.5 is GQA and does not exercise that path.
- `MoEBlock` was fixed by review only; no MoE model was run.
- Sequential reruns of the same attention input are not bit-reproducible for some batch sizes (about 1 ulp), so that test
  compares with a tolerance. The cause was not investigated.

Other instance state reviewed in `bitubi-inference-4j-core` and left unchanged: `MLPBlock.batchWeights` (`MLPBlock.java:45`)
and `CausalSelfAttention.qkvWeights` (`CausalSelfAttention.java:56`) are never written after construction; `wte` in
`LlamaModel`/`Gemma2Model`/`GemmaModel` is set at load time; `TensorCache` uses a `ConcurrentMap` of MPMC queues;
`TensorOperations.scratch` (`TensorOperations.java:26`) is a `ThreadLocal`; `KvBufferCache.getKvBuffer` uses
`computeIfAbsent` on a `ConcurrentHashMap` (`KvBufferCache.java:59`).

## 2. Concurrent requests on the same session

What breaks: `KvBufferCache.getKvBuffer(session)` returns one `KvBuffer` per session UUID (`KvBufferCache.java:59`).
Two generations on that UUID share the KV pages and the context position (`currentContextPosition`), so each
overwrites the other's keys and values and advances the shared position. The try-with-resources in
`AbstractModel.generate` (`AbstractModel.java:538`) also closes the pages when the first call ends while the other is
still reading them. Observed before the guard: one request failed with the same `BAD M`, or both finished with mixed,
wrong text (`"What is the capital of France?"` answered `"The sum of two equals four."`).

Options considered:

| Option | Behaviour | Cost |
|---|---|---|
| Lock per session UUID | Second request waits for the first | Threads (and HTTP connections) block for a whole generation, with no bound or timeout; the second turn runs on a context the client did not see; needs a queue policy and fairness |
| Reject (chosen) | Second request fails at once | Clients must retry or serialise their own turns; a cheap check, no waiting, no hidden latency |

## 3. Decision: reject

- `AbstractModel.generate` takes the session in a concurrent set of active sessions on entry and releases it in a
  `finally`. If the session is already there it throws `SessionBusyException`
  (`Session <uuid> is already generating; concurrent requests on the same session are rejected`). It never waits.
  Other sessions keep running in parallel.
- A rejected request does not touch the session: the follow-up turn gives the same output as on a session that never saw
  a rejection (checked in `ConcurrencyReentrancyTest`).
- `OpenAIChatService` answers HTTP 409 Conflict with an OpenAI-style body
  (`{"error": {"message": ..., "type": "invalid_request_error", "param": null, "code": "session_busy"}}`).
  For streaming, the service reserves the session before the `SseEmitter` is created, so a busy session gets a plain 409
  and no stream is opened. The reservation covers both modes (a streaming request is also refused while a non-streaming
  one is in flight). It is released when the generation ends. The engine check remains the authority for any other caller
  of the model; `@ExceptionHandler(SessionBusyException)` maps what leaks through.
- Tested in `SessionConflictTest` (stubbed `Generator`, no model). The gRPC services were not changed.
