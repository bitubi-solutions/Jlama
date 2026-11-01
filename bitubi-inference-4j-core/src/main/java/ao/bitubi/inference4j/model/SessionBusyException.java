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

import java.util.UUID;

/**
 * Thrown when a generation is requested on a session that is already generating. Concurrent requests on the
 * same session are rejected instead of queued, because they would share (and corrupt) the session's KV cache.
 */
public class SessionBusyException extends IllegalStateException {

    private final UUID sessionId;

    public SessionBusyException(UUID sessionId) {
        super("Session " + sessionId + " is already generating; concurrent requests on the same session are rejected");
        this.sessionId = sessionId;
    }

    public UUID getSessionId() {
        return sessionId;
    }
}
