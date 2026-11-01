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
 *
 * Modifications Copyright 2026 Bitubi Solutions
 */
package ao.bitubi.inference4j.util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads configuration from JVM system properties with the {@code bi4j.} prefix. The names inherited from Jlama
 * ({@code jlama.*} properties, {@code JLAMA_*} environment variables) were removed: they are never applied, but
 * each one found is reported once with a WARN that names its replacement.
 */
public final class Bi4jProperties {

    private static final Logger logger = LoggerFactory.getLogger(Bi4jProperties.class);

    private static final String PREFIX = "bi4j.";
    private static final String REMOVED_PREFIX = "jlama.";
    private static final String ENV_PREFIX = "BI4J_";
    private static final String REMOVED_ENV_PREFIX = "JLAMA_";

    /** Removed backend flags; the backend is chosen only with {@code bi4j.backend}. */
    private static final Map<String, String> REMOVED_FLAGS = Map.of(
        "jlama.use_native_tensor_operations",
        "bi4j.backend=native",
        "bi4j.use_native_tensor_operations",
        "bi4j.backend=native",
        "jlama.force_panama_tensor_operations",
        "bi4j.backend=panama"
    );

    private static final Set<String> warned = new HashSet<>();

    static {
        warnAboutRemovedNames(System.getProperties(), System.getenv(), logger::warn);
    }

    private Bi4jProperties() {}

    /** Value of {@code bi4j.<name>}, else {@code null}. */
    public static String getString(String name) {
        return System.getProperty(PREFIX + name);
    }

    /** True if {@code bi4j.<name>} is set to "true". */
    public static boolean getBoolean(String name) {
        return Boolean.getBoolean(PREFIX + name);
    }

    /** Value of {@code bi4j.<name>}, else {@code defaultValue}. */
    public static Integer getInteger(String name, int defaultValue) {
        return Integer.getInteger(PREFIX + name, defaultValue);
    }

    /** One message per removed property or environment variable that is set, naming its replacement. */
    static List<String> removedNames(Properties properties, Map<String, String> env) {
        List<String> messages = new ArrayList<>();
        for (String key : new TreeSet<>(properties.stringPropertyNames())) {
            if (REMOVED_FLAGS.containsKey(key)) messages.add(key + " was removed; use " + REMOVED_FLAGS.get(key));
            else if (key.startsWith(REMOVED_PREFIX)) messages.add(
                key + " was removed; use " + PREFIX + key.substring(REMOVED_PREFIX.length())
            );
        }
        for (String key : new TreeSet<>(env.keySet())) {
            if (key.startsWith(REMOVED_ENV_PREFIX)) messages.add(
                key + " was removed; use " + ENV_PREFIX + key.substring(REMOVED_ENV_PREFIX.length())
            );
        }
        return messages;
    }

    /** Sends each message of {@link #removedNames} to {@code sink}, at most once per process. */
    static synchronized void warnAboutRemovedNames(Properties properties, Map<String, String> env, Consumer<String> sink) {
        for (String message : removedNames(properties, env)) {
            if (warned.add(message)) sink.accept(message);
        }
    }
}
