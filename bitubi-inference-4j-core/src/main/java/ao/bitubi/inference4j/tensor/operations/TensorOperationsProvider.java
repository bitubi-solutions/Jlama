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
package ao.bitubi.inference4j.tensor.operations;

import ao.bitubi.inference4j.util.Bi4jProperties;
import ao.bitubi.inference4j.util.MachineSpec;
import java.lang.reflect.InvocationTargetException;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Picks the tensor operations backend. Select it with {@code -Dbi4j.backend=<auto|panama|naive|native>}
 * (the legacy {@code jlama.} prefix is also accepted). {@code auto}, the default, uses Panama (the Vector API)
 * when the machine has vectors and falls back to the scalar backend otherwise. The native backend is never
 * chosen automatically: on JDK 25 it produced wrong results in the smoke test.
 */
public class TensorOperationsProvider {
    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    public enum Backend {
        PANAMA,
        NAIVE,
        NATIVE
    }

    private static final Logger logger = LoggerFactory.getLogger(TensorOperationsProvider.class);

    private static final String lock = "lock";
    private static TensorOperationsProvider instance;

    public static TensorOperations get() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) instance = new TensorOperationsProvider();
            }
        }

        return instance.provider;
    }

    private final TensorOperations provider;

    private TensorOperationsProvider() {
        this.provider = pickImplementation();
    }

    /**
     * Resolves which backend to use.
     *
     * @param requested value of {@code bi4j.backend}; null, empty or "auto" means automatic
     * @param legacyNativeFlag value of the old {@code use_native_tensor_operations} flag (only used when auto)
     * @param vectorType the vector support detected on this machine
     */
    static Backend resolveBackend(String requested, boolean legacyNativeFlag, MachineSpec.Type vectorType) {
        String r = requested == null ? "auto" : requested.trim().toLowerCase(Locale.ROOT);
        if (r.isEmpty()) r = "auto";

        switch (r) {
            case "auto":
                if (legacyNativeFlag) return Backend.NATIVE;
                return vectorType == MachineSpec.Type.NONE ? Backend.NAIVE : Backend.PANAMA;
            case "panama":
                if (vectorType == MachineSpec.Type.NONE) throw new IllegalArgumentException(
                    "bi4j.backend=panama needs the Java Vector API, which is not available. "
                        + "Add --add-modules=jdk.incubator.vector to the JVM options, or use bi4j.backend=naive."
                );
                return Backend.PANAMA;
            case "naive":
                return Backend.NAIVE;
            case "native":
                return Backend.NATIVE;
            default:
                throw new IllegalArgumentException(
                    "Unknown bi4j.backend '" + requested + "'. Valid values: auto, panama, naive, native."
                );
        }
    }

    /** An explicit choice that cannot be met fails: {@code newInstance} wraps what the constructor throws. */
    static IllegalStateException nativeUnavailable(Throwable t) {
        Throwable cause = t instanceof InvocationTargetException && t.getCause() != null ? t.getCause() : t;
        if (cause instanceof UnsupportedOperationException) return new IllegalStateException(
            cause.getMessage() + ". Use bi4j.backend=panama.",
            cause
        );
        return new IllegalStateException(
            "The native backend was requested but is not available. Add 'ao.bitubi:bitubi-inference-4j-native' "
                + "to the classpath, or use bi4j.backend=panama.",
            t
        );
    }

    private TensorOperations pickImplementation() {
        Backend backend = resolveBackend(
            Bi4jProperties.getString("backend"),
            Bi4jProperties.getBoolean("use_native_tensor_operations"),
            MachineSpec.VECTOR_TYPE
        );

        TensorOperations pick;
        switch (backend) {
            case NATIVE:
                try {
                    Class<? extends TensorOperations> nativeClazz = (Class<? extends TensorOperations>) Class.forName(
                        "ao.bitubi.inference4j.tensor.operations.NativeSimdTensorOperations"
                    );
                    pick = nativeClazz.getConstructor().newInstance();
                } catch (Throwable t) {
                    throw nativeUnavailable(t);
                }
                break;
            case NAIVE:
                pick = new NaiveTensorOperations();
                break;
            default:
                pick = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
        }

        logger.info("Using {} ({})", pick.name(), "OffHeap");
        return pick;
    }
}
