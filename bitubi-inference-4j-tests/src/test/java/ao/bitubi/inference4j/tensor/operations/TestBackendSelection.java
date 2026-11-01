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
package ao.bitubi.inference4j.tensor.operations;

import static ao.bitubi.inference4j.tensor.operations.TensorOperationsProvider.Backend.NATIVE;
import static ao.bitubi.inference4j.tensor.operations.TensorOperationsProvider.Backend.PANAMA;
import static ao.bitubi.inference4j.tensor.operations.TensorOperationsProvider.resolveBackend;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import ao.bitubi.inference4j.util.MachineSpec;
import org.junit.Test;

public class TestBackendSelection {

    @Test
    public void autoPicksPanamaWhenVectorsExist() {
        assertEquals(PANAMA, resolveBackend(null, MachineSpec.Type.AVX_256));
        assertEquals(PANAMA, resolveBackend("auto", MachineSpec.Type.AVX_512));
        assertEquals(PANAMA, resolveBackend("", MachineSpec.Type.ARM_128));
    }

    @Test
    public void explicitChoiceIsHonoredAndCaseInsensitive() {
        assertEquals(PANAMA, resolveBackend("PANAMA", MachineSpec.Type.AVX_256));
        assertEquals(NATIVE, resolveBackend("Native", MachineSpec.Type.AVX_256));
    }

    @Test
    public void onlyTheBackendPropertyChoosesNative() {
        assertEquals(PANAMA, resolveBackend(null, MachineSpec.Type.AVX_256));
        assertEquals(NATIVE, resolveBackend("native", MachineSpec.Type.AVX_256));
    }

    @Test
    public void theRemovedNativeFlagDoesNotSelectNative() {
        System.setProperty("bi4j.use_native_tensor_operations", "true");
        System.setProperty("jlama.use_native_tensor_operations", "true");
        try {
            assertEquals(PANAMA, resolveBackend(null, MachineSpec.Type.AVX_256));
            assertEquals(PANAMA, resolveBackend("auto", MachineSpec.Type.AVX_256));
        } finally {
            System.clearProperty("bi4j.use_native_tensor_operations");
            System.clearProperty("jlama.use_native_tensor_operations");
        }
    }

    @Test
    public void withoutTheVectorApiAutoAndPanamaFailWithActionableMessage() {
        for (String requested : new String[] { null, "auto", "panama" }) {
            try {
                resolveBackend(requested, MachineSpec.Type.NONE);
                fail("expected IllegalArgumentException for " + requested);
            } catch (IllegalArgumentException e) {
                assertTrue(e.getMessage(), e.getMessage().contains("jdk.incubator.vector"));
            }
        }
    }

    @Test
    public void theNaiveBackendNoLongerExists() {
        try {
            resolveBackend("naive", MachineSpec.Type.AVX_256);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Valid values: auto, panama, native."));
        }
    }

    @Test
    public void unknownBackendListsTheValidOnes() {
        try {
            resolveBackend("gpu", MachineSpec.Type.AVX_256);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            String m = e.getMessage();
            assertTrue(m, m.contains("gpu") && m.contains("auto") && m.contains("panama") && m.contains("native") && !m.contains("naive,"));
        }
    }

    @Test
    public void onlyACpuBelowTheNativeFloorFallsBackToPanama() {
        assertTrue(TensorOperationsProvider.belowNativeFloor(new java.lang.reflect.InvocationTargetException(new UnsupportedOperationException("x"))));
        assertTrue(!TensorOperationsProvider.belowNativeFloor(new ClassNotFoundException("x")));
        assertTrue(!TensorOperationsProvider.belowNativeFloor(new java.lang.reflect.InvocationTargetException(new IllegalStateException("x"))));
    }

    @Test
    public void aMissingNativeModuleAsksForTheArtifact() {
        String m = TensorOperationsProvider.nativeUnavailable(new ClassNotFoundException("x")).getMessage();
        assertTrue(m, m.contains("classpath"));
    }
}
