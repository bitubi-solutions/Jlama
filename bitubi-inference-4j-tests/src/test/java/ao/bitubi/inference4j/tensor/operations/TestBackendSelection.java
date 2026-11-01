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

import static ao.bitubi.inference4j.tensor.operations.TensorOperationsProvider.Backend.NAIVE;
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
        assertEquals(PANAMA, resolveBackend(null, false, MachineSpec.Type.AVX_256));
        assertEquals(PANAMA, resolveBackend("auto", false, MachineSpec.Type.AVX_512));
        assertEquals(PANAMA, resolveBackend("", false, MachineSpec.Type.ARM_128));
    }

    @Test
    public void autoFallsBackToNaiveWithoutVectors() {
        assertEquals(NAIVE, resolveBackend(null, false, MachineSpec.Type.NONE));
        assertEquals(NAIVE, resolveBackend("auto", false, MachineSpec.Type.NONE));
    }

    @Test
    public void explicitChoiceIsHonoredAndCaseInsensitive() {
        assertEquals(NAIVE, resolveBackend("naive", false, MachineSpec.Type.AVX_256));
        assertEquals(PANAMA, resolveBackend("PANAMA", false, MachineSpec.Type.AVX_256));
        assertEquals(NATIVE, resolveBackend("Native", false, MachineSpec.Type.AVX_256));
    }

    @Test
    public void legacyNativeFlagStillSelectsNativeWhenBackendIsAuto() {
        assertEquals(NATIVE, resolveBackend(null, true, MachineSpec.Type.AVX_256));
        // an explicit backend wins over the legacy flag
        assertEquals(NAIVE, resolveBackend("naive", true, MachineSpec.Type.AVX_256));
    }

    @Test
    public void panamaWithoutVectorsFailsWithActionableMessage() {
        try {
            resolveBackend("panama", false, MachineSpec.Type.NONE);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("jdk.incubator.vector"));
        }
    }

    @Test
    public void unknownBackendListsTheValidOnes() {
        try {
            resolveBackend("gpu", false, MachineSpec.Type.AVX_256);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            String m = e.getMessage();
            assertTrue(m, m.contains("gpu") && m.contains("auto") && m.contains("panama") && m.contains("naive") && m.contains("native"));
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
