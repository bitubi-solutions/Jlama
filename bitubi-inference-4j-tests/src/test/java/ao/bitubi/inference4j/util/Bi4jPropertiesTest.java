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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.junit.After;
import org.junit.Test;

public class Bi4jPropertiesTest {

    @After
    public void clean() {
        System.clearProperty("bi4j.test_flag");
        System.clearProperty("jlama.test_flag");
        System.clearProperty("bi4j.test_int");
        System.clearProperty("jlama.test_int");
        System.clearProperty("bi4j.test_str");
        System.clearProperty("jlama.test_str");
    }

    @Test
    public void booleanDefaultsToFalse() {
        assertFalse(Bi4jProperties.getBoolean("test_flag"));
    }

    @Test
    public void booleanReadsBi4jPrefix() {
        System.setProperty("bi4j.test_flag", "true");
        assertTrue(Bi4jProperties.getBoolean("test_flag"));
    }

    @Test
    public void integerDefaultsThenBi4jPrefix() {
        assertEquals(Integer.valueOf(7), Bi4jProperties.getInteger("test_int", 7));
        System.setProperty("bi4j.test_int", "13");
        assertEquals(Integer.valueOf(13), Bi4jProperties.getInteger("test_int", 7));
    }

    @Test
    public void removedJlamaPrefixIsIgnored() {
        System.setProperty("jlama.test_flag", "true");
        System.setProperty("jlama.test_int", "11");
        System.setProperty("jlama.test_str", "x");
        assertFalse(Bi4jProperties.getBoolean("test_flag"));
        assertEquals(Integer.valueOf(7), Bi4jProperties.getInteger("test_int", 7));
        assertNull(Bi4jProperties.getString("test_str"));
    }

    @Test
    public void removedPropertyPointsToTheNewName() {
        Properties p = new Properties();
        p.setProperty("jlama.max_batch_size", "64");
        assertEquals(List.of("jlama.max_batch_size was removed; use bi4j.max_batch_size"), Bi4jProperties.removedNames(p, Map.of()));
    }

    @Test
    public void removedBackendFlagsPointToBi4jBackend() {
        Properties p = new Properties();
        p.setProperty("jlama.use_native_tensor_operations", "true");
        p.setProperty("bi4j.use_native_tensor_operations", "true");
        p.setProperty("jlama.force_panama_tensor_operations", "true");
        List<String> w = Bi4jProperties.removedNames(p, Map.of());
        assertTrue(w.toString(), w.contains("jlama.use_native_tensor_operations was removed; use bi4j.backend=native"));
        assertTrue(w.toString(), w.contains("bi4j.use_native_tensor_operations was removed; use bi4j.backend=native"));
        assertTrue(w.toString(), w.contains("jlama.force_panama_tensor_operations was removed; use bi4j.backend=panama"));
        assertEquals(3, w.size());
    }

    @Test
    public void removedEnvironmentVariablePointsToTheNewName() {
        List<String> w = Bi4jProperties.removedNames(new Properties(), Map.of("JLAMA_JVM_ARGS", "-Xmx1g", "BI4J_JVM_ARGS", "", "PATH", "/bin"));
        assertEquals(List.of("JLAMA_JVM_ARGS was removed; use BI4J_JVM_ARGS"), w);
    }

    @Test
    public void currentNamesProduceNoWarning() {
        Properties p = new Properties();
        p.setProperty("bi4j.backend", "panama");
        p.setProperty("bi4j.debug", "true");
        assertTrue(Bi4jProperties.removedNames(p, Map.of("BI4J_JVM_ARGS_EXTRA", "-Xmx1g")).isEmpty());
    }

    @Test
    public void eachRemovedNameIsWarnedOnlyOnce() {
        Properties p = new Properties();
        p.setProperty("jlama.warn_once_probe", "1");
        List<String> logged = new ArrayList<>();
        Bi4jProperties.warnAboutRemovedNames(p, Map.of("JLAMA_WARN_ONCE_PROBE", "1"), logged::add);
        Bi4jProperties.warnAboutRemovedNames(p, Map.of("JLAMA_WARN_ONCE_PROBE", "1"), logged::add);
        assertEquals(
            List.of("jlama.warn_once_probe was removed; use bi4j.warn_once_probe", "JLAMA_WARN_ONCE_PROBE was removed; use BI4J_WARN_ONCE_PROBE"),
            logged
        );
    }
}
