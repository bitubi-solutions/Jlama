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

import static ao.bitubi.inference4j.util.JsonSupport.om;

import ao.bitubi.inference4j.model.bert.BertConfig;
import ao.bitubi.inference4j.model.bert.BertModel;
import ao.bitubi.inference4j.model.bert.BertTokenizer;
import ao.bitubi.inference4j.model.gemma.GemmaConfig;
import ao.bitubi.inference4j.model.gemma.GemmaModel;
import ao.bitubi.inference4j.model.gemma.GemmaTokenizer;
import ao.bitubi.inference4j.model.gemma2.Gemma2Config;
import ao.bitubi.inference4j.model.gemma2.Gemma2Model;
import ao.bitubi.inference4j.model.gpt2.GPT2Config;
import ao.bitubi.inference4j.model.gpt2.GPT2Model;
import ao.bitubi.inference4j.model.gpt2.GPT2Tokenizer;
import ao.bitubi.inference4j.model.granite.GraniteConfig;
import ao.bitubi.inference4j.model.granite.GraniteModel;
import ao.bitubi.inference4j.model.llama.LlamaConfig;
import ao.bitubi.inference4j.model.llama.LlamaModel;
import ao.bitubi.inference4j.model.llama.LlamaTokenizer;
import ao.bitubi.inference4j.model.mistral.MistralConfig;
import ao.bitubi.inference4j.model.mistral.MistralModel;
import ao.bitubi.inference4j.model.mixtral.MixtralConfig;
import ao.bitubi.inference4j.model.mixtral.MixtralModel;
import ao.bitubi.inference4j.model.qwen2.Qwen2Config;
import ao.bitubi.inference4j.model.qwen2.Qwen2Model;
import ao.bitubi.inference4j.gguf.GgufConfigFactory;
import ao.bitubi.inference4j.gguf.GgufFile;
import ao.bitubi.inference4j.gguf.GgufTokenizerFactory;
import ao.bitubi.inference4j.gguf.GgufWeightLoader;
import ao.bitubi.inference4j.safetensors.Config;
import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.safetensors.SafeTensorSupport;
import ao.bitubi.inference4j.safetensors.WeightLoader;
import ao.bitubi.inference4j.safetensors.tokenizer.Tokenizer;
import ao.bitubi.inference4j.util.PhysicalCoreExecutor;
import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelSupport {
    private static final Logger logger = LoggerFactory.getLogger(ModelSupport.class);

    public enum ModelType {
        GEMMA(GemmaModel.class, GemmaConfig.class, GemmaTokenizer.class),
        GEMMA2(Gemma2Model.class, Gemma2Config.class, GemmaTokenizer.class),
        MISTRAL(MistralModel.class, MistralConfig.class, LlamaTokenizer.class),
        GRANITE(GraniteModel.class, GraniteConfig.class, GPT2Tokenizer.class),
        MIXTRAL(MixtralModel.class, MixtralConfig.class, LlamaTokenizer.class),
        LLAMA(LlamaModel.class, LlamaConfig.class, LlamaTokenizer.class),
        GPT2(GPT2Model.class, GPT2Config.class, GPT2Tokenizer.class),
        BERT(BertModel.class, BertConfig.class, BertTokenizer.class),
        QWEN2(Qwen2Model.class, Qwen2Config.class, LlamaTokenizer.class);

        public final Class<? extends AbstractModel> modelClass;
        public final Class<? extends Config> configClass;
        public final Class<? extends Tokenizer> tokenizerClass;

        ModelType(
            Class<? extends AbstractModel> modelClass,
            Class<? extends Config> configClass,
            Class<? extends Tokenizer> tokenizerClass
        ) {
            this.modelClass = modelClass;
            this.configClass = configClass;
            this.tokenizerClass = tokenizerClass;
        }
    }

    /** Shortcut for loading a model for token generation*/
    public static AbstractModel loadModel(File model, DType workingMemoryType, DType workingQuantizationType) {
        return loadModel(model, null, workingMemoryType, workingQuantizationType, Optional.empty(), Optional.empty());
    }

    /** Shortcut for loading a model for embeddings */
    public static AbstractModel loadEmbeddingModel(File model, DType workingMemoryType, DType workingQuantizationType) {
        return loadModel(
            AbstractModel.InferenceType.FULL_EMBEDDING,
            model,
            null,
            workingMemoryType,
            workingQuantizationType,
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            SafeTensorSupport::loadWeights
        );
    }

    /** Shortcut for loading a model for embeddings */
    public static AbstractModel loadClassifierModel(File model, DType workingMemoryType, DType workingQuantizationType) {
        return loadModel(
            AbstractModel.InferenceType.FULL_CLASSIFICATION,
            model,
            null,
            workingMemoryType,
            workingQuantizationType,
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            SafeTensorSupport::loadWeights
        );
    }

    public static AbstractModel loadModel(
        File model,
        File workingDirectory,
        DType workingMemoryType,
        DType workingQuantizationType,
        Optional<DType> modelQuantization,
        Optional<Integer> threadCount
    ) {
        return loadModel(
            AbstractModel.InferenceType.FULL_GENERATION,
            model,
            workingDirectory,
            workingMemoryType,
            workingQuantizationType,
            modelQuantization,
            threadCount,
            Optional.empty(),
            SafeTensorSupport::loadWeights
        );
    }

    public static AbstractModel loadModel(
        AbstractModel.InferenceType inferenceType,
        File model,
        File workingDirectory,
        DType workingMemoryType,
        DType workingQuantizationType,
        Optional<DType> modelQuantization,
        Optional<Integer> threadCount,
        Optional<Function<Config, DistributedContext>> distributedContextLoader,
        Function<File, WeightLoader> weightLoaderSupplier
    ) {

        if (!model.exists()) {
            throw new IllegalArgumentException("Model location does not exist: " + model);
        }

        File baseDir = model.isFile() ? model.getParentFile() : model;

        if (model.isFile() && model.getName().toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            try {
                threadCount.ifPresent(PhysicalCoreExecutor::overrideThreadCount);
                GgufFile gguf = new GgufFile(model.toPath());
                if (!"qwen2".equals(gguf.getMetadata().get("general.architecture")))
                    throw new IllegalArgumentException("Arquitetura GGUF nao suportada: " + gguf.getMetadata().get("general.architecture"));
                Config c = GgufConfigFactory.create(gguf.getMetadata());
                distributedContextLoader.ifPresent(loader -> c.setDistributedContext(loader.apply(c)));
                c.setWorkingDirectory(workingDirectory);
                Tokenizer t = new GPT2Tokenizer(GgufTokenizerFactory.create(gguf.getMetadata()));
                WeightLoader wl = new GgufWeightLoader(gguf);
                return Qwen2Model.class.getConstructor(AbstractModel.InferenceType.class, Config.class, WeightLoader.class,
                    Tokenizer.class, DType.class, DType.class, Optional.class)
                    .newInstance(inferenceType, c, wl, t, workingMemoryType, workingQuantizationType, modelQuantization);
            } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }

        // Find config
        if (!baseDir.isDirectory()) {
            throw new IllegalArgumentException("Model directory does not exist: " + baseDir);
        }

        File configFile = null;
        for (File f : Objects.requireNonNull(baseDir.listFiles())) {
            if (f.getName().equals("config.json")) {
                configFile = f;
                break;
            }
        }

        if (configFile == null) {
            throw new IllegalArgumentException("config.json in model directory does not exist: " + baseDir);
        }

        try {
            threadCount.ifPresent(PhysicalCoreExecutor::overrideThreadCount);

            ModelSupport.ModelType modelType = SafeTensorSupport.detectModel(configFile);
            Config c = om.readValue(configFile, modelType.configClass);
            distributedContextLoader.ifPresent(loader -> c.setDistributedContext(loader.apply(c)));

            c.setWorkingDirectory(workingDirectory);

            Tokenizer t = modelType.tokenizerClass.getConstructor(Path.class).newInstance(baseDir.toPath());
            WeightLoader wl = weightLoaderSupplier.apply(baseDir);

            return modelType.modelClass.getConstructor(
                AbstractModel.InferenceType.class,
                Config.class,
                WeightLoader.class,
                Tokenizer.class,
                DType.class,
                DType.class,
                Optional.class
            ).newInstance(inferenceType, c, wl, t, workingMemoryType, workingQuantizationType, modelQuantization);

        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
