<p align="center">
  <img src="docs/logo.svg" width="160" height="160" alt="bitubi-inference-4j">
</p>

# bitubi-inference-4j

Motor de inferência de LLMs para Java, mantido pela [Bitubi Solutions](https://github.com/bitubi-solutions).
Roda modelos diretamente na JVM, usando a [Vector API](https://openjdk.org/jeps/529) (Panama) para
acelerar a CPU, e expõe uma CLI, uma API compatível com OpenAI e inferência distribuída.

> **Origem.** Este projeto é um fork mantido do [Jlama](https://github.com/tjake/jlama)
> (© T Jake Luciani e colaboradores, Apache 2.0). Veja o arquivo [`NOTICE`](./NOTICE).

## Funcionalidades

Modelos:
  * Gemma e Gemma 2
  * Llama, Llama 2 e Llama 3
  * Mistral e Mixtral
  * Qwen2
  * IBM Granite
  * GPT-2
  * BERT
  * Tokenizers BPE e WordPiece

Implementa:
  * Paged attention
  * Mixture of Experts
  * Tool calling
  * Embeddings
  * Classificadores
  * Modelos e tokenizers no formato [SafeTensors](https://github.com/huggingface/safetensors) do Hugging Face
  * Tipos F32, F16 e BF16
  * Quantização Q8 e Q4
  * GEMM rápido (Panama / Vector API por padrão)
  * Inferência distribuída

Requer **JDK 25 ou superior** (a Vector API ainda é um módulo em incubação: `--add-modules jdk.incubator.vector`).
Só o JDK 25 é testado; versões mais novas podem alterar a Vector API.

Backends de tensores: o padrão é Panama (`-Dbi4j.backend=auto|panama|naive|native`). O nativo nunca é escolhido sozinho
(era incorreto por um defeito no cálculo dos blocos de borda, já corrigido; falta torná-lo portável entre CPUs); o `naive` falha com modelos Q4. Detalhes e medições em
[`docs/backends.md`](./docs/backends.md).

## Início rápido

Compile e rode a CLI (comando `bi4j`):

```shell
export JAVA_HOME=/caminho/para/jdk-25
./mvnw -B clean package -DskipTests
./run-cli.sh restapi tjake/Llama-3.2-1B-Instruct-JQ4 --auto-download
```

Abra http://localhost:8080/ no navegador.

```
Uso:

bi4j [COMANDO]

Inferência:
  chat                 Conversa com o modelo
  restapi              API REST compatível com OpenAI
  complete             Completa um prompt

Inferência distribuída:
  cluster-coordinator  API REST distribuída usando workers
  cluster-worker       Conecta a um coordinator para inferência distribuída

Outros:
  download             Baixa um modelo do Hugging Face (formato dono/nome)
  list                 Lista modelos locais
  quantize             Quantiza um modelo
```

Há modelos pré-quantizados em https://hf.co/tjake.

## Usar em um projeto Java

Os artefatos usam o `groupId` `ao.bitubi`. A publicação começa no GitHub Packages
(`https://maven.pkg.github.com/bitubi-solutions/bitubi-inference-4j`), que exige um token com `read:packages`.

```xml
<dependency>
  <groupId>ao.bitubi</groupId>
  <artifactId>bitubi-inference-4j-core</artifactId>
  <version>${bitubi-inference-4j.version}</version>
</dependency>
```

Ative a Vector API na JVM: `--add-modules jdk.incubator.vector --enable-preview`.

```java
public void sample() throws IOException {
    String model = "tjake/Llama-3.2-1B-Instruct-JQ4";
    String workingDirectory = "./models";
    String prompt = "What is the best season to plant avocados?";

    // Baixa o modelo, ou devolve o caminho local se já estiver baixado
    File localModelPath = SafeTensorSupport.maybeDownloadModel(workingDirectory, model);

    // Carrega o modelo quantizado
    AbstractModel m = ModelSupport.loadModel(localModelPath, DType.F32, DType.I8);

    PromptContext ctx;
    if (m.promptSupport().isPresent()) {
        ctx = m.promptSupport()
                .get()
                .builder()
                .addSystemMessage("You are a helpful chatbot who writes short responses.")
                .addUserMessage(prompt)
                .build();
    } else {
        ctx = PromptContext.of(prompt);
    }

    // O quarto argumento é o limite total de tokens (prompt incluído)
    Generator.Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, (s, f) -> {});
    System.out.println(r.responseText);
}
```

Ou com a **Builder API**:

```java
Generator.Response r = m.generateBuilder()
        .session(UUID.randomUUID())            // por padrão, UUID.randomUUID()
        .promptContext(ctx)                    // obrigatório, ou prompt(String text)
        .ntokens(256)                          // por padrão, 256
        .temperature(0.0f)                     // por padrão, 0.0f
        .onTokenWithTimings((s, t) -> {})      // por padrão, não faz nada
        .generate();
```

## Roteiro

* Suporte a GGUF e a mais modelos, com foco em hardware antigo
* API Anthropic e melhorias na API OpenAI
* UI com htmx + JTE sobre Spring Boot 4
* Coordinator/worker pronto para a cloud
* Backend de GPU opcional via WebGPU (em avaliação)
* Harness de agentes na CLI

## Licença e origem

O código está sob a [Apache License 2.0](./LICENSE). Contribuições seguem o
[guia de contribuição](./CONTRIBUTING.md) (commits assinados, DCO).

Se este trabalho for útil na sua pesquisa, cite o projeto original:

```
@misc{jlama2024,
    title = {Jlama: A modern Java inference engine for large language models},
    url = {https://github.com/tjake/jlama},
    author = {T Jake Luciani},
    month = {January},
    year = {2024}
}
```
