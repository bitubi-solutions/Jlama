# Migrating from Jlama to bitubi-inference-4j

bitubi-inference-4j is a maintained fork of [Jlama](https://github.com/tjake/jlama) (Apache 2.0, © T Jake Luciani
and contributors; see [`NOTICE`](../NOTICE)). It has **no backwards compatibility with Jlama**: every Jlama name
listed below was renamed or removed, and the old names are not accepted. This guide lists every change and how to
apply it, and [`docs/migration/rewrite.yml`](./migration/rewrite.yml) automates most of it (see
[Automated migration with OpenRewrite](#automated-migration-with-openrewrite)).

**How these tables were verified.** The baseline is Jlama 0.8.4 as it was when the fork was taken
(`<revision>0.8.4</revision>` in the upstream `pom.xml`, commit `c12e3f2`; commit `73c431d` is the last commit
with the Jlama names). The rename is the mechanical commit `80f5ab0` plus `4b4e276` (runtime contracts) and
`c34e188`/`7e3c91b` (JDK 25, backend default); the removal of the legacy names is the commit series of the
`chore/remove-legacy-contracts` branch. Every row was checked with `git grep` / `git diff -M` between those
commits and the current tree. In the 128 `.java` files touched by `80f5ab0`, every changed line is a package, import
or `Jlama` → `Bi4j` identifier change, apart from the renames in the tables below.

Contents: [Maven](#maven-coordinates) · [Packages](#java-packages) · [Classes](#renamed-classes) ·
[System properties](#system-properties) · [Environment variables](#environment-variables) ·
[HTTP](#http-headers) · [gRPC](#grpc) · [CLI](#cli) · [Deployment](#deployment-artifacts) ·
[Models](#supported-model-formats) · [Before/after](#before-and-after) · [OpenRewrite](#automated-migration-with-openrewrite) ·
[Roadmap: LangChain4j](#roadmap-langchain4j)

## What did not change

* The SDK API: method names, signatures and behavior of `ModelSupport.loadModel`, `AbstractModel.generate`,
  `generateBuilder()`, `PromptContext`, `Downloader`, `SafeTensorSupport.maybeDownloadModel`, and so on.
  Only package and class names changed.
* The CLI subcommands and their options: `chat`, `restapi`, `complete`, `list`, `download`, `quantize`,
  `cluster-coordinator`, `cluster-worker`.
* The HTTP routes (`/chat/completions`, ...) and the OpenAI-compatible payloads.
* The Hugging Face model ids (`tjake/...-JQ4`, `...-Jlama-Q4`) and `https://hf.co/tjake`: they are names of
  external repositories.
* The Apache 2.0 license headers and the credit to the original authors.

## Maven coordinates

Version: Jlama `0.8.4` → bitubi-inference-4j `0.1.0-SNAPSHOT` (`<revision>` in the parent `pom.xml`; a release is
published manually by the maintainer). Artifacts are published to GitHub Packages, not Maven Central: see the
README for the repository and the token (`read:packages`) it requires.

| Jlama | bitubi-inference-4j |
|---|---|
| `com.github.tjake:jlama-parent` | `ao.bitubi:bitubi-inference-4j-parent` |
| `com.github.tjake:jlama-core` | `ao.bitubi:bitubi-inference-4j-core` |
| `com.github.tjake:jlama-native` | `ao.bitubi:bitubi-inference-4j-native` |
| `com.github.tjake:jlama-net` | `ao.bitubi:bitubi-inference-4j-net` |
| `com.github.tjake:jlama-cli` | `ao.bitubi:bitubi-inference-4j-cli` |
| `com.github.tjake:jlama-tests` | `ao.bitubi:bitubi-inference-4j-tests` |

The native artifact keeps the platform classifier (`${os.detected.name}-${os.detected.arch}`); its shared library
is now `libbi4j` (was `libjlama`). JDK: Jlama needed Java 20+; this project needs **JDK 25** (the only target for
build, test and run), with `--add-modules jdk.incubator.vector --enable-preview`.

## Java packages

Every package moves by prefix: `com.github.tjake.jlama.` → `ao.bitubi.inference4j.`, keeping the sub-package.

| Jlama | bitubi-inference-4j |
|---|---|
| `com.github.tjake.jlama.math` | `ao.bitubi.inference4j.math` |
| `com.github.tjake.jlama.model` (and `.bert`, `.functions`, `.gemma`, `.gemma2`, `.gpt2`, `.granite`, `.llama`, `.mistral`, `.mixtral`, `.qwen2`) | `ao.bitubi.inference4j.model` (same sub-packages) |
| `com.github.tjake.jlama.safetensors` (and `.prompt`, `.tokenizer`) | `ao.bitubi.inference4j.safetensors` (same sub-packages) |
| `com.github.tjake.jlama.tensor` (and `.operations`) | `ao.bitubi.inference4j.tensor` (same sub-packages) |
| `com.github.tjake.jlama.util` | `ao.bitubi.inference4j.util` |
| `com.github.tjake.jlama.net` (and `.grpc`, `.openai`) | `ao.bitubi.inference4j.net` (same sub-packages) |
| `com.github.tjake.jlama.cli` (and `.commands`) | `ao.bitubi.inference4j.cli` (same sub-packages) |

Checked: every main-source `.java` file of Jlama at `73c431d` has a counterpart in the current tree after applying
this prefix and the class renames below; the only one without a counterpart is `NaiveTensorOperations` (see below).
`NativeSimd` and `MemorySegmentSupport` live in `src/main/java25` of the native module in both trees.

## Renamed classes

| Jlama | bitubi-inference-4j |
|---|---|
| `com.github.tjake.jlama.cli.JlamaCli` | `ao.bitubi.inference4j.cli.Bi4jCli` |
| `com.github.tjake.jlama.net.grpc.JlamaService` | `ao.bitubi.inference4j.net.grpc.Bi4jService` |
| `com.github.tjake.jlama.net.grpc.JlamaRingWorkerService` | `ao.bitubi.inference4j.net.grpc.Bi4jRingWorkerService` |
| `com.github.tjake.jlama.util.JsonSupport.JlamaPrettyPrinter` | `ao.bitubi.inference4j.util.JsonSupport.Bi4jPrettyPrinter` |
| `com.github.tjake.jlama.net.JlamaServiceGrpc` (generated) | `ao.bitubi.inference4j.net.Bi4jServiceGrpc` |
| `…JlamaServiceGrpc.JlamaServiceBlockingStub`, `.JlamaServiceStub`, `.JlamaServiceImplBase` | `…Bi4jServiceGrpc.Bi4jServiceBlockingStub`, `.Bi4jServiceStub`, `.Bi4jServiceImplBase` |
| `com.github.tjake.jlama.net.JlamaWorkerRingGrpc` (generated) | `ao.bitubi.inference4j.net.Bi4jWorkerRingGrpc` |
| `…JlamaWorkerRingGrpc.JlamaWorkerRingStub`, `.JlamaWorkerRingImplBase` | `…Bi4jWorkerRingGrpc.Bi4jWorkerRingStub`, `.Bi4jWorkerRingImplBase` |

Every other class keeps its simple name (for example `AbstractModel`, `ModelSupport`, `Generator`, `PromptContext`,
`DType`, `Downloader`, `TensorOperationsProvider`).

Added: `ao.bitubi.inference4j.util.Bi4jProperties` (reads `bi4j.*` properties; see below).

Removed: `com.github.tjake.jlama.tensor.operations.NaiveTensorOperations` (the scalar backend). It is gone from
the main code; a scalar oracle for tests only lives in the test sources. Setting `naive` as the backend fails with
`Unknown bi4j.backend 'naive'. Valid values: auto, panama, native.`

The `.proto` file is `Bi4jService.proto` (was `JlamaService.proto`).

## System properties

All `jlama.*` properties were removed and **are never applied**. Use the `bi4j.` prefix. When a removed name is set
(as a system property), the engine logs a single WARN per name, for example
`jlama.max_batch_size was removed; use bi4j.max_batch_size`, and still ignores the value.

| Jlama | bitubi-inference-4j | Type, default |
|---|---|---|
| `-Djlama.debug` | `-Dbi4j.debug` | boolean, `false` (CLI log level DEBUG) |
| `-Djlama.max_batch_size` | `-Dbi4j.max_batch_size` | int, `256` |
| `-Djlama.ideal_b_params` | `-Dbi4j.ideal_b_params` | int, `3` (billions of parameters per worker) |
| `-Djlama.use_hostname_as_workerid` | `-Dbi4j.use_hostname_as_workerid` | boolean, `false` |
| `-Djlama.use_native_tensor_operations=true` (fork-only flag, never in upstream Jlama) | `-Dbi4j.backend=native` | see below |
| `-Djlama.force_panama_tensor_operations=true` | `-Dbi4j.backend=panama` | see below |

Backend selection changed in behavior, not only in name:

* Jlama 0.8.4 used the native backend whenever `jlama-native` was on the classpath, unless
  `-Djlama.force_panama_tensor_operations=true`, and fell back to a scalar backend when the Vector API was missing.
* Here `-Dbi4j.backend=auto|panama|native` is the only switch. `auto` (the default) always uses Panama (the Vector
  API) and **never picks the native backend by itself**: if you relied on the classpath to get native, set
  `-Dbi4j.backend=native`. Without the Vector API (`--add-modules=jdk.incubator.vector`) `auto` and `panama` fail
  with a message that says what to add; there is no scalar fallback.
* `jlama.use_native_tensor_operations` was an intermediate flag of this fork (commit `7e3c91b`); it is removed too,
  as is its `bi4j.` spelling. Setting any of the flags in the table logs the WARN with the replacement and does
  nothing else.

The WARN is logged when the first `bi4j.*` setting is read, once per name. See [`backends.md`](./backends.md).

## Environment variables

Only the launcher scripts and the container image read them. The `JLAMA_*` names were removed (no fallback in
`run-cli.sh`); a `JLAMA_*` variable present in the process environment is reported by the same one-time WARN.

| Jlama | bitubi-inference-4j |
|---|---|
| `JLAMA_JVM_ARGS` | `BI4J_JVM_ARGS` |
| `JLAMA_JVM_ARGS_EXTRA` | `BI4J_JVM_ARGS_EXTRA` |
| `JLAMA_PREINSTALLED_JAR` | `BI4J_PREINSTALLED_JAR` |

`HF_TOKEN` (Hugging Face token) and `HOSTNAME` are unchanged.

## HTTP headers

| Jlama | bitubi-inference-4j |
|---|---|
| `X-Jlama-Session` | `X-Bi4j-Session` |

The value is still a UUID that identifies the chat session (an invalid UUID is answered with `400`). The old
header is ignored: a request that only sends it gets a new random session, with no error.

## gRPC

The protobuf package and service names changed, so **a Jlama worker cannot talk to a bitubi-inference-4j
coordinator, or the other way round**. Upgrade a whole cluster at once.

| Jlama | bitubi-inference-4j |
|---|---|
| package `com.github.tjake.jlama.net` | package `ao.bitubi.inference4j.net` |
| `service JlamaService` | `service Bi4jService` |
| `service JlamaWorkerRing` | `service Bi4jWorkerRing` |

## CLI

| Jlama | bitubi-inference-4j |
|---|---|
| `jlama <command>` | `bi4j <command>` |
| `jbang app install --force jlama@tjake` | not available yet (jbang is planned); use `./run-cli.sh <command>` from a build |
| `jlama-cli/target/jlama-cli.jar` | `bitubi-inference-4j-cli/target/bitubi-inference-4j-cli.jar` |
| `./run-cli.sh` (reads `JLAMA_*`) | `./run-cli.sh` (reads `BI4J_*`) |

The subcommands and their options are the same (`chat`, `restapi`, `complete`, `list`, `download`, `quantize`,
`cluster-coordinator`, `cluster-worker`).

## Deployment artifacts

| Jlama | bitubi-inference-4j |
|---|---|
| Helm chart `kubernetes/helm/jlama` (`name: jlama`) | `kubernetes/helm/bi4j` (`name: bi4j`) |
| image `docker.io/tjake/jlama` | `ghcr.io/bitubi-solutions/bitubi-inference-4j` |
| Docker `ENV JLAMA_PREINSTALLED_JAR`, `JLAMA_JVM_ARGS` | `ENV BI4J_PREINSTALLED_JAR`, `BI4J_JVM_ARGS` |
| compose services `jlama-coordinator`, `jlama-worker` | `bi4j-coordinator`, `bi4j-worker` |

## Supported model formats

Unchanged by the rename; verified by diffing `ModelSupport.ModelType` (identical in both trees).

| Item | Jlama 0.8.4 | bitubi-inference-4j today |
|---|---|---|
| Container format | SafeTensors (Hugging Face layout, `config.json`) | SafeTensors |
| Architectures (`ModelType`) | `GEMMA`, `GEMMA2`, `MISTRAL`, `GRANITE`, `MIXTRAL`, `LLAMA`, `GPT2`, `BERT`, `QWEN2` | same |
| Tokenizers | BPE, WordPiece | same |
| Weight types | F32, F16, BF16; runtime quantization Q8 and Q4 | same |
| GGUF | no | **not yet** (planned; see the roadmap in the README) |

The on-disk layout is the same, so the pre-quantized models published for Jlama (for example
`tjake/Llama-3.2-1B-Instruct-JQ4`) are the ones the README examples use; loading them was not re-tested for this guide.

## Before and after

### `pom.xml`

```xml
<!-- before -->
<dependency>
  <groupId>com.github.tjake</groupId>
  <artifactId>jlama-core</artifactId>
  <version>${jlama.version}</version>
</dependency>
<dependency>
  <groupId>com.github.tjake</groupId>
  <artifactId>jlama-native</artifactId>
  <classifier>${os.detected.name}-${os.detected.arch}</classifier>
  <version>${jlama.version}</version>
</dependency>

<!-- after -->
<dependency>
  <groupId>ao.bitubi</groupId>
  <artifactId>bitubi-inference-4j-core</artifactId>
  <version>${bitubi-inference-4j.version}</version>
</dependency>
<dependency>
  <groupId>ao.bitubi</groupId>
  <artifactId>bitubi-inference-4j-native</artifactId>
  <classifier>${os.detected.name}-${os.detected.arch}</classifier>
  <version>${bitubi-inference-4j.version}</version>
</dependency>
```

Add `bitubi-inference-4j-native` only if you intend to select it with `-Dbi4j.backend=native`.

### Java

```java
// before
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;

File local = SafeTensorSupport.maybeDownloadModel("./models", "tjake/Llama-3.2-1B-Instruct-JQ4");
AbstractModel m = ModelSupport.loadModel(local, DType.F32, DType.I8);
Generator.Response r = m.generate(UUID.randomUUID(), PromptContext.of("Hi"), 0.0f, 256, (s, f) -> {});

// after: only the imports change
import ao.bitubi.inference4j.model.AbstractModel;
import ao.bitubi.inference4j.model.ModelSupport;
import ao.bitubi.inference4j.model.functions.Generator;
import ao.bitubi.inference4j.safetensors.DType;
import ao.bitubi.inference4j.safetensors.SafeTensorSupport;
import ao.bitubi.inference4j.safetensors.prompt.PromptContext;

File local = SafeTensorSupport.maybeDownloadModel("./models", "tjake/Llama-3.2-1B-Instruct-JQ4");
AbstractModel m = ModelSupport.loadModel(local, DType.F32, DType.I8);
Generator.Response r = m.generate(UUID.randomUUID(), PromptContext.of("Hi"), 0.0f, 256, (s, f) -> {});
```

### Command line and launcher

```bash
# before
JLAMA_JVM_ARGS_EXTRA="-Xmx8g -Djlama.use_native_tensor_operations=true -Djlama.debug=true" \
  jlama restapi tjake/Llama-3.2-1B-Instruct-JQ4 --auto-download

# after
BI4J_JVM_ARGS_EXTRA="-Xmx8g -Dbi4j.backend=native -Dbi4j.debug=true" \
  ./run-cli.sh restapi tjake/Llama-3.2-1B-Instruct-JQ4 --auto-download
# (drop -Dbi4j.backend=native to get the default, Panama)
```

### HTTP client

```bash
# before
curl -H 'X-Jlama-Session: 3f1e9c5e-6e5b-4b1d-9d0c-2f3a1c1d7a10' ...
# after
curl -H 'X-Bi4j-Session: 3f1e9c5e-6e5b-4b1d-9d0c-2f3a1c1d7a10' ...
```

## Automated migration with OpenRewrite

[`docs/migration/rewrite.yml`](./migration/rewrite.yml) is a declarative [OpenRewrite](https://docs.openrewrite.org)
recipe, `ao.bitubi.inference4j.MigrateFromJlama`. It is not published; run it from a checkout of this repository
(or copy the file) against your project:

```bash
cd your-project
mvn org.openrewrite.maven:rewrite-maven-plugin:run \
  -Drewrite.configLocation=/path/to/bitubi-inference-4j/docs/migration/rewrite.yml \
  -Drewrite.activeRecipes=ao.bitubi.inference4j.MigrateFromJlama
git diff   # review the changes
```

Pin a plugin version if you want a reproducible run (the recipe was validated with `rewrite-maven-plugin:6.46.1`;
the recipes it uses ship with the plugin itself, so no extra `rewrite.recipe` dependency is needed).

What the recipe does:

| Recipe | Effect |
|---|---|
| `MigrateFromJlama.Dependencies` | renames `com.github.tjake:jlama-{core,native,net,cli}` to `ao.bitubi:bitubi-inference-4j-*` in `pom.xml` |
| `MigrateFromJlama.Types` | renames the classes in the [classes table](#renamed-classes), then every `com.github.tjake.jlama.*` package, in Java sources and imports |
| `MigrateFromJlama.Contracts` | `jlama.*` keys in `.properties` and `.yaml`, `-Djlama.*` in scripts, YAML and XML, `JLAMA_*` variables, and the `X-Jlama-Session` header literal |

What it does **not** do; finish these by hand:

* **The version.** It is left as it was (it names a Jlama release that does not exist for the new artifact).
  Set the bitubi-inference-4j version you adopt, and add the GitHub Packages repository to your build.
* **Flags that map to a value, not a key.** `-Djlama.use_native_tensor_operations=true` and
  `-Djlama.force_panama_tensor_operations=true` are rewritten in scripts, YAML and XML to `-Dbi4j.backend=native`
  and `-Dbi4j.backend=panama`, but only when written exactly like that; check the result, and check any
  `jlama.use_native_tensor_operations` set in code or in a `.properties` file.
* **Ids and URLs of models, Docker images and Helm charts** (see the tables above), and the JDK upgrade to 25.
* **gRPC**: regenerate stubs from `Bi4jService.proto` if you compile the proto yourself, and upgrade all cluster
  nodes together.
* Version properties such as `${jlama.version}` in your own `pom.xml` keep their name; rename them if you like.

### What was validated

The recipe was run (`mvn org.openrewrite.maven:rewrite-maven-plugin:6.46.1:run`) on a minimal project outside this
repository that depends on `jlama-core:0.8.4` and `jlama-net:0.8.4` from Maven Central, with a class that imports
the model classes, `JlamaServiceGrpc.JlamaServiceBlockingStub` and `JlamaService`, uses the `X-Jlama-Session`
literal, a `.properties` file with `jlama.*` keys and a shell script with `-Djlama.*` and `JLAMA_JVM_ARGS_EXTRA`.
The dependency coordinates, imports, class names, literal, property keys and script flags were rewritten as
described. Not covered by that run: `.yaml` property keys, the `jlama-native` and `jlama-cli` dependencies and
`JsonSupport.JlamaPrettyPrinter` (rules written from the code, not exercised).

## Roadmap: LangChain4j

A LangChain4j integration is planned. The SDK is meant to be ready for it, and its design starts only after the
SDK developer-experience review.
