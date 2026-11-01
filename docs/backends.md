# Backends de operações de tensor

Data da medição: 2026-10-12. Estado verificado no JDK 25, neste repositório.

## O que existe hoje

São **3 backends** de operações de tensor. Não há backend de GPU (o WebGPU é um spike por avaliar, sub-projeto 1b).

| Backend | Valor de `bi4j.backend` | O que é | Estado |
|---|---|---|---|
| Panama | `panama` | Java puro com a Vector API (`jdk.incubator.vector`). Caminhos para AVX-512, AVX2 (256 bits) e ARM de 128 bits | **Funciona e é o padrão** |
| Native | `native` | C com SIMD, chamado por FFM (módulo `bitubi-inference-4j-native`) | **Produz resposta errada no JDK 25**; só por opt-in |
| Naive | `naive` | Java escalar, sem vetores | **Falha** com modelos Q4 (ver abaixo); só entra no `auto` se a máquina não tiver vetores |

## Como escolher

```bash
java -Dbi4j.backend=panama ...      # auto (padrão) | panama | naive | native
```

- `auto` (padrão): Panama quando há vetores; senão Naive. **Nunca escolhe o nativo sozinho.**
- O prefixo antigo `jlama.backend` é aceito. A flag antiga `use_native_tensor_operations=true` continua a selecionar o nativo quando o backend é `auto`.
- Uma escolha explícita que não pode ser atendida falha com mensagem que diz o que fazer: `panama` sem a Vector API ativa pede `--add-modules=jdk.incubator.vector`; `native` sem o módulo no classpath pede o artefato; um nome desconhecido lista os válidos.
- O backend em uso aparece no log (`Using Panama Vector Operations (OffHeap)`) e na linha `PERF` do teste de desempenho.

## Medição

Qwen2.5-0.5B-Instruct quantizado (`tjake/Qwen2.5-0.5B-Instruct-JQ4`, memória de trabalho F32 e I8), prompt fixo,
temperatura 0, 2 execuções de aquecimento e 5 medidas por rodada. Máquina: Intel Core Ultra 7 255H, 16 CPUs,
AVX2 (sem AVX-512), WSL2. Medidas seguidas na mesma sessão, via `PERF_BASELINE=1`
(`bitubi-inference-4j-tests`, `PerfBaselineTest`).

| Backend | Rodada | Prefill tok/s | Decode tok/s | Pico de RSS (MB) | Resposta correta? |
|---|---|---|---|---|---|
| Panama | 1 | 24,64 | 8,03 | 822 | sim ("The capital of France is Paris.") |
| Panama | 2 | 24,21 | 7,77 | 902 | sim |
| Native | 1 | 46,42 | 8,47 | 755 | **não** ("I'm sorry, but I can't assist with that.") |
| Naive | 1 | n/d | n/d | n/d | **falhou** (`IndexOutOfBoundsException` em `Q4ByteBufferTensor.make`) |

Notas sobre as métricas:
- **Memória:** o pico de RSS (`VmHWM`, Linux) é o número que conta. Os tensores ficam fora do heap Java, então `heap_used_mb` (que o teste também imprime) subestima muito. Em outros sistemas o RSS aparece como -1.
- **Velocidade:** prefill = tokens do prompt por segundo; decode = tokens gerados por segundo.

## Qual é o mais eficiente

- **Entre os que funcionam, só há um: o Panama.** É o único que dá a resposta certa de ponta a ponta. Cerca de 24 tok/s no prefill e 8 tok/s no decode, com pico de RSS de 0,8 a 0,9 GB neste modelo de 309 MB.
- **O nativo é mais rápido no prefill (cerca de 1,9x) e praticamente igual no decode (+5% a +9% conforme a rodada do Panama), com RSS 8% a 16% menor, mas a resposta está errada.** Esses números não servem para decidir enquanto o defeito existir, porque um backend que calcula errado pode ser mais rápido por razões inválidas. Se o defeito for corrigido, vale remedir.
- **O Naive não serve** para modelos Q4 hoje. O erro está no fatiamento dos tensores Q4 quando o trabalho é dividido (stack: `Q4ByteBufferTensor.make` chamado de `CausalSelfAttention.forward`); não foi investigado, e não está claro se é do backend ou do particionamento. Não há medição.

## Limites desta medição (não tire conclusões além disto)

- Um modelo, uma máquina, uma sessão; Panama com 2 rodadas e Native com 1. Sem hardware antigo (sem AVX2).
- Esta máquina varia muito entre sessões: o baseline de 2026-10-11 (`docs/benchmarks/baseline-intel-core-ultra-7-255h.md`) mediu o Panama com prefill de 7,8 a 17,4 tok/s, abaixo das medições de hoje. Compare só números tirados em sequência, na mesma sessão.
- Na CI, em runners virtuais, o Panama deu (uma execução cada): Linux 20,3 / 12,6, Windows 21,4 / 14,1, macOS Intel 13,3 / 7,6 e macOS ARM (M1 virtual) 6,2 / 2,6 tok/s de prefill / decode. O Apple Silicon só funciona porque a PR #2 acrescentou o kernel `F32 x Q4` de 128 bits ao Panama.

## Defeitos abertos que esta medição expôs

1. **Backend nativo errado no JDK 25** (causa não investigada). Reproduzível com `-Dbi4j.backend=native` e o `SmokeTest`.
2. **Backend Naive quebra com Q4.** Reproduzível com `PERF_BASELINE=1 PERF_WARMUP=0 PERF_RUNS=1 PERF_MAX_TOKENS=80` e `-Dbi4j.backend=naive`.

## Reproduzir

```bash
export JAVA_HOME=$HOME/.sdkman/candidates/java/25.0.3-amzn
JDK_JAVA_OPTIONS="-Dbi4j.backend=panama" PERF_BASELINE=1 ./mvnw -B -pl bitubi-inference-4j-tests -am package \
  -Dtest=PerfBaselineTest -Dsurefire.failIfNoSpecifiedTests=false
# variáveis opcionais: PERF_WARMUP, PERF_RUNS, PERF_MAX_TOKENS (o total de tokens inclui o prompt)
```
