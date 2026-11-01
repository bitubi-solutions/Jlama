# Backends de operações de tensor

Medições de 2026-10-12, JDK 25, Intel Core Ultra 7 255H (AVX2 e AVX-VNNI, sem AVX-512), WSL2.

## O que existe hoje

São **2 backends** de operações de tensor (o `naive`, escalar, foi descartado por decisão do autor em 2026-10-12). Não há backend de GPU (o WebGPU é um spike por avaliar, sub-projeto 1b).

| Backend | Valor de `bi4j.backend` | O que é | Estado |
|---|---|---|---|
| Panama | `panama` | Java puro com a Vector API (`jdk.incubator.vector`). Caminhos para AVX-512, AVX2 (256 bits) e ARM de 128 bits | **Funciona e é o padrão** |
| Native | `native` | C com SIMD, chamado por FFM (módulo `bitubi-inference-4j-native`) | **Funciona depois da correção desta branch**; continua opt-in (ver "Antes de o tornar padrão") |

## Como escolher

```bash
java -Dbi4j.backend=panama ...      # auto (padrão) | panama | native
```

- `auto` (padrão): Panama, que exige a Vector API (`--add-modules=jdk.incubator.vector`); sem ela falha com uma mensagem que diz o que fazer. **Nunca escolhe o nativo sozinho.**
- O prefixo antigo `jlama.backend` é aceito. A flag antiga `use_native_tensor_operations=true` continua a selecionar o nativo quando o backend é `auto`.
- Uma escolha explícita que não pode ser atendida falha com mensagem que diz o que fazer: `auto` e `panama` sem a Vector API ativa pedem `--add-modules=jdk.incubator.vector`; `native` sem o módulo no classpath pede o artefato; um nome desconhecido lista os válidos.
- O backend em uso aparece no log (`Using Panama Vector Operations (OffHeap)`) e na linha `PERF` do teste de desempenho.

## Variantes do nativo

Em x86-64, a biblioteca contém uma unidade base x86-64 sem instruções AVX e unidades separadas AVX2+FMA, AVX-VNNI e AVX-512+VNNI. O runtime consulta CPUID e o estado AVX do sistema operativo (`XGETBV`) antes de escolher a variante. AVX2+FMA é o mínimo para executar o backend nativo. Sem a flag antiga `use_native_tensor_operations`, `auto` continua a escolher Panama quando a Vector API está disponível; pedir `native` (ou activar a flag antiga) numa CPU abaixo do piso regista um aviso no log e usa Panama.

`bi4j.native.isa` limita a variante máxima escolhida, para executar uma variante mais baixa na mesma máquina:

```bash
java -Dbi4j.backend=native -Dbi4j.native.isa=avx2 ...
java -Dbi4j.backend=native -Dbi4j.native.isa=avxvnni ...
java -Dbi4j.backend=native -Dbi4j.native.isa=auto ...
```

Valores aceites: `auto`, `avx2`, `avxvnni`, `avx512`; em ARM64, `neon` e `neon-dotprod`. Um limite acima do suportado selecciona a variante mais alta disponível abaixo desse limite e deixa um aviso no log. A lista da CPU aparece no log ao criar o backend. Em ARM64, a unidade base é NEON; dotprod/i8mm ainda não têm variantes independentes.

No Linux x86-64, `check-isa.sh` usa `objdump` para exigir todos os objectos, confirmar que a unidade base não contém VEX/EVEX, que AVX2 não contém AVX-VNNI/AVX-512, que AVX-VNNI não contém AVX-512 e que as unidades avançadas contêm as instruções esperadas. A variante AVX-512 é compilada e inspeccionada em CPUs sem AVX-512, mas não deve ser executada nelas.

Na CI, confirmar em Linux x86-64 o `check-isa.sh`, `TestNativeRemainderTiles`, `TestOperations` e `SmokeTest` com cada ISA executável. Confirmar também build, `TestOperations` e smoke nativo em Windows x86-64, macOS Intel e macOS ARM64; em ARM64 verificar que a NEON base compila sem dotprod e que os kernels produzem a saída de referência.

## Por que o nativo dava resposta errada (causa-raiz, verificada)

**Não era o JDK 25.** O driver de blocos do GEMM, em C (`gemm`, em `vector_simd.c`) e a cópia dele em Java
(`Gemmer.mnpack`, em `PanamaTensorOperations`), dividem a matriz de saída em blocos de até 5x5. As células que
sobram (linhas que não completam um bloco, colunas que não completam um bloco) são tratadas por duas chamadas
recursivas: a faixa de baixo e a faixa da direita. **O canto onde as duas sobras se cruzam nunca era calculado.**
Essas células ficavam com o valor que já estava na memória.

- No decode (M = 1 linha) não há sobra de linhas, então o canto não existe e tudo dava certo.
- No prefill (M = tokens do prompt, ex.: 37) sobram linhas e colunas, e as últimas linhas do resultado ficavam sem calcular. O modelo recebia lixo no prompt e respondia com uma recusa.
- Isto afetava as quatro combinações de tipos do nativo (`F32xF32`, `F32xQ4`, `F32xBF16`, `I8xQ4`).
- **O Panama, o backend padrão, tinha o mesmo defeito** em `F32xF32` (12 de 70 formas testadas falhavam, por exemplo M=5, N=5). Os outros tipos do Panama escapavam porque os kernels deles usam blocos de 1 coluna ou 1 linha.

Correção: uma terceira chamada recursiva para o canto, nos dois lugares. Teste: `TestNativeRemainderTiles` compara os dois
backends, em 4 combinações de tipos e 70 formas (M em 1..37, N em 1..112), contra um oráculo de laços simples; falhava antes
(nativo: 15 formas erradas por tipo; Panama: 12 em `F32xF32`) e passa agora.

Como o defeito foi achado: `ShadowNativeInvestigation` (opt-in com `SHADOW=1`) roda o modelo no Panama e repete cada operação no
nativo sobre cópias, relatando onde divergem. Uma primeira versão deu falsos positivos porque cada chamada comparava o resultado
inteiro enquanto outras threads preenchiam outros pedaços; a versão atual usa buffers privados com valor sentinela.

## Medição (depois da correção)

Qwen2.5-0.5B-Instruct quantizado (`tjake/Qwen2.5-0.5B-Instruct-JQ4`, memória de trabalho F32 e I8), prompt fixo, temperatura 0,
2 execuções de aquecimento e 5 medidas por rodada (`PERF_BASELINE=1`, `PerfBaselineTest`). Rodadas **intercaladas na mesma sessão**,
com a saída conferida contra a referência (o teste de fumaça passa nos dois).

| Rodada | Backend | Prefill tok/s | Decode tok/s | Pico de RSS (MB) | Carga (ms) |
|---|---|---|---|---|---|
| 1 | Panama | 47,5 | 17,4 | 1132 | 2254 |
| 2 | Native | 109,3 | 20,1 | 786 | 4187 |
| 3 | Panama | 39,4 | 15,9 | 1143 | 3613 |
| 4 | Native | 102,1 | 17,8 | 856 | 4750 |

| | Panama (média de 2) | Native (média de 2) | Razão |
|---|---|---|---|
| Prefill | 43,5 | 105,7 | **cerca de 2,4x** |
| Decode | 16,7 | 19,0 | +14% (as faixas se sobrepõem: 15,9 a 17,4 contra 17,8 a 20,1) |
| Pico de RSS | 1138 MB | 821 MB | **cerca de 28% menos** |
| Carga do modelo | 2,9 s | 4,5 s | nativo mais lento a carregar |

Notas:
- **Memória:** o número que conta é o pico de RSS (`VmHWM`, Linux). Os tensores ficam fora do heap Java, então `heap_used_mb` subestima muito. Em outros sistemas o RSS aparece como -1.
- **Velocidade:** prefill = tokens do prompt por segundo; decode = tokens gerados por segundo.
- **Ruído:** esta máquina varia muito entre sessões (uma hora antes o Panama media 24 e 8 tok/s). Compare só rodadas intercaladas na mesma sessão. Duas rodadas por backend dão a ordem de grandeza, não um intervalo de confiança.

## Qual é o mais eficiente

Entre os que funcionam: o **nativo**, neste hardware e neste modelo, no prefill (cerca de 2,4x) e na memória (cerca de 28% menos de RSS); no decode a vantagem é pequena e está dentro do ruído. O Panama continua o padrão por não depender de uma biblioteca compilada por sistema operacional.

## Onde vai a memória (Native Memory Tracking e JFR)

- A diferença de pico de RSS entre Panama e nativo foi quase toda **heap Java comprometido** (585 MB contra 328 MB numa medição); JIT (Code 19,6 contra 20,0 MB), classes e threads são iguais. Os tensores ficam fora do heap e são iguais nos dois.
- **O RSS varia muito entre execuções do mesmo backend** (o Panama deu de 793 a 1143 MB), porque depende da expansão do heap pelo G1. Para comparar memória é preciso fixar o heap ou medir a taxa de alocação.
- Perfil de alocações do Panama: **84% a 87% eram `int[]` criados no laço quente do GEMM** (parâmetros variádicos de `getVector`, `get` e `set`). O ramo `perf/no-varargs-alloc` acrescenta sobrecargas de dois inteiros e esse valor cai para 5%. **O efeito em tok/s não está demonstrado:** os A/B feitos sofreram carga do hospedeiro (a carga do modelo foi de 4 s para 18 s) e não permitem concluir. A mudança tem testes verdes (28) e saída idêntica à referência, mas não deve ser integrada como "otimização" sem um A/B numa máquina calma.

## Antes de tornar o nativo o padrão

1. **Portabilidade entre CPUs:** o módulo nativo passou a compilar a base x86-64 sem AVX e a escolher kernels AVX2+FMA, AVX-VNNI e AVX-512+VNNI por CPUID. O piso do backend nativo é AVX2+FMA; em CPUs abaixo dele o pedido de `native` cai em Panama, com um aviso no log. `check-isa.sh` verifica os objectos x86 no build Linux.
2. **Apple Silicon e Windows:** o nativo foi medido só em Linux x86.
3. **Cobertura de testes:** `TestOperations` só inclui o nativo quando ele é o backend ativo; agora há também `TestNativeRemainderTiles`, que o executa sempre.
4. **Carga do modelo:** o nativo demorou mais a carregar (4,5 s contra 2,9 s); causa não investigada.

## Defeitos que continuam abertos

Nenhum defeito de correção conhecido nos dois backends. Abertos: o build do nativo não é portável entre CPUs (`-march=native`), e o laço quente do Panama ainda aloca (ver abaixo).

## Reproduzir

```bash
export JAVA_HOME=$HOME/.sdkman/candidates/java/25.0.3-amzn
JDK_JAVA_OPTIONS="-Dbi4j.backend=native" PERF_BASELINE=1 ./mvnw -B -pl bitubi-inference-4j-tests -am package \
  -Dtest=PerfBaselineTest -Dsurefire.failIfNoSpecifiedTests=false
# variáveis opcionais: PERF_WARMUP, PERF_RUNS, PERF_MAX_TOKENS (o total de tokens inclui o prompt)
# diagnóstico operação a operação (nativo contra Panama): SHADOW=1 ... -Dtest=ShadowNativeInvestigation
```
