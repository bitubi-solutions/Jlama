# Direção do motor: backends, build portável, C ou Rust, Valhalla

Data: 2026-10-12. Estado: **decisões do autor registadas; o desenho do build portável aguarda aprovação**.

## 1. Decisões do autor

- O backend `naive` é descartado. Ficam **Panama** (Java, Vector API) e **Native** (C com SIMD). A classe escalar virou um oráculo só de testes (`ScalarReferenceOperations`).
- A **compilação portável** do código nativo é desejada.
- A ideia é usar o poder do Java no motor e, depois, montar um cluster e um produto para empresas com modelos abertos pequenos (SLM) otimizados e ajustados (fine-tuned).

## 2. Evidências medidas (nesta máquina: Core Ultra 7 255H, JDK 25, WSL2)

- **O nativo é híbrido.** Só os GEMMs correm em C. `accumulate`, `saxpy`, `scale` e `quantize` são delegados ao Panama (Java).
- **A resposta errada do nativo vinha de um defeito no driver de blocos do GEMM**, que não calculava o canto das sobras. O mesmo defeito existia no Panama, em `F32xF32`. Corrigido em C e em Java (`docs/backends.md`).
- **Onde está a memória do Panama.** Com o Native Memory Tracking, a diferença de pico de RSS entre o Panama e o nativo foi quase toda **heap Java comprometido** (585 MB contra 328 MB numa medição); JIT, classes e threads são iguais nos dois. Os tensores (pesos, ativações, KV) ficam fora do heap e são iguais.
- **O RSS varia muito entre execuções do mesmo backend** (o Panama deu de 793 a 1143 MB), porque depende de quanto o G1 expande o heap. Não se deve comparar backends por um único RSS; é preciso fixar o heap (`-Xmx`/`-Xms`) ou medir a taxa de alocação.
- **Perfil de alocações (JFR) do Panama:** 84% a 87% da pressão de alocação eram `int[]` criados no laço quente do GEMM, pelos parâmetros variádicos de `getVector(species, int... )`, `get(int...)` e `set(float, int...)`. Não eram objetos `Vector`. Com sobrecargas de dois inteiros, esse `int[]` caiu para 5% (os `Float256Vector`, 12% de um volume bem menor, passaram a ser visíveis). **Efeito no tok/s: ainda não demonstrado** (ver secção 6).

## 3. As value classes (Valhalla) reduzem o consumo do Panama?

Fatos verificados ([JEP 529](https://openjdk.org/jeps/529), JDK 26, "Closed / Delivered"):
- A Vector API "vai ficar em incubação até que as funcionalidades necessárias do Projeto Valhalla estejam disponíveis como preview". A meta de longo prazo é tornar as classes atuais (baseadas em valor) em *value classes*.
- Hoje o C2 trata as instâncias de `Vector` de forma especial para "superar limitações da análise de escape e evitar boxing"; o JEP afirma que nos benchmarks "todas as alocações de objetos são eliminadas".
- O JEP **não diz nada sobre o consumo de memória**.
- As datas do Valhalla (JEP 401 em preview no JDK 28 ou 29) vêm de fontes secundárias e conflitam; não há data confirmada. Não existe no JDK 25.

Conclusão (inferência, não medida): **não é uma alavanca de memória.** Os pesos, o KV cache e as ativações estão fora do heap, e as value classes não os tocam. O que as value classes podem dar é **previsibilidade de desempenho** (menos dependência de o C2 conseguir eliminar o boxing). Os ganhos de memória e de GC que conseguimos hoje vêm de tirar alocações do laço quente, o que é possível já, em Java 25.

## 4. C ou Rust para o código nativo?

O que o nativo precisa de ser: poucos milhares de linhas de kernels SIMD (GEMM quantizado), chamados por FFM, em Linux, Windows e macOS (x86_64 e ARM64), compilados sem ligação à CPU de quem compila.

| Critério | C (hoje) | Rust |
|---|---|---|
| Custo agora | zero: o código existe e passou a estar correto e testado | reescrita e nova validação de cada kernel |
| SIMD x86 | intrínsecos (`immintrin.h`); variantes por ficheiro com `-mavx2`, `-mavx512f`... | `std::arch` com `#[target_feature]` por função, sem ficheiros separados; AVX-512 estável desde o Rust 1.89 (ago. 2025, segundo a imprensa técnica; confirmar nas notas oficiais) |
| Seleção em tempo de execução | `__builtin_cpu_supports` ou CPUID | `is_x86_feature_detected!` (padrão da biblioteca padrão) |
| Segurança de memória | manual (os defeitos de bordas que achámos são deste tipo) | verificação de limites e `unsafe` localizado, mas nos kernels o `unsafe` é inevitável |
| Ligação com Java | `jextract` lê o cabeçalho C diretamente | `extern "C"` + `cbindgen` para gerar o cabeçalho, depois `jextract`: um passo a mais |
| Compilação cruzada e CI | `gcc`/`clang` por plataforma, `Makefile` e `antrun` já existentes | `cargo` simplifica muito a cruzada e o empacotamento |
| Equipa | C é o que existe | exige competência em Rust |

**Recomendação (opinião, ainda sem medição):** manter **C** e fazer o build portável agora; não reescrever em Rust por precaução. Justificação: o ganho de Rust (segurança e empacotamento) é real mas não está provado que compense a reescrita, e o risco imediato (CPUs antigas) resolve-se igual nos dois. Propor um **spike com limite de tempo**: portar só o kernel `F32xQ4` para Rust, medir desempenho e esforço contra o C, e decidir com evidência. Se o nativo continuar estratégico e o time quiser, Rust passa a opção forte.

Uma pergunta que pesa mais do que a linguagem: quanto do ganho do nativo o Java consegue alcançar? O prefill do nativo foi cerca de 2,4x mais rápido que o do Panama, mas o Panama ainda tem alocações no laço quente. Convém **esgotar o Java** (secção 6) antes de investir em reescrever nativo.

## 5. Build portável do nativo (proposta, aguarda aprovação)

Problema verificado: o Linux compila com `-march=native`, ou seja, para a CPU de quem compila; um JAR da CI pode terminar com `SIGILL` em CPUs mais antigas.

Desenho:
1. **Base fixa:** o código comum compila para um alvo mínimo (x86-64-v2, ou v3 se AVX2 for o piso decidido; ARM64 com NEON).
2. **Uma unidade de compilação por ISA**, cada uma com as suas flags: AVX2+FMA+F16C; AVX-VNNI; AVX-512 (+VNNI). No ARM: NEON, e dotprod/i8mm como variantes.
3. **Seleção em tempo de execução** na inicialização, por CPUID, devolvendo uma tabela de funções; a flag Java que hoje só serve para escolher o caminho 512 passa a ser a máscara real de capacidades.
4. **Testar todas as variantes numa única máquina** forçando a máscara de capacidades para baixo (os testes `TestNativeRemainderTiles` e `TestOperations` correm por variante).
5. **Verificação automática:** `objdump` na unidade base para garantir que não contém instruções acima do piso; e execução na CI no runner de menor capacidade.
6. Um artefato por sistema operacional e arquitetura (classificador já existente), publicado como hoje.

Critérios de aceitação: o mesmo JAR dá o mesmo texto (referência do `SmokeTest`) em todas as variantes selecionáveis; nenhuma instrução acima do piso na unidade base; sem regressão de desempenho na variante mais alta em relação ao `-march=native` atual (medido em rodadas intercaladas).

## 6. Experimento: alocações no laço quente do Panama (ramo `perf/no-varargs-alloc`)

Mudança: sobrecargas sem variádicos (`getOffset(i, j)`, `getVector(species, i, j)`, `get(i, j)`, `set(v, i, j)`) em `TensorShape`, `AbstractTensor`, `FloatBufferTensor` e `Q4ByteBufferTensor`; os métodos antigos ficam.

- **Provado:** a pressão de alocação de `int[]` no kernel caiu de 87% para 5% (JFR). Os testes por operação, os blocos de borda, a seleção de backend e o teste de fumaça (saída igual à referência) passam: 28 testes, 0 falhas.
- **Não provado:** o efeito em velocidade. Dois A/B (4 e depois 4 amostras) deram resultados contraditórios (os "depois" saíram piores, 12 contra 17 a 31 tok/s) mas num hospedeiro com carga média de 8 a 12 em 16 CPUs, com a carga do modelo a variar de 4 s a 18 s. Não permitem concluir nada, nem a favor nem contra. Repetir numa máquina calma, em cópias compiladas lado a lado, alternando antes e depois.
- Terceiro A/B (2026-10-16, 3 rodadas alternadas de cada lado, só Panama): **inconclusivo por ruído** (load de 5,4 a 10,7; medianas de prefill 13,99 antes e 11,57 depois, faixas sobrepostas).
- Decisão do autor (2026-10-18): **o `d5504cf` fica como refactor**, porque o GGUF e outras branches já usam as sobrecargas de dois inteiros. Não é apresentado como otimização: o efeito em tok/s **não está medido**. O ganho de alocação (87% para 5% de `int[]`) é o único efeito provado.

## 7. Direção do produto (hipóteses a validar, não fatos)

- Para modelos pequenos (0,5B a 8B), o modelo cabe num nó. **"Cluster" tende a significar réplicas, roteamento, filas e isolamento entre clientes**, mais do que dividir um modelo por vários nós (que é o que o código distribuído atual faz e serve a modelos de dezenas de bilhões de parâmetros). Convém validar isto antes de priorizar o sub-projeto 3.
- Modelos ajustados chegam como SafeTensors (LoRA já fundido) ou GGUF. O suporte a GGUF (spec em `2026-10-12-gguf-slice1-design.md`) e, mais tarde, a **adaptadores LoRA** (carregar vários adaptadores sobre um modelo base partilhado) podem reduzir muito a memória por cliente. O README antigo já listava LoRA como pendente.
- **Lote contínuo de pedidos** (várias sessões em simultâneo no mesmo modelo) é o que decide a vazão por máquina. Não verifiquei se o motor atual o suporta; é uma pergunta aberta importante.

## 8. Próximos passos propostos

1. Concluir o A/B do Panama e decidir sobre as sobrecargas.
2. Aprovação do desenho do build portável (secção 5), depois plano e implementação, com o spike de Rust em paralelo e com limite de tempo.
3. GGUF fatia 1 (spec já escrito, aguarda revisão).
4. Investigar o lote contínuo e a viabilidade de adaptadores LoRA, antes de desenhar o cluster.
