# Build portável do módulo nativo — plano

Data: 2026-10-16. Origem: secção 5 de `docs/superpowers/specs/2026-10-12-engine-direction-decisions.md` (aprovada
pelo autor) e "Antes de tornar o nativo o padrão" de `docs/backends.md`.

Base do trabalho: `fix/gemm-corner-tiles` (traz a correção do GEMM, `bi4j.backend` e `TestNativeRemainderTiles`,
de que este trabalho depende).

## Objetivo

Remover `-march=native`. Um único `libbi4j` por sistema operacional e arquitetura, com:

- uma **unidade base** (`vector_simd.c`: driver de blocos, API exportada, deteção de CPU) compilada para o
  x86-64 genérico, sem nenhuma instrução AVX;
- **uma unidade por ISA** (`kernels_avx2.c`, `kernels_avxvnni.c`, `kernels_avx512.c`; `kernels_neon.c` no ARM64),
  cada uma com as suas flags e a sua tabela de funções;
- **seleção em tempo de execução** por CPUID (com a verificação do SO via `xgetbv`), limitada por um teto
  vindo de Java: `-Dbi4j.native.isa=auto|avx2|avxvnni|avx512|neon`;
- **verificação por `objdump`** no build Linux: a unidade base não pode ter instruções VEX/EVEX, nem as
  unidades AVX2/AVX-VNNI instruções AVX-512.

Piso x86: **AVX2 + FMA** (é o que os kernels `*_256` usam; o código não usa F16C). CPU sem isso: o construtor
Java falha e o `TensorOperationsProvider` cai para o Panama.
O AVX-512 exige F, BW, DQ, VL e VNNI (Skylake-X sem VNNI usa o AVX2).

## Decisões

- A API pública C não muda (`vector_simd.h`). O argumento `flags` passa a transportar o teto de ISA nos bits 8..11
  (`BI4J_ISA_CAP_SHIFT`). O bit antigo `HAS_AVX2` (que só ativava o caminho 512) deixa de ser usado.
- Duas funções novas: `bi4j_supported_isas()` (máscara `1 << nível`) e `bi4j_selected_isa(flags)` (nível escolhido).
  Níveis: 1 = AVX2 (ou NEON no ARM), 2 = AVX-VNNI, 3 = AVX-512.
- Os kernels ficam em `kernels_x86_256.inc` / `kernels_x86_512.inc` (movidos sem alteração de lógica), incluídos
  por cada unidade; os `static` evitam conflito de símbolos entre unidades.
- ARM64: o código NEON só é movido (não tenho como compilá-lo aqui). Variantes dotprod/i8mm ficam fora deste plano.

## Tarefas (TDD, pequenas)

1. **Baseline.** Guardar o `libbi4j.so` atual (`-march=native`) e o resultado do `SmokeTest` nativo. (feito)
2. **Teste que falha primeiro.**
   - `TestNativeRemainderTiles` passa a correr por cada ISA suportada (`NativeSimdTensorOperations.supportedIsas()`);
   - `bitubi-inference-4j-native/src/main/c/check-isa.sh` (verificação `objdump`) falha com o build atual.
3. **Partir o C.** `kernels.h` (struct de parâmetros e tabela), `kernels_x86_256.inc`, `kernels_x86_512.inc`,
   `kernels_neon.c`, quatro unidades finas, `vector_simd.c` só com driver, API e despacho.
4. **Makefile e pom.** Unidades por ISA com flags próprias; remover `-mavx2 -march=native` dos três perfis x86
   (Linux, Windows, macOS Intel); chamar `check-isa.sh` no perfil Linux.
5. **Java.** Teto lido de `bi4j.native.isa`; `NativeSimd` com as duas funções novas; falha clara se a CPU não tem o
   piso; log da ISA escolhida.
6. **Verificar.** `SmokeTest` nativo por ISA (`avx2`, `avxvnni`; `avx512` só compila), `TestNativeRemainderTiles`
   por ISA, `objdump` nas unidades, `clean package` completo (`-Dtest='!TestSample'`).
7. **Desempenho.** Nao medido: a maquina estava com carga elevada (load average acima de 5, com outros agentes a correr).
8. **Documentar.** `docs/backends.md` (como mascarar, o que a CI tem de confirmar).

## O que não se consegue verificar nesta máquina

- Execução das variantes AVX-512 (a máquina tem AVX2 + AVX-VNNI): só compilação e `objdump`.
- Windows, macOS Intel e macOS ARM: só a CI.
- Linux ARM64 e variantes dotprod/i8mm.
