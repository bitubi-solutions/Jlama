# bitubi-inference-4j

Motor de inferência de LLMs em Java (comando da CLI e harness de agentes: `bi4j`), fork mantido do [Jlama](https://github.com/tjake/jlama)
(Apache 2.0) pela Bitubi Solutions. Modelo de negócio: open core (núcleo aberto, extras enterprise
num repositório separado que depende do aberto via SPI, nunca o contrário).

Spec e plano: `docs/superpowers/specs/` e `docs/superpowers/plans/`. Ordem dos sub-projetos:
0 fundação, 1 motor e SDK (modelos novos, hardware antigo, GGUF), 1b WebGPU (spike primeiro),
2 servidor e UI, 3 cluster para cloud, 4 harness de agentes (por último).

## Regras (do autor)

- **Nunca adicionar trailers do Claude** (`Co-Authored-By`, `Claude-Session`, "Generated with...") em
  commits ou PRs.
- Idioma: **CLI e documentação em inglês** (decisão do autor, 2026-10-18; docs antigos em português estão a ser traduzidos). Conversa com o autor em português europeu.
- **Sem retrocompatibilidade com o Jlama** (decisão do autor, 2026-10-18): contratos legados (`jlama.*`, `JLAMA_*`, `X-Jlama-Session`, `use_native_tensor_operations`) são removidos; a migração de quem usa o SDK do Jlama é facilitada por guia e receita OpenRewrite, e o SDK deve ficar preparado para integração com o LangChain4j (desenho só depois da revisão de DX do SDK).
- JDK 25 é o mínimo e único alvo de build, teste e execução. Bytecode fica em 21 até o upgrade do
  Spring (o Spring Boot 3.3 não lê classes versão 69).
- **Panama (Vector API) é a implementação padrão.** O backend se escolhe com `-Dbi4j.backend=auto|panama|native`
  (ver `docs/backends.md`). O nativo nunca é escolhido sozinho: dava resposta errada por um defeito no
  cálculo dos blocos de borda do GEMM (corrigido em C e em Java; não era o JDK 25) e ainda é compilado com
  `-march=native`, não portável entre CPUs.
- Fase de fundação: nenhuma mudança de comportamento do motor.
- **DX é requisito do produto inteiro** (SDK, API, UI, CLI, harness, deploy, docs). Cada componente tem
  revisão própria quando chegar a sua vez; por enquanto só a intenção está registada em
  `docs/superpowers/specs/2026-10-12-dx-product-wide-intent.md`. Não desenhe nem implemente DX de um
  componente sem essa revisão.
- A **release é manual e feita pelo autor** (workflow `Release` por `workflow_dispatch`). Nunca dispare
  releases, tags de versão nem publicações de pacotes.
- Ações visíveis externamente (push, renomear repositório, publicar pacotes) só com autorização.

## Build e testes

```bash
export JAVA_HOME=$HOME/.sdkman/candidates/java/25.0.3-amzn
./mvnw -B clean package                       # use package, não test: o bitubi-inference-4j-tests precisa do JAR multi-release
./mvnw -B -pl bitubi-inference-4j-tests -am package -Dtest=SmokeTest -Dsurefire.failIfNoSpecifiedTests=false
```

- `SMOKE_MODEL=1` liga o teste de fumaça; `SMOKE_RECORD=1` grava a referência; `SMOKE_RELAXED=1` só exige texto não vazio.
- `PERF_BASELINE=1` liga `PerfBaselineTest` (tokens/s).
- `TestSample` e alguns testes do `bitubi-inference-4j-net` baixam modelos do Hugging Face; `Connection reset` é rede,
  repita ou baixe com `curl -L -C -` (o downloader pula arquivos com o tamanho esperado).
- `git reset --hard` e `rm` em massa são bloqueados neste ambiente; edite arquivos CRLF por script.

## Equipa de agentes (`.claude/agents/`)

| Agente | Responsabilidade |
|---|---|
| `engine-perf` | kernels Panama/Vector API, quantização, hardware antigo, benchmarks do motor |
| `model-formats` | GGUF, SafeTensors, tokenizers, arquiteturas novas |
| `server-api` | API OpenAI e Anthropic, Spring Boot 4, UI htmx + JTE |
| `cluster-cloud` | coordinator/worker, gRPC, Kubernetes/Helm, HA e autoscaling |
| `build-release` | Maven, toolchains, CI (Linux/Windows/macOS), GitHub Packages, rename |
| `qa-bench` | testes, referências de saída, baselines de desempenho, flakiness |
| `gpu-webgpu` | spike e backend opcional de GPU via WebGPU (FFM) |
| `brand-compliance` | rebranding, logo, README, NOTICE, conformidade Apache 2.0, DCO |
| `agent-harness` | harness de agentes (só depois de o motor estabilizar) |

Fluxo: `qa-bench` valida antes e depois de qualquer mudança no motor; `build-release` é o dono do
`pom.xml` e da CI; `brand-compliance` revê toda mudança de nome, licença ou atribuição.
