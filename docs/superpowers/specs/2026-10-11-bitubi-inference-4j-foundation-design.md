# bitubi-inference-4j: visão geral e fundação (sub-projeto 0)

Data: 2026-10-11. Estado: aguardando revisão do autor.

## 1. Intenção

Continuar o Jlama (motor de inferência de LLMs em Java, upstream `tjake/jlama`, parado pelo dono) como um produto novo, **bitubi-inference-4j**, mantido pela organização `bitubi-solutions`.

- **Identidade:** rebrand com nome novo, creditando o Jlama original (Apache 2.0).
- **Modelo de negócio:** open core. Núcleo aberto (motor, SDK, servidor, cluster básico) em Apache 2.0. Extras enterprise pagos (operador Kubernetes, multi-tenant, SSO, cotas, auditoria, hospedagem, suporte) em repositório separado que depende do aberto via SPI, nunca o contrário.
- **Critério de sucesso global:** um motor Java competitivo em hardware antigo, utilizável como SDK, servidor self-hosted (APIs OpenAI e Anthropic) e cluster que justifique migrar para a cloud.

Premissas (do autor): `groupId` `ao.bitubi`, CLI `bi4j`, repositório migrado para `bitubi-solutions`.

## 2. Decomposição em sub-projetos

Cada um tem seu próprio ciclo spec, plano e implementação.

| # | Sub-projeto | Depende de |
|---|---|---|
| 0 | Fundação: rename, atribuição, CI, migração, publicação de teste | nada |
| 1 | Motor e SDK: modelos novos, otimização para hardware antigo, API estável | 0 |
| 1b | Backend de GPU via WebGPU (spike primeiro) | 1 |
| 2 | Servidor e UI: API OpenAI melhorada, API Anthropic, htmx + JTE sobre Spring Boot 4 | 1 |
| 3 | Cluster para cloud: coordinator/worker evoluído (HA, autoscaling, descoberta de workers) | 2 |
| 4 | Harness de agentes `bi4j` (tipo Claude Code), **por último**, após estabilizar o motor | 2 |

Este spec detalha só o sub-projeto 0. Os demais têm intenção e restrições registradas abaixo, mas serão especificados quando chegar a vez.

### 2.1 Restrições já decididas para os próximos sub-projetos

- **Sub-projeto 1.** Alvo: JDK 25 LTS como mínimo e único suportado (decisão do autor; JDK 21 sai). A pesquisa (2026-10) não achou ganho de desempenho novo e comprovado entre JDK 21 e 26: a Vector API segue em incubação (JEP 529, JDK 26, sem mudanças substanciais desde o 25), e Valhalla (JEP 401) está previsto só como preview em JDK 28/29. Os ganhos para hardware antigo virão do motor: kernels de GEMM com fallback para CPUs sem AVX-512/AVX2, quantização agressiva, tiling para cache e threads, e modelos pequenos e eficientes (Phi, Gemma 3, Qwen3, Llama 3.2). Critério de pronto: benchmarks em CPU sem AVX-512.
  - **Suporte a GGUF (requisito do autor).** O motor hoje lê apenas SafeTensors; GGUF é o formato mais usado para modelos quantizados e dá acesso direto ao ecossistema existente (Hugging Face, llama.cpp). Escopo: leitor de GGUF (metadados, tokenizer embutido, tensores), os tipos de quantização mais comuns (Q4_0, Q8_0 e as K-quants como Q4_K_M, Q5_K_M, Q6_K), com kernels de dequantização/GEMM para eles, e carregamento direto sem converter. Precisa de uma camada de abstração de formato de modelo (SafeTensors e GGUF atrás da mesma interface). Decisão em aberto para o spec do sub-projeto 1: quais arquiteturas ligar primeiro (Llama, Qwen, Gemma, Phi) e se o formato JQ4 próprio continua sendo suportado.
- **Sub-projeto 1b (WebGPU).** Não verificado que exista binding Java maduro para WebGPU. Hipótese a testar: ligar `wgpu-native` via FFM (API final desde Java 22), com kernels de matmul/atenção em WGSL. **Começa com um spike** (carregar um modelo pequeno e medir tokens/s contra o caminho de CPU, num hardware antigo real) antes de qualquer compromisso de escopo. O backend deve ser opcional e cair para CPU se não houver GPU.
- **Sub-projeto 2.** UI com htmx + JTE sobre Spring Boot 4 (base Java 17). Existem starter JTE para Boot 4 e htmx-spring-boot 5.x declarando compatibilidade, mas não há anúncio oficial juntando os três: validar versões no Maven Central antes de fixar. APIs: OpenAI (streaming, tools, embeddings, `/v1/models`) e Anthropic (`/v1/messages`, streaming, tool use).
- **Sub-projeto 3.** O coordinator/worker open source deve evoluir a ponto de o self-hosted já justificar migrar para cloud (HA, autoscaling, descoberta, filas). O operador gerenciado fica no lado pago.
- **Sub-projeto 4.** O harness consome os modelos pela API do servidor (sub-projeto 2). Risco conhecido: modelos locais pequenos são fracos em tool use; o harness deve usar prompts enxutos e poucas ferramentas bem definidas, com modelo remoto opcional.

## 3. Sub-projeto 0: fundação

Estado atual verificado: branch `main`, `origin` em `DumiJDev/Jlama`. O `git status` mostra quase todos os arquivos modificados, mas `git diff --ignore-cr-at-eol --stat` está vazio, ou seja, são só diferenças de fim de linha (CRLF/LF). 154 arquivos referenciam `com.github.tjake`. Versão atual `0.8.4`. Workflows: `unit-tests.yaml` e `release.yaml`. O `pom.xml` raiz referencia JDK 20, 21 e 22 em perfis.

### 3.1 Passos (um commit ou PR revisável cada)

1. **Normalizar fim de linha.** Adicionar `.gitattributes` (`* text=auto eol=lf`) e renormalizar.
2. **Migrar para `bitubi-solutions`.** Transferir ou recriar o repositório na organização, mantendo o histórico (preserva a autoria original) e adicionando `upstream` apontando para `tjake/jlama`.
3. **Rename mecânico, num commit, sem mudança de comportamento:**
   - Pacote `com.github.tjake.jlama` para `ao.bitubi.inference4j` (o hífen não é válido em pacote Java).
   - `groupId` `ao.bitubi`; `artifactId`s `bitubi-inference-4j-core`, `-native`, `-net`, `-cli`, `-tests`, `-parent`.
   - CLI `bi4j` (`run-cli.sh`, `run-cli.bat`, jbang, Dockerfile, compose, kubernetes).
   - `<revision>` reinicia em `0.1.0-SNAPSHOT`.
4. **Atribuição (Apache 2.0).** `NOTICE` ("Este produto inclui software derivado do Jlama, © Tim Jake e colaboradores"), seção "Origem" no README, cabeçalhos de copyright originais preservados, arquivos modificados marcados, `conf/license-header` atualizado. Não usar o nome ou a marca do autor original para sugerir endosso.
5. **Build e CI.** JDK 25 como mínimo e único alvo; CI em Linux, Windows e macOS (ARM e Intel); remover os perfis e fontes de JDK 20/21/22 (`jlama-native/src/main/java22` vira `java25`); workflow de release para o GitHub Packages.
6. **Publicação.** Por decisão do autor, começa no GitHub Packages (`maven.pkg.github.com/bitubi-solutions/bitubi-inference-4j`); o Maven Central fica para depois (exige verificar o domínio `bitubi.ao`, ou usar `io.github.bitubi-solutions`).
7. **Governança mínima.** `CONTRIBUTING`, DCO ou CLA (necessário antes de aceitar PR externo, para preservar a opção comercial), templates de issue.

### 3.2 Critério de pronto

- `./mvnw verify` verde na CI em JDK 25, em Linux, Windows e macOS.
- `grep -ri tjake` fora de `NOTICE`, README e cabeçalhos de atribuição não retorna nada, exceto ids de modelo do Hugging Face (`tjake/...`) e o clone `tjake/sdkman-for-toolchains` no Dockerfile, que são nomes externos.
- Artefato publicado no GitHub Packages.
- Texto gerado idêntico ao de referência gravada antes do rename (temperatura 0, por JDK), e tokens/s dentro da faixa de variação do baseline (`docs/benchmarks/`).
- Teste de fumaça: carrega um modelo pequeno e gera tokens, rodando na CI.

### 3.3 Riscos

- **`jlama-native`:** nomes de classe ou símbolos JNI ligados ao pacote antigo podem quebrar no rename. O teste de fumaça com modelo real detecta.
- **Namespace Maven:** sem controle de `bitubi.ao`, usar o fallback.
- **Divergência do upstream:** após o rename, merges do upstream ficam trabalhosos. Aceito, dado o rebrand.
- **Modelos pré-quantizados:** os de `hf.co/tjake` usam o formato JQ4. Manter leitura compatível e decidir depois onde hospedar os próprios; com GGUF (sub-projeto 1) a dependência de modelos próprios diminui.

### 3.4 Testes

Os testes existentes passam sem alteração de lógica; mais o teste de fumaça acima.

## 4. Fora de escopo deste spec

Qualquer implementação dos sub-projetos 1 a 4 e do lado pago. Cada um terá seu spec.

## 5. Decisões posteriores (2026-10-11 e 2026-10-12)

- **Nome do produto:** `bitubi-inference-4j` (substitui `aby-inference`). `groupId` `ao.bitubi`, pacote Java `ao.bitubi.inference4j`, artefatos `bitubi-inference-4j-{parent,core,native,net,cli,tests}`, comando da CLI `bi4j`. `bi4j` é também o harness de agentes (a CLI), decisão do autor em 2026-10-12.
- **Panama por padrão no JDK 25.** O backend nativo (`jlama-native`) deu resposta errada no teste de fumaça no JDK 25 ("I'm sorry, but I can't assist with that." contra "The capital of France is Paris." com Panama). Passou a ser opt-in com `-Dbi4j.use_native_tensor_operations=true`. Causa-raiz (2026-10-12): o driver de blocos do GEMM não calculava o canto das sobras de linhas e colunas, em C e em Java; não era o JDK 25. Detalhes em `docs/backends.md`.
- **Bytecode 21 com JDK 25 como único JDK.** O Spring Boot 3.3.x não lê classes versão 69; o alvo sobe para 25 junto com o upgrade do Spring (sub-projeto 2).
- **Equipa de agentes** em `.claude/agents/` (motor, formatos, servidor, cluster, build, QA, GPU, marca, harness) e regras do projeto em `CLAUDE.md`. Regra do autor: nunca adicionar trailers do Claude em commits ou PRs.
- **Baseline de desempenho** (Intel Core Ultra 7 255H, WSL2, JDK 25, Panama): `docs/benchmarks/baseline-intel-core-ultra-7-255h.md`.
