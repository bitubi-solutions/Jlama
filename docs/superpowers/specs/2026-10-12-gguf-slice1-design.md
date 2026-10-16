# Suporte a GGUF, fatia 1: Qwen2 com F16, Q8_0 e Q4_0

Data: 2026-10-12. Estado: **revisto; decisões do autor da secção 7 aplicadas; falta aprovar o plano** (nada implementado). Faz parte do sub-projeto 1
(motor e SDK); ver `2026-10-11-bitubi-inference-4j-foundation-design.md`.

## 1. Intenção

O motor só lê SafeTensors. O GGUF é o formato mais usado para modelos quantizados e dá acesso direto
ao ecossistema (Hugging Face, llama.cpp). Decisões do autor (2026-10-12): **abordagem A** (adaptador
sobre os modelos existentes) e **primeira fatia = Qwen2 com F16, Q8_0 e Q4_0**.

Critério de sucesso da fatia: carregar `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (Apache-2.0) em F16, Q8_0 e
Q4_0, e gerar texto correto em temperatura 0, comprovado pelos testes da secção 6.

## 2. Fatos verificados (fontes primárias)

Especificação `ggml-org/ggml/docs/gguf.md` e structs de `ggml/src/ggml-common.h`:

- Cabeçalho: magic `GGUF` (bytes `47 47 55 46`), `uint32 version` (a spec descreve a versão 3), `uint64 tensor_count`, `uint64 metadata_kv_count`. Little-endian por padrão.
- Ordem: cabeçalho, informações dos tensores, preenchimento, dados dos tensores. Alinhamento `general.alignment` (uint32, múltiplo de 8; padrão 32); preenchimento com `0x00`.
- Metadados: chave `gguf_string_t` (`uint64 len` + bytes UTF-8), tipo `uint32` (UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12). Array = tipo do elemento + `uint64` contagem + elementos (pode ser aninhado).
- Informação de tensor: nome (string, até 64 bytes), `uint32 n_dimensions` (até 4), `uint64 dims[n]`, `uint32 type`, `uint64 offset` **relativo ao início dos dados**, múltiplo do alinhamento.
- `ggml_type`: F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9, Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, I8=24, I16=25, I32=26, I64=27, F64=28, BF16=30 (e famílias IQ/TQ/MXFP4).
- Blocos (elementos / bytes): Q4_0 32 / 18 (`half d` + 16 bytes de nibbles); Q8_0 32 / 34 (`half d` + 32 `int8`). Para a fatia 2: Q4_K 256 / 144, Q5_K 256 / 176, Q6_K 256 / 210.
- Tokenizer: `tokenizer.ggml.model` (`llama`, `replit`, `gpt2`, `rwkv`), `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.token_type` (1 normal, 2 desconhecido, 3 controlo, 4 definido pelo utilizador, 5 não usado, 6 byte), IDs especiais (`bos`, `eos`, `padding`...), `tokenizer.chat_template`.
- Arquitetura: `general.architecture` e chaves `<arch>.*` (ex.: `<arch>.context_length`, `embedding_length`, `block_count`, `feed_forward_length`, `attention.head_count`, `attention.layer_norm_rms_epsilon`).

**A spec não diz** a ordem das dimensões nem os tamanhos de bloco (estes vêm do código de referência).

Do nosso código (lido): o `Q4ByteBufferTensor` usa blocos de 32 valores, um byte = valor *j* (nibble baixo) e valor *j+16* (nibble alto), escala `max / -8` e valor `(q - 8) * escala`, com as escalas num `FloatBufferTensor` à parte. É o mesmo esquema do Q4_0 do GGUF, que guarda a escala (16 bits) inline no bloco. O `Q8ByteBufferTensor` tem blocos de 32 e escalas à parte.

## 3. Âmbito

**Dentro:** leitor GGUF v3 little-endian de arquivo único; tipos F32, F16, BF16, Q8_0 e Q4_0; arquitetura `qwen2`; configuração, tokenizer e template de chat a partir dos metadados; carregamento por `ModelSupport.loadModel` ao receber um `.gguf`.

**Fora (fatias seguintes ou nunca):** K-quants (Q4_K, Q5_K, Q6_K), Q5_0/Q5_1, famílias IQ; arquiteturas Llama, Mistral, Gemma etc. (Llama exige tratar a permutação de Q/K do RoPE); GGUF em partes (`-00001-of-0000N`); big-endian; exportar para GGUF; inferência distribuída com GGUF (`DistributedContext` não nulo falha com mensagem clara).

## 4. Desenho

Pacote novo `ao.bitubi.inference4j.gguf` no `bitubi-inference-4j-core`:

| Unidade | Responsabilidade | Depende de |
|---|---|---|
| `GgufType` | enum do `ggml_type` (id, elementos por bloco, bytes por bloco) | nada |
| `GgufFile` | abre o arquivo (mmap), valida magic e versão, lê metadados e informações dos tensores, calcula o início dos dados e o alinhamento | `GgufType` |
| `GgufTensorInfo` | nome, dimensões, tipo, offset absoluto | `GgufType` |
| `GgufWeightLoader` (implementa `WeightLoader`) | traduz o nome HF pedido pelo modelo para o nome GGUF, lê o tensor e devolve o `AbstractTensor` certo | `GgufFile`, `GgufTensorNames`, `GgufRepack` |
| `GgufTensorNames` | tabela por arquitetura: nome HF para nome GGUF | nada |
| `GgufRepack` | converte blocos GGUF para os tensores internos (F16 direto; Q8_0 e Q4_0 separando a escala `half` para `float`) | tensores existentes |
| `GgufConfigFactory` | metadados `qwen2.*` para `Qwen2Config` | `GgufFile` |
| `GgufTokenizerFactory` | `tokenizer.ggml.*` para um `TokenizerModel` e o template de chat | `GgufFile` |

Mudanças em código existente (mínimas):
- `BPETokenizer` e `LlamaTokenizer`: construtor a partir de um `TokenizerModel` pronto (hoje só aceitam um `Path` com `tokenizer.json`).
- `ModelSupport.loadModel`: se o arquivo termina em `.gguf`, usa o caminho GGUF (sem exigir `config.json`), escolhendo o `ModelType` por `general.architecture`.

Tradução de nomes para `qwen2` (**a confirmar listando os tensores do arquivo real**): `model.embed_tokens.weight` = `token_embd.weight`; `model.norm.weight` = `output_norm.weight`; `lm_head.weight` = `output.weight` (se ausente, usar `token_embd.weight` apenas quando os embeddings são partilhados); por camada `N`: `self_attn.{q,k,v}_proj.{weight,bias}` = `blk.N.attn_{q,k,v}.{weight,bias}`, `self_attn.o_proj.weight` = `blk.N.attn_output.weight`, `mlp.gate_proj` / `up_proj` / `down_proj` = `blk.N.ffn_gate` / `ffn_up` / `ffn_down`, `input_layernorm` = `blk.N.attn_norm`, `post_attention_layernorm` = `blk.N.ffn_norm`.

Configuração `Qwen2Config` a partir dos metadados: `qwen2.context_length`, `embedding_length`, `feed_forward_length`, `attention.head_count`, `attention.head_count_kv`, `block_count`, `attention.layer_norm_rms_epsilon`, `rope.freq_base`; vocabulário = tamanho de `tokenizer.ggml.tokens`; `bos`/`eos` dos IDs especiais (se `bos` faltar, usar o do `Config` do Qwen2 e documentar); ativação SiLU (fixa para a arquitetura, não está nos metadados).

## 5. Riscos conhecidos

1. **Ordem das dimensões.** Confirmado no Q4_0 real: GGML guarda a dimensão interna primeiro. Por exemplo, `blk.0.ffn_down.weight` tem `[4864,896]` no GGUF e a forma interna é `[896,4864]`. O loader inverte as dimensões.
2. **Tokenizer.** O Qwen2 usa BPE ao nível de byte com um pré-tokenizador por regex; os metadados trazem `tokenizer.ggml.pre`, não o regex. Tem de reproduzir exatamente o do `tokenizer.json`. Mitigação: teste de paridade (secção 6).
3. **Tokens especiais.** Os de controlo (`token_type = 3`, ex.: `<|im_start|>`) têm de virar *added tokens* para o texto do template de chat ser segmentado igual.
4. **Template de chat.** Vem em `tokenizer.chat_template` (Jinja); tem de entrar no `PromptSupport` como o do `tokenizer_config.json`.
5. **Embeddings partilhados.** A inspeção do ficheiro Q4_0 real `Qwen/Qwen2.5-0.5B-Instruct-GGUF` encontrou `output.weight` Q8_0 e `token_embd.weight` Q4_0; manter o fallback para variantes com pesos partilhados.
6. **Arquivos grandes.** `ByteBuffer` limita a 2 GB; mapear por tensor (os do 0.5B são pequenos, mas o desenho não pode assumir isso).
7. **Falso positivo de correção.** Um modelo pequeno pode "responder bem" por acaso; por isso a verificação compara contra referências, não só olha o texto.

## 6. Verificação (critérios de aceitação)

1. **Leitor, sem rede:** um GGUF mínimo gerado pelo próprio teste (escritor de teste) cobre todos os tipos de metadados, arrays, strings, preenchimento de alinhamento e offsets de tensores; magic ou versão inválidos dão erro claro.
2. **Paridade do tokenizer:** para um conjunto fixo de textos (ASCII, acentos, emoji, código, números, espaços e quebras de linha, marcadores do template de chat), os IDs do tokenizer GGUF são **idênticos** aos do tokenizer feito do `tokenizer.json` do mesmo modelo (`tjake/Qwen2.5-0.5B-Instruct-JQ4`). Diferença zero.
3. **Paridade dos pesos:** para todos os tensores Q8_0 e Q4_0 dos ficheiros reais, um descodificador de teste independente, lendo os blocos GGML crus, produz floats exatamente iguais aos tensores do `GgufWeightLoader`. A comparação com GGUF F16 é apenas estatística informativa (erro médio/máximo em passos, fração até meio passo e saturação), não é critério de falha.
4. **Ponta a ponta:** com `qwen2.5-0.5b-instruct-fp16.gguf`, o prompt "What is the capital of France?" em temperatura 0 produz uma resposta contendo "Paris" e coerente; o Q8_0 produz o mesmo texto ou diverge só depois de um prefixo longo (registrar); o Q4_0 é coerente (registrar divergências). As referências de saída entram como `smoke-golden-gguf-*.txt`.
5. **Sem regressão:** o teste de fumaça atual e a suíte existente continuam verdes; CI verde nos 4 sistemas.
6. **Desempenho:** registrar tokens/s do GGUF F16, Q8_0 e Q4_0 no hardware do baseline. Não há meta numérica antes de medir.

Os arquivos GGUF de teste (F16 1,27 GB, Q8_0 676 MB, Q4_0 429 MB) são baixados sob demanda e ficam fora do Git; os testes que dependem deles são opt-in por variável de ambiente, como o `SmokeTest`.

### Revisão do critério de paridade Q4_0/Q8_0 (2026-10-16)

O antigo limite universal de meio passo contra o GGUF F16 foi removido: não verifica o repack e não é válido para todos os quantizadores. O Q4_0 de referência do ggml escolhe `d = max / -8` com o máximo assinado, converte com `x / d + 8.5` e limita o resultado superior a 15; há saturação unilateral, cujo erro pode chegar a um passo. Ver [quantização Q4_0 de referência](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c#L2357-L2404). Uma divergência face a um GGUF F16 de referência também pode reflectir decisões de quantização distintas; a proveniência/imatrix dos ficheiros usados não foi confirmada.

A prova do repack compara, em cada tensor quantizado, os floats carregados com um descodificador de teste independente dos bytes crus: Q4_0 lê half `d` e os 16 bytes de nibbles (índices `j` e `j+16`); Q8_0 lê half `d` e 32 int8. A igualdade é exata. No ficheiro Q4_0 testado, os 169 tensores Q4_0 (493 961 216 valores) e o tensor Q8_0 `output.weight` (136 134 656 valores) passaram após corrigir o endereçamento de blocos para índices grandes. A comparação informativa Q4_0/F16 teve média 0,24477 passos, máximo 1 passo, 99,4083% até meio passo; 10 179 596 valores foram identificados como saturados (máximo de 1 passo), enquanto os não saturados ficaram até meio passo. A comparação Q8_0/F16 no ficheiro Q8_0 teve média 0,24274 passos, máximo 0,67213 e 99,2894% até meio passo; não mostrou saturação Q4. A comparação Q8_0/F16 no ficheiro Q8_0 teve média 0,24274 passos, máximo 0,67213 e 99,2894% até meio passo; não mostrou saturação Q4.

## 7. Decisões do autor (2026-10-13) e achado de kernels

Aceitas as recomendações:
- **Memória das escalas:** Q4_0 e Q8_0 ficam maiores (escala `half` inline vira `float` separada: Q4_0 18 para 20 bytes por bloco, +11%; Q8_0 34 para 36). Aceite na fatia 1.
- **mmap por tensor** (sem o limite de 2 GB).
- **F16 copiado** para um tensor interno primeiro; só se passa a mmap sem cópia se o tempo de carga doer.
- **SentencePiece fora:** só `gpt2` (BPE de byte).
- Ordem do plano: leitor e escritor de teste, paridade dos pesos, tokenizer, ponta a ponta.

**Achado (verificado no código, `PanamaTensorOperations.batchDotProduct`):** os kernels de GEMM só aceitam, como pesos (`b`), `F32`, `BF16` e `Q4` (com ativação `F32`; `I8` só com `Q4`; `BF16` só com `BF16`). **Não há kernel para pesos F16 nem Q8**; o nativo tem as mesmas 4 combinações (`F32xF32`, `F32xQ4`, `F32xBF16`, `I8xQ4`). Consequências para a fatia:
- **Q4_0:** funciona com os kernels existentes (Panama e nativo).
- **Q8_0:** exige kernel novo `F32xQ8` (e `I8xQ8` se a memória de trabalho for I8) no Panama. No nativo, só se for portado; até lá o nativo falha com mensagem clara ou delega ao Panama (decidir no plano, depois de ler como o `NativeSimdTensorOperations` trata combinações sem kernel).
- **F16:** exige kernel novo `F32xF16`, ou alargar para F32 no carregamento (simples, mas dobra a memória: o 0,5B passa de 1,27 GB para cerca de 2,5 GB). Recomendação: alargar para F32 na fatia 1 como caminho de referência, documentado, e deixar o kernel `F32xF16` para depois.
- O critério de sucesso (secção 1) mantém F16, Q8_0 e Q4_0, mas o Q8_0 passa a incluir o kernel novo, com teste contra o oráculo escalar (`ScalarReferenceOperations`, como o `TestNativeRemainderTiles`), incluindo blocos de borda.

## 8. Questões em aberto

- Como o nativo trata combinações de tipos sem kernel (falha ou delega)? Verificar no plano.
- `I8xQ8`: a memória de trabalho I8 é opcional; basta `F32xQ8` na fatia 1? Preferência: sim.
