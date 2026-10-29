# TCC — TODO & Status

## Objetivo
DNN com saídas antecipadas (early-exit) para inferência híbrida edge/cloud usando **MobileNetV2 em CIFAR-10**. Saídas confiantes classificam na borda; amostras incertas vão pra nuvem.

**Estado atual: geração v5** (retreino de 04 a 06/10/2026, receita revisada, 5 sementes).
Todos os números e conclusões válidos estão no [README.md](README.md).

---

## Concluído

### Infraestrutura
- [x] **Modelo paramétrico** — `N_BRANCHES` 1-5, posições por FLOPs acumulados → [src/model.py](src/model.py)
- [x] **Cascata edge→cloud com `p_tar`** — logits coletados 1× por modelo → [src/inference.py](src/inference.py)
- [x] **Estratégias de loss weights** — `equal` / `linspace` / `inv_linspace` / CSV
- [x] **Temperature scaling** — `overall` + `branches`, LBFGS, salvo no checkpoint → [src/calibrate.py](src/calibrate.py)
- [x] **Métrica de custo** — MACs acumulados por saída e `E[MACs]` médio por amostra
- [x] **Reliability diagrams + ECE** → [src/reliability.py](src/reliability.py)
- [x] **Variantes de arquitetura** — `EXTRA_STRIDE1` com proteção no load (o state_dict é intercambiável entre variantes sem erro)
- [x] **Pipeline não supervisionado** → [run_retrain_v4.sh](run_retrain_v4.sh)
- [x] **uv** como package manager

### Experimentos (geração v4)
- [x] **Grade b2–b5 × {equal, linspace}** × 3 calibrações × 10 `p_tar`, com duplo checkpoint (val loss / val acc) = 60 CSVs
- [x] **Ablation de arquitetura** — v3a (mapa 4×4) e v3b (8×8): v3b eleva o teto para **93.01% ± 0.22** (+1.93 ± 0.30 pp em três treinos), v3a inconclusiva (+0.49 ± 0.54 pp)
- [x] **Estudo de seleção de checkpoint** — efeito acompanha quão cedo a val loss atinge o mínimo, não a capacidade do modelo
- [x] **Correções de metodologia (v4)** — classificador final destravado no warmup, dropout na saída final, validação dividida em seleção/calibração, `min_delta`

## Backlog

- [x] **Consolidação estatística** (10/08) — `p_tar` escolhido na validação, McNemar pareado, grade estendida a 0.99 → [src/stats.py](src/stats.py), README §7. Aposentados por falta de significância: overall vs branches, ganho de calibração no ponto ótimo, v3a, efeito de seleção nos modelos `equal`
- [x] **Infraestrutura de serving** (18/08) — servidores Flask de borda e nuvem em [src/serving/](src/serving/) com decomposição de tempo (pré-proc / borda / rede / nuvem) e payload medido. **Limitação:** os dois serviços rodam no mesmo host e o enlace é simulado por atraso proporcional ao payload, então os números em `results/latency/` não são medida de implantação real. Por isso saíram do README (19/09)
- [ ] **Validação em borda e nuvem reais** (WIP, README §7, abordagem definida em 08/10) — borda no PC só com CPU e poucos núcleos; nuvem em EC2 só CPU, mesmo tipo, em São Paulo, Virgínia, Frankfurt e Tóquio; conexão da borda por rede doméstica e celular (4G/5G); RTT e jitter de cada sessão medidos com `ping`; enlace simulado desligado. Estratégias: só nuvem (PNG), cascata enviando features (float16), cascata enviando a imagem, só borda. Modelo b3 `linspace` r1-last, `overall`, `p_tar` 0.80/0.90/0.99, cortes após as saídas 2 e 3. Medidas: latência média, mediana, p95, p99 e distribuição, decomposição do tempo, bytes, acurácia e fração na borda; 1000 imagens por configuração, ordem sorteada, duas sessões por região em horários diferentes. Pendente no código: modo de envio da imagem na borda, modo só borda, conexão reaproveitada e tempos por amostra no cliente, script de análise (percentis e distribuição), roteiro de instalação na AWS
- [x] **README reescrito para leitor externo** (19/09) — texto impessoal, sem histórico de gerações, sem a seção de latência e sem o TL;DR do topo; glossário de estudo em [docs/glossario.md](docs/glossario.md)
- [x] **README visual** (25/08) — TL;DR de 4 números no topo, forest plot das 27 comparações pareadas (`plots/forest_mcnemar.png`), figuras redistribuídas para as seções a que pertencem com legenda de leitura, §9 comprimida de 54 linhas de prosa para uma tabela de 8 achados
- [x] **Revisão do README contra os dados** (27/09): 9 de 27 comparações significativas (eram citadas 6); limiares escolhidos na validação vão de 0.80 a 0.99; época 133 do b5 linspace preenchida; figuras passam a usar o checkpoint por acurácia; `linspace` deixa de ser atribuído ao BranchyNet (o artigo recomenda o contrário, mais peso nas saídas rasas); novas seções de ponto de operação com borda e nuvem, análise por classe, comparação com a literatura, limitações e ambiente → [src/per_class.py](src/per_class.py), [src/profile_models.py](src/profile_models.py), `results/stats_operating_points.csv`
- [x] **Réplicas com outras sementes e baseline sem saídas antecipadas** (29/09): 23 treinos, três por configuração → [run_replicas.sh](run_replicas.sh), [src/replicas.py](src/replicas.py). Confirmado: v3b > padrão, `linspace` > `equal` (11 de 12 pares), seleção por acurácia quando a loss atinge o mínimo cedo (13 de 13), calibração sem efeito no limiar escolhido, saídas antecipadas melhoram a saída final sobre o b0. Não se confirmou: b3 > b2 (+1.14 pp no treino original, +0.52 ± 0.73 nos três). v3a passou de "não difere" para inconclusivo
- [x] **Retreino com receita revisada** (concluído em 06/10, 60 treinos, 46 h, nenhuma falha) — decidido em 01 a 03/10, comparando com o framework do orientador: weight decay 0.00005, lr em cosseno, cabeça das saídas `BatchNorm2d → pool → Linear` (como o orientador), 5 sementes com divisão de dados própria por semente (`SPLIT_SEED`, já implementado), log da loss de validação por saída. Mantidos: warmup e lr atuais, augmentation, 32×32, calibração atual. Escolhido cosseno com uma única queda em 200 épocas fixas, sem early stopping (novos defaults de [src/train.py](src/train.py); a receita v4 se reproduz com `WEIGHT_DECAY=0 LR_SCHEDULE=constant EPOCHS=150 PATIENCE=30 EXIT_HEAD=mlp`). Pendente: rodar o piloto [run_pilot_v5.sh](run_pilot_v5.sh) (b0, b3 `equal` e b3 `linspace`, semente 1, mesma divisão das réplicas r1, comparação em [src/pilot.py](src/pilot.py)) e depois o retreino completo. Piloto concluído em 04/10: saída final melhorou (b0 +0.93, b3 `equal` +0.39, b3 `linspace` +0.59 pp), primeira saída piorou (−1.8 e −3.2 pp); o teste com a cabeça antiga (04/10) mostrou que a queda vem da receita, não da cabeça (as duas cabeças empatam), então fica a do orientador. Retreino completo pronto em [run_retrain_v5.sh](run_retrain_v5.sh): 5 sementes, grade b2 a b5, b0, v3b, b0 sem pré-treino e b3 `inv_linspace`; cada treino salva também o modelo da última época (`-last`) e a loss de treino por saída; um treino por vez, saída em `saved_model_v5/` e `results_v5/`, cerca de 40 h. Modelos do piloto movidos para `saved_model_v5/pilot/`. Dados e augmentation passaram para a GPU (`GPU_DATA=1`, padrão), cerca de 1.4 vez mais rápido
- [x] **Regras de inferência** (03/10) — quando nenhuma saída atinge o `p_tar`, a classe vem da saída mais confiante (como o orientador); o ponto de operação passa a ser o menor `p_tar` cuja acurácia na validação não fica abaixo da saída final sozinha → [src/stats.py](src/stats.py), [src/inference.py](src/inference.py), [src/serving/](src/serving/). Nos modelos v5 a saída mais confiante empata com a saída final (−0.03 pp em média)
- [x] **README reescrito com os resultados v5** (06/10) — médias de 5 treinos, comparações pareadas por treino com IC95% por t, McNemar só dentro do mesmo modelo com Holm, comparação no mesmo custo, calibração como garantia do limiar, tamanho do mapa enviado por ponto de corte, b0 do zero, `inv_linspace`, última época como modelo → [src/replicas.py](src/replicas.py), [src/plots.py](src/plots.py), `results_v5/`. Mudanças de conclusão em relação à v4: o ganho da saída final sobre o b0 deixou de ser geral (só b3 `linspace`); `linspace` contra `equal` depende do orçamento
- [x] **Revisão do README** (21/10) — fontes movidas para [docs/referencias.md](docs/referencias.md); garantia do limiar medida por modelo (40 modelos: até 0.90 sem calibração, 0.95 `overall`, 0.97 `branches`); tabela de ECE corrigida (as colunas depois da saída 2 estavam deslocadas) e trocada para `linspace`; modelo de exemplo passa a ser o b3 `linspace` (por classe e reliability); nova comparação cascata contra saída fixa de custo igual ou maior → `results_v5/replicas_fixed_exit.csv` ([src/replicas.py](src/replicas.py))
- [x] **Desempate do ponto de operação** (26/10) — quando nenhum limiar alcança a saída final em `val_select` (15 dos 50 modelos com `overall`), o de maior acurácia passa a ser desempatado pelo menor `p_tar`, não pelo maior → [src/stats.py](src/stats.py). Mesma acurácia (0.00 a −0.01 pp), até 6.6% menos MACs na grade e 20% na v3b (69.4 → 55.4 M, nuvem 18.5% → 7.8%). `stats.py`, `replicas.py`, `per_class.py` e `plots.py` rodados de novo; README atualizado nas seções 3 a 6, 8, 9 e 10
- [ ] **Escrita da dissertação** — fonte primária: [README.md](README.md) + CSVs em `results_v5/`

---

## Checkpoints

`saved_model_v5/`: 180 checkpoints v5 — `mobilenet_[variante]r{1..5}[-acc|-last]_b{N}_{pesos}_best.pth`
(sem sufixo = menor val loss, `-acc` = maior média de val acc, `-last` = última época, usado
nos resultados); variantes `s7-14-` (v3b) e `scratch-` (b0 sem ImageNet). `history/` tem o
histórico por época de cada treino; `pilot/` guarda os modelos do piloto.

`saved_model/`: 20 checkpoints v4 — `mobilenet_[variante_]b{2..5}_{equal,linspace}_best.pth`,
onde a variante vazia = seleção por val loss e `acc` = seleção por val acc; `s14`/`s7-14`
(+ `-acc`) são as variantes de arquitetura. Metadados de cada um: posições dos ramos,
`extra_stride1_blocks`, normalização, temperaturas, época e critério de seleção.

Backups das gerações anteriores: `saved_model_backup_v1_adam/`, `_v2_sgd/`, `_v3_experiment/`.

## Resultados

`results_v5/` — 900 CSVs de cascata, `preds/` com os logits e as tabelas consolidadas
(`replicas_*.csv`, `stats_*.csv`, `per_class.csv`, `payload.csv`, `model_profile.csv`).

`results/` — 60 CSVs da geração v4; gerações anteriores em `results/legacy_prev4/`.

---

*Última atualização: 2026-10-26*
