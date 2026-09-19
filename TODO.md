# TCC — TODO & Status

## Objetivo
DNN com saídas antecipadas (early-exit) para inferência híbrida edge/cloud usando **MobileNetV2 em CIFAR-10**. Saídas confiantes classificam na borda; amostras incertas vão pra nuvem.

**Estado atual: geração v4** (retreino de 10/08/2026, pipeline corrigido). Todos os
números e conclusões válidos estão no [README.md](README.md).

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
- [x] **Ablation de arquitetura** — v3a (mapa 4×4) e v3b (8×8): v3b eleva o teto para **92.98% ± 0.50** (+1.65 pp), v3a empata com o padrão
- [x] **Estudo de seleção de checkpoint** — efeito acompanha quão cedo a val loss atinge o mínimo, não a capacidade do modelo
- [x] **Correções de metodologia (v4)** — classificador final destravado no warmup, dropout na saída final, validação dividida em seleção/calibração, `min_delta`

## Backlog

- [x] **Consolidação estatística** (10/08) — `p_tar` escolhido na validação, McNemar pareado, grade estendida a 0.99 → [src/stats.py](src/stats.py), README §7. Aposentados por falta de significância: overall vs branches, ganho de calibração no ponto ótimo, v3a, efeito de seleção nos modelos `equal`
- [x] **Infraestrutura de serving** (18/08) — servidores Flask de borda e nuvem em [src/serving/](src/serving/) com decomposição de tempo (pré-proc / borda / rede / nuvem) e payload medido. **Limitação:** os dois serviços rodam no mesmo host e o enlace é simulado por atraso proporcional ao payload, então os números em `results/latency/` não são medida de implantação real. Por isso saíram do README (19/09)
- [ ] **Validação em borda e nuvem reais** — a arquitetura nunca foi executada em nós separados de verdade; nenhuma conclusão sobre latência de implantação pode ser tirada até isso ser medido. Pendente: rodar borda e nuvem em máquinas distintas, medir o enlace real em vez de simulá-lo, varrer condições (Wi-Fi, 4G, 5G) e estender à v3b e à grade b2–b5. Enquanto não houver medição, o README não reporta latência
- [x] **README reescrito para leitor externo** (19/09) — texto impessoal, sem histórico de gerações, sem a seção de latência e sem o TL;DR do topo; glossário de estudo em [docs/glossario.md](docs/glossario.md)
- [x] **README visual** (25/08) — TL;DR de 4 números no topo, forest plot das 27 comparações pareadas (`plots/forest_mcnemar.png`), figuras redistribuídas para as seções a que pertencem com legenda de leitura, §9 comprimida de 54 linhas de prosa para uma tabela de 8 achados
- [ ] **Escrita da dissertação** — fonte primária: [README.md](README.md) + CSVs em `results/`

---

## Checkpoints

`saved_model/`: 20 checkpoints v4 — `mobilenet_[variante_]b{2..5}_{equal,linspace}_best.pth`,
onde a variante vazia = seleção por val loss e `acc` = seleção por val acc; `s14`/`s7-14`
(+ `-acc`) são as variantes de arquitetura. Metadados de cada um: posições dos ramos,
`extra_stride1_blocks`, normalização, temperaturas, época e critério de seleção.

Backups das gerações anteriores: `saved_model_backup_v1_adam/`, `_v2_sgd/`, `_v3_experiment/`.

## Resultados

`results/` — 60 CSVs da geração v4; gerações anteriores em `results/legacy_prev4/`
(`src/plots.py` varre só o topo do diretório, então não se misturam).

---

*Última atualização: 2026-09-19*
