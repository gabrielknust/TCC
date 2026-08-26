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
- [ ] **Análises pós-hoc restantes** (opcionais):
  - [ ] **ECE condicional** — calibração na população que de fato chega a cada saída, não na marginal
  - [ ] Refinos: `E[MACs]` contando heads atravessados, IC de Wilson, log do `calibrate.py` (o "NLL antes" é em T=1.5, não T=1)
- [x] **Integração Flask edge/cloud** (18/08) — servidores próprios em [src/serving/](src/serving/) com decomposição de tempo (pré-proc / borda / rede / nuvem), payload medido e simulação de enlace. Achados: particionar após o 1º ramo é contraproducente (mapa de features 5,5× maior que a imagem); no ponto certo, 19.2 ms vs 94.4 ms do só-nuvem sob 10 Mbps/50 ms → README §8
- [ ] **Latência: extensões** — medir a v3b (custo maior, payload maior) e a grade b2–b5; rodar borda e nuvem em máquinas distintas de verdade; varrer condições de enlace (Wi-Fi, 4G, 5G)
- [x] **README visual** (25/08) — TL;DR de 4 números no topo, forest plot das 27 comparações pareadas (`plots/forest_mcnemar.png`), figuras redistribuídas para as seções a que pertencem com legenda de leitura, §9 comprimida de 54 linhas de prosa para uma tabela de 8 achados
- [ ] **Escrita da dissertação** — fonte primária: [README.md](README.md) + CSVs em `results/`
- [ ] *(opcional)* Exit heads com convolução antes do pooling — melhoraria os ramos rasos; é melhoria, não correção, então merece experimento próprio
- [ ] *(opcional)* Expandir v3 para b2/b4/b5 e `linspace`
- [ ] *(opcional)* Outros datasets — o orientador testou Caltech-256 e CIFAR-100

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

*Última atualização: 2026-08-25*
