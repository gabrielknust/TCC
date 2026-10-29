# Referências e comparação com a literatura

Material retirado do README em 08/10/2026, guardado para consulta na escrita da
dissertação. As seções citadas são as do README atual.

## Onde cada fonte era citada no README

| Seção do README | Fonte | Texto original |
|---|---|---|
| 1, Saídas | Pacheco et al. | "É a cabeça do framework de referência do grupo (`exit_type = "bnpool"`) e a camada totalmente conectada única descrita por Pacheco et al." |
| 1, O que seria enviado à nuvem | Kang et al., 2017 (Neurosurgeon) | "É o mesmo princípio do Neurosurgeon (Kang et al., 2017): só vale particionar onde o volume de dados encolhe, e em imagens de 32×32 isso acontece apenas nos blocos profundos." |
| 2, Pesos da loss | Pacheco et al.; Laskaridis et al., 2020 (SPINN) | "`linspace`: pesos crescentes de 0.15 na saída mais rasa a 1.0 na final. É o esquema de Pacheco et al., que o atribuem ao SPINN (Laskaridis et al., 2020), e o do framework de referência." |
| 2, Pesos da loss | Teerapittayanon et al., 2016 (BranchyNet) | "`inv_linspace`: pesos decrescentes de 1.0 a 0.15, mais peso nas saídas rasas, como recomenda o BranchyNet." |
| 3, Calibração | Guo et al., 2017 | "Temperature scaling (Guo et al., 2017): os logits são divididos por uma temperatura `T`..." |
| 6, Qual modelo de cada treino usar | Guo et al., 2017 | "...e depois sobe enquanto a acurácia ainda melhora: é o overfitting de NLL descrito por Guo et al." |
| 8, Análise estatística | Dietterich, 1998 | "O teste de McNemar compara dois classificadores nas mesmas imagens e controla a amostragem do teste, mas não o sorteio do treino (Dietterich, 1998)." |
| 9, Resumo dos achados | BranchyNet | achado 4: "Mais peso nas saídas rasas (BranchyNet) piora a cascata" |
| 10, Limitações | Ma et al., 2018 (ShuffleNet V2) | "O custo é medido em MACs, não em tempo, e MACs preveem mal a latência de convoluções depthwise como as da MobileNet (Ma et al., 2018)." |

## Comparação com a literatura

Antiga seção 9 do README.

Não há referência publicada com a mesma rede, o mesmo dataset e a mesma resolução, então a
comparação é qualitativa.

* **BranchyNet** (Teerapittayanon et al., 2016) avaliou saídas antecipadas em CIFAR10 com
  AlexNet e ResNet110. Com saídas antecipadas, a AlexNet acerta 79.19% contra 78.38% da
  original, resolvendo 90.8% das amostras antes da última saída; a ResNet110 acerta 79.17%
  contra 80.70%, resolvendo 55.3% antes. O critério de saída é a entropia, não a confiança
  máxima, e o ganho foi medido em tempo. A fração resolvida antes da saída final aqui (de
  81% a 90%) fica perto da AlexNet. O BranchyNet recomenda dar mais peso às saídas rasas;
  aqui esse esquema (`inv_linspace`) fortalece a primeira saída, mas piora a saída final e a
  cascata.
* **SPINN** (Laskaridis et al., 2020). Pacheco et al. inserem as saídas de forma
  equidistante seguindo a metodologia do SPINN e usam pesos de loss que crescem linearmente
  com a posição da saída, também citando o SPINN. O framework de referência implementa os
  dois, e são as escolhas de posicionamento e do `linspace` aqui.
* **Deep supervision.** Classificadores auxiliares treinados junto costumam melhorar a saída
  final (Lee et al., 2015, DSN). Aqui isso só aparece no b3 `linspace`; no b4 `equal` a
  saída final piora, o que se aproxima do relato do MSDNet (Huang et al., 2018) de que
  classificadores rasos podem atrapalhar os profundos.
* **Pacheco, Couto e Simeone (2023)** estudam o temperature scaling em MobileNetV2 e outras
  redes com saídas antecipadas, em Caltech256 e CIFAR100. Eles encontram saídas
  overconfident, redução do ECE com a calibração, menos amostras classificadas na borda
  depois de calibrar e, com limiar fixo, maior probabilidade de atender ao requisito de
  acurácia. Isso concorda com a seção 3, em que a calibração faz o acerto na borda
  acompanhar `p_tar`. Duas diferenças: quando cada configuração escolhe o limiar na
  validação, a calibração não muda a acurácia; e aqui a temperatura única piora as saídas
  rasas, que já nascem calibradas. Eles treinam até a loss de validação parar de cair por
  10 épocas. A seção 6 mostra que a loss da saída final atinge o mínimo, quase sempre, entre
  as épocas 12 e 27, então esse critério interromperia o treino com a acurácia ainda subindo.
* **Guo et al. (2017)** descrevem o overfitting de NLL: a loss de validação sobe enquanto a
  acurácia ainda melhora. É o que a seção 6 observa na saída final.
* **Neurosurgeon** (Kang et al., 2017) mostra que os bons pontos de partição são aqueles em
  que o volume de dados encolhe. A seção 1 mostra que, em 32×32, isso só acontece a partir
  do bloco 14.
* **Kornblith, Shlens e Le (2019)** avaliam a transferência do ImageNet ampliando as imagens
  para a resolução do pré treino. Aqui, sem ampliar, os pesos do ImageNet ainda valem 16.6
  pp.
* **Decaimento em cosseno** (Loshchilov e Hutter, 2017) é o que torna a última época uma boa
  escolha de modelo (seção 6).
* O framework de referência do grupo, em [early_exit_dnn_analysis/](../early_exit_dnn_analysis/),
  usa entradas de 300×300 no Caltech256. Com 32×32, a MobileNetV2 precisa da adaptação da
  seção 1, o que impede comparar números absolutos com trabalhos em resolução maior.

## Lista de referências

* Teerapittayanon et al., *BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks* (2016)
* Laskaridis et al., *SPINN: Synergistic Progressive Inference of Neural Networks over Device and Cloud* (MobiCom 2020)
* Pacheco, Couto e Simeone, *On the impact of deep neural network calibration on adaptive edge offloading for image classification*, Journal of Network and Computer Applications (2023)
* Guo et al., *On Calibration of Modern Neural Networks* (2017)
* Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (2018)
* Lee et al., DSN (AISTATS 2015)
* Huang et al., MSDNet (ICLR 2018)
* Kang et al., *Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge* (ASPLOS 2017)
* Kornblith, Shlens e Le, *Do Better ImageNet Models Transfer Better?* (CVPR 2019)
* Loshchilov e Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (ICLR 2017)
* Dietterich, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms* (Neural Computation 1998)
* Ma et al., *ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design* (ECCV 2018)

PDFs em `fontes/`, pasta local fora do git.

## Acurácia de referência da MobileNetV2 no CIFAR10

Conferido em 21/10/2026, para a limitação sobre acurácia absoluta.

* [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar): MobileNetV2 com 94.43%
  no CIFAR10, treinada do zero em 32×32. A adaptação tira o stride do primeiro conv e do
  segundo estágio, o que deixa o mapa final em 4×4 (aqui é 2×2). Treino com SGD, lr 0.1,
  momentum 0.9, weight decay 5e-4, cosseno e 200 épocas, com o mesmo augmentation daqui. É
  um repositório, não um artigo.
* A diferença para este trabalho é uma redução de resolução a menos e outra receita. A v3b
  (mapa 8×8) chega a 93.03%, o que sugere que a resolução explica a maior parte da
  diferença.
* Outros números publicados variam muito com a resolução, o pré treino e a parte da rede que
  é treinada, então qualquer comparação precisa dizer qual adaptação foi usada.
