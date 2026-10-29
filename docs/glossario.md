# Glossário de estudo

Arquivo de apoio interno: explica os termos que aparecem no [README](../README.md), no
código e nos artigos de referência. Não faz parte da documentação do projeto.

## Índice

**Rede e arquitetura:** [MobileNetV2](#mobilenetv2) · [backbone](#backbone) ·
[bloco e features[]](#bloco-e-features) · [cabeça (head)](#cabeça-head) ·
[mapa de features](#mapa-de-features) · [canal](#canal) · [stride](#stride) ·
[pooling](#pooling) · [flatten](#flatten) · [camada linear](#camada-linear) ·
[ReLU](#relu) · [saída antecipada (early exit)](#saída-antecipada-early-exit) ·
[cascata edge/cloud](#cascata-edgecloud) · [p_tar](#p_tar) · [bN e v3b](#bn-e-v3b) ·
[BatchNorm](#batchnorm) · [deep supervision](#deep-supervision)

**Treino:** [CIFAR10](#cifar10) · [treino, validação e teste](#treino-validação-e-teste) ·
[val_select e val_calib](#val_select-e-val_calib) · [augmentation](#augmentation) ·
[normalização](#normalização) · [época](#época) · [batch](#batch) ·
[learning rate](#learning-rate) · [Adam](#adam) · [loss](#loss) ·
[cross entropy](#cross-entropy) · [pesos da loss](#pesos-da-loss) ·
[transfer learning e ImageNet](#transfer-learning-e-imagenet) · [warmup](#warmup) ·
[ajuste fino](#ajuste-fino) · [decaimento em cosseno](#decaimento-em-cosseno) ·
[weight decay](#weight-decay) · [overfitting](#overfitting) ·
[regularização](#regularização) · [dropout](#dropout) ·
[early stopping](#early-stopping) · [checkpoint](#checkpoint) · [seed](#seed) ·
[treino do zero](#treino-do-zero)

**Probabilidade e calibração:** [logits](#logits) · [softmax](#softmax) ·
[confiança](#confiança) · [entropia](#entropia) · [NLL](#nll) · [temperatura](#temperatura) ·
[temperature scaling](#temperature-scaling) · [overall e branches](#overall-e-branches) ·
[LBFGS](#lbfgs) · [overconfident](#overconfident) · [calibração](#calibração) ·
[ECE](#ece) · [reliability diagram](#reliability-diagram)

**Custo e tempo:** [MAC](#mac) · [FLOP](#flop) · [MACs acumulados](#macs-acumulados) ·
[E[MACs]](#emacs) · [thop](#thop) · [latência](#latência) · [RTT](#rtt) ·
[payload](#payload) · [float16](#float16) · [pp](#pp)

**Estatística:** [acurácia](#acurácia) · [IC95%](#ic95) · [p valor](#p-valor) ·
[McNemar](#mcnemar) · [réplica e desvio padrão](#réplica-e-desvio-padrão) ·
[significância](#significância) · [correção de Holm](#correção-de-holm) ·
[data snooping](#data-snooping)

**Ferramentas:** [PyTorch e torchvision](#pytorch-e-torchvision) · [uv](#uv) ·
[Flask](#flask) · [serving](#serving) · [npz](#npz)

## Rede e arquitetura

### MobileNetV2
Rede convolucional criada para rodar em celular, publicada por Sandler et al. em 2018. É
menor e mais barata que as redes tradicionais porque separa a convolução em duas etapas
mais simples. Aqui ela é o ponto de partida do modelo.

### Backbone
O corpo principal da rede, que transforma a imagem em uma representação numérica. É tudo
menos os classificadores do final. Neste projeto o backbone é a MobileNetV2 sem a cabeça
original.

### Bloco e features[]
A MobileNetV2 é uma sequência de blocos, numerados de 0 a 18, guardados na lista
`features[]` do torchvision. A imagem passa por eles em ordem. Cada bloco entende um pouco
mais da imagem e entrega um mapa menor e mais "abstrato" que o anterior.

### Cabeça (head)
O pedacinho no fim de um caminho que transforma a representação da rede em uma resposta:
10 números, um por classe. Este projeto tem várias cabeças, uma por saída. As das saídas
antecipadas fazem `BatchNorm → média de cada canal → camada linear`, como no framework do
orientador.

### Mapa de features
A saída de um bloco: uma pilha de imagens pequenas, uma por canal. No fim da rede padrão
deste projeto sobra um mapa de 2×2 pixels com 1280 canais. É pouca informação espacial, e
é isso que limita a acurácia (seção 6 do README).

### Canal
Cada "camada de informação" de um mapa de features. A imagem de entrada tem 3 canais
(vermelho, verde e azul). No meio da rede há dezenas ou centenas, cada um reagindo a um
padrão diferente. O `C` no código das cabeças é o número de canais naquele ponto.

### Stride
De quantos em quantos pixels a convolução anda ao varrer a imagem. Stride 1 anda de um em
um e mantém o tamanho; stride 2 pula de dois em dois e corta o tamanho pela metade. Reduzir
strides é o que a variante v3b faz para manter os mapas maiores.

### Pooling
Resumir uma região em um número só. `AdaptiveAvgPool2d(1×1)` tira a média de cada canal
inteiro, transformando um mapa de 2×2 por 1280 canais em 1280 números.

### Flatten
Enfileirar os números em uma lista única, para que uma camada linear possa usá los.

### Camada linear
Também chamada de densa ou fully connected. Multiplica cada número de entrada por um peso,
soma tudo e entrega os números de saída. `Linear(1280 → 10)` recebe 1280 números e devolve 10.

### ReLU
Função que zera valores negativos e deixa os positivos como estão. Serve para a rede
aprender relações que não são uma simples linha reta.

### Saída antecipada (early exit)
Um classificador colocado no meio da rede, que pode responder sem esperar o resto. A
alternativa seria sempre atravessar a rede inteira.

### Cascata edge/cloud
O arranjo proposto: as primeiras saídas rodam perto do usuário (borda) e as últimas em um
servidor (nuvem). Cada imagem para na primeira saída confiante. Neste repositório o arranjo
foi avaliado em custo computacional; a execução em máquinas separadas ainda não foi medida.

### p_tar
O limiar de confiança. Se a saída está com confiança maior ou igual a `p_tar`, ela responde;
senão a imagem segue adiante. Limiar alto significa mais acerto por saída e mais imagens
indo longe.

### bN e v3b
`bN` é o modelo com N saídas antecipadas (b3 tem 3, mais a final). O `b0` é a mesma rede
sem nenhuma saída antecipada, usada como referência. A `v3b` é a variante que reduz dois
strides internos para manter o mapa final em 8×8, com cerca de 3.4 vezes o custo da cascata
padrão.

### BatchNorm
Camada que padroniza cada canal (média zero e desvio um, com ajuste aprendido). Deixa a
escala dos números estável e facilita o treino. Aqui aparece no início de cada cabeça das
saídas antecipadas.

### Deep supervision
Treinar classificadores auxiliares no meio da rede junto com o principal, na esperança de
que o sinal extra melhore também a saída final. Aqui isso só aconteceu no b3 com
`linspace`.

## Treino

### CIFAR10
Conjunto de 60 mil fotos coloridas de 32×32 pixels, divididas em 10 classes (avião,
automóvel, pássaro, gato, veado, cachorro, sapo, cavalo, navio, caminhão). São 50 mil para
treino e 10 mil para teste.

### Treino, validação e teste
Treino: as imagens que o modelo usa para aprender. Validação: imagens separadas, usadas
para tomar decisões sobre o treino (quando parar, qual versão guardar). Teste: imagens que
só são olhadas no fim, para medir o resultado. Misturar os três é o erro clássico.

### val_select e val_calib
As duas metades da validação neste projeto, de 5 mil imagens cada. `val_select` decide
quando parar e qual checkpoint guardar. `val_calib` ajusta as temperaturas. Separadas para
que o mesmo conjunto não escolha e valide o modelo.

### Augmentation
Alterar as imagens de treino de propósito para a rede não decorar. `RandomCrop(32, pad=4)`
recorta a imagem de um jeito diferente a cada vez; `RandomHorizontalFlip` espelha metade
delas. Só no treino, nunca na validação ou no teste.

### Normalização
Reescalar os valores dos pixels para uma faixa padrão, subtraindo a média e dividindo pelo
desvio. Aqui usam se os valores do ImageNet, para casar com os pesos herdados.

### Época
Uma passada completa por todas as imagens de treino. Cada treino aqui tem 200 épocas.

### Batch
O grupo de imagens processadas de uma vez antes de atualizar os pesos. Aqui, 128 imagens.

### Learning rate
O tamanho do passo na hora de corrigir os pesos. Grande demais, o treino oscila; pequeno
demais, demora. Aqui é 0.001 no warmup e começa em 0.0001 no ajuste fino, caindo até zero
em [cosseno](#decaimento-em-cosseno).

### Adam
O otimizador, ou seja, a regra que decide como mexer nos pesos a cada passo. Ajusta o
tamanho do passo de cada peso automaticamente.

### Loss
A nota de castigo do modelo. Quanto maior, pior. O treino inteiro é uma tentativa de
diminuir esse número.

### Cross entropy
A loss usada em classificação. É a mesma conta da [NLL](#nll): pega a probabilidade dada à
classe certa e castiga o modelo pelo logaritmo dela com o sinal invertido.

### Pesos da loss
Como o modelo tem várias saídas, a loss final é a soma das losses de cada uma, cada qual
multiplicada por um peso. `equal` usa peso 1.0 em todas. `linspace` usa pesos crescentes,
de 0.15 na primeira a 1.0 na última, o que faz o treino priorizar as saídas profundas; é o
esquema que Pacheco et al. atribuem ao SPINN. `inv_linspace` faz o contrário, como recomenda
o BranchyNet.

### Transfer learning e ImageNet
ImageNet é um conjunto de milhões de fotos em mil categorias. Uma rede treinada nele já
sabe reconhecer bordas, texturas e formas. Aproveitar esses pesos em outro problema é
transfer learning, e é por isso que o treino aqui parte dos pesos `IMAGENET1K_V1`.

### Warmup
As primeiras 5 épocas, em que o backbone fica congelado (pesos não mudam) e só as cabeças
novas treinam. Serve para as cabeças, que começam aleatórias, não bagunçarem os pesos bons
herdados do ImageNet.

### Ajuste fino
Fase seguinte, em que a rede inteira volta a treinar, com learning rate menor, para
adaptar o backbone ao CIFAR10 sem destruir o que ele já sabia.

### Decaimento em cosseno
Forma de reduzir o learning rate ao longo do treino seguindo meia onda de cosseno: começa no
valor inicial, cai devagar, depois mais rápido, e termina em zero na última época. Com o
passo quase zero no fim, as últimas épocas mudam pouco o modelo, e a última vira uma boa
escolha.

### Weight decay
Penalidade que puxa os pesos para perto de zero a cada passo. Pesos menores costumam dar
confianças menos extremas e menos overfitting. Aqui o valor é 0.00005, o mesmo do framework
do orientador.

### Overfitting
Quando o modelo decora o treino em vez de aprender o padrão. Acerta muito nas imagens que
já viu e mal nas novas.

### Regularização
Qualquer técnica que atrapalhe o modelo de propósito para ele não decorar. Dropout e
augmentation são exemplos.

### Dropout
Durante o treino, desliga ao acaso uma fração dos neurônios a cada passo (0.2 quer dizer um
quinto). Assim a rede não pode depender de um neurônio específico. Na hora de usar o
modelo, nada é desligado. Aqui só a saída final usa dropout.

### Early stopping
Parar o treino quando a validação para de melhorar, depois de uma paciência de N épocas sem
melhora. Não é usado aqui: os treinos têm 200 épocas fixas, porque o decaimento em cosseno
precisa de duração fixa e porque a loss de validação da saída final atinge o mínimo cedo
demais (seção 6 do README).

### Checkpoint
O arquivo com os pesos do modelo salvos em um momento do treino. Aqui cada treino salva
três: o da última época (usado nos resultados), o da maior acurácia de validação e o da
menor loss de validação.

### Seed
Número que fixa o sorteio do computador, para que o experimento possa ser repetido. Aqui o
treino s usa `TRAIN_SEED` s (pesos iniciais, ordem das imagens, augmentation) e
`SPLIT_SEED` 41 + s (quais imagens treinam e quais validam).

### Treino do zero
Treinar a rede a partir de pesos aleatórios, sem aproveitar o ImageNet. Aqui o b0 treinado
do zero acerta 16.6 pp menos que o b0 com os pesos do ImageNet.

## Probabilidade e calibração

### Logits
Os números crus que a rede produz no fim, um por classe, antes de virarem probabilidade.
Podem ter qualquer valor e não somam 1. Exemplo com 3 classes: `[4.0, 2.0, 1.0]`.

### Softmax
A conta que transforma logits em probabilidades que somam 1. Eleva `e` a cada logit e
divide pela soma de todos. Os logits `[4, 2, 1]` viram aproximadamente `[0.84, 0.11, 0.04]`.

### Confiança
A maior das probabilidades depois do softmax. No exemplo acima, 0.84. É esse número que a
cascata compara com `p_tar`.

### Entropia
Outra forma de medir incerteza, usada pelo BranchyNet como critério de saída. Olha todas as
probabilidades, não só a maior: é baixa quando uma classe concentra quase tudo e alta
quando as probabilidades estão espalhadas. Aqui o critério é a confiança máxima.

### NLL
Negative log likelihood. Para cada imagem, pega a probabilidade dada à classe certa e
calcula o logaritmo com o sinal invertido: 0.9 vira 0.11, 0.5 vira 0.69, 0.1 vira 2.3. A
NLL é a média desses valores. Ela pune muito quem erra com muita certeza.

### Temperatura
Um número `T` pelo qual os logits são divididos antes do softmax. `T` maior que 1 aproxima
as probabilidades umas das outras e derruba a confiança; `T` menor que 1 faz o contrário.
Com `T = 2`, os logits `[4, 2, 1]` viram `[0.63, 0.23, 0.14]`, e a confiança cai de 0.84
para 0.63. `T` não muda qual classe ganha, só o quanto o modelo se diz seguro.

### Temperature scaling
A técnica de calibrar um modelo achando o melhor `T`, proposta por Guo et al. em 2017.
Acontece depois do treino e não mexe em nenhum peso da rede.

### overall e branches
Os dois modos de calibrar aqui. `overall` usa um único `T` para todas as saídas.
`branches` usa um `T` por saída. `none` é sem calibração.

### LBFGS
O algoritmo que procura o valor de `T`. Funciona como uma busca guiada: testa um valor,
vê como a NLL reage, usa os últimos passos para estimar a curvatura da função e dar o
próximo passo. Aqui começa em `T = 1.5` e faz 50 iterações, em
[src/calibrate.py](../src/calibrate.py).

### Overconfident
Modelo que diz 95% de certeza e acerta 80% das vezes. O contrário, dizer 60% e acertar
80%, é underconfident. Os dois atrapalham a cascata, porque a decisão de parar depende da
confiança ser honesta.

### Calibração
Deixar a confiança do modelo bater com o acerto real: entre as respostas dadas com 90% de
certeza, cerca de 90% devem estar certas.

### ECE
Expected calibration error. Um número que resume o quanto a confiança mente. Separa as
respostas em faixas de confiança, compara a confiança média com o acerto real de cada
faixa e tira a média das diferenças. ECE de 1.5% é bom; de 13%, ruim.

### Reliability diagram
O gráfico da calibração. No eixo horizontal, a confiança declarada; no vertical, o acerto
real. A diagonal é o ideal. Pontos abaixo da diagonal indicam excesso de confiança.

## Custo e tempo

### MAC
Multiply accumulate: uma multiplicação seguida de uma soma (`a × b + c`). É a conta básica
de uma rede neural, e contar MACs mede o custo sem depender do computador usado.

### FLOP
Floating point operation. Cada MAC são duas operações, então FLOPs equivalem mais ou menos
ao dobro dos MACs. Os dois nomes aparecem misturados na literatura.

### MACs acumulados
O total de contas desde a entrada até uma saída: todos os blocos do caminho, mais a cabeça
daquela saída. No b3: 7.4 / 14.2 / 20.5 / 26.6 milhões. Quem sai na segunda saída custou
14.2 M, porque passou pela primeira no caminho.

### E[MACs]
O custo médio por imagem na cascata: soma dos MACs de cada saída multiplicados pela fração
de imagens que sai ali. Se metade sai com 7.4 M e metade com 26.6 M, o custo médio é 17 M.
É a métrica honesta, porque "fração resolvida na borda" sozinha ignora quanto cada saída
custou.

### thop
Biblioteca que conta os MACs de um módulo PyTorch. É com ela que o projeto mede o custo de
cada bloco e decide onde colocar as saídas.

### Latência
Tempo entre mandar a imagem e receber a resposta. Diferente de custo computacional, porque
inclui transporte e espera.

### RTT
Round trip time: o tempo que um pacote leva para ir e voltar na rede, sem contar o tamanho
do conteúdo. É o "pedágio fixo" de cada troca de mensagem.

### Payload
O conteúdo enviado pela rede em uma requisição. Aqui pode ser a imagem (3.0 KB crua, 2.2 KB
em PNG) ou um mapa de features intermediário (de 2.5 a 24 KB em float32, dependendo do
ponto de corte).

### float16
Guardar cada número em 16 bits em vez de 32. Ocupa metade do espaço, com menos casas
decimais. Usado para encolher o mapa de features enviado pela rede.

### pp
Ponto percentual. A diferença entre 91% e 93% é de 2 pp. Evita a confusão de dizer "2% de
diferença", que poderia significar 2% de 91.

## Estatística

### Acurácia
Porcentagem de respostas certas. Com 10 mil imagens de teste, 91.5% são 9150 acertos.

### IC95%
Intervalo de confiança de 95%: a faixa em que o valor real deve estar. Aqui ele aparece
nas comparações entre configurações: com as 5 diferenças pareadas de um par de
configurações, a distribuição t dá a faixa da diferença média. Se a faixa não contém zero,
a diferença conta como confirmada.

### p valor
A chance de ver uma diferença tão grande quanto a observada por puro acaso, supondo que na
verdade não exista diferença nenhuma. Pequeno quer dizer difícil de explicar por sorte. O
corte usual é 0.05.

### McNemar
Teste para comparar dois classificadores que responderam exatamente as mesmas perguntas.
Ignora as imagens em que os dois acertaram ou os dois erraram e olha só as discordâncias,
como contar caras e coroas numa moeda. Ele mede o acaso das imagens de teste, mas não o do
treino, por isso aqui só compara modos de calibração de um mesmo modelo.

### Réplica e desvio padrão
Réplica é o mesmo experimento treinado de novo com outra semente. Aqui cada configuração tem
5 treinos, cada um com sua divisão de dados. A média diz o resultado típico e o desvio
padrão diz quanto ele oscila entre treinos. Com só 5 valores, o próprio desvio é incerto: o
verdadeiro pode ficar entre 0.6 e 2.9 vezes o estimado.

### Significância
Uma diferença é chamada de significativa quando o p valor fica abaixo de 0.05. Não quer
dizer que a diferença seja grande ou importante, só que ela dificilmente é fruto do acaso.

### Correção de Holm
Quando se fazem muitos testes, alguns dão significativo por acaso: com 100 testes a 5%,
cerca de 5 dariam significativo mesmo sem efeito real. A correção de Holm ajusta os p valores para que a chance de pelo
menos um falso positivo no conjunto todo fique em 5%.

### Data snooping
Escolher a configuração olhando o resultado do teste, e depois reportar esse mesmo
resultado. Como sempre existe alguma combinação boa por sorte, o número sai inflado. A
correção é escolher na validação e olhar o teste uma vez só.

## Ferramentas

### PyTorch e torchvision
PyTorch é a biblioteca em que a rede é escrita e treinada. Torchvision é o pacote
complementar com modelos prontos (a MobileNetV2 e seus pesos) e datasets (o CIFAR10).

### uv
Gerenciador de pacotes e ambientes Python usado no projeto. `uv sync` instala as
dependências travadas no `uv.lock`, e `uv run` executa dentro desse ambiente.

### Flask
Biblioteca para escrever servidores HTTP pequenos em Python. É com ela que a borda e a
nuvem rodam como dois serviços separados.

### serving
A pasta [src/serving/](../src/serving/) com o protótipo da divisão borda/nuvem:

* `edge.py`: recebe a imagem, roda as primeiras saídas e decide entre responder ou repassar.
* `cloud.py`: recebe o mapa de features e termina a inferência (ou roda a rede inteira, no
  caso de comparação).
* `client.py`: manda as imagens, varre limiares e pontos de corte e grava os tempos em CSV.
* `common.py`: o que os dois servidores compartilham, como carregar o checkpoint e
  empacotar tensores.

Rodando os dois servidores no mesmo computador, o atraso de rede pode ser imitado com uma
pausa proporcional ao tamanho do payload (`LINK_MBPS`, `LINK_RTT_MS`), o que serve para
estudar o comportamento, não para afirmar tempos reais. A medição da seção 7 do README usa
máquinas separadas: a borda no computador local, só com CPU, e a nuvem em instâncias da AWS
em várias regiões, com o enlace simulado desligado.

### npz
Formato do NumPy para guardar vários arrays em um arquivo só. Em `results/preds/` ficam as
predições salvas assim, para o `stats.py` comparar modelos sem rodar tudo de novo.
