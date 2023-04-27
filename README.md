
---
Material referente ao modelo gerador de layouts de documento.
---



**1) Introdução**
<p>
O gerador de layout foi construído tendo como referência o Artigo “LayoutTransformer: Layout Generation and Completion with Self-attention”, disponível em https://assets.amazon.science/4a/32/8f0fc55145889db4fd2feb12f99c/layout-transformer-layout-generation-and-completion-with-self-attention.pdf, cujo download do código pode ser feito em https://github.com/kampta/DeepLayout

É composto de dois módulos: Treinamento e Inferência, que estão nas pastas "code" e "predict", respectivamente.

Na pasta "auxiliary" estão todos os códigos-fonte python utilizados ao longo do projeto para manipulação de datasets, como conversões de arquivos csv ou xml para json e análises exploratórias.
<p>



---

**2) Ambiente**

O container está disponível no docker hub em flamorim/01layout e replicado nas GPUs V e P do gpucluster do ICA.

<p>
Os arquivos de trabalho estão em \\gpucluster\share_alpha_2\NLP\generator_layout

<p>

---

**3) Módulo Treinamento - Pasta Code**

 A pasta 'code' contém os arquivos de código-fonte Python para o treinamento e criação do modelo Pytorch para geração do layouts.<p>
São eles: 

**3.1) main.py**

O arquivo main.py contém o código principal, ponto de início do modelo, e contém o tratamento dos argumentos de entrada.
  
A plataforma [Weights & Biases](http://wandb.ai/home) para monitoramento e visualização de experimentos de deep learning foi integrada e seu acionamento é  feito através da presença do argumento --exp contendo o nome do experimento em andamento.<p>
Os demais argumentos estão discriminados abaixo por funcionalidade.

<p> Argumentos relativos aos arquivos de entrada e saída:
<p>


<style>
  table {
    border-collapse: collapse;
    border: 1px solid #000;
  }
  th, td {
    border: 1px solid #000;
    padding: 8px;
    text-align: left;
  }
</style>

<table  border="1">
  <thead>
    <tr>
      <th colspan="3"><center>manipulação de arquivos</th>
    </tr>
  </thead>
  <tr>
    <th width="150"> Argumento</th>
    <th width="400">Descrição</th>
    <th width="200"><center>Exemplo</th>
  </tr>

  <tr>
    <td width="150"><center> --train_json</td>
    <td width="400">arquivo json para o treinamento do modelo</td>
    <td width="200"><center>../dataset/train.json</td>
  </tr>

  <tr>
    <td width="150"><center> --val_json</td>
    <td width="400">arquivo json para validação do modelo</td>
    <td width="200"><center>../dataset/valid.json</td>
  </tr>

  <tr>
    <td width="150"><center> --ckpt_dir</td>
    <td width="400">local existente onde os checkpoints e modelo final criados serão armazenados</td>
    <td width="200"><center>../checkpoint</td>
  </tr>

  <tr>
    <td width="150"><center> --samples_dir</td>
    <td width="400">quando especificado, ao final de cada época de treinamento, é o local onde amostras das 12 primeiras imagens do arquivo de validação serão geradas e armazenadas no formato png</td>
    <td width="200"><center>../samples</td>
  </tr>

</table>

Argumentos relativos a configuração do modelo que, quando aplicáveis também no módulo de inferência, precisam ter o mesmo valor:

<table  border="1">
  <thead>
    <tr>
      <th colspan="3"><center>configuração do modelo</th>
    </tr>
  </thead>
  <tr>
    <th width="150"><center> Argumento</th>
    <th width="400">Descrição</th>
    <th width="200"><center>Exemplo</th>
  </tr>

  <tr>
    <td width="150"><center> --epochs</td>
    <td width="400">número de épocas</td>
    <td width="200"><center>40</td>
  </tr>

  <tr>
    <td width="150"><center> --batchsize</td>
    <td width="400">numero de amostras em um batch</td>
    <td width="200"><center>64</td>
  </tr>

  <tr>
    <td width="150"><center> --nlayer</td>
    <td width="400">número de blocos transformers</td>
    <td width="200"><center>8</td>
  </tr>

  <tr>
    <td width="150"><center> --nhead</td>
    <td width="400">número de cabeças por bloco transformer</td>
    <td width="200"><center>8</td>
  </tr>

  <tr>
    <td width="150"><center> --precision</td>
    <td width="400">discretização das coordenadas, sendo 2**precision</td>
    <td width="200"><center>8</td>
  </tr>

  <tr>
    <td width="150"><center> --max_length </td>
    <td width="400">tamanho máximo da sequência de entrada do modelo, sabendo que cada bbx gera 5 entradas (x1,y1,x2,y2 e classe). Neste número estão incluídos o BOS e EOS</td>
    <td width="200"><center>200</td>
  </tr>

  <tr>
    <td width="150"><center> --lr_decay </td>
    <td width="400">sua existência indica que a diminuição do learning rate deve ser aplicada</td>
    <td width="200"><center>'N/A'</td>
  </tr>


</table>

<p>

Argumentos relativos a geração de amostragens durante o treinamento, quando habilitado (não influenciam na criação e treinamento do modelo):


<table  border="1">
  <thead>
    <tr>
      <th colspan="3"><center>geração de amostragens</th>
    </tr>
  </thead>
  <tr>
    <th width="100"> Argumento</th>
    <th width="300">Descrição</th>
    <th width="100"><center>Exemplo</th>
  </tr>

  <tr>
    <td width="100"><center> --topk</td>
    <td width="300">A acurácia Top-K indica que qualquer uma das K respostas de maior probabilidade do modelo deve corresponder à resposta esperada.</td>
    <td width="100"><center>5</td>
  </tr>

  <tr>
    <td width="100"><center> --temp</td>
    <td width="300">Entre 0 e 1, onde valores mais altos tornam a saída mais flexível e valores mais baixos tornam a saída mais determinística.</td>
    <td width="100"><center>0.3</td>
  </tr>

</table>


Como referência, segue um exemplo:

```html
python main.py --exp layout --train_json ../dataset/train.json --val_json ../dataset/valid.json --epochs 40 --batch_size 64 --ckpt_dir ../checkpoint --n_layer 8 --precision 8 --n_head 8 --topk 1 --tempe 0.3 --max_length 200 --samples_dir ../samples --lr_decay
```





**3.2) dataset.py**
<p> O arquivo dataset.py é responsável pela manipulação dos datasets de treino e validação, que são lidos e transformados em classes do tipo JSONLayout. Também cria os objetos que representam os marcadores de início e fim de uma sequência de entrada, o BOS e o EOS, respectivamente.

Os datasets de entrada devem possuir o formato semelhante ao dataset PUBLAYNET, com objetos "images" e "annotations".
<p>
Os objetos "images" contém informações dos arquivos imagens, sendo a chave id a mais importante, pois é através dela que as anotações são associadas a uma mesma imagem.<p>

A tabela abaixo ilustra as chaves do objeto images:

<table  border="1">
  <thead>
    <tr>
      <th colspan="3"><center>Objeto "images"</th>
    </tr>
  </thead>
  <tr>
    <th width="100"> Chave</th>
    <th width="300">Descrição</th>
    <th width="100"><center>Exemplo</th>
  </tr>

  <tr>
    <td width="100"><center> id</td>
    <td width="300">identificador do arquivo imagem, sendo um número inteiro.</td>
    <td width="100"><center>1</td>
  </tr>

  <tr>
    <td width="100"><center> file_name</td>
    <td width="300">nome do arquivo imagem, usado para auxiliar na solução de problemas
    <td width="100"><center>output.xml</td>
  </tr>
  <tr>
    <td width="100"><center> height</td>
    <td width="300">altura em pixel da imagem
    <p>**reservado para uso futuro**</td>
    <td width="100"><center>800</td>

  </tr>
  <tr>
    <td width="50"><center> width</td>
    <td width="300">largura em pixel da imagem
    <p>**reservado para uso futuro**</td>
    <td width="100"><center>600</td>
  </tr>
</table>

Os objetos "annotations" contém anotações de imagens, sendo que a chave image_id indica a qual imagem ela pertence.<p>
De acordo com o último treinamento realizado, a classe de um bonding box (chave category-id) pode ser:<p>

<ul>
  <li>1 para "image"</li>
  <li>2 para "text"</li>
  <li>3 para "table"</li>
  <li>4 para "equation</li>
</ul>
Essa correlação "category-id" para uma palavra está feita estaticamente no código.

A tabela abaixo ilustra as chaves do objeto annotation:
<table border="1">
  <thead>
    <tr>
      <th colspan="3"><center>Objeto "annotations"</th>
    </tr>
  </thead>
  <tr>
    <th width="50"> Chave</th>
    <th width="300">Descrição</th>
    <th width="100"><center>Exemplo</th>
  </tr>

  <tr>
    <td width="50"><center> image_id</td>
    <td width="300">identificador da imagem a qual a anotação faz referência.</td>
    <td width="300"><center>1</td>
  </tr>

  <tr>
    <td width="50"><center> bbox</td>
    <td width="300">coordenadas das extremidades do bounding box (x1,y1,x2,y2) 
    </td>
    <td width="300"><center>[0.20392157137393951, 0.7529411911964417, 0.8039215803146362, 0.8039215803146362]</td>
  </tr>

  <tr>
    <td width="50"><center> category_id</td>
    <td width="300">número inteiro que representa a classe do bounding box, começando por 1.</td>
    <td width="300"><center>1</td>
  </tr>

  <tr>
    <td width="50"><center> module</td>
    <td width="300">indentificador para área da imagem, podendo ser header, footer ou bodymodule.
    <p>**reservado para uso futuro**</td>
    <td width="300"><center>99</td>

  </tr>
</table>


**3.3) trainer.py**
<p> O arquivo trainer.py é responsável pelo treinamento e criação do modelo Pytorch. É uma estrutura básica em loop de uma rede neural regular com mecanismos para:


<ul>
  <li>loss: cross_entropy</li>
  <li>otimizador: escolhido AdamW com betas = (0.9, 0.99) </li>
  <li>learning rate: escolhido MultiStepLR e, ciente que a melhor época é próxima a 12, foram definidas as épocas no intervalo [8,10,12,14,16,18,20,22] onde a taxa de aprendizado é multiplicada pelo fator 0.1
</li>
  <li>checkpoint: em cada época que o erro de teste tenha obtido um valor menor, um novo modelo é salvo no diretório 'checkpoint'</li>
</ul>

O mecanismo early stop não foi implementado para que o modelo possa ser avaliado até o número de épocas definido no argumento de entrada.

O arquivo criado referente ao modelo Pytorch tem como nome a seguinte estrutura:

<pre>   modPARALLEL-epoch{epoch}-{rand:04d}.pth</pre><p>
onde *epoch* é o número da época e *rand* é um número randômico entre 1 e 9999.

<p>

**3.4) model.py**
<p>
O arquivo model.py é responsável pela criação do modelo GPT, contendo uma sequência de blocos transformers, cada um com 1-hidden-layer MLP block e um self-attention block. O decodificador final é uma projeção linear em um classificador Softmax 'vanilla'.<p>
Não foi utilizado torch.nn.MultiheadAttention, mas uma implementação em detalhes do mecanismo de attention.


**3.5) util.py**
<p>
O arquivo util.py é responsável por funções auxiliares, como a função para geração de amostragens durante o treinamento e a definição das cores para cada classe no arquivo png gerado.

---
**4) Módulo Inferência - Pasta Predict**
<p>
A pasta 'predict' contém os arquivos de código-fonte Python do módulo de inferência que, através do modelo Pytorch criado pelo módulo de treinamento, gera novos layouts sintéticos semelhantes a layouts recebidos em sua entrada.<p>
Originalmente a inferência era executada no módulo de treinamento então, quando esta separação foi feita, a elaboração dos códigos-fonte de inferência utilizou a  mesma estrutura do módulo de treinamento.<p>
Os arquivos que a compõem são:


**4.1) mpmain.py**

O arquivo mpmain.py contém o código principal, ponto de início do modelo, e contém o tratamento dos argumentos de entrada.

A plataforma [Weights & Biases](http://wandb.ai/home) também foi integrada e sua utilização é feita quando da presença do argumento --exp contendo o nome do experimento em andamento.<p>
Os demais argumentos estão discriminados abaixo por funcionalidade.

Argumentos relativos a configuração do modelo, que precisam ter o mesmmo valor do módulo de treinamento:

<table  border="1">
  <thead>
    <tr>
      <th colspan="3"><center>configuração do modelo</th>
    </tr>
  </thead>
  <tr>
    <th width="150"> Argumento</th>
    <th width="300">Descrição</th>
    <th width="100"><center>Exemplo</th>
  </tr>

  <tr>
    <td width="150"><center> --nlayer</td>
    <td width="300">número de blocos transformers</td>
    <td width="100"><center>8</td>
  </tr>

  <tr>
    <td width="150"><center> --nhead</td>
    <td width="300">número de cabeças por bloco transformer</td>
    <td width="100"><center>8</td>
  </tr>

  <tr>
    <td width="150"><center> --precision</td>
    <td width="300">discretização das coordenadas, sendo 2**precision</td>
    <td width="100"><center>8</td>
  </tr>

  <tr>
    <td width="150"><center> --max_length </td>
    <td width="300">tamanho máximo da sequência de entrada do modelo, sabendo que cada bbx gera 5 entradas (x1,y1,x2,y2 e classe). Neste número estão incluídos o BOS e EOS</td>
    <td width="100"><center>200</td>
  </tr>

</table>

<p> Argumentos relativos a inferência:
<p>

<table  border="1">
  <thead>
    <tr>
      <th colspan="3"><center>módulo inferência</th>
    </tr>
  </thead>
  <tr>
    <th width="150"> Argumento</th>
    <th width="300">Descrição</th>
    <th width="100"><center>Exemplo</th>
  </tr>

  <tr>
    <td width="150"><center> --predict_json</td>
    <td width="300">arquivo json para entrada do modelo de inferência</td>
    <td width="100"><center>../dataset/test.json</td>
  </tr>

  <tr>
    <td width="150"><center> --samples_dir</td>
    <td width="300">local onde são armazenados os arquivos png gerados quando a ferramente WandB está ativa</td></td>
    <td width="100"><center>../dataset/samples</td>
  </tr>

  <tr>
    <td width="150"><center> --amostras</td>
    <td width="300">quantidade de amostragens geradas por cada imagem de entrada</td>
    <td width="100"><center> 1 </td>
  </tr>

  <tr>
    <td width="150"><center> --topk</td>
    <td width="300">A acurácia Top-K indica que qualquer uma das K respostas de maior probabilidade do modelo deve corresponder à resposta esperada.</td>
    <td width="100"><center>5</td>
  </tr>

  <tr>
    <td width="150"><center> --temp</td>
    <td width="300">Entre 0 e 1, onde valores mais altos tornam a saída mais flexível e valores mais baixos tornam a saída mais determinística.</td>
    <td width="100"><center>0.3</td>

</table>



Como referência, segue um exemplo:


```html
python mpmain.py --exp layout --predict_json ../dataset/test.json --n_layer 8 --amostras 20 --topk 3 --temp 0.3
```




**4.2 mpdataset.py**

<p> O arquivo mpdataset.py é responsável pela leitura do arquivo com o(s) layouts de entrada e sua transformação em classe do tipo JSONLayout.
<p>
A principal diferença do mpdataset.py para o dataset.py é a inclusão de objetos "nome" e "id_imagem" para auxiliar na solução de problemas. No futuro, eses dois códigos python podem ser integrados como apenas um.<p>

Atualmente, o formato do arquivo de entrada do módulo predição é o mesmo do dataset de entrada do módulo de treinamento descrito no item 3.2

**4.3 mpredict.py**
<p>
O arquivo mpredict.py é responsável por carregar um modelo existente, fazer a inferência e salvar os layouts sintéticas gerados num arquivo no formato json.

Uma vez que esses layouts gerados são utilizados como entrada para o Gerador de Documentos Sintéticos, foi necessário que eles possuíssem o mesmo formato, que é uma chave "bodymodule" contendo uma lista de bounding box e classe.<p>
O bounding box é representado pelas coordenadas [x1,y1,x2,y2] e a classe pelas palavras "image", "text", "table" ou "equation".
<p>
Esta estrutura for definida na possibilidade de haver no futuro as chaves "header" e "footer", que representarão bbx presentes no cabeçalho e rodapé, respectivamente.<p>
Segue um exemplo de um layout contendo 3 bounding boxes da classe texto:

<pre>
{"bodymodule": [
[[0.18431372940540314, 0.15294118225574493, 0.8823529481887817, 0.24705882370471954], "text"],
[[0.18431372940540314, 0.2549019753932953, 0.8823529481887817, 0.3686274588108063], "text"],
[[0.18431372940540314, 0.3843137323856354, 0.8823529481887817, 0.5215686559677124], "text"]]}
</pre>


O local onde os arquivos json são armazenados é o diretório '../json' e seu nome segue a seguinte estrutura:

<pre>   data-hora-imageid-{contador}-top{k}.json,</pre><p>
onde *contador* é um número entre 1 e o argumento "--amostras" e *K* o argumento "--topk".

Os arquivos png gerados, quando o WandB está ativo, é o diretório "../samples"

**4.4 mpmodel.py**<p>
Este arquivo tem exatamente o mesmo conteúdo do arquivo model.py, sem absolutamente nenhuma alteração.<p>
Isso acontece pois ambos módulos treinamento e inferência usam o mesmo modelo.
Em uma versão futura, isto será unificado. 

**4.5 mputil.py**<p>

Este arquivo tem o mesmo conteúdo do arquivo util.py acrescentado de uma função, cujo o objetivo é copiar o arquivo json de entrada para um arquivo json de saída, utilizado quando o arquivo de entrada não possui nenhum bounding box.


---
**5) Atualizações importantes**
<p>

- [x] março/2023: coordenadas de entrada normalizadas e formato xyxy :tada:
- [x] março/2023: ajustes dos hiperparâmetros :tada:
- [x] abril/2023: inclusão de learning rate decay :tada:
  
  
  
---

<p>
fim
---
