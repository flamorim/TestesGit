# readme.md
### Informações do repositório https://gitlab.com/ica-lab/big-oil-nlp/layout-generator com o gerador de layouts de documento.
---


1) Introdução / Pastas disponíveis:

* Pasta 01: arquivos para treinar o modelo....
* Pasta 02: arquivos para inferência....

container do docker hub

---

2) ambiente
pip list

---

3) **Pasta Code**

 A pasta contém os arquivos referentes para criação e treinamento do modelo pytorch para geração do layouts. São eles: 



* main.py
<p> Arquivo principal
<p>
Argumentos importantes:

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
      <th colspan="3"><center>dataset options</th>
    </tr>
  </thead>
  <tr>
    <th width="50"> Argumento</th>
    <th width="1000">Descrição</th>
    <th width="1000">Exemplo</th>
  </tr>

  <tr>
    <td width="50"><center> --train_json</td>
    <td width="1000">arquivo json para o treinamento do modelo</td>
    <td width="1000"><center>../dataset/train.json</td>
  </tr>

  <tr>
    <td width="50"><center> --val_json</td>
    <td width="1000">arquivo json para validação do modelo</td>
    <td width="1000"><center>../dataset/valid.json</td>
  </tr>

</table>




''''''   



----------------



* dataset.py
<p> Arquivo responsável pela manipulação dos datasets de treino e validação, que são lidos e transformados em classes do tipo JSONLayout. Também cria os objetos que representam os marcadores de início e fim de uma sequência de entrada, o BOS e o EOS, respectivamente.

Os datasets de entrada devem possuir o formato semelhante ao dataset PUBLAYNET, contendo os objetos "images" e "annotations" conforme abaixo:


<table  border="1">
  <thead>
    <tr>
      <th colspan="3">Objeto "images"</th>
    </tr>
  </thead>
  <tr>
    <th width="50"> Chave</th>
    <th width="1000">Descrição</th>
    <th width="1000">Exemplo</th>
  </tr>

  <tr>
    <td width="50"><center> id</td>
    <td width="1000">identificador do arquivo imagem, sendo um número inteiro.</td>
    <td width="1000"><center>1</td>
  </tr>

  <tr>
    <td width="50"><center> file_name</td>
    <td width="1000">nome do arquivo imagem.
    <td width="1000"><center>output.xml</td>
  </tr>
  <tr>
    <td width="50"><center> height</td>
    <td width="1000">altura em pixel da imagem
    <p>**VERIFICAR**</td>
    <td width="1000"><center>800</td>

  </tr>
  <tr>
    <td width="50"><center> width</td>
    <td width="1000">largura em pixel da imagem
    <p>**VERIFICAR**</td>
    <td width="1000"><center>600</td>
  </tr>
</table>


<table border="1">
  <thead>
    <tr>
      <th colspan="3">Objeto "annotations"</th>
    </tr>
  </thead>
  <tr>
    <th width="50"> Chave</th>
    <th width="1000">Descrição</th>
    <th width="1000">Exemplo</th>
  </tr>

  <tr>
    <td width="50"><center> image_id</td>
    <td width="1000">identificador da imagem a qual a anotação faz referência.</td>
    <td width="1000"><center>1</td>
  </tr>

  <tr>
    <td width="50"><center> bbox</td>
    <td width="1000">coordenadas das extremidades do bounding box (x1,y1,x2,y2 
    </td>
    <td width="1000"><center>[0.20392157137393951, 0.7529411911964417, 0.8039215803146362, 0.8039215803146362]</td>
  </tr>

  <tr>
    <td width="50"><center> category_id</td>
    <td width="1000">número inteiro que representa a classe do bounding box, começando por 1.</td>
    <td width="1000"><center>1</td>
  </tr>

  <tr>
    <td width="50"><center> module</td>
    <td width="1000">indentificador para área da imagem, podendo ser header, footer ou bodymodule.
    <p>**reservado para uso futuro**</td>
    <td width="1000"><center>99</td>

  </tr>
</table>


De acordo com o último treinameto feito, a classe de um bonding box pode ser:
* 1 para "image",
* 2 para "text",
* 3 para "table"}
* 4 para "equation"






, ambos no formado JSON.  
* main.py
* model.py
* trainer.py
* trainer.py.CosineAnnealingLR
* trainer.py.Plateau
* utils.py





 Para executar, fazer:

```html
python predict ....
```

```

```js
console.log('WriteMe.md');
```

---

To learn the basics of using Markdown, **[read this](http://daringfireball.net/projects/markdown/basics)**.
