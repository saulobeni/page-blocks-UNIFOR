# 📊 Page Blocks Classification — UNIFOR

Projeto individual desenvolvido na disciplina **Inteligência Artificial Computacional**  
Centro de Ciências Tecnológicas — Universidade de Fortaleza (UNIFOR)

Este repositório contém a implementação manual de diversos classificadores de Machine Learning para realizar a classificação de blocos de páginas de documentos digitalizados, utilizando o dataset **Page Blocks** disponível no OpenML.

---

## 🎯 Objetivo do Projeto

O objetivo deste projeto é avaliar e comparar o desempenho de diferentes algoritmos de aprendizado supervisionado no problema de **classificação de layout de documentos**, identificando blocos como:

- Texto
- Linha horizontal
- Imagem
- Linha vertical
- Gráfico

Os algoritmos são avaliados usando:
- Validação Cruzada Estratificada
- Acurácia
- Precisão
- F1-Score
- Tempo de execução (treino e teste)

---

## 🗂️ Dataset Utilizado

**Nome:** Page Blocks  
**Fonte:** OpenML (ID: 30)  
**Instâncias:** 5.473  
**Atributos:** 10 atributos numéricos  
**Classes:** 5 (classificação nominal)

Cada instância representa um bloco de uma página extraído de documentos reais.  
As variáveis descrevem características como altura, largura, área, quantidade de pixels pretos, transições branco-preto, entre outras.

---

## 🧠 Algoritmos Implementados

Neste projeto, **nenhuma biblioteca pronta de Machine Learning foi utilizada** (como scikit-learn).  
Todos os algoritmos foram implementados manualmente em Python.

### 🚀 Classificadores avaliados:

- **KNN (k-Nearest Neighbors)**
  - Distância Euclidiana
  - Distância Manhattan

- **Perceptron Multiclasse**

- **MLP (Multi-Layer Perceptron)**  
  Rede neural com:
  - Camada de entrada
  - Camada oculta
  - Camada de saída

- **Naive Bayes**
  - Univariado (variância independente)
  - Multivariado (covariância entre atributos)

---

## 🧪 Processo Experimental

O fluxo de execução segue:

1. Leitura do arquivo `.arff`
2. Codificação das classes
3. Normalização dos dados (Z-score)
4. Divisão por validação cruzada estratificada
5. Treinamento dos modelos
6. Teste em cada fold
7. Cálculo das métricas
8. Geração da tabela final com média e desvio padrão
9. Exportação dos resultados para arquivo Excel

---

## 📁 Estrutura do Projeto

```bash
page-blocks-UNIFOR/
│
├── main.py                 # Arquivo principal de execução
├── reader.py               # Leitura do dataset ARFF
├── utils.py                # Normalização e codificação
├── metrics.py              # Implementação das métricas
├── cross_validation.py     # Validação cruzada estratificada
│
├── knn.py                  # Implementação do KNN
├── perceptron.py           # Implementação do Perceptron
├── mlp.py                  # Implementação da rede neural MLP
├── naive_bayes.py          # Implementação do Naive Bayes
│
├── dataset_30.arff         # Dataset Page Blocks
├── resultados_pageblocks.xlsx  # Tabela de resultados gerada
└── README.md               # Documentação do projeto
````

---

## ▶️ Como executar o projeto

### 🔹 1. Requisitos

Você precisa de Python instalado (recomendado: Python 3.8+)

---

### 🔹 2. Executar análise completa

No terminal, dentro da pasta do projeto:

```bash
python main.py --data dataset_30.arff
```

---

### 🔹 3. Parâmetros opcionais

Você pode personalizar os parâmetros assim:

```bash
python main.py --data dataset_30.arff --folds 5 --k 5 --hidden 32 --epochs_perceptron 50 --epochs_mlp 100
```

**Significado:**

| Parâmetro             | Descrição                         |
| --------------------- | --------------------------------- |
| `--folds`             | Nº de folds da validação cruzada  |
| `--k`                 | Valor de K no KNN                 |
| `--hidden`            | Neurônios na camada oculta da MLP |
| `--epochs_perceptron` | Épocas de treino do Perceptron    |
| `--epochs_mlp`        | Épocas da MLP                     |

---

## 📈 Resultados

Os resultados são apresentados em uma planilha Excel:

```
resultados_pageblocks.xlsx
```

Contendo:

* Média e desvio padrão da Acurácia
* Média e desvio padrão da Precisão
* Média e desvio padrão do F1-Score
* Tempo médio de treino
* Tempo médio de teste

---

## 🏆 Conclusão

O classificador **KNN com distância Manhattan** apresentou o melhor equilíbrio entre:

* Desempenho
* Estabilidade
* Eficiência computacional

Enquanto isso, redes neurais simples (MLP), apesar de mais complexas, não apresentaram bom desempenho nesse dataset específico.

---

## 📚 Referência do Dataset

Malerba, D., Esposito, F., & Semeraro, G. (1994).
Multistrategy Learning for Document Recognition. *Applied Artificial Intelligence.*

---

## 👨‍🎓 Autor

**Saulo Benício**
Universidade de Fortaleza — UNIFOR
Curso: Ciência da Computação
Disciplina: Inteligência Artificial Computacional

---

## ⚠️ Observações

✔️ Implementação manual dos classificadores
✔️ Sem uso de bibliotecas como scikit-learn ou pandas
✔️ Código didático e acadêmico

---