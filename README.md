# Pipeline Preditivo de Risco de Crédito — Machine Learning

## 1. Visão Geral do Projeto
Este projeto desenvolve um pipeline preditivo de Machine Learning para classificação de risco de crédito no setor financeiro. O objetivo é prever a inadimplência de clientes (`loan_status = 1`) para mitigar perdas financeiras decorrentes da concessão inadequada de empréstimos.

### Problema de Negócio e Assimetria do Erro
No cenário bancário, os erros de classificação possuem custos financeiros distintos:
* **Falso Positivo (FP):** Classificar um bom pagador como risco de calote. Custa ao banco a perda da margem de lucro sobre os juros do empréstimo.
* **Falso Negativo (FN):** Classificar um cliente inadimplente como seguro e conceder o crédito. Custa ao banco a **perda integral do capital emprestado (100% do principal)**.

Portanto, o foco estratégico do projeto é a **minimização dos Falsos Negativos (maximização do Recall na classe 1)**.

---

## 2. Dicionário de Dados

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `person_age` | Inteiro | Idade do cliente |
| `person_income` | Inteiro | Renda anual do cliente |
| `person_home_ownership` | Categórico | Tipo de moradia (RENT, OWN, MORTGAGE, OTHER) |
| `person_emp_length` | Contínuo | Tempo de emprego em anos |
| `loan_intent` | Categórico | Intenção/motivo do empréstimo |
| `loan_grade` | Categórico | Graduação do risco de crédito atribuída |
| `loan_amnt` | Inteiro | Valor total do empréstimo solicitado |
| `loan_int_rate` | Contínuo | Taxa de juros do empréstimo |
| `loan_status` | Binário | **Variável Alvo:** (0 = Adimplente, 1 = Inadimplente) |
| `loan_percent_income` | Contínuo | Percentual da renda comprometido pelo empréstimo |
| `cb_person_default_on_file` | Categórico | Histórico prévio de calote (Y/N) |
| `cb_person_cred_hist_length` | Inteiro | Anos de histórico de crédito |
| **`comprometimento_renda`** | **Contínuo** | **Variável Calculada:** `(loan_amnt / person_income) * 100` |

---

## 3. Principais Insights da Análise Exploratória (EDA)
1. **Desbalanceamento de Classes:** Apenas ~22% dos registros representavam inadimplência (`loan_status = 1`), justificando a aplicação de reamostragem sintética (SMOTE) no treino.
2. **Qualidade dos Dados:** Identificados valores nulos em `person_emp_length` e `loan_int_rate`, além de *outliers* irrealistas na idade (`person_age >= 100`) e tempo de emprego (`person_emp_length >= 60`).
3. **Colinearidade:** Alta correlação de Pearson (0.86) entre a idade do cliente e a extensão do seu histórico de crédito.

---

## 4. Metodologia e Preparação de Dados (Data Prep)
* **Duplicados e Outliers:** Remoção de registros duplicados e eliminação de 7 registros com dados atípicos.
* **Imputação de Nulos:** Preenchimento de dados ausentes via **Mediana**, devido à presença de assimetria e valores atípicos nas distribuições.
* **Encoding:** Aplicação de One-Hot Encoding (`pd.get_dummies(drop_first=True)`) gerando 23 variáveis preditoras.
* **Divisão dos Dados:** Split de 80% treino e 20% teste com `stratify=y` para preservação da proporção das classes.
* **Balanceamento Seguro (Anti-Leakage):** Aplicação do **SMOTE** estritamente sobre os dados de treino (`X_train`, `y_train`), equalizando as classes em 20.257 registros cada.
* **Escalonamento Seguro:** Aplicação do `StandardScaler` treinado (`fit_transform`) apenas na base de treino do modelo KNN. O modelo de Árvore de Decisão foi mantido em escala original.

---

## 5. Resultados e Comparação de Modelos

| Algoritmo | Hiperparâmetro | Recall (Treino) | Recall (Teste) | Falsos Negativos (Teste) |
| :--- | :--- | :--- | :--- | :--- |
| **KNN** | $K = 7$ | 89.94% | 68.05% | 453 |
| **Árvore de Decisão** | `max_depth = 5` | 83.58% | **74.75%** | **358** |

---

## 6. Resumo Executivo e Veredito Final
O modelo recomendado para implementação em produção é a **Árvore de Decisão com `max_depth = 5`**.

**Justificativa de Negócio:**
1. **Proteção Financeira:** Reduziu os Falsos Negativos para **358** (contra 453 do KNN), evitando que 95 clientes inadimplentes adicionais recebessem crédito indevido.
2. **Transparência e Auditabilidade:** Produz regras lógicas claras para o setor de compliance e risco do banco.
3. **Simplicidade Operacional:** Dispensa etapas de escalonamento numérico em tempo de inferência.
