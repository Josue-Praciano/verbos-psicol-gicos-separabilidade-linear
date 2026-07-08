# Investigando a Separabilidade Linear de Verbos Psicológicos no XLM-RoBERTa

Este repositório contém o pipeline completo feito para investigar da separabilidade linear de verbos psicológicos do Português utilizando embeddings do modelo Transformer **XLM-RoBERTa-base** e classificadores lineares.

O objetivo do projeto é avaliar se representações vetoriais de modelos de linguagem capturam nuances sintático-semânticas profundas (como papéis temáticos) a ponto de distinguir classes de verbos psicológicos sem o auxílio de metadados.

---

## Visão Geral e Fundamentação

### A Complexidade dos Verbos Psicológicos
Diferente dos verbos de ação tradicionais, os verbos psicológicos descrevem estados mentais e envolvem uma dinâmica complexas entre dois papéis temáticos principais: o **Experienciador** (quem sente) e o **Tema** (o que causa o sentimento).

Seguindo a tipologia clássica de Cançado (1997), dividimos as sentenças em:
* **Classe I (Sujeito Experienciador):** *"O aluno (Exp) gosta da aula (Tema)"*
* **Classe II (Objeto Experienciador):** *"A aula (Tema) agrada o aluno (Exp)"*
* **Controle:** Sentenças com verbos agentivos para calibração.

## Arquitetura do Pipeline

O pipeline de execução é dividido em três fases principais:

```text
  [ Extração Manual ] ──> [ XLM-RoBERTa + Pooling ] ──> [ Classificação & Plots ]
  (processed.json)         (Média Camadas 9-12)          (SVM, Regressão Logística)

### 1. Preparação do Corpus
* **Origem e Extração:** Extração manual realizada a partir do *Corpus do Português*, gerando o arquivo processado original `processed.json`.
* **Volume de Dados:** O dataset foi estruturado e balanceado com 300 sentenças no total, divididas igualmente entre Classe 1, Classe 2 e Controle (Agentivos), contendo 20 verbos e 15 sentenças por verbo ($20 \times 15 = 300$).
* **Estrutura do JSON:** As frases foram salvas em formato JSON estruturado com metadados para indexação e identificação visual facilitada:
    ```json
    {
        "id": 1,
        "texto": "Houve um exagero em abominar a política.",
        "verbo_alvo": "abominar",
        "posicao_verbo": 4
    }
    ```

### 2. Extração de Embeddings
Para codificar a semântica profunda das sentenças, foi utilizado o modelo de linguagem multilíngue `xlm-roberta-base` integrado à seguinte estratégia de engenharia de características:
* **Fusão de Camadas Ocultas:** Extração dos estados ocultos (*hidden states*) correspondentes à média aritmética das últimas 4 camadas internas (camadas 9 a 12).
* **Masked Mean Pooling:** Consolidação dos tokens de tamanho variável em um único vetor representativo por sentença. O algoritmo utiliza a máscara de atenção (`attention_mask`) para ignorar estritamente os tokens de preenchimento (padding usados no processamento em batches dinâmicos). Isso impede que elementos artificiais diluam ou distorçam a representação vetorial final.

### 3. Modelos de Classificação
A hipótese de separabilidade linear foi testada por meio de dois classificadores matematicamente distintos:

| Classificador | Função Base / Estratégia | Hiperparâmetros | Artefato Gerado | Relatório de Saída |
| :--- | :--- | :--- | :--- | :--- |
| **Support Vector Machine (SVM)** | Separação de margem máxima | Kernel Linear (Abordagem One-vs-One cumulativa) | `modelo_svm_linear.joblib` | `resultados_Support_Vector_Machine.txt` |
| **Logistic Regression (LR)** | Ajuste probabilístico | Regularização L2 (Abordagem One-vs-Rest) | `modelo_regressao_logistica.joblib` | `resultados_Logistic_Regression.txt` |

---
## 🛡️ Proveniência dos Dados e Reprodutibilidade Scientífica

Para garantir a total transparência, auditoria e reprodutibilidade do experimento, o projeto adota duas frentes fundamentais:

### 1. Modelo de Proveniência W3C PROV
Todo o ciclo de vida dos dados e artefatos gerados obedece às diretrizes de proveniência do padrão **W3C PROV**.:
```text
  [processed.json] -> (Fase 1: Preparação) -> [Matriz Binária] -> (Fase 2: XLM-R) -> [Modelos .joblib] -> (Fase 3: Classificação) -> [Relatórios .txt]

### 2. Reprodutibilidade
Validação Cruzada: Aplicação de 5-Fold Cross-Validation para mitigar vieses de partição e garantir robustez estatística frente à variância dos dados.

Seeds Fixas: Fixação de seeds fixas (random_state) em todos os algoritmos de partição de dados e inicialização de classificadores para garantir que execuções futuras gerem exatamente os mesmos hiperplanos e acurácias.

  