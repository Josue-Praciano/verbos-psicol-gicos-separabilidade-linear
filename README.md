![image](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/images/PPGI-ufrj-logo.png)

# Trabalho final em Fundamentos de Ciência de Dados - PPGI/UFRJ

## __Professores:__

|     Sergio Serra	  | Jorge Zavaleta         |
|---------------------|------------------------|
|serra@pet-si.ufrrj.br|zavaleta@pet-si.ufrrj.br|


# Separabilidade Linear de Verbos Psicológicos no XLM-RoBERTa-base

Este esstudo contém o pipeline completo feito para investigar a separabilidade linear de verbos psicológicos do Português utilizando embeddings do modelo Transformer **XLM-RoBERTa-base** e classificadores lineares.

O objetivo do projeto é avaliar se representações vetoriais de modelos de linguagem capturam nuances sintático-semânticas profundas (como papéis temáticos) a ponto de distinguir classes de verbos psicológicos sem o auxílio de metadados.

**Josué David Praciano* [Programa de Pós-Graduação em Informática/UFRJ]

-----
## Arquivos disponibilizados
* Dataset bruto em txt e tratado em json
* Dataset pré-processado contendo as 4 classes de verbos psicológicos e a classe de controle
* Resultados da validação cruzada e do teste final obtidos via SVM Linear e Regressão Logística

-----
## Imagens disponibilizadas
* Grafo de proveniência do projeto
![image](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/images/pipeline_provenance.png)
* Gráfico 2d dos embeddinggs via t-SNE
![image](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/images/visualizacao_embeddings_tsne_controle.png)

-----
## Autoria:
* Josué David Praciano, Sergio Serra e Jorge Zavaleta
* Contato: josuepraciano@gmail.com
* Página: https://www.linkedin.com/in/josue-praciano/
 
-----
Artigo: Em andamento

## Uso de IA Generativa
Este projeto utilizou o modelo **Gemini 1.5 Pro** (Google) de forma complementar para suporte técnico e desenvolvimento, aplicado nas seguintes tarefas:

* **Documentação do Código:** Geração de comentários explicativos nas funções para facilitar a leitura e manutenção do código.
* **Depuração e Rastreabilidade:** Otimização de marcações e instruções de *print* ao longo do código para mapeamento rápido de falhas e rastreabilidade de erros.

> **Nota:** Toda a concepção conceitual da pesquisa, a fundamentação teórica de linguística computacional e a análise crítica dos resultados do modelo XLM-RoBERTa-base foram de autoria inteiramente humana.

  
