import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# Importações para o design acadêmico
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# --- Configurações Tipográficas e Acadêmicas do Matplotlib ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,            
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "pdf.fonttype": 42,          
    "ps.fonttype": 42
})

# Mapeamento de classes
MAPEAMENTO_CLASSES = {
    # Classe 1
    "admirar": "Classe 1", "adorar": "Classe 1", "amar": "Classe 1", "desejar": "Classe 1", 
    "detestar": "Classe 1", "estranhar": "Classe 1", "invejar": "Classe 1", "odiar": "Classe 1", 
    "almejar": "Classe 1", "apreciar": "Classe 1",
    # Classe 2
    "aborrecer": "Classe 2", "afligir": "Classe 2", "abalar": "Classe 2", "chatear": "Classe 2", 
    "comover": "Classe 2", "decepcionar": "Classe 2", "encantar": "Classe 2", "escandalizar": "Classe 2", 
    "apaixonar": "Classe 2", "constranger": "Classe 2",
    # Classe 3
    "acalmar": "Classe 3", "apaziguar": "Classe 3", "convencer": "Classe 3", "enganar": "Classe 3", 
    "humilhar": "Classe 3", "ludibriar": "Classe 3", "pacificar": "Classe 3", "derrotar": "Classe 3", 
    "ridicularizar": "Classe 3", "tranquilizar": "Classe 3",
    # Classe 4
    "acovardar": "Classe 4", "agradar": "Classe 4", "alarmar": "Classe 4", "alegrar": "Classe 4", 
    "animar": "Classe 4", "apavorar": "Classe 4", "assombrar": "Classe 4", "assustar": "Classe 4", 
    "aterrorizar": "Classe 4", "atormentar": "Classe 4",
    # Classe Controle
    "dar": "Classe controle", "atribuir": "Classe controle", "devolver": "Classe controle", "distribuir": "Classe controle", 
    "emprestar": "Classe controle", "entregar": "Classe controle", "fornecer": "Classe controle", "oferecer": "Classe controle", 
    "pagar": "Classe controle", "transferir": "Classe controle"
}

# Paleta de cores (convertida e organizada para consistência no Seaborn)
PALETA_CORES = {
    "Classe 1": "#1f77b4",        # Azul
    "Classe 2": "#ff7f0e",        # Laranja
    "Classe 3": "#2ca02c",        # Verde
    "Classe 4": "#d62728",        # Vermelho
    "Classe controle": "#000000"   # Preto
}

def main():
    pasta_src = Path(__file__).resolve().parent
    raiz_projeto = pasta_src.parent
    pasta_outputs = raiz_projeto / 'outputs'
    pasta_outputs.mkdir(parents=True, exist_ok=True) # Garante que a pasta existe
    
    json_path = raiz_projeto / 'data' / 'corpus_processado.json'
    features_path = pasta_outputs / 'features_xlmr_mean_pooling.npy'
    
    # 1. Carregar os dados
    print("Carregando embeddings e metadados...")
    X = np.load(features_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        dados_json = json.load(f)
        
    verbos = [item['verbo_alvo'].strip().lower() for item in dados_json]
    
    # 2. Agrupar os embeddings calculando a média para cada verbo único
    print("Agrupando embeddings por verbo...")
    verbos_unicos = sorted(list(set(verbos)))
    embeddings_medios = []
    classes_verbos = []
    
    for v in verbos_unicos:
        indices = [i for i, verbo in enumerate(verbos) if verbo == v]
        vetor_medio = np.mean(X[indices], axis=0)
        embeddings_medios.append(vetor_medio)
        classes_verbos.append(MAPEAMENTO_CLASSES.get(v, "Desconhecido"))
        
    X_medios = np.vstack(embeddings_medios)
    
    # 3. Aplicar t-SNE
    print("Executando redução de dimensionalidade com t-SNE...")
    tsne = TSNE(
        n_components=2, 
        perplexity=10, 
        random_state=42, 
        init='pca', 
        max_iter=1000
    )
    X_2d = tsne.fit_transform(X_medios)
    
    df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'Verbo': verbos_unicos,
        'Classe': classes_verbos
    })
    
    # 4. Configurar e Criar o Gráfico Estático (Padrão Publicação)
    print("Gerando visualização de qualidade científica com Matplotlib e Seaborn...")
    
    # Proporção ideal para páginas de artigos (geralmente largura entre 6 e 8 polegadas)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    
    # Fundo branco e sem linhas de grade chamativas (estilo limpo)
    ax.set_facecolor('white')
    
    # Plot dos pontos com Seaborn
    scatter = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='Classe',
        palette=PALETA_CORES,
        hue_order=["Classe 1", "Classe 2", "Classe 3", "Classe 4", "Classe controle"],
        s=45,                   
        edgecolor='white',      
        linewidth=0.7,
        alpha=0.95,
        ax=ax
    )
    
    # Rótulos dos eixos limpos
    ax.set_xlabel("Dimensão t-SNE 1", fontweight='regular')
    ax.set_ylabel("Dimensão t-SNE 2", fontweight='regular')
    
    # Tufte-style: Remove bordas de cima e da direita (despine)
    sns.despine(ax=ax, top=True, right=True)
    
    # Ajuste fino da legenda acadêmica (fora da área dos dados para não sobrepor nada)
    ax.legend(
        title="Classes de Verbos",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False
    )
    
    # 5. Aplicar o adjustText para repelir os rótulos de forma inteligente
    print("Ajustando posicionamento dos textos com o algoritmo adjustText...")
    texts = []
    for i in range(len(df)):
        # Criamos o objeto de texto no gráfico
        t = ax.text(
            df['x'].iloc[i], 
            df['y'].iloc[i], 
            df['Verbo'].iloc[i],
            fontsize=8,
            fontfamily='serif'
        )
        texts.append(t)
    
    # Executa a repulsão física dos textos automaticamente
    adjust_text(
        texts,
        x=df['x'].values,
        y=df['y'].values,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        expand=(1.2, 1.4), 
        force_text=(0.1, 0.25)
    )
    
    # Ajuste de margens do layout
    plt.tight_layout()
    
    # 6. Exportação de Formatos Vetoriais (Sem perda de resolução)
    caminho_pdf = pasta_outputs / 'visualizacao_embeddings_tsne_controle.pdf'
    caminho_svg = pasta_outputs / 'visualizacao_embeddings_tsne_controle.svg'
    caminho_png = pasta_outputs / 'visualizacao_embeddings_tsne_controle.png' # Caso precise de uma prévia rápida
    
    # Salvando em múltiplos formatos
    plt.savefig(caminho_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(caminho_svg, format='svg', bbox_inches='tight')
    plt.savefig(caminho_png, format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"\nSucesso! Gráficos prontos para publicação salvos em:")
    print(f" -> PDF: {caminho_pdf}")
    print(f" -> SVG: {caminho_svg}")
    print(f" -> PNG: {caminho_png}")

if __name__ == "__main__":
    main()