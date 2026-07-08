import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs"

try:
    le = joblib.load(DATA_DIR / 'label_encoder.joblib')
    modelo_lr = joblib.load(DATA_DIR / 'modelo_regressao_logistica.joblib')
    modelo_svm = joblib.load(DATA_DIR / 'modelo_svm_linear.joblib')
    classes_model = list(le.classes_)
    print("✨ Todos os arquivos foram carregados com sucesso localmente!\n")
except FileNotFoundError as e:
    print(f"❌ Erro: Não foi possível encontrar o arquivo.")
    print(f"Certifique-se de que a pasta 'output' existe em: {BASE_DIR}")
    print(f"E que o arquivo '{e.filename}' está dentro dela.")
    exit()


# 1. Definição dos Grupos verbais

grupo1 = ['abominar', 'admirar', 'adorar', 'amar', 'cobiçar', 'desejar', 'detestar', 'estranhar', 'invejar', 'odiar', 'aguentar', 'almejar', 'apreciar', 'contemplar', 'curtir', 'depreciar', 'desprezar', 'estimar', 'idolatrar', 'louvar']
grupo2 = ['aborrecer', 'afligir', 'abalar', 'chatear', 'comover', 'decepcionar', 'deprimir', 'encantar', 'escandalizar', 'horrorizar', 'acanhar', 'afetar', 'agitar', 'desinteressar', 'alucinar', 'desgostar', 'apaixonar', 'deslumbrar', 'desanimar', 'emocionar', 'enfurecer']
grupo3 = ['quebrar', 'chutar', 'cortar', 'esmagar', 'rasgar', 'perfurar', 'bater', 'amassar', 'derrubar', 'arranhar', 'esmurrar', 'triturar', 'furar', 'trincar', 'estraçalhar', 'moer', 'ralar', 'partir', 'despedaçar', 'estilhaçar']

def identificar_grupo(verbo):
    v = verbo.lower().strip()
    if v in grupo1: return 'Grupo 1 (Exp-Suj)'
    if v in grupo2: return 'Grupo 2 (EXP-Obg)'
    if v in grupo3: return 'Grupo 3 (Controle)'
    return 'Outros'

# Base de dados base
df_classes = pd.DataFrame({'Verbo': classes_model})
df_classes['Grupo'] = df_classes['Verbo'].apply(identificar_grupo)
valid_idx = df_classes['Grupo'] != 'Outros'

# DataFrames para os plots
df_lr = df_classes[valid_idx].copy()
df_svm = df_classes[valid_idx].copy()

# Paleta de cores
paleta_cores = {
    'Grupo 1 (Exp-Suj)': '#2ecc71',
    'Grupo 2 (EXP-Obg)': '#FF0000',
    'Grupo 3 (Controle)': '#3498db'
}

# 2. Processamento - Regressão Logística

coefs_lr = modelo_lr.coef_[valid_idx.values]
pca_lr = PCA(n_components=2, random_state=42)
coords_lr = pca_lr.fit_transform(coefs_lr)

df_lr['PCA 1'] = coords_lr[:, 0]
df_lr['PCA 2'] = coords_lr[:, 1]


# 3. Processamento - SVM (Reconstrução OVO)

n_classes = len(classes_model)
n_features = modelo_svm.n_features_in_
ovo_coef = modelo_svm.coef_.toarray() if hasattr(modelo_svm.coef_, "toarray") else modelo_svm.coef_

svm_coefs_per_class = np.zeros((n_classes, n_features))
k = 0
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        svm_coefs_per_class[i] += ovo_coef[k]
        svm_coefs_per_class[j] -= ovo_coef[k]
        k += 1

coefs_svm_filtrados = svm_coefs_per_class[valid_idx.values]
pca_svm = PCA(n_components=2, random_state=42)
coords_svm = pca_svm.fit_transform(coefs_svm_filtrados)

df_svm['PCA 1'] = coords_svm[:, 0]
df_svm['PCA 2'] = coords_svm[:, 1]


# 4. Gráficos Lado a Lado (Subplots)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'Regressão Logística (OVR)', 
        'SVM Linear (Hiperplanos OVO Reconstruídos)'
    ),
    horizontal_spacing=0.08
)

# Adiciona os pontos de cada grupo no subplot correspondente
for grupo, cor in paleta_cores.items():
    # --- Regressão Logística (Coluna 1) ---
    df_g_lr = df_lr[df_lr['Grupo'] == grupo]
    fig.add_trace(
        go.Scatter(
            x=df_g_lr['PCA 1'], y=df_g_lr['PCA 2'],
            mode='markers',
            name=grupo,
            marker=dict(size=12, color=cor, opacity=0.85, line=dict(width=1, color='black')),
            text=df_g_lr['Verbo'],
            hovertemplate="<b>%{text}</b><br>Grupo: " + grupo + "<br>PCA 1: %{x:.2f}<br>PCA 2: %{y:.2f}<extra></extra>",
            legendgroup=grupo, 
            showlegend=True
        ),
        row=1, col=1
    )
    
    # --- SVM (Coluna 2) ---
    df_g_svm = df_svm[df_svm['Grupo'] == grupo]
    fig.add_trace(
        go.Scatter(
            x=df_g_svm['PCA 1'], y=df_g_svm['PCA 2'],
            mode='markers',
            name=grupo,
            marker=dict(size=12, color=cor, opacity=0.85, line=dict(width=1, color='black')),
            text=df_g_svm['Verbo'],
            hovertemplate="<b>%{text}</b><br>Grupo: " + grupo + "<br>PCA 1: %{x:.2f}<br>PCA 2: %{y:.2f}<extra></extra>",
            legendgroup=grupo,
            showlegend=False
        ),
        row=1, col=2
    )


# 5. Ajustes de Layout e Estilização

fig.update_layout(
    title_text='<b>Comparativo de Separabilidade Linear dos Verbos (PCA)</b>',
    template='simple_white',
    width=1400,
    height=700,
    title_font=dict(size=18),
    legend=dict(title_text='Grupos Semânticos', orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
)


fig.update_xaxes(title_text=f'CP 1 ({pca_lr.explained_variance_ratio_[0]*100:.1f}%)', showgrid=True, gridcolor='LightGray', gridwidth=0.5, row=1, col=1)
fig.update_yaxes(title_text=f'CP 2 ({pca_lr.explained_variance_ratio_[1]*100:.1f}%)', showgrid=True, gridcolor='LightGray', gridwidth=0.5, row=1, col=1)

fig.update_xaxes(title_text=f'CP 1 ({pca_svm.explained_variance_ratio_[0]*100:.1f}%)', showgrid=True, gridcolor='LightGray', gridwidth=0.5, row=1, col=2)
fig.update_yaxes(title_text=f'CP 2 ({pca_svm.explained_variance_ratio_[1]*100:.1f}%)', showgrid=True, gridcolor='LightGray', gridwidth=0.5, row=1, col=2)


fig.show()