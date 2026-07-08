import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score #
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, silhouette_score

def carregar_dados_e_labels(json_path, features_path):
    json_path = Path(json_path)
    features_path = Path(features_path)

    X = np.load(features_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        dados_json = json.load(f)
        
    assert len(dados_json) == X.shape[0], "O número de linhas do JSON difere da matriz de features!"
    
    labels_texto = [item['verbo_alvo'] for item in dados_json]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_texto)
    
    return X, y, label_encoder

def calcular_separabilidade(X, y):
    score_geral = silhouette_score(X, y, metric='cosine')
    print("\n" + "="*50)
    print(" CÁLCULO DE SEPARABILIDADE (ESPAÇO VETORIAL)")
    print("="*50)
    print(f"Silhouette Score Geral (Métrica de Cosseno): {score_geral:.4f}")
    print("="*50 + "\n")

def avaliar_e_salvar_modelo(modelo, X_train, X_test, y_train, y_test, label_encoder, nome_modelo, pasta_outputs):
    print(f"\n" + "="*50)
    print(f" AVALIANDO MODELO: {nome_modelo}")
    print("="*50)
    
    # VALIDAÇÃO CRUZADA (k=5)
    # Criado o gerador de folds garantindo a estratificação e a reproduzibilidade com random_state
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=26)
    
    scores_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')
    
    texto_cv = (
        f"Resultados da Validação Cruzada (5 Folds):\n"
        f"Acurácias por fold: {', '.join([f'{s:.2%}' for s in scores_cv])}\n"
        f"Média do CV: {scores_cv.mean():.2%} (+/- {scores_cv.std():.2%})\n"
    )
    print(texto_cv)
    
    #  Treino e Teste
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc_teste = accuracy_score(y_test, y_pred)
    relatorio_texto = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    
    print(f"Resultados no Conjunto de Teste Final")
    print(f"Acurácia no Teste: {acc_teste:.4%}")
    print(relatorio_texto)
    
    # Outputs 
    nome_limpo = nome_modelo.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    # Salva o binário do modelo (.joblib)
    joblib.dump(modelo, pasta_outputs / f"modelo_{nome_limpo}.joblib")
    
    # Salva o relatório textual contendo o CV e o Teste Final (.txt)
    with open(pasta_outputs / f"resultado_{nome_limpo}.txt", 'w', encoding='utf-8') as f:
        f.write(f"Resultados do {nome_modelo}\n")
        f.write(f"{texto_cv}\n")
        f.write(f"Acurácia no Teste Final: {acc_teste:.4%}\n\n{relatorio_texto}")

def main():
    pasta_src = Path(__file__).resolve().parent
    raiz_projeto = pasta_src.parent
    pasta_outputs = raiz_projeto / 'outputs'
    pasta_outputs.mkdir(parents=True, exist_ok=True)
    
    json_path = raiz_projeto / 'data' / 'processed.json'
    features_path = pasta_outputs / 'features_xlmr_mean_pooling.npy'
    
    X, y, label_encoder = carregar_dados_e_labels(json_path, features_path)
    
    # Salva o label encoder para uso na visualização
    joblib.dump(label_encoder, pasta_outputs / 'label_encoder.joblib')
    
    calcular_separabilidade(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    modelo_svm = SVC(kernel='linear', C=1.0, random_state=42)
    modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
    
    avaliar_e_salvar_modelo(modelo_svm, X_train, X_test, y_train, y_test, label_encoder, "SVM Linear", pasta_outputs)
    avaliar_e_salvar_modelo(modelo_lr, X_train, X_test, y_train, y_test, label_encoder, "Regressao Logistica", pasta_outputs)

if __name__ == "__main__":
    main()