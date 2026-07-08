import json
import numpy as np
import torch  
from pathlib import Path
from utils_pipeline import inicializar_modelos

def extrair_embedding_mean_pooling(textos, tokenizer, model, device):
    """
    Extrai embeddings usando Mean Pooling (média de todos os tokens válidos)
    a partir da média das últimas 4 camadas do XLM-R. Suporta lotes (batches).
    """
    with torch.no_grad():
        # Tokeniza o lote de textos com padding e truncamento dinâmico
        encoded = tokenizer(
            textos, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
        # O attention_mask pode ter sido interessante para ignorar tokens de [PAD] no cálculo da média
        attention_mask = encoded['attention_mask']
        
        outputs = model(**encoded, output_hidden_states=True)
        
        # Média aritmética das últimas 4 camadas do Transformer (A camada 12 também serviria, porém me pareceu ser mais seguro fazer uma média das últimas 4 camadas)
        ultimas_4 = torch.stack(outputs.hidden_states[-4:])
        media_camadas = ultimas_4.mean(dim=0)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Expandim a máscara de atenção para bater com o tamanho do embedding
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(media_camadas.size()).float()
        
        # Multiplquei os embeddings pela máscara (zera os tokens de PAD) e somei na dimensão da sequência
        sum_embeddings = torch.sum(media_camadas * input_mask_expanded, 1)
        
        # Somei a máscara para saber quantos tokens reais cada sentença possui
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Dividi a soma dos embeddings pelo total de tokens válidos
        embeddings_finais = (sum_embeddings / sum_mask).cpu().numpy()
        
    return embeddings_finais

def main():
    base_path = Path(__file__).parent.parent
    json_path = base_path / 'data' / 'processed.json'
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dados_corpus = json.load(f)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {json_path}")
        return
    
    print(f"Total de registros carregados: {len(dados_corpus)}")
    
    # Inicializa o pipeline do XLM-R-base
    nlp_stanza, tokenizer, model, device = inicializar_modelos('xlm-roberta-base')
    
    features_list = []
    
    # Configuração de Batching
    BATCH_SIZE = 32  # Alterar para 16 ou 64 dependendo da memória GPU/CPU, 32 funcionou bem para uma RX570 - 4gb
    
    print(f"Iniciando a extração com Mean Pooling (Batch Size: {BATCH_SIZE})...")
    
    for i in range(0, len(dados_corpus), BATCH_SIZE):
        batch_itens = dados_corpus[i:i + BATCH_SIZE]
        textos_batch = [item['texto'] for item in batch_itens]
        
        # Extrai os embeddings para o lote inteiro de uma vez
        embeddings_batch = extrair_embedding_mean_pooling(textos_batch, tokenizer, model, device)
        
        features_list.append(embeddings_batch)
        
        if (i + len(batch_itens)) % 100 == 0 or (i + len(batch_itens)) == len(dados_corpus):
            print(f"Processados {i + len(batch_itens)}/{len(dados_corpus)} itens...")

    # Consolida e exporta os embeddings extraídos
    if features_list:
        features_matrix = np.vstack(features_list)
        print("\nProcessamento concluído com sucesso!")
        print("Formato final da matriz de features (Mean Pooling + 4 camadas):", features_matrix.shape)
        
        # Salva a matriz em um arquivo binário do NumPy
        np.save('features_xlmr_mean_pooling.npy', features_matrix)
        print("Embeddings salvos em 'features_xlmr_mean_pooling.npy'")
    else:
        print("Nenhuma feature foi extraída.")

if __name__ == "__main__":
    main()