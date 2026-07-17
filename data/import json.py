import json
import re
import os

# Mapeamento dos verbos para suas respectivas classes
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
    "humilhar": "Classe 3", "ludibriar": "Classe 3", "pacificar": "Classe 3", "provocar": "Classe 3", 
    "ridicularizar": "Classe 3", "tranquilizar": "Classe 3",
    
    # Classe 4
    "acovardar": "Classe 4", "agradar": "Classe 4", "alarmar": "Classe 4", "alegrar": "Classe 4", 
    "animar": "Classe 4", "apavorar": "Classe 4", "assombrar": "Classe 4", "assustar": "Classe 4", 
    "aterrorizar": "Classe 4"
}

def extrair_dados_verbo(caminho_arquivo):
    resultado = []
    verbo_atual = None
    contador_verbo = {}
    
    # Conjunto para rastrear frases únicas já processadas
    frases_processadas = set()
    # Lista para armazenar as informações sobre as duplicatas encontradas
    duplicatas_encontradas = []

    regex_source = re.compile(r'\\')
    regex_pontuacao = re.compile(r'[.,\/#!$%\^&\*;:{}=\-_`~()?]')

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            linha_limpa = regex_source.sub('', linha).strip()
            
            if not linha_limpa:
                continue

            # Identifica se a linha define o Verbo Alvo
            match_vp = re.match(r'^VP\(([^)]+)\):$', linha_limpa)
            if match_vp:
                verbo_atual = match_vp.group(1).strip().lower()
                if verbo_atual not in contador_verbo:
                    contador_verbo[verbo_atual] = 0
                continue
            
            # Processa a frase se o verbo estiver definido
            if verbo_atual:
                chave_duplicata = (verbo_atual, linha_limpa)
                
                # Se a frase JÁ EXISTE para este verbo, nós a guardamos no relatório de duplicadas
                if chave_duplicata in frases_processadas:
                    duplicatas_encontradas.append({
                        "verbo": verbo_atual,
                        "frase_repetida": linha_limpa
                    })
                    continue  # Pula para não colocar duplicado no JSON principal
                
                # Se for inédita, adicionamos ao controle e seguimos o fluxo normal
                frases_processadas.add(chave_duplicata)
                
                contador_verbo[verbo_atual] += 1
                id_gerado = f"{verbo_atual}-{contador_verbo[verbo_atual]}"
                
                frase_para_busca = regex_pontuacao.sub('', linha_limpa)
                palavras = frase_para_busca.split()
                
                posicao_verbo = -1
                for idx, palavra in enumerate(palavras):
                    if palavra.lower() == verbo_atual:
                        posicao_verbo = idx
                        break
                
                classe_verbo = MAPEAMENTO_CLASSES.get(verbo_atual, "Não categorizado")
                
                resultado.append({
                    "id": id_gerado,
                    "texto": linha_limpa,
                    "verbo_alvo": verbo_atual,
                    "classe": classe_verbo,
                    "posicao_verbo": posicao_verbo
                })

    return resultado, duplicatas_encontradas

# --- Execução do Script ---
diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
arquivo_entrada = os.path.join(diretorio_do_script, "corpus_puro.txt")
arquivo_saida = os.path.join(diretorio_do_script, "corpus_processado.json")
arquivo_duplicatas = os.path.join(diretorio_do_script, "duplicatas.txt")

try:
    dados_json, duplicadas = extrair_dados_verbo(arquivo_entrada)
    
    # 1. Salva o JSON limpo (sem duplicadas)
    with open(arquivo_saida, 'w', encoding='utf-8') as f_out:
        json.dump(dados_json, f_out, indent=4, ensure_ascii=False)
        
    # 2. Salva o relatório de duplicatas encontradas em um arquivo .txt legível
    with open(arquivo_duplicatas, 'w', encoding='utf-8') as f_dup:
        if duplicadas:
            f_dup.write(f"=== FORAM ENCONTRADAS {len(duplicadas)} FRASES DUPLICADAS ===\n\n")
            for item in duplicadas:
                f_dup.write(f"Verbo: {item['verbo']}\n")
                f_dup.write(f"Frase Repetida: {item['frase_repetida']}\n")
                f_dup.write("-" * 50 + "\n")
        else:
            f_dup.write("Nenhuma frase duplicada foi encontrada no arquivo de origem.")

    print(f"Sucesso!")
    print(f"-> O arquivo '{arquivo_saida}' foi gerado com {len(dados_json)} itens únicos.")
    print(f"-> O arquivo de relatório '{arquivo_duplicatas}' foi gerado com {len(duplicadas)} duplicatas para você analisar.")

except FileNotFoundError:
    print(f"Erro: O arquivo '{arquivo_entrada}' não foi encontrado.")