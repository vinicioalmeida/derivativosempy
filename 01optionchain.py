# Opções na B3
import pandas as pd
import requests

subjacente = 'PETR4'   # não vem do Yfinance! esqueça o ".SA" ao final do ticker!

# Um vencimento
vencimento = '2026-10-16'      # YYYY-MM-DD
def optionchaindate(subjacente, vencimento):
    url = f'https://opcoes.net.br/listaopcoes/completa?idAcao={subjacente}&listarVencimentos=false&cotacoes=true&vencimentos={vencimento}'
    r = requests.get(url).json()
    x = ([subjacente, vencimento, i[0].split('_')[0], i[2], i[3], i[5], i[8], i[9], i[10]] for i in r['data']['cotacoesOpcoes'])
    return pd.DataFrame(x, columns=['subjacente', 'vencimento', 'ativo', 'tipo', 'modelo', 'strike', 'preco', 'negocios', 'volume'])

optionchaindate(subjacente, vencimento)

# Todos os vencimentos
def optionchain(subjacente):
    url2 = f'https://opcoes.net.br/listaopcoes/completa?idLista=ML&idAcao={subjacente}&listarVencimentos=true&cotacoes=true'
    r = requests.get(url2).json()
    vencimentos = [i['value'] for i in r['data']['vencimentos']]
    df = pd.concat([optionchaindate(subjacente, vencimento) for vencimento in vencimentos])
    return df

optionchain(subjacente)