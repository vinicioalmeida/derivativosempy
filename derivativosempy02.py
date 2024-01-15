# Black-Scholes-Merton - Derivativos em Py #02

import math

def black_scholes_call_put(S, K, T, r, sigma, option_type):
    '''
    Para calcular o preço de uma call ou put (europeias) usando o modelo de Black-Scholes-Merton
        
    parâmetro S: preço spot do ativo subjacente
    parâmetro K: preço de exercício da opção
    parâmetro T: vencimento (em anos)
    parâmetro r: taxa de juros livre de risco (anual)
    parâmetro sigma: volatilidade do ativo subjacente (anual)
    parâmetro option_type: 'call' para uma opção de compra, 'put' para uma opção de venda
    retorna: preço da opção
    '''
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        option_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("Tipo de opção inválido. Use 'call' ou 'put'.")
    
    return option_price

def norm_cdf(x):
    """
    Para calcular a função de distribuição de probabilidade cumulativa (CDF) para uma variável com distribuição normal padrão
       
    parâmetro x: valor para calcular a CDF
    retorna: o valor da CDF
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# Um exemplo
S = 37.24
K = 35.52  
T = 0.05158730158
r = 0.116571 
sigma = 0.26 

precoCall = black_scholes_call_put(S, K, T, r, sigma, 'call')
precoPut = black_scholes_call_put(S, K, T, r, sigma, 'put')

print(f"Preço da call: {precoCall:.2f}")
print(f"Preço da put: {precoPut:.2f}")