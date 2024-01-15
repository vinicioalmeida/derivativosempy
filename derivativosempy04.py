# Árvores binomiais para precificação de opções - Derivativos em Py #04

import numpy as np

# Função para cálculo do preço da opção via árvore binomial
def binomial_tree_option_pricing(S, K, T, r, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    # Matriz dos preços das opções em cada nó
    option_values = np.zeros((n + 1, n + 1))

    # Cálculo do valor da opção nos nós de vencimento
    for j in range(n + 1):
        option_values[n, j] = max(0, S * (u**j) * (d**(n - j)) - K)

    # Cálculo dos preços retrocedendo na árvore
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (q * option_values[i + 1, j] + (1 - q) * option_values[i + 1, j + 1])

    # Preço da opção no nó inicial
    option_price = option_values[0, 0]
    
    return option_price

# Parâmetros
S0 = 20      # Preço spot inicial da ação
K = 21       # Preço de exercício
T = 0.25     # Vencimento em anos
r = 0.12     # Taxa de juros livre de risco (anual)
sigma = 0.2  # Volatilidade anual

# Número de passos
n = int(2)

# Cálculo do preço da opção
option_price = binomial_tree_option_pricing(S0, K, T, r, sigma, n)

print(f"O preço da opção com {n} passos é: {option_price:.2f}")


# Black-Scholes-Merton - Derivativos em Py #02

import math

def black_scholes_call_put(S, K, T, r, sigma, option_type):
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