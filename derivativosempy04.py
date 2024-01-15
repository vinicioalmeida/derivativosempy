# Árvores binomiais para precificação de opções - Derivativos em Py #04

import numpy as np

# Função para cálculo do preço da opção via árvore binomial
def binomial_tree_option_pricing(S, K, T, r, sigma, n, option_type='call'):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    # Matriz dos preços das opções em cada nó
    option_values = np.zeros((n + 1, n + 1))

    # Cálculo do valor da opção nos nós de vencimento
    for j in range(n + 1):
        if option_type == 'call':
            option_values[n, j] = max(0, S * (u**j) * (d**(n - j)) - K)
        elif option_type == 'put':
            option_values[n, j] = max(0, K - S * (u**j) * (d**(n - j)))

    # Cálculo dos preços retrocedendo na árvore
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (q * option_values[i + 1, j] + (1 - q) * option_values[i + 1, j + 1])

    # Preço da opção no nó inicial
    option_price = option_values[0, 0]
    
    return option_price

# Parâmetros
S = 38.17      # Preço spot inicial da ação
K = 40.01       # Preço de exercício
T = 22/252      # Vencimento em anos
r = 0.115685    # Taxa de juros livre de risco (anual)
sigma = 0.25    # Volatilidade anual

# Número de passos
n = int(5)

# Cálculo do preço da opção de compra (call)
call_price = binomial_tree_option_pricing(S, K, T, r, sigma, n, option_type='call')

# Cálculo do preço da opção de venda (put)
put_price = binomial_tree_option_pricing(S, K, T, r, sigma, n, option_type='put')

print(f"O preço da opção de compra (call) com {n} passos é: {call_price:.2f}")
print(f"O preço da opção de venda (put) com {n} passos é: {put_price:.2f}")