# Precificando um Straddle - Derivativos em Py #03

import matplotlib.pyplot as plt

# Função para calcular o payoff do Straddle
def straddle_payoff(stock_price, strike_price, call_price, put_price):
    call_payoff = max(stock_price - strike_price, 0) - call_price
    put_payoff = max(strike_price - stock_price, 0) - put_price
    return call_payoff + put_payoff

# K e preços das calls e puts
strike_price = 74.32
call_price = 1.47
put_price = 1.09

# Cálculo do payoff do straddle para um intervalo dos preços do ativo subjacente
stock_prices = range(int(0.5 * strike_price), int(1.5 * strike_price), 0.1)
payoffs = [straddle_payoff(price, strike_price, call_price, put_price) for price in stock_prices]

# Plot do payoff do straddle
plt.plot(stock_prices, payoffs)
plt.title('Payoff do Straddle')
plt.xlabel('Preço do ativo subjacente')
plt.ylabel('Payoff')
plt.grid(True)
plt.show()

# Stradle - Gregas
##################

import numpy as np
from scipy.stats import norm

# Modelo de precificação de Black-Scholes-Merton
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Funções para cálculo das gregas - delta, gamma, vega, theta
def delta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def theta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

# Parâmetros
S = 74.27           # Preço atual do ativo subjacente
K = strike_price    # Preço de exercício
T = 8/252           # Vencimento em anos
r = 0.116571        # Taxa de juros livre de risco
sigma = 0.21        # Volatilidade

# Cálculo das gregas para call e put
call_delta = delta(S, K, T, r, sigma, 'call')
put_delta = delta(S, K, T, r, sigma, 'put')
call_gamma = gamma(S, K, T, r, sigma)
put_gamma = gamma(S, K, T, r, sigma)
call_vega = vega(S, K, T, r, sigma)
put_vega = vega(S, K, T, r, sigma)
call_theta = theta(S, K, T, r, sigma, 'call')
put_theta = theta(S, K, T, r, sigma, 'put')

# Cálculo das gregas do straddle, somando as gregas de call e put
straddle_delta = call_delta + put_delta
straddle_gamma = call_gamma + put_gamma
straddle_vega = call_vega + put_vega
straddle_theta = call_theta + put_theta

# Critério de conveniência da operação baseada em valores das gregas do straddle
convenient_delta = abs(straddle_delta) >= 0.5  # Exemplo. Ajustar de acordo com análise individual
convenient_gamma = abs(straddle_gamma) >= 0.1  # Exemplo. Ajustar de acordo com análise individual
convenient_vega = abs(straddle_vega) >= 0.01   # Exemplo. Ajustar de acordo com análise individual
convenient_theta = abs(straddle_theta) >= 0.05 # Exemplo. Ajustar de acordo com análise individual

# Checagem da conveniência do straddle
if convenient_delta and convenient_gamma and convenient_vega and convenient_theta:
    print("Montar o straddle é conveniente de acordo com as regras de sua estratégia.")
else:
    print("Montar o straddle NÃO é conveniente de acordo com as regras de sua estratégia.")
