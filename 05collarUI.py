# Fence vale3 ok
# Fence petr4
# collarui petr4 ok
# 1 collar de 2022

# Exotic options pricing example
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
PETR4 Collar UI
Lucro máximo: $1.349.000
Prejuízo máximo: $341.000
Preço de referência: $34,12

1. Compra Put - Strike 90%
2. Venda Call - Strike 110% up and in 139,53%
3. Compra da Ação

Data de vencimento 1/10/24
Data de fixing 30/9/24
Valor atribuido à estrutura de opcoes $0
Exposicao da operacao $ 3.412.000

'''

# 1. PUT VANILLA COMPRADA

def black_scholes_call_put(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        option_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

S = 34.12    # Current price of the underlying asset
K = S * 0.90 # Strike price of the option
T = 252/252 * 2  # Time to expiration (year based)
r = 0.11     # Risk-free interest rate
sigma = 0.25 # Volatility (25%)

call_price = black_scholes_call_put(S, K, T, r, sigma, 'call')
put_price = black_scholes_call_put(S, K, T, r, sigma, 'put')

print(f"Put Option Price: {put_price:.2f}")

# 2. CALL UP AND IN VENDIDA

Ke = S * 1.100  # Strike price of the option
H = S * 1.3953 
q = 0 # Dividend yield


lambda1 = (r - q + (sigma**2)/2)/(sigma**2)
y = math.log((H**2)/(S*Ke))/(sigma*math.sqrt(T)) + lambda1*sigma*math.sqrt(T)
x1 = math.log(S/H)/(sigma*math.sqrt(T)) + lambda1*sigma*math.sqrt(T)
y1 = math.log(H/S)/(sigma*math.sqrt(T)) + lambda1*sigma*math.sqrt(T)

cui_price = S*norm_cdf(x1)*math.exp(-q * T) - Ke*math.exp(-r * T) * norm_cdf(x1 - sigma*math.sqrt(T))      \
            - S*math.exp(-q * T)*(H/S)**(2*lambda1)*(norm_cdf(y)-norm_cdf(y1))                            \
            + Ke*math.exp(-r * T)*(H/S)**(2*lambda1-2)*(norm_cdf(y-sigma*math.sqrt(T)) - norm_cdf(y1 - sigma*math.sqrt(T)))

print(f"Exotic Call Option Price: {cui_price:.2f}")

# Lembrando, compra put vanilla, vende call exótica
portfolio_value = put_price - cui_price 
print(f"Portfolio value: {portfolio_value:.2f}")


### PAYOFF DA PUT COMPRADA

def put_payoff(S, K, premium):
    return np.where(S < K, K - S, 0) - premium

# Preços do ativo subjacente
S_values = np.linspace(0, 2 * S, 100)

# Preço da put
put_premium = put_price

# Calculando o payoff da put
payoff_put = put_payoff(S_values, K, put_premium)

# Plotando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(S_values, payoff_put, label='Payoff da Put Comprada')
plt.xlabel('Preço do Ativo Subjacente')
plt.ylabel('Payoff')
plt.title('Payoff da Opção de Venda (Put)')
plt.grid(True)
plt.legend()
plt.show()

### PAYOFF DA CALL EXOTICA VENDIDA
# Preços do ativo subjacente
S_values = np.linspace(0, 2 * S, 100)

# Strike e barreira da call up-and-in
K = Ke  # Strike price da opção
H = H  # Barreira da opção

# Preço da call up-and-in vendida
call_premium = cui_price

# Função para calcular o payoff da call up-and-in vendida
def call_up_and_in_payoff(S, K, H, premium):
    # Calculando o payoff da call up-and-in
    payoff = np.where(S <= H, 0, S - K) - premium
    return payoff

# Calculando o payoff da call up-and-in vendida
payoff_call_up_and_in = - call_up_and_in_payoff(S_values, K, H, call_premium)

# Plotando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(S_values, payoff_call_up_and_in, label='Payoff da Call Up-and-In Vendida')
plt.xlabel('Preço do Ativo Subjacente')
plt.ylabel('Payoff')
plt.title('Payoff da Opção de Compra (Call) Up-and-In Vendida')
plt.grid(True)
plt.legend()
plt.show()

# Somando os payoffs das opcoes
payoff_total = payoff_put + payoff_call_up_and_in

# Plotando o gráfico do payoff total
plt.figure(figsize=(10, 6))
plt.plot(S_values, payoff_total, label='Payoff Total')
plt.xlabel('Preço do Ativo Subjacente')
plt.ylabel('Payoff')
plt.title('Payoff Total da Estrutura de Opções')
plt.grid(True)
plt.legend()
plt.show()

# Plotando o gráfico com os três payoffs juntos
plt.figure(figsize=(10, 6))
# Payoff da put
plt.plot(S_values, payoff_put, label='Payoff da Put Comprada')
# Payoff da call exótica vendida
plt.plot(S_values, payoff_call_up_and_in, label='Payoff da Call Up-and-In Vendida')
# Payoff total
plt.plot(S_values, payoff_total, label='Payoff Total')
plt.xlabel('Preço do Ativo Subjacente')
plt.ylabel('Payoff')
plt.title('Payoff da Estrutura de Opções')
plt.grid(True)
plt.legend()
plt.show()


### Payoff da ação apenas

# Preços do ativo subjacente para o gráfico
S_values = np.linspace(0, 2 * S, 100)

# Payoff de ter apenas a ação
payoff_acao = S_values - S

# Plotando o gráfico do payoff de ter a ação
plt.figure(figsize=(10, 6))
plt.plot(S_values, payoff_acao, label='Payoff da Ação Comprada')
plt.xlabel('Preço do Ativo Subjacente')
plt.ylabel('Payoff')
plt.title('Payoff da Ação Comprada')
plt.grid(True)
plt.legend()
plt.show()

# Calculando o payoff total da estrutura
payoff_total = payoff_put + payoff_call_up_and_in + payoff_acao

# Plotando o gráfico dos payoffs
plt.figure(figsize=(10, 6))

# Payoff da put
plt.plot(S_values, payoff_put, label='Payoff da Put Comprada')

# Payoff da call exótica vendida
plt.plot(S_values, payoff_call_up_and_in, label='Payoff da Call Up-and-In Vendida')

# Payoff de ter a ação
plt.plot(S_values, payoff_acao, label='Payoff da Ação Comprada')

# Payoff total
plt.plot(S_values, payoff_total, label='Payoff Total')

plt.xlabel('Preço do Ativo Subjacente')
plt.ylabel('Lucro/Prejuízo do Collar UI')
plt.title('Lucro/Prejuízo da Operação para o Emissor')
plt.grid(True)
plt.legend()
plt.show()

# Agora a Tabela de Payoffs

# Variações percentuais do ativo subjacente
percent_changes = np.linspace(-0.40, 0.4953, 100)
S_values = S * (1 + percent_changes)

# Calculando o retorno percentual do payoff total
return_percent = ((S + payoff_total) / S - 1) * 100

# Criando a tabela de retorno percentual
tabela_retorno = pd.DataFrame({
    'Variação %': percent_changes * 100,
    'Preço do Ativo': S_values,
    'Payoff Total': payoff_total,
    'Retorno %': return_percent
})

pd.set_option('display.max_rows', None)
print(tabela_retorno)

