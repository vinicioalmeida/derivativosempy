# =============================================================================
# SEMINARIO - VOLATILIDADE IMPLICITA
# Da volatilidade realizada a superficie SVI e a estrategia de straddles
# Baseado no working paper "SVI Surface / Trade Options - MT5"
#
# Roteiro (segue a ordem dos slides):
#   PARTE 1  Volatilidade realizada (4 estimadores), IV e spread IV-RV
#   PARTE 2  Black-Scholes e calculo da IV por Newton-Raphson
#   PARTE 3  Cadeia de opcoes, sorriso e calibracao SVI: superficie unica x dual
#   PARTE 4  Estrutura a termo e superficie de volatilidade
#   PARTE 5  Payoff de straddles (long e short vol)
#   PARTE 6  Sinal z-score nas duas pernas e P&L do straddle
#
# Por padrao roda com dados simulados (funciona offline, resultado reproduzivel).
# Taxa de juros tratada em termos nominais.
# =============================================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
np.random.seed(7)

# -----------------------------------------------------------------------------
# PARAMETROS GERAIS
# -----------------------------------------------------------------------------
USAR_DADOS_REAIS = False          # True: baixa o ativo no Yahoo Finance (parte 1)
TICKER = "BOVA11.SA"
R_NOMINAL = 0.1425                # taxa nominal anual (ajuste para a Selic vigente)
DU = 252
JANELA_RV = 21
JANELA_Z_DIARIO = 63
LIMIAR_DIARIO = 1.5

LIMIAR_PISO = 1.5                 # regra do sistema: piso 1.5, teto 3.0
LIMIAR_TETO = 3.0
MIN_OBS = 30                      # minimo de observacoes por serie
CAPITAL_MAX = 100_000             # capital maximo por straddle (R$)
LOTE = 100
ALVO_LUCRO = 0.08                 # alvo sobre o premio pago (didatico)
STOP_PERDA = 0.10                 # stop sobre o premio pago (didatico)
Z_SAIDA = 0.5                     # saida por reversao do residuo

SALVAR_FIGURAS = True
PASTA = Path(r"C:\repo\derivativosempy")
PASTA.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# IDENTIDADE VISUAL
# -----------------------------------------------------------------------------
AZUL = "#1f4e79"
AZUL_CLARO = "#8fb3d9"
LARANJA = "#d9822b"
VERDE = "#2e8b57"
VERMELHO = "#c0392b"
ROXO = "#6c3483"
CINZA = "#7f8c8d"
FUNDO = "#fbfbfd"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": FUNDO,
    "axes.edgecolor": "#c8ccd4",
    "axes.grid": True,
    "grid.color": "#e3e6ec",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Formulas de Black-Scholes (lambdas curtas: sao necessarias para o otimizador
# e evitam repetir a mesma expressao dezenas de vezes)
bs_d1 = lambda S, K, T, r, s: (np.log(S / K) + (r + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
bs_call = lambda S, K, T, r, s: S * norm.cdf(bs_d1(S, K, T, r, s)) - K * np.exp(-r * T) * norm.cdf(bs_d1(S, K, T, r, s) - s * np.sqrt(T))
bs_put = lambda S, K, T, r, s: bs_call(S, K, T, r, s) - S + K * np.exp(-r * T)
bs_vega = lambda S, K, T, r, s: S * np.sqrt(T) * norm.pdf(bs_d1(S, K, T, r, s))
svi_w = lambda p, k: p[0] + p[1] * (p[2] * (k - p[3]) + np.sqrt((k - p[3]) ** 2 + p[4] ** 2))

print("=" * 78)
print("SEMINARIO - VOLATILIDADE IMPLICITA E SUPERFICIE SVI")
print("=" * 78)

# =============================================================================
# PARTE 1 - VOLATILIDADE REALIZADA, IMPLICITA E SPREAD
# =============================================================================
ohlc = None
if USAR_DADOS_REAIS:
    try:
        import yfinance as yf
        bruto = yf.download(TICKER, period="2y", auto_adjust=True, progress=False)
        if isinstance(bruto.columns, pd.MultiIndex):
            bruto.columns = bruto.columns.get_level_values(0)
        ohlc = bruto[["Open", "High", "Low", "Close"]].dropna().copy()
        print(f"Dados reais carregados: {TICKER}, {len(ohlc)} pregoes")
    except Exception as erro:
        print(f"Falha ao baixar dados reais ({erro}). Seguindo com simulacao.")

if ohlc is None:
    # Precos intradiarios com volatilidade estocastica (log-OU), saltos e um
    # episodio de estresse, para gerar OHLC realista
    n_dias, passos = 504, 78
    log_vol = np.empty(n_dias)
    log_vol[0] = np.log(0.22)
    for t in range(1, n_dias):
        log_vol[t] = log_vol[t - 1] + 0.04 * (np.log(0.22) - log_vol[t - 1]) + 0.09 * np.random.randn()
    vol_dia = np.exp(log_vol)
    vol_dia[340:365] *= 1.9
    dt = 1 / (DU * passos)
    ret_intra = -0.5 * vol_dia[:, None] ** 2 * dt + vol_dia[:, None] * np.sqrt(dt) * np.random.randn(n_dias, passos)
    ret_intra[:, 0] += (np.random.rand(n_dias) < 0.02) * np.random.normal(-0.015, 0.025, n_dias)
    caminho = np.exp(np.log(130.0) + np.cumsum(ret_intra.ravel())).reshape(n_dias, passos)
    datas = pd.bdate_range(end=pd.Timestamp("2026-09-14"), periods=n_dias)
    ohlc = pd.DataFrame({
        "Open": caminho[:, 0], "High": caminho.max(axis=1),
        "Low": caminho.min(axis=1), "Close": caminho[:, -1],
    }, index=datas)
    print(f"Dados simulados: {n_dias} pregoes com {passos} observacoes intradiarias cada")

# Estimadores de volatilidade realizada (anualizados)
ret = np.log(ohlc["Close"] / ohlc["Close"].shift(1))
hl = np.log(ohlc["High"] / ohlc["Low"])
co = np.log(ohlc["Close"] / ohlc["Open"])
ohlc["RV_cc"] = ret.rolling(JANELA_RV).std() * np.sqrt(DU)
ohlc["RV_park"] = np.sqrt((hl ** 2).rolling(JANELA_RV).mean() / (4 * np.log(2)) * DU)
ohlc["RV_gk"] = np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(JANELA_RV).mean() * DU)
ohlc["RV_ewma"] = np.sqrt((ret ** 2).ewm(alpha=1 - 0.94).mean() * DU)

# IV ATM simulada: mistura da RV corrente com a RV futura (o mercado antecipa),
# um premio de risco de variancia e ruido persistente
rv_futura = ohlc["RV_cc"].shift(-JANELA_RV).fillna(ohlc["RV_cc"])
ruido = np.zeros(len(ohlc))
for t in range(1, len(ohlc)):
    ruido[t] = 0.85 * ruido[t - 1] + 0.012 * np.random.randn()
ohlc["IV_atm"] = (0.45 * ohlc["RV_cc"] + 0.55 * rv_futura + 0.025 + ruido).clip(lower=0.08)

ohlc["spread"] = ohlc["IV_atm"] - ohlc["RV_cc"]
media_movel = ohlc["spread"].rolling(JANELA_Z_DIARIO).mean().shift(1)
desvio_movel = ohlc["spread"].rolling(JANELA_Z_DIARIO).std().shift(1)
ohlc["z_spread"] = (ohlc["spread"] - media_movel) / desvio_movel
ohlc["sinal"] = np.select(
    [ohlc["z_spread"] > LIMIAR_DIARIO, ohlc["z_spread"] < -LIMIAR_DIARIO],
    ["VENDER VOL", "COMPRAR VOL"], default="NEUTRO")
ohlc = ohlc.dropna()

fig, eixos = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Parte 1 | Volatilidade realizada, implicita e o spread IV - RV", fontsize=15, fontweight="bold")

ax = eixos[0, 0]
ax.plot(ohlc.index, ohlc["Close"], color=AZUL, lw=1.4)
ax2 = ax.twinx()
ax2.fill_between(ohlc.index, 0, ohlc["RV_cc"] * 100, color=LARANJA, alpha=0.18)
ax2.set_ylabel("RV 21d (%)", color=LARANJA)
ax2.grid(False)
ax.set_title("Preco do ativo e regime de volatilidade")
ax.set_ylabel("Preco (R$)")

ax = eixos[0, 1]
ax.plot(ohlc.index, ohlc["RV_cc"] * 100, color=AZUL, lw=1.5, label="Close-to-close")
ax.plot(ohlc.index, ohlc["RV_park"] * 100, color=LARANJA, lw=1.1, label="Parkinson (H-L)")
ax.plot(ohlc.index, ohlc["RV_gk"] * 100, color=VERDE, lw=1.1, label="Garman-Klass (OHLC)")
ax.plot(ohlc.index, ohlc["RV_ewma"] * 100, color=ROXO, lw=1.1, ls="--", label="EWMA (lambda=0.94)")
ax.set_title("Quatro estimadores de volatilidade realizada")
ax.set_ylabel("% a.a.")
ax.legend(ncol=2)

ax = eixos[1, 0]
ax.plot(ohlc.index, ohlc["IV_atm"] * 100, color=VERMELHO, lw=1.6, label="IV ATM (expectativa)")
ax.plot(ohlc.index, ohlc["RV_cc"] * 100, color=AZUL, lw=1.4, label="RV 21d (observada)")
ax.fill_between(ohlc.index, ohlc["IV_atm"] * 100, ohlc["RV_cc"] * 100,
                where=ohlc["IV_atm"] >= ohlc["RV_cc"], color=VERMELHO, alpha=0.12, label="IV > RV")
ax.fill_between(ohlc.index, ohlc["IV_atm"] * 100, ohlc["RV_cc"] * 100,
                where=ohlc["IV_atm"] < ohlc["RV_cc"], color=VERDE, alpha=0.18, label="IV < RV")
ax.set_title("IV olha para frente, RV olha para tras")
ax.set_ylabel("% a.a.")
ax.legend(ncol=2)

ax = eixos[1, 1]
ax.plot(ohlc.index, ohlc["z_spread"], color=CINZA, lw=1.0)
ax.axhline(LIMIAR_DIARIO, color=VERMELHO, ls="--", lw=1)
ax.axhline(-LIMIAR_DIARIO, color=VERDE, ls="--", lw=1)
ax.axhspan(LIMIAR_DIARIO, 6, color=VERMELHO, alpha=0.05)
ax.axhspan(-6, -LIMIAR_DIARIO, color=VERDE, alpha=0.06)
vendas = ohlc[ohlc["sinal"] == "VENDER VOL"]
compras = ohlc[ohlc["sinal"] == "COMPRAR VOL"]
ax.scatter(vendas.index, vendas["z_spread"], color=VERMELHO, s=14, zorder=3, label=f"Vender vol ({len(vendas)})")
ax.scatter(compras.index, compras["z_spread"], color=VERDE, s=14, zorder=3, label=f"Comprar vol ({len(compras)})")
ax.set_ylim(-5, 5)
ax.set_title(f"Z-score do spread (janela {JANELA_Z_DIARIO}d, sem look-ahead)")
ax.set_ylabel("Z")
ax.legend(loc="upper left")

for ax in eixos.ravel():
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b/%y"))
fig.tight_layout()
if SALVAR_FIGURAS:
    fig.savefig(PASTA / "01_volatilidades_spread.png")

print("\nPARTE 1 - situacao no ultimo pregao")
ultimo = ohlc.iloc[-1]
print(f"  Preco: R$ {ultimo['Close']:.2f} | RV 21d: {ultimo['RV_cc']:.1%} | IV ATM: {ultimo['IV_atm']:.1%}")
print(f"  Spread: {ultimo['spread']:+.2%} | Z: {ultimo['z_spread']:+.2f} | Sinal: {ultimo['sinal']}")
print(f"  Correlacao IV x RV: {ohlc['IV_atm'].corr(ohlc['RV_cc']):.2f}")

# =============================================================================
# PARTE 2 - BLACK-SCHOLES E NEWTON-RAPHSON
# =============================================================================
S_demo, K_demo, T_demo = 130.0, 150.0, 30 / DU
sigma_verdadeiro = 0.27
preco_mercado = bs_call(S_demo, K_demo, T_demo, R_NOMINAL, sigma_verdadeiro)

sigma_it = [0.50]
for _ in range(8):
    s = sigma_it[-1]
    erro_preco = bs_call(S_demo, K_demo, T_demo, R_NOMINAL, s) - preco_mercado
    sigma_it.append(s - erro_preco / bs_vega(S_demo, K_demo, T_demo, R_NOMINAL, s))
sigma_it = np.array(sigma_it)

print("\nPARTE 2 - Newton-Raphson")
print(f"  Preco de mercado da call: R$ {preco_mercado:.4f}")
for i, s in enumerate(sigma_it[:6]):
    print(f"  iteracao {i}: sigma = {s:.6f} | erro = {abs(s - sigma_verdadeiro):.2e}")

fig, eixos = plt.subplots(1, 2, figsize=(14, 5.2))
fig.suptitle("Parte 2 | Invertendo Black-Scholes: a volatilidade implicita por Newton-Raphson",
             fontsize=14, fontweight="bold")

ax = eixos[0]
grade_sigma = np.linspace(0.08, 0.50, 300)
ax.plot(grade_sigma * 100, bs_call(S_demo, K_demo, T_demo, R_NOMINAL, grade_sigma), color=AZUL, lw=2.2,
        label="Preco BS da call")
ax.axhline(preco_mercado, color=VERMELHO, ls="--", lw=1.2, label=f"Preco de mercado = R$ {preco_mercado:.2f}")
cores_it = plt.cm.Oranges(np.linspace(0.45, 0.95, 4))
for i in range(4):
    s0 = sigma_it[i]
    p0 = bs_call(S_demo, K_demo, T_demo, R_NOMINAL, s0)
    v0 = bs_vega(S_demo, K_demo, T_demo, R_NOMINAL, s0)
    xs = np.array([s0, sigma_it[i + 1]])
    ax.plot(xs * 100, p0 + v0 * (xs - s0), color=cores_it[i], lw=1.4)
    s1 = sigma_it[i + 1]
    ax.plot([s1 * 100, s1 * 100], [preco_mercado, bs_call(S_demo, K_demo, T_demo, R_NOMINAL, s1)],
            color=cores_it[i], ls=":", lw=1.2)
    ax.scatter(s0 * 100, p0, color=cores_it[i], s=45, zorder=4)
    if i < 2:
        ax.annotate(f"n={i}", (s0 * 100, p0), textcoords="offset points", xytext=(-24, 6), fontsize=9)
ax.scatter(sigma_verdadeiro * 100, preco_mercado, marker="*", s=220, color=VERMELHO, zorder=5,
           label=f"IV = {sigma_verdadeiro:.0%}")
ax.set_xlabel("Volatilidade (% a.a.)")
ax.set_ylabel("Preco da opcao (R$)")
ax.set_title("Cada passo segue a tangente (inclinacao = vega)")
ax.legend(loc="upper left")

ax = eixos[1]
erros = np.abs(sigma_it - sigma_verdadeiro)[:6] + 1e-16
ax.semilogy(range(len(erros)), erros, "o-", color=ROXO, lw=2, ms=7)
ax.set_xlabel("Iteracao")
ax.set_ylabel("|sigma_n - IV| (escala log)")
ax.set_title("Convergencia quadratica: digitos corretos dobram a cada passo")
fig.tight_layout()
if SALVAR_FIGURAS:
    fig.savefig(PASTA / "02_newton_raphson.png")

# =============================================================================
# PARTE 3 - CADEIA DE OPCOES, SORRISO E CALIBRACAO SVI
# =============================================================================
spot = 130.0
vencimentos_du = np.array([21, 42, 63, 105])
strikes = np.round(spot * np.arange(0.80, 1.201, 0.0125), 2)

# Superficie "verdadeira" (desconhecida pelo mercado) com skew tipico de acoes
linhas = []
for T in vencimentos_du / DU:
    fwd = spot * np.exp(R_NOMINAL * T)
    p_true = [0.0, 0.11 * np.sqrt(T), -0.55, 0.01, 0.08 + 0.10 * T]
    p_true[0] = 0.0625 * T - p_true[1] * (-p_true[2] * p_true[3] + np.sqrt(p_true[3] ** 2 + p_true[4] ** 2))
    for K in strikes:
        k = np.log(K / fwd)
        iv_true = np.sqrt(svi_w(p_true, k) / T)
        for tipo in ["C", "P"]:
            otm = (tipo == "P" and k < 0) or (tipo == "C" and k >= 0)
            volume = np.exp(-6 * abs(k) / np.sqrt(T * 4)) * (5000 if otm else 600) * (1.6 if tipo == "P" else 1.0)
            volume = max(volume * np.random.lognormal(0, 0.4), 1.0)
            # cunha estrutural: puts mais demandadas (protecao) ficam mais caras
            iv_mkt = iv_true + (0.006 if tipo == "P" else -0.004) + np.random.randn() * 0.012 / volume ** 0.25
            linhas.append(dict(T=T, du=int(round(T * DU)), K=K, k=k, fwd=fwd, tipo=tipo,
                               volume=volume, iv_true=iv_true, iv_mkt_gerada=iv_mkt))
cadeia = pd.DataFrame(linhas)

# Deslocamentos no vencimento curto: pernas baratas (buy vol) e caras (sell vol)
curto = cadeia["du"] == vencimentos_du[0]
alvo_barato = cadeia.loc[curto, "K"].sub(spot * 0.975).abs().idxmin()
alvo_caro = cadeia.loc[curto, "K"].sub(spot * 1.0625).abs().idxmin()
K_barato = cadeia.loc[alvo_barato, "K"]
K_caro = cadeia.loc[alvo_caro, "K"]
cadeia.loc[curto & (cadeia["K"] == K_barato), "iv_mkt_gerada"] -= 0.035
cadeia.loc[curto & (cadeia["K"] == K_caro), "iv_mkt_gerada"] += 0.030

# Precos com bid-ask que alarga onde a liquidez e baixa
eh_call = cadeia["tipo"] == "C"
mid_teorico = np.where(eh_call,
                       bs_call(spot, cadeia["K"], cadeia["T"], R_NOMINAL, cadeia["iv_mkt_gerada"]),
                       bs_put(spot, cadeia["K"], cadeia["T"], R_NOMINAL, cadeia["iv_mkt_gerada"]))
meio_spread = np.maximum(mid_teorico * (0.01 + 0.25 / np.sqrt(cadeia["volume"])), 0.005)
cadeia["bid"] = np.maximum(np.round(mid_teorico - meio_spread, 2), 0.0)
cadeia["ask"] = np.round(mid_teorico + meio_spread, 2)
cadeia["mid"] = (cadeia["bid"] + cadeia["ask"]) / 2
cadeia["spread_rel"] = (cadeia["ask"] - cadeia["bid"]) / cadeia["mid"]

# Filtros de qualidade
intrinseco = np.where(eh_call, np.maximum(spot - cadeia["K"] * np.exp(-R_NOMINAL * cadeia["T"]), 0),
                      np.maximum(cadeia["K"] * np.exp(-R_NOMINAL * cadeia["T"]) - spot, 0))
filtro = (cadeia["bid"] >= 0.02) & (cadeia["spread_rel"] < 0.40) & (cadeia["mid"] > intrinseco + 0.005)
print(f"\nPARTE 3 - cadeia simulada: {len(cadeia)} opcoes, {filtro.sum()} aprovadas nos filtros")
cadeia = cadeia[filtro].reset_index(drop=True)
eh_call = cadeia["tipo"] == "C"

# IV por bissecao vetorizada (robusta nas asas, onde a vega e pequena)
baixo = np.full(len(cadeia), 0.01)
alto = np.full(len(cadeia), 3.00)
for _ in range(60):
    meio = (baixo + alto) / 2
    preco_meio = np.where(eh_call, bs_call(spot, cadeia["K"], cadeia["T"], R_NOMINAL, meio),
                          bs_put(spot, cadeia["K"], cadeia["T"], R_NOMINAL, meio))
    acima = preco_meio > cadeia["mid"].values
    alto = np.where(acima, meio, alto)
    baixo = np.where(acima, baixo, meio)
cadeia["iv"] = (baixo + alto) / 2
cadeia["w_mkt"] = cadeia["iv"] ** 2 * cadeia["T"]

# Pesos de liquidez assimetricos: a asa esquerda confia nas puts (book mais
# profundo), a direita nas calls; a perna ITM pesa menos
peso_base = np.log1p(cadeia["volume"]) / (cadeia["spread_rel"] + 0.02)
fator_assim = np.select(
    [(cadeia["tipo"] == "P") & (cadeia["k"] < 0), (cadeia["tipo"] == "C") & (cadeia["k"] < 0),
     (cadeia["tipo"] == "C") & (cadeia["k"] >= 0), (cadeia["tipo"] == "P") & (cadeia["k"] >= 0)],
    [1.00, 0.35, 1.00, 0.60])
cadeia["peso"] = peso_base * fator_assim
cadeia["peso"] = cadeia["peso"] / cadeia.groupby(["du", "tipo"])["peso"].transform("mean")

# Calibracao SVI multi-start: superficie conjunta (so no vencimento curto,
# para comparacao) e superficies duais (calls e puts) em todos os vencimentos
limites_inf = [-0.5, 1e-4, -0.999, -0.5, 1e-3]
limites_sup = [0.5, 2.0, 0.999, 0.5, 1.5]
parametros = {}
tarefas = [(vencimentos_du[0], "CONJ")] + [(du, lado) for du in vencimentos_du for lado in ["C", "P"]]
for du, lado in tarefas:
    sub = cadeia[(cadeia["du"] == du) & ((cadeia["tipo"] == lado) | (lado == "CONJ"))]
    kk, ww, pp = sub["k"].values, sub["w_mkt"].values, np.sqrt(sub["peso"].values)
    melhor = None
    for tentativa in range(6):
        chute = [ww.min() * 0.5, np.random.uniform(0.02, 0.3), np.random.uniform(-0.8, 0.2),
                 np.random.uniform(-0.1, 0.1), np.random.uniform(0.02, 0.3)]
        ajuste = least_squares(lambda p: pp * (svi_w(p, kk) - ww), chute,
                               bounds=(limites_inf, limites_sup), method="trf")
        if melhor is None or ajuste.cost < melhor.cost:
            melhor = ajuste
    iv_fit = np.sqrt(np.maximum(svi_w(melhor.x, kk), 1e-8) / sub["T"].values)
    rmse = np.sqrt(np.average((iv_fit - sub["iv"].values) ** 2, weights=sub["peso"].values))
    parametros[(du, lado)] = dict(a=melhor.x[0], b=melhor.x[1], rho=melhor.x[2], m=melhor.x[3],
                                  sigma=melhor.x[4], rmse=rmse, p=melhor.x)

tabela_svi = pd.DataFrame(
    [dict(du=du, superficie=lado, **{c: v for c, v in par.items() if c != "p"}) for (du, lado), par in parametros.items()])
print("\nParametros SVI calibrados (w(k) = a + b[rho(k-m) + sqrt((k-m)^2 + sigma^2)])")
print(tabela_svi.round(4).to_string(index=False))

# Residuos (em pontos de vol) no vencimento curto
sm = cadeia[cadeia["du"] == vencimentos_du[0]].copy()
p_conj = parametros[(vencimentos_du[0], "CONJ")]["p"]
sm["iv_conj"] = np.sqrt(svi_w(p_conj, sm["k"]) / sm["T"])
sm["iv_dual"] = np.where(sm["tipo"] == "C",
                         np.sqrt(svi_w(parametros[(vencimentos_du[0], "C")]["p"], sm["k"]) / sm["T"]),
                         np.sqrt(svi_w(parametros[(vencimentos_du[0], "P")]["p"], sm["k"]) / sm["T"]))
sm["res_conj"] = sm["iv"] - sm["iv_conj"]
sm["res_dual"] = sm["iv"] - sm["iv_dual"]

pares = sm.pivot_table(index="K", columns="tipo", values=["res_conj", "res_dual"]).dropna()
mesmo_sinal_conj = (np.sign(pares[("res_conj", "C")]) == np.sign(pares[("res_conj", "P")])).mean()
mesmo_sinal_dual = (np.sign(pares[("res_dual", "C")]) == np.sign(pares[("res_dual", "P")])).mean()
print(f"\nStrikes com residuos de call e put no mesmo sentido:")
print(f"  superficie conjunta: {mesmo_sinal_conj:.0%}   |   superficies duais: {mesmo_sinal_dual:.0%}")

fig = plt.figure(figsize=(15, 9.5))
grade = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.32, wspace=0.18)
fig.suptitle(f"Parte 3 | Sorriso de volatilidade ({vencimentos_du[0]} d.u.): por que duas superficies SVI?",
             fontsize=15, fontweight="bold")
k_grade = np.linspace(sm["k"].min() - 0.02, sm["k"].max() + 0.02, 300)
T0 = vencimentos_du[0] / DU
escala = 25 + 180 * sm["peso"] / sm["peso"].max()

for coluna, (titulo, modo) in enumerate([("Superficie unica (calls + puts juntas)", "conj"),
                                          ("Superficies duais com pesos assimetricos", "dual")]):
    ax = fig.add_subplot(grade[0, coluna])
    calls, puts = sm[sm["tipo"] == "C"], sm[sm["tipo"] == "P"]
    ax.scatter(calls["k"], calls["iv"] * 100, s=escala[calls.index], color=AZUL, alpha=0.65,
               edgecolor="white", label="Calls (tamanho = peso)")
    ax.scatter(puts["k"], puts["iv"] * 100, s=escala[puts.index], color=LARANJA, alpha=0.65,
               edgecolor="white", label="Puts (tamanho = peso)")
    if modo == "conj":
        ax.plot(k_grade, np.sqrt(svi_w(p_conj, k_grade) / T0) * 100, color=ROXO, lw=2.4, label="SVI conjunta")
    else:
        ax.plot(k_grade, np.sqrt(svi_w(parametros[(vencimentos_du[0], "C")]["p"], k_grade) / T0) * 100,
                color=AZUL, lw=2.4, label="SVI calls")
        ax.plot(k_grade, np.sqrt(svi_w(parametros[(vencimentos_du[0], "P")]["p"], k_grade) / T0) * 100,
                color=LARANJA, lw=2.4, label="SVI puts")
    for K_alvo, cor in [(K_barato, VERDE), (K_caro, VERMELHO)]:
        ax.axvline(sm.loc[sm["K"] == K_alvo, "k"].iloc[0], color=cor, ls=":", lw=1.4)
    ax.axvline(0, color=CINZA, lw=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("log-moneyness k = ln(K/F)")
    ax.set_ylabel("Volatilidade implicita (% a.a.)")
    ax.legend(loc="upper right")

for coluna, (col_res, titulo) in enumerate([("res_conj", f"Sentidos opostos em {1 - mesmo_sinal_conj:.0%} dos strikes: a cunha put-call contamina o sinal"),
                                            ("res_dual", "Residuos centrados em zero: deslocamentos saltam aos olhos")]):
    ax = fig.add_subplot(grade[1, coluna])
    largura = (strikes[1] - strikes[0]) * 0.38
    ax.bar(pares.index - largura / 2, pares[(col_res, "C")] * 100, width=largura, color=AZUL, label="Call")
    ax.bar(pares.index + largura / 2, pares[(col_res, "P")] * 100, width=largura, color=LARANJA, label="Put")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvspan(K_barato - 0.9, K_barato + 0.9, color=VERDE, alpha=0.12)
    ax.axvspan(K_caro - 0.9, K_caro + 0.9, color=VERMELHO, alpha=0.10)
    ax.set_title(titulo, fontsize=10.5)
    ax.set_xlabel("Strike (R$)")
    ax.set_ylabel("IV mercado - IV SVI (p.p.)")
    ax.legend(loc="lower left")
if SALVAR_FIGURAS:
    fig.savefig(PASTA / "03_sorriso_svi_dual.png")

# =============================================================================
# PARTE 4 - ESTRUTURA A TERMO E SUPERFICIE
# =============================================================================
k_sup = np.linspace(-0.20, 0.15, 60)
T_sup = vencimentos_du / DU
w_sup = np.array([(svi_w(parametros[(du, "C")]["p"], k_sup) + svi_w(parametros[(du, "P")]["p"], k_sup)) / 2
                  for du in vencimentos_du])
violacoes = int((np.diff(w_sup, axis=0) < 0).sum())
iv_sup = np.sqrt(w_sup / T_sup[:, None]) * 100
atm_c = [np.sqrt(svi_w(parametros[(du, "C")]["p"], 0.0) / (du / DU)) * 100 for du in vencimentos_du]
atm_p = [np.sqrt(svi_w(parametros[(du, "P")]["p"], 0.0) / (du / DU)) * 100 for du in vencimentos_du]
print(f"\nPARTE 4 - violacoes de calendar spread (w decrescente em T): {violacoes} de {w_sup[1:].size} pontos")

fig = plt.figure(figsize=(16, 5.6))
fig.suptitle("Parte 4 | Estrutura a termo e superficie de volatilidade (media das superficies duais)",
             fontsize=14, fontweight="bold")
paleta_t = plt.cm.viridis(np.linspace(0.1, 0.85, len(vencimentos_du)))

ax = fig.add_subplot(1, 3, 1)
for i, du in enumerate(vencimentos_du):
    ax.plot(k_sup, w_sup[i], color=paleta_t[i], lw=2.2, label=f"{du} d.u.")
ax.set_title("Variancia total w(k,T): curvas nao se cruzam")
ax.set_xlabel("k = ln(K/F)")
ax.set_ylabel("w = IV^2 x T")
ax.legend()

ax = fig.add_subplot(1, 3, 2)
ax.plot(vencimentos_du, atm_p, "o-", color=LARANJA, lw=2.2, ms=8, label="ATM puts")
ax.plot(vencimentos_du, atm_c, "s-", color=AZUL, lw=2.2, ms=7, label="ATM calls")
ax.fill_between(vencimentos_du, atm_c, atm_p, color=CINZA, alpha=0.15, label="cunha put-call")
ax.set_title("Estrutura a termo da IV ATM")
ax.set_xlabel("Dias uteis ate o vencimento")
ax.set_ylabel("IV (% a.a.)")
ax.legend()

ax = fig.add_subplot(1, 3, 3, projection="3d")
KK, TT = np.meshgrid(k_sup, vencimentos_du)
ax.plot_surface(KK, TT, iv_sup, cmap="viridis", edgecolor="none", alpha=0.92)
ax.set_xlabel("k")
ax.set_ylabel("d.u.")
ax.set_zlabel("IV (%)")
ax.set_title("Superficie IV(k, T)")
ax.view_init(elev=24, azim=-128)
ax.tick_params(labelsize=8)
fig.tight_layout()
if SALVAR_FIGURAS:
    fig.savefig(PASTA / "04_estrutura_termo_superficie.png")

# =============================================================================
# PARTE 5 - PAYOFF DE STRADDLES
# =============================================================================
K_st, T_st, iv_st = 130.0, 21 / DU, 0.25
premio = bs_call(spot, K_st, T_st, R_NOMINAL, iv_st) + bs_put(spot, K_st, T_st, R_NOMINAL, iv_st)
S_grade = np.linspace(105, 155, 400)
payoff_venc = np.abs(S_grade - K_st) - premio
valor_hoje = bs_call(S_grade, K_st, T_st, R_NOMINAL, iv_st) + bs_put(S_grade, K_st, T_st, R_NOMINAL, iv_st) - premio
valor_iv_alta = bs_call(S_grade, K_st, T_st, R_NOMINAL, iv_st + 0.04) + bs_put(S_grade, K_st, T_st, R_NOMINAL, iv_st + 0.04) - premio

fig, eixos = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
fig.suptitle(f"Parte 5 | Straddle ATM (K = {K_st:.0f}, {int(T_st * DU)} d.u., IV {iv_st:.0%}, premio R$ {premio:.2f})",
             fontsize=14, fontweight="bold")
for ax, sinal, titulo, cor in [(eixos[0], 1, "Long straddle = comprar volatilidade", VERDE),
                               (eixos[1], -1, "Short straddle = vender volatilidade", VERMELHO)]:
    ax.plot(S_grade, sinal * payoff_venc, color=cor, lw=2.6, label="No vencimento")
    ax.plot(S_grade, sinal * valor_hoje, color=AZUL, lw=1.8, ls="--", label="Hoje (IV constante)")
    ax.plot(S_grade, sinal * valor_iv_alta, color=ROXO, lw=1.6, ls=":", label="Hoje com IV +4 p.p.")
    ax.fill_between(S_grade, 0, sinal * payoff_venc, where=sinal * payoff_venc > 0, color=VERDE, alpha=0.10)
    ax.fill_between(S_grade, 0, sinal * payoff_venc, where=sinal * payoff_venc < 0, color=VERMELHO, alpha=0.10)
    ax.axhline(0, color="black", lw=0.8)
    for be in [K_st - premio, K_st + premio]:
        ax.axvline(be, color=CINZA, ls=":", lw=1)
    ax.set_title(titulo)
    ax.set_xlabel("Preco do ativo no vencimento (R$)")
    ax.legend(loc="upper center")
eixos[0].set_ylabel("Resultado por unidade (R$)")
eixos[0].text(K_st, -premio - 2.5, "risco: theta", ha="center", color=VERMELHO, fontsize=10)
eixos[1].text(K_st, premio + 1.5, "ganho: theta", ha="center", color=VERDE, fontsize=10)
eixos[1].text(110, -8, "risco: caudas\ne salto de IV", ha="center", color=VERMELHO, fontsize=10)
fig.tight_layout()
if SALVAR_FIGURAS:
    fig.savefig(PASTA / "05_payoff_straddles.png")

# =============================================================================
# PARTE 6 - SINAL NAS DUAS PERNAS E P&L DO STRADDLE
# =============================================================================
# Um pregao em ciclos de coleta (ex.: a cada 2 minutos entre 10h00 e 17h10)
n_ciclos = 215
horarios = pd.date_range("2026-09-14 10:00", periods=n_ciclos, freq="2min")
K_op = K_barato
T_op = vencimentos_du[0] / DU
p_call = parametros[(vencimentos_du[0], "C")]["p"]
p_put = parametros[(vencimentos_du[0], "P")]["p"]

# Residuos AR(1) correlacionados + deslocamento conjunto que reverte
choque = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_ciclos)
res_c = np.zeros(n_ciclos)
res_p = np.zeros(n_ciclos)
for t in range(1, n_ciclos):
    res_c[t] = 0.70 * res_c[t - 1] + 0.0045 * choque[t, 0]
    res_p[t] = 0.70 * res_p[t - 1] + 0.0040 * choque[t, 1]
inicio_evento = 120
perfil = np.zeros(n_ciclos)
perfil[inicio_evento:] = -0.040 * np.exp(-np.arange(n_ciclos - inicio_evento) / 30)
res_c += perfil * 1.05
res_p += perfil

# Caminho intradiario do ativo
spot_t = spot * np.exp(np.cumsum(np.r_[0, 0.25 * np.sqrt(2 / (60 * 7 * DU)) * np.random.randn(n_ciclos - 1)]))

# Z-score com historico ate t-1 e limiar calibrado por serie (piso 1.5, teto 3.0)
z_c = np.full(n_ciclos, np.nan)
z_p = np.full(n_ciclos, np.nan)
lim_c = np.full(n_ciclos, np.nan)
lim_p = np.full(n_ciclos, np.nan)
for t in range(MIN_OBS, n_ciclos):
    for serie, z_arr, lim_arr in [(res_c, z_c, lim_c), (res_p, z_p, lim_p)]:
        hist = serie[:t]
        z_arr[t] = (serie[t] - hist.mean()) / hist.std(ddof=1)
        cv = np.abs(hist).std(ddof=1) / np.abs(hist).mean()
        lim_arr[t] = np.clip(LIMIAR_PISO * (1 + cv), LIMIAR_PISO, LIMIAR_TETO)

# Maquina de estados do straddle
posicao = None
registro = []
for t in range(n_ciclos):
    T_rest = T_op - t * 2 / (60 * 7 * DU)
    fwd_t = spot_t[t] * np.exp(R_NOMINAL * T_rest)
    k_t = np.log(K_op / fwd_t)
    iv_c = np.sqrt(svi_w(p_call, k_t) / T_op) + res_c[t]
    iv_p = np.sqrt(svi_w(p_put, k_t) / T_op) + res_p[t]
    mid_c = bs_call(spot_t[t], K_op, T_rest, R_NOMINAL, iv_c)
    mid_p = bs_put(spot_t[t], K_op, T_rest, R_NOMINAL, iv_p)
    meio_c, meio_p = max(mid_c * 0.012, 0.01), max(mid_p * 0.012, 0.01)
    sinal_t = "NEUTRO"
    if not np.isnan(z_c[t]):
        if z_c[t] < -lim_c[t] and z_p[t] < -lim_p[t]:
            sinal_t = "BUY_VOL"
        elif z_c[t] > lim_c[t] and z_p[t] > lim_p[t]:
            sinal_t = "SELL_VOL"
    pnl = np.nan
    evento = ""
    if posicao is None and sinal_t == "BUY_VOL":
        custo_unit = (mid_c + meio_c) + (mid_p + meio_p)
        contratos = int(CAPITAL_MAX / custo_unit // LOTE * LOTE)
        posicao = dict(t=t, custo=custo_unit, n=contratos)
        evento = "ENTRADA"
        pnl = contratos * ((mid_c - meio_c) + (mid_p - meio_p) - custo_unit)
    elif posicao is not None:
        valor_saida = (mid_c - meio_c) + (mid_p - meio_p)
        pnl = posicao["n"] * (valor_saida - posicao["custo"])
        retorno = valor_saida / posicao["custo"] - 1
        if retorno >= ALVO_LUCRO:
            evento = "SAIDA: alvo"
        elif retorno <= -STOP_PERDA:
            evento = "SAIDA: stop"
        elif abs(z_c[t]) < Z_SAIDA and abs(z_p[t]) < Z_SAIDA and t - posicao["t"] > 3:
            evento = "SAIDA: reversao"
        elif t == n_ciclos - 1:
            evento = "SAIDA: fim do pregao"
        if evento:
            posicao["saida"] = t
            posicao["retorno"] = retorno
            posicao["pnl"] = pnl
            posicao["motivo"] = evento
            operacao = posicao
            posicao = None
    registro.append(dict(hora=horarios[t], spot=spot_t[t], iv_c=iv_c, iv_p=iv_p, sinal=sinal_t,
                         pnl=pnl, evento=evento))
trilha = pd.DataFrame(registro)

print("\nPARTE 6 - straddle no strike R$ {:.2f} ({} d.u.)".format(K_op, vencimentos_du[0]))
print(f"  Ciclos com sinal BUY_VOL: {(trilha['sinal'] == 'BUY_VOL').sum()} de {n_ciclos}")
if "operacao" in dir():
    print(f"  Entrada: {horarios[operacao['t']]:%H:%M} | {operacao['n']} contratos de cada perna"
          f" | premio pago R$ {operacao['custo'] * operacao['n']:,.0f}")
    print(f"  Saida:   {horarios[operacao['saida']]:%H:%M} | {operacao['motivo']}"
          f" | retorno {operacao['retorno']:+.1%} | P&L R$ {operacao['pnl']:,.0f}")

fig, eixos = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                          gridspec_kw=dict(height_ratios=[1, 1.1, 1]))
fig.suptitle(f"Parte 6 | Estrategia: sinal simultaneo nas duas pernas e straddle no strike R$ {K_op:.2f}",
             fontsize=14, fontweight="bold")
zona = trilha["sinal"] == "BUY_VOL"

ax = eixos[0]
ax.plot(trilha["hora"], res_c * 100, color=AZUL, lw=1.5, label="Residuo call vs SVI calls")
ax.plot(trilha["hora"], res_p * 100, color=LARANJA, lw=1.5, label="Residuo put vs SVI puts")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("p.p. de vol")
ax.set_title("Residuos em relacao as superficies duais: as duas pernas ficam baratas ao mesmo tempo")
ax.legend(loc="lower left")

ax = eixos[1]
ax.plot(trilha["hora"], z_c, color=AZUL, lw=1.4, label="z call")
ax.plot(trilha["hora"], z_p, color=LARANJA, lw=1.4, label="z put")
ax.plot(trilha["hora"], -lim_c, color=AZUL, ls="--", lw=1, alpha=0.7, label="limiar call")
ax.plot(trilha["hora"], -lim_p, color=LARANJA, ls="--", lw=1, alpha=0.7, label="limiar put")
y0, y1 = ax.get_ylim()
ax.axvspan(horarios[0], horarios[MIN_OBS], color=CINZA, alpha=0.12)
ax.text(horarios[MIN_OBS // 2], y1 - 0.25 * (y1 - y0), f"aquecimento\n({MIN_OBS} obs)",
        ha="center", fontsize=9, color=CINZA)
ax.fill_between(trilha["hora"], y0, y1, where=zona, color=VERDE, alpha=0.18, step="mid")
ax.set_ylim(y0, y1)
ax.set_ylabel("Z-score")
ax.set_title("Sinal BUY_VOL (faixa verde) so quando z_call e z_put rompem os respectivos limiares")
ax.legend(loc="lower left", ncol=4)

ax = eixos[2]
ax.plot(trilha["hora"], trilha["pnl"], color=ROXO, lw=2)
ax.fill_between(trilha["hora"], 0, trilha["pnl"], where=trilha["pnl"] > 0, color=VERDE, alpha=0.2)
ax.fill_between(trilha["hora"], 0, trilha["pnl"], where=trilha["pnl"] < 0, color=VERMELHO, alpha=0.2)
ax.axhline(0, color="black", lw=0.8)
for _, linha in trilha[trilha["evento"] != ""].iterrows():
    cor = VERDE if linha["evento"] == "ENTRADA" else AZUL
    ax.scatter(linha["hora"], linha["pnl"], s=90, color=cor, zorder=5, edgecolor="white")
    ax.annotate(f"{linha['evento']}\nR$ {linha['pnl']:,.0f}", (linha["hora"], linha["pnl"]),
                textcoords="offset points", xytext=(10, -28) if linha["evento"] == "ENTRADA" else (-150, -8),
                fontsize=9, color=cor, fontweight="bold")
ax.set_ylabel("P&L (R$)")
ax.set_title(f"P&L marcado a mercado (compra no ask, saida no bid, capital maximo R$ {CAPITAL_MAX:,.0f})")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
ax.set_xlabel("Horario do pregao")
fig.tight_layout()
if SALVAR_FIGURAS:
    fig.savefig(PASTA / "06_sinal_e_pnl_straddle.png")

print("\n" + "=" * 78)
print(f"Figuras salvas em: {PASTA.resolve()}")
print("Material didatico: dados simulados, sem custos de corretagem e emolumentos.")
print("=" * 78)
plt.show()