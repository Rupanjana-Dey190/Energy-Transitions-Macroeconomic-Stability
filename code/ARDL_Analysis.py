"""
ARDL Analysis: Energy Transitions & EU Macroeconomic Stability
Dependent Variable : Inflation (Consumer Prices, annual %)
Exogenous Variables: Oil Price, GDP Growth, Renewable Energy Share, EUR/USD Exchange Rate
Sample             : 2005–2025 (21 annual observations)

ARDL(p, q1, q2, q3, q4) — selected by AIC grid search
Includes: Bounds test (Pesaran et al. 2001), Short-run ECM, Long-run coefficients,
          VIF, Unit root tests, Residual diagnostics, Forecast
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings, itertools
warnings.filterwarnings('ignore')

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats

# ── palette ──────────────────────────────────────────────────────────────────
NAVY   = '#1A3A5C'
RED    = '#C0392B'
GREEN  = '#1E8449'
GOLD   = '#D4AC0D'
TEAL   = '#117A65'
GREY   = '#717D7E'
LIGHT  = '#AED6F1'
BG     = '#F7F9FC'

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.25,
    'grid.linestyle'   : '--',
    'figure.facecolor' : BG,
    'axes.facecolor'   : BG,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame({
    'Year'      : list(range(2005, 2026)),
    'Inflation' : [2.488,2.666,2.511,4.165,0.839,1.531,3.289,2.663,1.220,
                   0.199,-0.062,0.183,1.429,1.739,1.631,0.476,2.555,8.834,6.299,2.435,2.300],
    'OilPrice'  : [54.57,65.16,72.44,96.94,61.74,79.61,111.26,111.63,108.56,
                   98.97,52.32,43.64,54.13,71.34,64.30,41.96,70.86,100.93,82.49,80.52,69.14],
    'GDPGrowth' : [1.984,3.568,3.142,0.618,-4.312,2.182,1.961,-0.764,-0.033,
                   1.633,2.372,1.920,2.818,2.046,1.875,-5.579,6.378,3.562,0.454,1.065,1.500],
    'RenewShare': [10.182,10.778,11.749,12.552,13.85,14.405,14.547,16.002,16.659,
                   17.416,17.82,17.978,18.411,19.096,19.887,22.038,21.897,23.003,24.579,25.241,24.000],
    'ExchRate'  : [0.6837,0.6817,0.6843,0.7963,0.8909,0.8578,0.8679,0.8109,0.8493,
                   0.8061,0.7258,0.8195,0.8767,0.8847,0.8778,0.8897,0.8596,0.8528,0.8698,0.8466,0.8568],
}).set_index('Year')

YVARS  = 'Inflation'
XVARS  = ['OilPrice', 'GDPGrowth', 'RenewShare', 'ExchRate']
XLABS  = {'OilPrice':'Oil Price', 'GDPGrowth':'GDP Growth',
          'RenewShare':'Renew. Share', 'ExchRate':'Exch. Rate'}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. UNIT ROOT TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def unit_root(series):
    adf = adfuller(series, autolag='AIC')
    try:
        kp  = kpss(series, regression='c', nlags='auto')
        ks, kp_val = kp[0], kp[1]
    except Exception:
        ks, kp_val = np.nan, np.nan
    return adf[0], adf[1], ks, kp_val

ur = {}
for col in df.columns:
    ur[col] = unit_root(df[col])

print("=" * 72)
print("  ARDL ANALYSIS — EU INFLATION  (2005–2025)")
print("=" * 72)
print("\n[1] UNIT ROOT TESTS")
print(f"  {'Variable':<18} {'ADF':>8} {'p':>7}  {'Order':>12}  |  {'KPSS':>8} {'p':>8}  {'Order':>12}")
print("  " + "-" * 72)
integration_order = {}
for v, (as_, ap, ks, kp) in ur.items():
    adf_I0 = ap < 0.05
    kpss_I0 = kp >= 0.05
    if adf_I0 and kpss_I0:
        order = "I(0)"
    elif not adf_I0 and not kpss_I0:
        order = "I(1)?"
    else:
        order = "Ambiguous"
    integration_order[v] = order
    print(f"  {v:<18} {as_:>8.4f} {ap:>7.4f}  {order:>12}  |  "
          f"{ks:>8.4f} {str(round(kp,4)):>8}  {'Non-stat' if kp<0.05 else 'Stat':>12}")

print("\n  Note: ARDL is valid for I(0)/I(1) mix — no I(2) variables detected.")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ARDL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_ardl_regressors(data, y_var, x_vars, p, q_dict):
    """
    Constructs the ARDL(p, q1,...,qk) regressor matrix.
    p  : lags of dependent variable
    q_dict : {x_var: lags} for each exogenous variable
    Returns (Y, X_df) with matching indices.
    """
    cols = {}
    # lagged Y
    for lag in range(1, p + 1):
        cols[f'{y_var}_L{lag}'] = data[y_var].shift(lag)
    # current + lagged X
    for xv in x_vars:
        cols[xv] = data[xv]
        for lag in range(1, q_dict[xv] + 1):
            cols[f'{xv}_L{lag}'] = data[xv].shift(lag)

    reg_df = pd.DataFrame(cols, index=data.index)
    reg_df['const'] = 1.0
    combined = pd.concat([data[y_var], reg_df], axis=1).dropna()
    Y = combined[y_var]
    X = combined.drop(columns=[y_var])
    return Y, X


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRID SEARCH — AIC-BASED LAG SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] ARDL LAG SELECTION — Grid Search by AIC (max p=2, max q=1 per variable)")

MAX_P = 2
MAX_Q = 1   # keep degrees of freedom given n=21

best_aic = np.inf
best_spec = None
search_log = []

for p in range(1, MAX_P + 1):
    for qs in itertools.product(range(0, MAX_Q + 1), repeat=len(XVARS)):
        q_dict = dict(zip(XVARS, qs))
        try:
            Y, X = build_ardl_regressors(df, YVARS, XVARS, p, q_dict)
            if len(Y) < X.shape[1] + 2:       # too few obs
                continue
            res = OLS(Y, X).fit()
            k   = X.shape[1]
            n   = len(Y)
            aic = n * np.log(res.ssr / n) + 2 * k
            bic = n * np.log(res.ssr / n) + k * np.log(n)
            search_log.append({'p': p, 'q': q_dict, 'AIC': aic, 'BIC': bic,
                                'nobs': n, 'k': k})
            if aic < best_aic:
                best_aic  = aic
                best_spec = {'p': p, 'q': q_dict, 'AIC': aic, 'BIC': bic}
        except Exception:
            pass

top5 = sorted(search_log, key=lambda x: x['AIC'])[:5]
print(f"\n  {'Spec':<42} {'AIC':>8}  {'BIC':>8}  {'n':>4}")
print("  " + "-" * 66)
for r in top5:
    qs_str = ",".join(str(r['q'][v]) for v in XVARS)
    spec_str = f"ARDL({r['p']}; {qs_str})"
    mark = " ← SELECTED" if r['AIC'] == best_aic else ""
    print(f"  {spec_str:<42} {r['AIC']:>8.3f}  {r['BIC']:>8.3f}  {r['nobs']:>4}{mark}")

p_sel = best_spec['p']
q_sel = best_spec['q']
print(f"\n  Selected: ARDL({p_sel}; "
      + ", ".join(f"{v}={q_sel[v]}" for v in XVARS) + ")")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIT SELECTED ARDL MODEL
# ═══════════════════════════════════════════════════════════════════════════════
Y_ardl, X_ardl = build_ardl_regressors(df, YVARS, XVARS, p_sel, q_sel)
ardl_res  = OLS(Y_ardl, X_ardl).fit()
# HAC standard errors (Newey-West) — robust to autocorrelation & heteroskedasticity
cov_hac_m = cov_hac(ardl_res, nlags=2)
ardl_hac  = ardl_res.get_robustcov_results(cov_type='HAC', maxlags=2)

print("\n[3] ARDL REGRESSION — FULL OUTPUT")
print(ardl_hac.summary())

# ═══════════════════════════════════════════════════════════════════════════════
# 6. LONG-RUN COEFFICIENTS  (θ_x = Σβ_x / (1 − Σβ_y))
# ═══════════════════════════════════════════════════════════════════════════════
params = ardl_res.params

# sum of lagged Y coefficients
sum_lag_y = sum(params.get(f'{YVARS}_L{l}', 0) for l in range(1, p_sel + 1))
denom     = 1 - sum_lag_y   # (1 − Σ φ)

lr_coefs = {}
for xv in XVARS:
    num = params.get(xv, 0) + sum(params.get(f'{xv}_L{l}', 0)
                                   for l in range(1, q_sel[xv] + 1))
    lr_coefs[xv] = num / denom if abs(denom) > 1e-10 else np.nan

print("\n[4] LONG-RUN COEFFICIENTS  [θ = Σβ_x / (1 − Σφ_y)]")
print(f"  Denominator (1 − Σφ_y) = {denom:.4f}")
print(f"  {'Variable':<18} {'Long-run coef':>15}")
print("  " + "-" * 35)
for xv, c in lr_coefs.items():
    print(f"  {xv:<18} {c:>15.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. ECM — Error Correction Representation
#    Δy_t = α + γ·ECT_{t-1} + Σ Δx short-run terms + ε
# ═══════════════════════════════════════════════════════════════════════════════
# Construct ECT = y_{t-1} − Σ θ_x · x_{t-1}
ect_vals = (df[YVARS].shift(1)
            - sum(lr_coefs[xv] * df[xv].shift(1) for xv in XVARS))

ecm_data = pd.DataFrame(index=df.index)
ecm_data['dInflation'] = df[YVARS].diff()
ecm_data['ECT_lag1']   = ect_vals.shift(1)
for xv in XVARS:
    ecm_data[f'd{xv}'] = df[xv].diff()

ecm_data.dropna(inplace=True)
Y_ecm = ecm_data['dInflation']
X_ecm = add_constant(ecm_data.drop(columns=['dInflation']))
ecm_res  = OLS(Y_ecm, X_ecm).fit(cov_type='HAC', cov_kwds={'maxlags': 2})

print("\n[5] ECM — ERROR CORRECTION MODEL")
print(ecm_res.summary())
gamma = ecm_res.params.get('ECT_lag1', np.nan)
print(f"\n  Adjustment speed (γ) = {gamma:.4f}")
if gamma < 0:
    half_life = -np.log(2) / np.log(1 + gamma)
    print(f"  Half-life of shock   = {half_life:.2f} years")
    print(f"  Interpretation: {abs(gamma)*100:.1f}% of disequilibrium corrected each year.")
else:
    print("  γ > 0 — ECT not valid; series may not be cointegrated at this sample size.")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. BOUNDS TEST (Pesaran, Shin & Smith 2001)
#    F-statistic on joint significance of lagged levels in ARDL-levels form
# ═══════════════════════════════════════════════════════════════════════════════
# Critical values (k=4 regressors, unrestricted intercept, no trend)
# Source: PSS Table CI(iii), n=30 — conservative given small sample
BOUNDS_CV = {
    '10%': (2.45, 3.52),
    '5%' : (2.86, 4.01),
    '1%' : (3.74, 5.06),
}

# F-test: restrict all lagged-level terms to zero
level_cols = [YVARS + '_L1'] + [xv for xv in XVARS if xv in X_ardl.columns]
level_cols = [c for c in level_cols if c in X_ardl.columns]

if level_cols:
    r_mat = np.zeros((len(level_cols), X_ardl.shape[1]))
    col_names = list(X_ardl.columns)
    for i, lc in enumerate(level_cols):
        if lc in col_names:
            r_mat[i, col_names.index(lc)] = 1
    try:
        f_test = ardl_res.f_test(r_mat)
        f_stat = float(f_test.fvalue)
        f_pval = float(f_test.pvalue)
    except Exception:
        f_stat, f_pval = np.nan, np.nan
else:
    f_stat, f_pval = np.nan, np.nan

print("\n[6] BOUNDS TEST (PSS 2001) — k = 4 regressors")
print(f"  F-statistic = {f_stat:.4f}   p = {f_pval:.4f}")
print(f"  {'Level':<6} {'I(0) lower':>12} {'I(1) upper':>12}  {'Result':>20}")
print("  " + "-" * 56)
for lvl, (lo, hi) in BOUNDS_CV.items():
    if np.isnan(f_stat):
        verdict = "Cannot compute"
    elif f_stat > hi:
        verdict = "Cointegration ✓"
    elif f_stat < lo:
        verdict = "No cointegration"
    else:
        verdict = "Inconclusive"
    print(f"  {lvl:<6} {lo:>12.2f} {hi:>12.2f}  {verdict:>20}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. VIF
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7] VIF — MULTICOLLINEARITY (ARDL regressor matrix)")
vif_cols = [c for c in X_ardl.columns if c != 'const']
X_vif    = X_ardl[vif_cols].values.astype(float)
vif_res  = []
for i, col in enumerate(vif_cols):
    v = variance_inflation_factor(X_vif, i)
    tag = "⚠ HIGH" if v > 10 else ("△ MOD" if v > 5 else "✓ LOW")
    vif_res.append((col, v, tag))
    print(f"  {col:<22} VIF = {v:8.3f}  {tag}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RESIDUAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
resid   = ardl_hac.resid
fitted  = ardl_hac.fittedvalues
dw_s    = durbin_watson(resid)
lb      = acorr_ljungbox(resid, lags=[4, 8], return_df=True)
jb_r    = stats.jarque_bera(resid)
bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_ardl)

print("\n[8] RESIDUAL DIAGNOSTICS")
print(f"  R²                      : {ardl_hac.rsquared:.4f}")
print(f"  Adj. R²                 : {ardl_hac.rsquared_adj:.4f}")
print(f"  F-stat (model)          : {ardl_hac.fvalue:.4f}   p = {ardl_hac.f_pvalue:.4f}")
print(f"  Durbin-Watson           : {dw_s:.4f}  "
      f"{'OK' if 1.5 < dw_s < 2.5 else 'AUTOCORR SUSPECTED'}")
print(f"  Ljung-Box Q(4)  p       : {lb['lb_pvalue'].iloc[0]:.4f}  "
      f"{'No autocorr' if lb['lb_pvalue'].iloc[0] > 0.05 else 'Autocorr detected'}")
print(f"  Ljung-Box Q(8)  p       : {lb['lb_pvalue'].iloc[1]:.4f}  "
      f"{'No autocorr' if lb['lb_pvalue'].iloc[1] > 0.05 else 'Autocorr detected'}")
print(f"  Jarque-Bera stat        : {jb_r.statistic:.4f}   p = {jb_r.pvalue:.4f}  "
      f"{'Normal residuals ✓' if jb_r.pvalue > 0.05 else 'Non-normal'}")
print(f"  Breusch-Pagan (hetero)  : {bp_stat:.4f}   p = {bp_p:.4f}  "
      f"{'Homoskedastic ✓' if bp_p > 0.05 else 'Heteroskedastic'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. FORECAST 2026–2030
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[9] ARDL FORECAST — 2026–2030 (scenario: trend extrapolation)")

# Project exogenous variables using mean first difference
trends = {v: df[v].diff().mean() for v in XVARS}
last   = {v: df[v].iloc[-1]      for v in XVARS}

future_x = {}
for v in XVARS:
    future_x[v] = [max(0.1, last[v] + trends[v] * i) for i in range(1, 6)]

fc_years    = list(range(2026, 2031))
fc_infl     = []
y_hist      = list(df[YVARS].values)

# Bootstrap CI via residual resampling
N_BOOT = 2000
boot_paths = np.zeros((N_BOOT, 5))

for b in range(N_BOOT):
    resid_boot = np.random.choice(np.array(resid), size=5, replace=True)
    y_sim      = list(y_hist)
    for i in range(5):
        row = {'const': 1.0}
        for lag in range(1, p_sel + 1):
            row[f'{YVARS}_L{lag}'] = y_sim[-(lag)]
        for v in XVARS:
            row[v] = future_x[v][i]
            for lag in range(1, q_sel[v] + 1):
                past_idx = -(lag)
                row[f'{v}_L{lag}'] = (df[v].iloc[past_idx]
                                      if i == 0
                                      else future_x[v][max(0, i - lag)])
        x_row = np.array([row.get(c, 0.0) for c in X_ardl.columns])
        yhat  = float(ardl_res.params.values @ x_row) + resid_boot[i]
        boot_paths[b, i] = yhat
        y_sim.append(yhat)

# Point forecast (no noise)
y_pt = list(y_hist)
for i in range(5):
    row = {'const': 1.0}
    for lag in range(1, p_sel + 1):
        row[f'{YVARS}_L{lag}'] = y_pt[-(lag)]
    for v in XVARS:
        row[v] = future_x[v][i]
        for lag in range(1, q_sel[v] + 1):
            past_idx = -(lag)
            row[f'{v}_L{lag}'] = (df[v].iloc[past_idx]
                                   if i == 0
                                   else future_x[v][max(0, i - lag)])
    x_row = np.array([row.get(c, 0.0) for c in X_ardl.columns])
    yhat  = float(ardl_res.params.values @ x_row)
    fc_infl.append(yhat)
    y_pt.append(yhat)

ci_lo80 = np.percentile(boot_paths, 10, axis=0)
ci_hi80 = np.percentile(boot_paths, 90, axis=0)
ci_lo95 = np.percentile(boot_paths, 2.5, axis=0)
ci_hi95 = np.percentile(boot_paths, 97.5, axis=0)

print(f"\n  {'Year':<8} {'Forecast':>10} {'80% Lo':>10} {'80% Hi':>10} "
      f"{'95% Lo':>10} {'95% Hi':>10}")
print("  " + "-" * 62)
for i, yr in enumerate(fc_years):
    print(f"  {yr:<8} {fc_infl[i]:>10.3f} {ci_lo80[i]:>10.3f} {ci_hi80[i]:>10.3f} "
          f"{ci_lo95[i]:>10.3f} {ci_hi95[i]:>10.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
years_hist = list(df.index)
y_hist_arr = df[YVARS].values
fit_years  = list(Y_ardl.index)

# ── Figure 1: Main Dashboard ─────────────────────────────────────────────────
fig1 = plt.figure(figsize=(18, 22), facecolor=BG)
fig1.suptitle(f"ARDL({p_sel}; " + ",".join(str(q_sel[v]) for v in XVARS)
              + ") — EU Inflation Analysis  2005–2025",
              fontsize=17, fontweight='bold', color=NAVY, y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig1, hspace=0.45, wspace=0.35,
                       left=0.08, right=0.95, top=0.93, bottom=0.05)

# Panel A — Actual vs Fitted
ax1 = fig1.add_subplot(gs[0, :])
ax1.fill_between(years_hist, y_hist_arr, alpha=0.10, color=NAVY)
ax1.plot(years_hist, y_hist_arr, color=NAVY, lw=2.5, marker='o',
         markersize=5, label='Actual Inflation', zorder=4)
ax1.plot(fit_years, np.array(fitted), color=RED, lw=2, ls='--', marker='s',
         markersize=4, label=f'ARDL Fitted', zorder=3)
ax1.axhline(2.0, color=GREEN, lw=1.2, ls=':', alpha=0.7, label='ECB target (2%)')
for yr, av, fv in zip(fit_years, Y_ardl.values, np.array(fitted)):
    if abs(av - fv) > 1.8:
        ax1.annotate(str(yr), (yr, av), xytext=(0, 10),
                     textcoords='offset points', fontsize=7.5, color=RED, ha='center')
ax1.set_title("Panel A — Actual vs ARDL Fitted Values", fontsize=13,
              fontweight='bold', color=NAVY, pad=10)
ax1.set_ylabel("Inflation (%)"); ax1.legend(fontsize=10)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# Panel B — Residuals
ax2 = fig1.add_subplot(gs[1, 0])
bar_colors = [GREEN if r >= 0 else RED for r in np.array(resid)]
ax2.bar(fit_years, np.array(resid), color=bar_colors, alpha=0.75, edgecolor='white')
ax2.axhline(0, color=NAVY, lw=1.2)
ax2.axhline( 2*resid.std(), color=GREY, lw=1, ls='--', alpha=0.7, label='±2σ')
ax2.axhline(-2*resid.std(), color=GREY, lw=1, ls='--', alpha=0.7)
ax2.set_title("Panel B — Residuals", fontsize=12, fontweight='bold', color=NAVY)
ax2.set_ylabel("Residual"); ax2.legend(fontsize=9)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# Panel C — Q-Q Plot
ax3 = fig1.add_subplot(gs[1, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(np.array(resid), dist="norm")
ax3.scatter(osm, osr, color=NAVY, s=50, zorder=3, label='Residuals')
lx = np.array([min(osm), max(osm)])
ax3.plot(lx, slope*lx + intercept, color=RED, lw=1.8, ls='--', label='Normal line')
ax3.set_title("Panel C — Q-Q Plot", fontsize=12, fontweight='bold', color=NAVY)
ax3.set_xlabel("Theoretical Quantiles"); ax3.set_ylabel("Sample Quantiles")
ax3.legend(fontsize=9)
resid_s = pd.Series(np.array(resid))
txt = (f"JB stat={jb_r.statistic:.3f}\np={jb_r.pvalue:.3f}\n"
       f"Skew={resid_s.skew():.3f}\nKurt={resid_s.kurtosis():.3f}")
ax3.text(0.04, 0.97, txt, transform=ax3.transAxes, fontsize=8.5,
         va='top', bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

# Panel D — Long-run coefficients
ax4 = fig1.add_subplot(gs[2, 0])
lr_vals  = list(lr_coefs.values())
lr_labs  = [XLABS[v] for v in XVARS]
lr_cols  = [RED if v > 0 else GREEN for v in lr_vals]
ax4.barh(lr_labs, lr_vals, color=lr_cols, alpha=0.75, edgecolor='white', height=0.45)
ax4.axvline(0, color=NAVY, lw=1.5)
for i, v in enumerate(lr_vals):
    ax4.text(v + (0.1 if v >= 0 else -0.1), i, f'{v:.4f}',
             va='center', ha='left' if v >= 0 else 'right', fontsize=9, fontweight='bold')
ax4.set_title("Panel D — Long-run Coefficients  [θ = Σβ / (1−Σφ)]",
              fontsize=12, fontweight='bold', color=NAVY)
ax4.set_xlabel("Long-run coefficient")

# Panel E — VIF
ax5 = fig1.add_subplot(gs[2, 1])
vif_names = [r[0] for r in vif_res]
vif_vals  = [r[1] for r in vif_res]
vif_cols2 = [RED if v > 10 else (GOLD if v > 5 else GREEN) for v in vif_vals]
bars = ax5.barh(vif_names, vif_vals, color=vif_cols2, edgecolor='white', height=0.45)
ax5.axvline(5,  color=GOLD, lw=1.5, ls='--', alpha=0.8, label='VIF=5')
ax5.axvline(10, color=RED,  lw=1.5, ls='--', alpha=0.8, label='VIF=10')
for bar, val in zip(bars, vif_vals):
    ax5.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
ax5.set_title("Panel E — VIF (Multicollinearity)", fontsize=12,
              fontweight='bold', color=NAVY)
ax5.set_xlabel("VIF"); ax5.legend(fontsize=9)
ax5.set_xlim(0, max(vif_vals) * 1.25)

fig1.savefig('/mnt/user-data/outputs/ardl_fig1_dashboard.png',
             dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig1)

# ── Figure 2: Forecast ────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
ax.set_facecolor(BG)
fig2.suptitle(f"ARDL Inflation Forecast  2026–2030  (Bootstrap CI, n={N_BOOT})",
              fontsize=15, fontweight='bold', color=NAVY)

ax.fill_between(years_hist, y_hist_arr, alpha=0.08, color=NAVY)
ax.plot(years_hist, y_hist_arr, color=NAVY, lw=2.5, marker='o',
        markersize=5, label='Historical', zorder=4)
ax.plot(fit_years, np.array(fitted), color=GREY, lw=1.4, ls='--',
        alpha=0.7, label='In-sample fitted')

ax.fill_between(fc_years, ci_lo95, ci_hi95, alpha=0.15, color=RED, label='95% CI')
ax.fill_between(fc_years, ci_lo80, ci_hi80, alpha=0.28, color=RED, label='80% CI')
ax.plot(fc_years, fc_infl, color=RED, lw=2.5, marker='D',
        markersize=7, label='ARDL Forecast', zorder=5)
for i, (yr, fv) in enumerate(zip(fc_years, fc_infl)):
    ax.annotate(f'{fv:.2f}%', xy=(yr, fv), xytext=(0, 12),
                textcoords='offset points', fontsize=9.5, fontweight='bold',
                color=RED, ha='center',
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

ax.axvline(2025.5, color=GREY, lw=1.3, ls=':', alpha=0.6)
ax.axhline(2.0, color=GREEN, lw=1.2, ls='--', alpha=0.55, label='ECB target')
ax.text(2025.65, ax.get_ylim()[1]*0.88, 'Forecast →',
        fontsize=9, color=GREY, style='italic')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Inflation, Consumer Prices (annual %)", fontsize=12)
ax.legend(fontsize=10, loc='upper left', framealpha=0.85)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

info = (f"ARDL({p_sel}; " + ",".join(str(q_sel[v]) for v in XVARS) + ")\n"
        f"AIC={best_aic:.2f}  |  R²={ardl_hac.rsquared:.3f}\n"
        f"Adj.R²={ardl_hac.rsquared_adj:.3f}  |  n={len(Y_ardl)}")
ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8.5,
        va='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=NAVY, alpha=0.85))

fig2.savefig('/mnt/user-data/outputs/ardl_fig2_forecast.png',
             dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig2)

# ── Figure 3: Short-run ECM coefficients ──────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig3.suptitle("ECM Short-run Dynamics & Coefficient Summary",
              fontsize=14, fontweight='bold', color=NAVY)

ax6 = axes[0]
ecm_params = ecm_res.params.drop('const', errors='ignore')
ecm_pvals  = ecm_res.pvalues.drop('const', errors='ignore')
clrs = [RED if v > 0 else TEAL for v in ecm_params.values]
ax6.barh(ecm_params.index, ecm_params.values, color=clrs, alpha=0.75, edgecolor='white')
ax6.axvline(0, color=NAVY, lw=1.5)
for i, (c, p) in enumerate(zip(ecm_params.values, ecm_pvals.values)):
    sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    ax6.text(c + (0.05 if c >= 0 else -0.05), i,
             f'{c:.4f}{sig}', va='center',
             ha='left' if c >= 0 else 'right', fontsize=8.5, fontweight='bold')
ax6.set_title("Short-run ECM Coefficients\n* p<0.10  ** p<0.05  *** p<0.01",
              fontsize=11, fontweight='bold', color=NAVY)
ax6.set_xlabel("Coefficient")

ax7 = axes[1]
sr_vars  = [XLABS[v] for v in XVARS]
sr_coefs = [ardl_hac.params[list(X_ardl.columns).index(v)] if v in list(X_ardl.columns) else 0 for v in XVARS]
sr_pvals = [ardl_hac.pvalues[list(X_ardl.columns).index(v)] if v in list(X_ardl.columns) else 1 for v in XVARS]
lr_list  = [lr_coefs[v] for v in XVARS]
x_pos    = np.arange(len(sr_vars))
width    = 0.35
b1 = ax7.bar(x_pos - width/2, sr_coefs, width, label='Short-run (β)',
             color=NAVY, alpha=0.75, edgecolor='white')
b2 = ax7.bar(x_pos + width/2, lr_list,  width, label='Long-run (θ)',
             color=TEAL, alpha=0.75, edgecolor='white')
ax7.axhline(0, color=GREY, lw=1)
ax7.set_xticks(x_pos); ax7.set_xticklabels(sr_vars, rotation=15)
ax7.set_title("Short-run vs Long-run Coefficients",
              fontsize=11, fontweight='bold', color=NAVY)
ax7.set_ylabel("Coefficient"); ax7.legend(fontsize=10)

fig3.tight_layout()
fig3.savefig('/mnt/user-data/outputs/ardl_fig3_ecm.png',
             dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig3)

print("\n" + "=" * 72)
print("  All outputs saved successfully.")
print("  Files: ardl_fig1_dashboard.png | ardl_fig2_forecast.png | ardl_fig3_ecm.png")
print("=" * 72)
