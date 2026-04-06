"""
β无量纲化验证：有量纲 vs 无量纲公式对比
========================================
铁新发现：NPM用(ρ-ρ_c)^β拟合，经典渗流用((ρ-ρ_c)/ρ_c)^β。
少了除以ρ_c这一步，可能导致β被压低。

本脚本用原论文14个模型数据，分别拟合两版公式，比较β值。
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr

# ══════════════════════════════════════════════════════
# 原论文14个模型数据（和npm_param_fit.py一致）
# ══════════════════════════════════════════════════════
models = [
    ("GPT-2 (1.5B)",          40,    0.70,   1.5,   35),
    ("Pythia-1B",             300,   0.70,   1.0,   27),
    ("Pythia-6.9B",           300,   0.70,   6.9,   35),
    ("Pythia-12B",            300,   0.70,  12.0,   38),
    ("LLaMA-7B",             1000,   0.75,   7.0,   35),
    ("LLaMA-13B",            1000,   0.75,  13.0,   47),
    ("LLaMA-33B",            1400,   0.75,  33.0,   58),
    ("LLaMA-65B",            1400,   0.75,  65.0,   64),
    ("GPT-3 (175B)",          300,   0.75, 175.0,   44),
    ("Chinchilla (70B)",     1400,   0.75,  70.0,   68),
    ("LLaMA-2-70B",          2000,   0.75,  70.0,   69),
    ("CodeLlama-7B",          500,   0.30,   7.0,   31),
    ("CodeLlama-34B",         500,   0.30,  34.0,   42),
    ("BLOOM-176B",            366,   0.85, 176.0,   39),
]

N_data = np.array([m[1] for m in models], dtype=float)
sigma_data = np.array([m[2] for m in models], dtype=float)
mmlu_data = np.array([m[4] for m in models], dtype=float)
names = [m[0] for m in models]


# ══════════════════════════════════════════════════════
# 方案A：有量纲公式（当前NPM）
# C_eff = (ρ - ρ_c)^β
# ══════════════════════════════════════════════════════
def ceff_raw(N_B, sigma, alpha, beta, delta, k):
    rho = N_B / np.exp(alpha * sigma)
    rho_c = k * sigma**delta
    if rho <= rho_c:
        return 0.0
    return (rho - rho_c)**beta

def ceff_raw_vec(N_arr, s_arr, alpha, beta, delta, k):
    return np.array([ceff_raw(n, s, alpha, beta, delta, k)
                     for n, s in zip(N_arr, s_arr)])


# ══════════════════════════════════════════════════════
# 方案B：无量纲公式（约化变量）
# C_eff = ((ρ - ρ_c) / ρ_c)^β
# ══════════════════════════════════════════════════════
def ceff_reduced(N_B, sigma, alpha, beta, delta, k):
    rho = N_B / np.exp(alpha * sigma)
    rho_c = k * sigma**delta
    if rho <= rho_c:
        return 0.0
    return ((rho - rho_c) / rho_c)**beta

def ceff_reduced_vec(N_arr, s_arr, alpha, beta, delta, k):
    return np.array([ceff_reduced(n, s, alpha, beta, delta, k)
                     for n, s in zip(N_arr, s_arr)])


# ══════════════════════════════════════════════════════
# 统一优化目标：最大化Spearman + Kaplan斜率约束
# ══════════════════════════════════════════════════════
sigma_web = 0.75
N_scan = np.array([10, 30, 100, 300, 1000, 3000, 10000], dtype=float)
target_slope = 0.076

def make_objective(ceff_fn):
    def objective(params):
        alpha, beta, delta, k = params
        ceffs = ceff_fn(N_data, sigma_data, alpha, beta, delta, k)
        if np.all(ceffs == 0) or len(np.unique(ceffs)) < 3:
            return 10.0
        r_s, _ = spearmanr(ceffs, mmlu_data)

        c_scan = ceff_fn(N_scan, np.full_like(N_scan, sigma_web),
                         alpha, beta, delta, k)
        mask = c_scan > 0
        slope_err = 0
        if mask.sum() >= 3:
            slope = np.polyfit(np.log10(N_scan[mask]), np.log10(c_scan[mask]), 1)[0]
            slope_err = (slope - target_slope)**2 * 200
        else:
            slope_err = 5.0

        active_frac = (ceffs > 0).mean()
        if active_frac < 0.5:
            return 10.0 - active_frac

        return -r_s + slope_err
    return objective


def fit_model(name, ceff_fn):
    print(f"\n{'='*60}")
    print(f"拟合: {name}")
    print(f"{'='*60}")

    obj = make_objective(ceff_fn)
    bounds = [
        (0.3, 4.0),   # α
        (0.01, 2.0),  # β — 放宽上界到2.0，给无量纲版空间
        (0.3, 3.0),   # δ
        (0.5, 100),   # k
    ]

    result = differential_evolution(obj, bounds, seed=42,
                                    maxiter=2000, tol=1e-10, popsize=30)

    # 精细优化
    for x0 in [result.x,
               [1.5, 0.1, 1.0, 10],
               [2.0, 0.5, 0.5, 5],
               [1.0, 1.0, 1.5, 20]]:
        res2 = minimize(obj, x0, method='Nelder-Mead',
                        options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res2.fun < result.fun:
            result = res2

    alpha, beta, delta, k = result.x
    ceffs = ceff_fn(N_data, sigma_data, alpha, beta, delta, k)
    r_s, p_s = spearmanr(ceffs, mmlu_data)

    # Kaplan斜率
    c_scan = ceff_fn(N_scan, np.full_like(N_scan, sigma_web),
                     alpha, beta, delta, k)
    mask = c_scan > 0
    slope = np.polyfit(np.log10(N_scan[mask]), np.log10(c_scan[mask]), 1)[0] if mask.sum() >= 3 else None

    print(f"  α = {alpha:.4f}")
    print(f"  β = {beta:.4f}")
    print(f"  δ = {delta:.4f}")
    print(f"  k = {k:.4f}")
    print(f"  Spearman = {r_s:.4f} (p = {p_s:.4f})")
    if slope:
        print(f"  Kaplan斜率 = N^{slope:.4f} (目标: N^0.076)")

    return alpha, beta, delta, k, r_s, slope


# ══════════════════════════════════════════════════════
# 执行对比
# ══════════════════════════════════════════════════════
print("β无量纲化验证")
print("="*60)
print("用同一组14个模型数据，拟合两版公式，比较β值")
print(f"数据点: {len(models)}个模型")

a1, b1, d1, k1, r1, s1 = fit_model("方案A: 有量纲 (ρ-ρ_c)^β", ceff_raw_vec)
a2, b2, d2, k2, r2, s2 = fit_model("方案B: 无量纲 ((ρ-ρ_c)/ρ_c)^β", ceff_reduced_vec)


# ══════════════════════════════════════════════════════
# 对比总结
# ══════════════════════════════════════════════════════
print("\n")
print("="*60)
print("对比总结")
print("="*60)
print()
print(f"  {'指标':<20} {'有量纲(当前)':<15} {'无量纲(约化)':<15} {'3D渗流理论':<12}")
print(f"  {'-'*62}")
print(f"  {'β':<20} {b1:<15.4f} {b2:<15.4f} {'0.41':<12}")
print(f"  {'α':<20} {a1:<15.4f} {a2:<15.4f} {'-':<12}")
print(f"  {'δ':<20} {d1:<15.4f} {d2:<15.4f} {'-':<12}")
print(f"  {'k':<20} {k1:<15.4f} {k2:<15.4f} {'-':<12}")
print(f"  {'Spearman':<20} {r1:<15.4f} {r2:<15.4f} {'-':<12}")
if s1 and s2:
    print(f"  {'Kaplan斜率':<20} {'N^'+f'{s1:.4f}':<15} {'N^'+f'{s2:.4f}':<15} {'N^0.076':<12}")

print()
if b2 > b1 * 1.5:
    print(f"  ★ β无量纲化后从 {b1:.4f} 回升到 {b2:.4f}，提升 {b2/b1:.1f}x")
    if abs(b2 - 0.41) < abs(b1 - 0.41):
        print(f"  ★ 更接近3D渗流理论值0.41")
    if b2 > 0.3:
        print(f"  ★ 铁新的工程直觉可能是对的——公式少了一步无量纲化")
elif b2 > b1:
    print(f"  β略有回升（{b1:.4f} → {b2:.4f}），但幅度不大")
    print(f"  无量纲化可能不是β偏低的主要原因")
else:
    print(f"  β没有回升（{b1:.4f} → {b2:.4f}）")
    print(f"  无量纲化不是β偏低的原因，需要寻找其他解释")

print()
print("结果是什么就写什么。")
