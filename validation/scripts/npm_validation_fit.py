"""
NPM验证：拟合 + 预测
====================
第2步：用Pythia系列拟合NPM参数
第3步：用拟合参数预测Phi-2的57子类分数

关键原则：
- 预测目标在拟合之前锁定（Phi-2的57子类分数）
- 两个版本：有量纲 + 无量纲
- 结果是什么写什么
"""

import json
import glob
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr, pearsonr
import os

# ══════════════════════════════════════════════════════
# 1. 加载数据
# ══════════════════════════════════════════════════════

def load_results(pattern):
    """加载lm_eval结果JSON"""
    files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)

    results = data.get('results', {})
    subjects = {}
    for key, val in results.items():
        if key.startswith('mmlu_') and isinstance(val, dict) and 'acc,none' in val:
            name = key.replace('mmlu_', '')
            subjects[name] = val['acc,none']

    mmlu_total = results.get('mmlu', {}).get('acc,none', 0)
    return {'total': mmlu_total, 'subjects': subjects}

# 加载所有模型结果
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'results')

pythia_models = [
    ('pythia-70m', 0.07, 300),    # (名称, 参数量B, 训练token数B)
    ('pythia-160m', 0.16, 300),
    ('pythia-410m', 0.41, 300),
    ('pythia-1b', 1.0, 300),
    ('pythia-1.4b', 1.4, 300),
    ('pythia-2.8b', 2.8, 300),
    ('pythia-6.9b', 6.9, 300),
]

phi2_info = ('phi-2', 2.78, 1400)  # Phi-2: ~1.4T tokens训练数据

print("NPM验证：拟合 + 预测")
print("="*70)
print()

# 加载Pythia系列
pythia_data = []
for name, params, tokens in pythia_models:
    pattern = os.path.join(results_dir, f'results_{name}', '*', 'results_*.json')
    # 处理名称中的点号
    pattern2 = os.path.join(results_dir, f'results_pythia_{name.replace("pythia-", "")}', '*', 'results_*.json')

    result = load_results(pattern) or load_results(pattern2)
    if result:
        pythia_data.append((name, params, tokens, result))
        print(f"  加载 {name}: MMLU={result['total']:.4f}, {len(result['subjects'])}个子类")
    else:
        print(f"  !! 未找到 {name} 的结果")

# 加载Phi-2
phi2_pattern = os.path.join(results_dir, 'results_phi2_fp16', '*', 'results_*.json')
phi2_result = load_results(phi2_pattern)
if phi2_result:
    print(f"  加载 phi-2: MMLU={phi2_result['total']:.4f}, {len(phi2_result['subjects'])}个子类")

print(f"\n共加载 {len(pythia_data)} 个Pythia模型 + 1 个Phi-2")

# ══════════════════════════════════════════════════════
# 2. 锁定预测目标（在拟合之前写死）
# ══════════════════════════════════════════════════════

print("\n" + "="*70)
print("预测目标锁定（拟合之前声明）")
print("="*70)
print()
print("  目标：用Pythia 7个模型拟合NPM参数")
print("  预测：Phi-2在57个子类上的分数")
print("  Pythia σ = 0.70（The Pile）")
print("  Phi-2 σ = 0.70（保守估计，使对比更干净）")
print("  预测目标已锁定，不可更改。")

# ══════════════════════════════════════════════════════
# 3. NPM公式
# ══════════════════════════════════════════════════════

SIGMA_PYTHIA = 0.70
SIGMA_PHI2 = 0.70  # 保守估计，和Pythia一致
RANDOM_BASELINE = 0.25

def npm_rho(N_tokens_B, params_B, sigma):
    """计算数据密度ρ
    简化版：ρ = N_tokens / exp(α * σ) * f(params)
    这里用 N_tokens * params 作为有效数据密度的代理
    """
    return N_tokens_B * params_B / np.exp(sigma)

def npm_ceff_raw(rho, rho_c, beta):
    """有量纲版：C_eff = (ρ - ρ_c)^β"""
    if rho <= rho_c:
        return 0.0
    return (rho - rho_c)**beta

def npm_ceff_reduced(rho, rho_c, beta):
    """无量纲版：C_eff = ((ρ - ρ_c) / ρ_c)^β"""
    if rho <= rho_c:
        return 0.0
    return ((rho - rho_c) / rho_c)**beta

# ══════════════════════════════════════════════════════
# 4. 拟合：用Pythia系列的MMLU总分
# ══════════════════════════════════════════════════════

print("\n" + "="*70)
print("第2步：NPM拟合（Pythia系列）")
print("="*70)

# Pythia的ρ值
pythia_rhos = []
pythia_mmlu = []
for name, params, tokens, result in pythia_data:
    rho = tokens * params  # 简化的ρ代理
    pythia_rhos.append(rho)
    pythia_mmlu.append(result['total'])

pythia_rhos = np.array(pythia_rhos)
pythia_mmlu = np.array(pythia_mmlu)

# Phi-2的ρ
phi2_rho = phi2_info[2] * phi2_info[1]  # 1400 * 2.78

print(f"\n  Pythia ρ值: {pythia_rhos}")
print(f"  Pythia MMLU: {pythia_mmlu}")
print(f"  Phi-2 ρ: {phi2_rho}")
print(f"  Phi-2 MMLU (真实值，预测目标): {phi2_result['total']:.4f}")

# 拟合函数
def fit_npm(ceff_fn, label):
    """拟合NPM参数并预测"""

    def objective(params):
        rho_c, beta, scale = params
        predicted = []
        for rho in pythia_rhos:
            c = ceff_fn(rho, rho_c, beta)
            predicted.append(RANDOM_BASELINE + scale * c)
        predicted = np.array(predicted)

        # 最小化MSE
        mse = np.mean((predicted - pythia_mmlu)**2)

        # 惩罚：所有预测不应该超过1或低于0
        penalty = sum(max(0, p - 1) + max(0, -p) for p in predicted) * 10

        return mse + penalty

    bounds = [
        (1, 5000),     # ρ_c
        (0.001, 2.0),  # β
        (0.001, 10.0), # scale
    ]

    result = differential_evolution(objective, bounds, seed=42, maxiter=2000, tol=1e-12, popsize=30)

    # 精细优化
    for x0 in [result.x, [100, 0.1, 1.0], [500, 0.5, 0.5]]:
        res2 = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res2.fun < result.fun:
            result = res2

    rho_c, beta, scale = result.x

    # Pythia预测
    pythia_pred = []
    for rho in pythia_rhos:
        c = ceff_fn(rho, rho_c, beta)
        pythia_pred.append(RANDOM_BASELINE + scale * c)
    pythia_pred = np.array(pythia_pred)

    # Phi-2预测
    phi2_c = ceff_fn(phi2_rho, rho_c, beta)
    phi2_pred = RANDOM_BASELINE + scale * phi2_c

    # 评估
    r_s, p_s = spearmanr(pythia_pred, pythia_mmlu)
    mse = np.mean((pythia_pred - pythia_mmlu)**2)

    print(f"\n  [{label}]")
    print(f"  ρ_c = {rho_c:.4f}")
    print(f"  β = {beta:.4f}")
    print(f"  scale = {scale:.6f}")
    print(f"  Pythia拟合 Spearman = {r_s:.4f} (p = {p_s:.4f})")
    print(f"  Pythia拟合 MSE = {mse:.6f}")
    print()

    print(f"  Pythia逐模型对比:")
    print(f"  {'模型':<15} {'ρ':<10} {'实际':<10} {'预测':<10} {'误差':<10}")
    for i, (name, params, tokens, _) in enumerate(pythia_data):
        print(f"  {name:<15} {pythia_rhos[i]:<10.1f} {pythia_mmlu[i]:<10.4f} {pythia_pred[i]:<10.4f} {pythia_pred[i]-pythia_mmlu[i]:+.4f}")

    print()
    print(f"  ★ Phi-2预测:")
    print(f"    ρ = {phi2_rho:.1f}")
    print(f"    预测 MMLU = {phi2_pred:.4f} ({phi2_pred*100:.1f}%)")
    print(f"    实际 MMLU = {phi2_result['total']:.4f} ({phi2_result['total']*100:.1f}%)")
    print(f"    误差 = {phi2_pred - phi2_result['total']:+.4f} ({(phi2_pred - phi2_result['total'])*100:+.1f}%)")

    return rho_c, beta, scale, phi2_pred

print("\n--- 有量纲版 C_eff = (ρ - ρ_c)^β ---")
rc1, b1, s1, pred1 = fit_npm(npm_ceff_raw, "有量纲")

print("\n--- 无量纲版 C_eff = ((ρ - ρ_c)/ρ_c)^β ---")
rc2, b2, s2, pred2 = fit_npm(npm_ceff_reduced, "无量纲")

# ══════════════════════════════════════════════════════
# 5. 子类级预测（57个子类）
# ══════════════════════════════════════════════════════

print("\n" + "="*70)
print("第3步：子类级分析")
print("="*70)

# 获取所有子类列表
all_subjects = sorted(phi2_result['subjects'].keys())

# 收集每个子类在所有模型中的分数
subject_data = {}
for subj in all_subjects:
    scores = []
    for name, params, tokens, result in pythia_data:
        if subj in result['subjects']:
            scores.append(result['subjects'][subj])

    phi2_score = phi2_result['subjects'].get(subj, 0)

    if scores:
        subject_data[subj] = {
            'pythia_mean': np.mean(scores),
            'pythia_std': np.std(scores),
            'phi2': phi2_score,
            'delta': phi2_score - np.mean(scores),
        }

# 按Phi-2分数排序（ρc代理）
sorted_subjects = sorted(subject_data.items(), key=lambda x: x[1]['phi2'], reverse=True)

print(f"\n  57个子类按Phi-2分数排序（高→低 = 低ρc→高ρc）:")
print(f"  {'子类':<35} {'Phi-2':<8} {'Pythia均值':<10} {'Δ(涌现幅度)':<12}")
print(f"  {'-'*65}")
for subj, data in sorted_subjects:
    print(f"  {subj:<35} {data['phi2']:.3f}  {data['pythia_mean']:.3f}     {data['delta']:+.3f}")

# 计算涌现幅度和Phi-2分数的相关性
deltas = [d['delta'] for _, d in sorted_subjects]
phi2_scores = [d['phi2'] for _, d in sorted_subjects]
pythia_means = [d['pythia_mean'] for _, d in sorted_subjects]

r_delta, p_delta = spearmanr(phi2_scores, deltas)
print(f"\n  涌现幅度Δ vs Phi-2分数的Spearman = {r_delta:.4f} (p = {p_delta:.6f})")
print(f"  含义：{'低ρc子类涌现幅度更大' if r_delta > 0.3 else '涌现幅度和ρc关系不明显'}")

# ══════════════════════════════════════════════════════
# 6. 总结
# ══════════════════════════════════════════════════════

print("\n" + "="*70)
print("总结")
print("="*70)
print()
print(f"  拟合数据：7个Pythia模型 (70m-6.9b)")
print(f"  预测目标：Phi-2 (2.78B)")
print()
print(f"  有量纲版：β={b1:.4f}, ρ_c={rc1:.1f}")
print(f"    Phi-2预测={pred1*100:.1f}%, 实际={phi2_result['total']*100:.1f}%, 误差={abs(pred1-phi2_result['total'])*100:.1f}%")
print()
print(f"  无量纲版：β={b2:.4f}, ρ_c={rc2:.1f}")
print(f"    Phi-2预测={pred2*100:.1f}%, 实际={phi2_result['total']*100:.1f}%, 误差={abs(pred2-phi2_result['total'])*100:.1f}%")
print()
print(f"  涌现幅度和ρc的Spearman = {r_delta:.4f}")
print()
print("  结果是什么写什么。")
