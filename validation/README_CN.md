# NPM验证实验

神经渗流模型（NPM）框架预测的实证验证，基于MMLU基准测试。

[English](README.md)

---

## 已发表成果

**NPM框架研究#1：涌现排序预测**
- Zenodo: [DOI: 10.5281/zenodo.19440042](https://doi.org/10.5281/zenodo.19440042)
- arXiv: submit/7447724 (cs.LG)

### 核心发现

1. **涌现顺序是任务的内禀属性。** 用accuracy和Brier Score两种指标衡量，53个MMLU子类的难度排序高度一致（Spearman = 0.984, p ~ 10⁻³⁹）。
2. **数据质量对涌现具有决定性作用。** Phi-2（2.78B）得分54.5%，Pythia-2.8B（2.80B）得分25.2%——同参数量差29个百分点。
3. **单纯增加参数不能替代数据质量。** Pythia系列170倍参数增长（70M→12B），MMLU始终在随机水平附近（22.9%–25.6%）。

---

## 目录结构

```
validation/
├── scripts/
│   ├── npm_validation_fit.py        # NPM参数拟合+预测
│   └── beta_dimensionless_test.py   # 无量纲β验证
├── results/
│   ├── results_pythia_*/            # Pythia系列8个模型的MMLU结果
│   ├── results_phi2_fp16/           # Phi-2 FP16结果
│   └── pilot*/                      # 试验性实验
├── figures/
│   ├── fig1_acc_vs_brier_CN.png     # 散点图（中文版）
│   ├── fig1_acc_vs_brier_EN.png     # 散点图（英文版）
│   └── cover_wechat.png             # 公众号封面
├── configs/                         # 实验配置
└── requirements.txt                 # Python依赖
```

## 评测模型

| 模型 | 参数(B) | 训练数据 | MMLU |
|------|--------|---------|------|
| Pythia-70m | 0.07 | The Pile 300B | 22.9% |
| Pythia-160m | 0.16 | The Pile 300B | 23.0% |
| Pythia-410m | 0.41 | The Pile 300B | 23.1% |
| Pythia-1b | 1.00 | The Pile 300B | 23.1% |
| Pythia-1.4b | 1.40 | The Pile 300B | 24.3% |
| Pythia-2.8b | 2.80 | The Pile 300B | 25.2% |
| Pythia-6.9b | 6.90 | The Pile 300B | 25.6% |
| Pythia-12b | 12.00 | The Pile 300B | 24.2% |
| Phi-2 | 2.78 | 精选合成 ~1.4T | 54.5% |

## 计算环境

- AutoDL RTX 4090（24GB）
- lm-evaluation-harness v0.4.11
- MMLU全部57个子类，0-shot评测

---

**许可证：** CC BY-NC-ND 4.0 | **作者：** 丁铁新 | **NeuralCAE**
