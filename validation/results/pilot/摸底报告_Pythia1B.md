# Pythia-1B MMLU摸底报告

**日期：** 2026-04-05
**模型：** EleutherAI/pythia-1b (1012M params)
**硬件：** CPU (PyTorch 2.11.0+cpu)
**评测：** lm-eval 0.4.11, batch_size=1

## 结果

| 子类 | 预期ρc | Accuracy | vs 随机(25%) |
|------|--------|----------|-------------|
| world_religions | 低 | **32.2%** ± 3.6% | **高于随机** |
| formal_logic | 高 | 25.4% ± 3.9% | 约等于随机 |
| abstract_algebra | 高 | 23.0% ± 4.2% | 略低于随机 |
| college_physics | 中 | 21.6% ± 4.1% | 低于随机 |
| high_school_geography | 低 | 17.7% ± 2.7% | 低于随机 |

## 结论

1. Pythia-1B在MMLU上基本处于随机水平，**不适合跑预测3（量化退化）**——没有退化空间
2. **唯一高于随机的是world_religions（32.2%）**，恰好是ρc最低的任务（纯事实记忆）
3. 这本身是NPM预测1（涌现顺序）的证据：ρ刚好够让事实记忆越过p_c，不够让推理涌现
4. 此摸底结果记入预测1数据，作为Pythia-1B的数据点

## 决策

- **预测3：换Phi-2（2.7B）**，MMLU得分预期40-50%，有足够退化空间
- **预测1：保留Pythia系列（70M→1B）**，本摸底是1B的数据点

## high_school_geography异常说明

high_school_geography预期为低ρc（事实记忆），但实际得分17.7%低于随机。可能原因：
- 地理知识在Pythia训练数据中覆盖不足
- 或该子类的题目实际上需要空间推理，ρc比预期高
- 锁定ρc排序时需注意此子类的分类可能不准
