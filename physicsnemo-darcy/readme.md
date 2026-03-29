# PhysicsNeMo Darcy FNO / PhysicsNeMo达西流FNO示例

PhysicsNeMo框架自带的2D Darcy流FNO训练示例。达西渗流是物理AI最基础的benchmark之一。

## What this shows / 这个示例说明了什么

PhysicsNeMo本质上是把现有物理求解方法（FNO傅里叶神经算子）封装成Python模块，用GPU加速训练和推理。用户只需修改config.yaml并运行train_fno_darcy.py。

## Results / 运行结果

- validation_step_004: 训练早期，相对误差±0.5，预测粗糙
- validation_step_008: 训练后期，相对误差降至±0.3，但误差仍沿渗透率场界面分布

注：即使在这个最简单的2D问题上，界面处的误差仍然可见。工业级3D问题的挑战会大得多。

## Connection to NPM / 与NPM论文的联系

达西渗流（Darcy flow）正是NPM论文的核心物理背景——孔隙网络模型（PNM）求解的就是达西方程。PhysicsNeMo用FNO学习达西算子，NPM证明了GNN与PNM守恒方程的精确等价（误差=0）。同一个物理问题，两种AI方法。

## Reference

- [FNO Paper](https://arxiv.org/abs/2010.08895)
- [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)
- [NPM Paper - Zenodo](https://doi.org/10.5281/zenodo.19209722)
