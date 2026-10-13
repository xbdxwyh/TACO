# TACO / MathRepairBench 仓库预备版

这是单独整理的代码与纯文本数据目录，原工作区、协议和实验结果没有改动。
GitHub 私有仓库为 [xbdxwyh/TACO](https://github.com/xbdxwyh/TACO)，归属 Yihao Wang 的个人账号。
尚未公开发布；代码、数据和自身模型的许可均待作者决定。

## 内容

- `src/taco/`：诊断编码、Track、候选排序、执行修复、独立评分和训练入口。
- `data/mathrepairbench/`：3,456 条 text/LaTeX 数据；公开输入、标签、题族划分分开存放。
- `checkpoints/`：两个种子的诊断模型和排序模型，共四份冻结权重，约 30 MB。
- `baselines/`：七个外部 PRM 的原生推理适配，以及 Rule / +TACO 修复接口。
- `reference_results/`：论文核心表及保护、开销、错误类型、图 4/5 的精简结果与来源。
- `scripts/`：数据校验、文件完整性检查，以及无需图像依赖的文本生成器。
- `docs/`：数据、训练、指标和外部方法的复现说明。

历史云端登录信息、临时试验脚本、图片、模拟学习者轨迹、外部大模型权重和
大量中间缓存都不属于发布文件。文本数据与我们自己的小权重可以放在同一个仓库。

## 本机检查

当前 `image_edit` 环境已经完成可编辑安装。在本仓库根目录运行：

```powershell
python scripts/verify_dataset.py
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

更换机器后，先使用 Python 3.11+，安装适配机器的 PyTorch，再执行：

```powershell
python -m pip install -e .
```

四条数据的运行演示，依次执行：

```powershell
python -m taco predict --split validation --seeds 17 --configs taco --limit 4 --device cpu --output outputs/demo
python -m taco score --predictions outputs/demo
```

结果写入 `outputs/demo/evaluation/summary.json`，过程保留在
`predictions.jsonl`。这只是工程检查，不用来报告模型效果。

## 完整复现

完整运行仍由使用者自行启动；本次整理没有重跑完整训练或评估。
README.md 中提供验证集、测试集的两种子八配置命令。
`--limit 0` 才运行整个划分；默认只处理四条。
训练入口默认也是每组件两步的 smoke，完整训练必须显式指定 `--mode full`。

使用随附冻结权重复现论文结果；重新训练可能受缓存顺序、硬件与数值库影响，
不能承诺逐位一致。预测过程不读取标签，完成封存后再独立评分。

## 目前检查结果

- 数据数量、题族不交叉、四变体齐全、文件与公开输入哈希检查通过。
- 四份权重与原实验逐张量一致。
- 20 项核心行为与完整性回归通过。
- 一个函数复合题族：4 变体 × 2 种子 × 8 配置，与原封存结果的动作、完成判断、
  工具开销及最终评分完全一致；这是限定工程回放，不是全量复现。
- 训练入口通过 8 条 train、每组件 2 更新的短测。
- 外部方法接口通过离线导入和已保存诊断的少量恢复核对；没有重新运行七个大模型。

后续确定许可和 Git 地址，再填写论文里的数据/代码链接。
