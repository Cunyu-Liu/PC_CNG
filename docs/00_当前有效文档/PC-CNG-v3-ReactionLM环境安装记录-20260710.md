# PC-CNG v3 Reaction LM 环境安装记录

## 目标

为后续接入真实 Reaction LM scorer 准备 Chemformer 和 Molecular Transformer 运行依赖。

## 安装策略

没有直接修改稳定训练环境：

```text
/home/cunyuliu/miniconda3/envs/pc_cng_gpu
```

而是创建隔离 venv：

```text
/home/cunyuliu/pc_cng_research/envs/reaction_lm
```

原因：

1. Chemformer 官方依赖较老，要求 Python 3.7 / torch 1.8 / pytorch-lightning 1.2。
2. Molecular Transformer 原始代码基于 OpenNMT-py 0.4.1 / torchtext 0.3。
3. 直接安装进 `pc_cng_gpu` 有较高破坏现有实验链路的风险。

当前方案：

```text
reaction_lm venv --system-site-packages
复用 pc_cng_gpu 中的 torch / rdkit / transformers
额外安装 legacy-compatible Lightning / torchtext / hydra / pysmilesutils
```

## 源码路径

```text
/home/cunyuliu/pc_cng_research/external/reaction_lm/Chemformer
/home/cunyuliu/pc_cng_research/external/reaction_lm/MolecularTransformer
```

仓库：

```text
https://github.com/MolecularAI/Chemformer
https://github.com/pschwllr/MolecularTransformer
```

## 关键依赖版本

```text
python: 3.10.20
torch: 2.6.0+cu124
rdkit: 2025.03.6
transformers: 4.57.6
pytorch-lightning: 1.5.10
torchmetrics: 0.10.3
torchtext: 0.6.0
hydra-core: 1.3.2
omegaconf: 2.3.0
tensorboard: 2.21.0
PySMILESutils: git+https://github.com/MolecularAI/pysmilesutils.git@b5b3b7d...
```

完整 freeze：

```text
/home/cunyuliu/pc_cng_research/results/reaction_lm_scorer_smoke/reaction_lm_pip_freeze.txt
```

## 兼容处理

Chemformer 的旧代码使用：

```python
from pytorch_lightning.plugins import Plugin
```

当前 Lightning 版本没有直接暴露该符号，因此在 reaction_lm venv 中加入了一个 `sitecustomize.py` 兼容 shim：

```text
/home/cunyuliu/pc_cng_research/envs/reaction_lm/lib/python3.10/site-packages/sitecustomize.py
```

内容：

```python
try:
    import pytorch_lightning.plugins as _pl_plugins
    from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin as _TrainingTypePlugin
    if not hasattr(_pl_plugins, "Plugin"):
        _pl_plugins.Plugin = _TrainingTypePlugin
except Exception:
    pass
```

源码包没有通过 pip editable 安装，而是通过 `.pth` 暴露源码路径：

```text
chemformer_molbart.pth
molecular_transformer_onmt.pth
```

这样避免 Chemformer `pyproject.toml` 中 `python == 3.7.11` 的硬性 pin 阻塞。

## 验证命令

```bash
cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
ROOT=/home/cunyuliu/pc_cng_research bash scripts_check_reaction_lm_env.sh
```

验证结果：

```text
torch OK
rdkit OK
transformers OK
pytorch_lightning OK
torchtext OK
molbart OK
molbart.models.chemformer OK
molbart.predict OK
onmt OK
onmt.opts OK
onmt.translate.translator OK
```

CLI smoke：

```text
python -m molbart.predict --help
python translate.py -h
```

均已通过。

## 当前限制

当前只完成了依赖和代码环境：

```text
Chemformer checkpoint: not installed
Molecular Transformer checkpoint: not installed
```

HuggingFace 当前在 A100 上不可达，Chemformer 官方预训练权重在 Box 链接上，需要后续单独下载或从本地同步。

## 下一步

1. 获取 Chemformer forward prediction checkpoint 和 vocabulary。
2. 获取 Molecular Transformer `.pt` checkpoint 与 vocab/data preprocessing 文件。
3. 在 `pc_cng/reaction_lm_scorer.py` 中实现：
   - `ChemformerScorer`
   - `MolecularTransformerScorer`
4. 用已完成的 candidate CSV 协议运行真实 LM scorer：

```text
build_reaction_lm_candidate_set.py
-> reaction_lm_scorer.py
-> evaluate_reaction_lm_scores.py
```

5. 将真实 LM scorer 的 top-1 / MRR / NDCG 与当前 PC-CNG reranker 结果对齐比较。
