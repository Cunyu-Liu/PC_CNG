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

## Checkpoint 状态

### Chemformer USPTO-50K forward checkpoint

已完成下载、校验、同步与推理版转换：

```text
本地下载文件:
/Users/bytedance/Downloads/chemformer_forward.zip

A100:
/home/cunyuliu/pc_cng_research/models/reaction_lm/chemformer_forward_uspto50k/chemformer_forward.zip
/home/cunyuliu/pc_cng_research/models/reaction_lm/chemformer_forward_uspto50k/last.ckpt
/home/cunyuliu/pc_cng_research/models/reaction_lm/chemformer_forward_uspto50k/model_sanitized.ckpt
```

校验：

```text
chemformer_forward.zip sha256:
1ee180c898d87b770b98d2ee60035cf594cb897990d675230c7e9453e8a4cab4

zip content:
last.ckpt
```

下载说明：

- `curl https://figshare.com/ndownloader/files/42012708` 在本机/A100 均返回 403；
- 通过浏览器访问 Figshare 页面并触发 Download 后，Figshare 会经 AWS WAF 生成 10 秒有效的 signed S3 URL；
- 浏览器下载成功后再 `scp` 到 A100。

转换说明：

- 原始 `last.ckpt` 是旧 Chemformer / Lightning checkpoint；
- PyTorch 2.6 的 `weights_only=True` 无法直接解析旧 pickle opcode；
- 在确认来源为 Figshare/Syntheseus 公开 Apache 2.0 checkpoint 后，使用 `TRUST_EXTERNAL_CHECKPOINTS=1` 进行一次受控 `weights_only=False` 读取；
- 随后立即保存为只含 `state_dict` 和 `hyper_parameters` 的 `model_sanitized.ckpt`，用于推理。

### Chemformer combined pretrained fallback

已从 Hugging Face 镜像下载并生成 sanitized checkpoint：

```text
/home/cunyuliu/pc_cng_research/models/reaction_lm/chemformer_pretrained_hf/pretrained.ckpt
/home/cunyuliu/pc_cng_research/models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt
```

该 checkpoint 是 combined pretraining，不是 forward fine-tuned；只作为环境加载 smoke 和后续 fine-tune 初始化备选，不作为论文主 benchmark。

### Molecular Transformer checkpoint

仍未完成自动下载：

```text
Official model page:
https://ibm.box.com/v/MolecularTransformerModels
```

当前状态：

- IBM Box 页面可见模型文件；
- 直接 file-content endpoint 需要 Box/IBM authentication；
- 浏览器点击下载会跳转到 IBM / w3id 登录；
- 待后续通过登录浏览器或可公开镜像获取 `.pt` 文件。

## 下一步

1. 获取 Molecular Transformer `.pt` checkpoint 与 vocab/data preprocessing 文件。
2. 用已完成的 candidate CSV 协议运行真实 LM scorer：

```text
build_reaction_lm_candidate_set.py
-> reaction_lm_scorer.py
-> evaluate_reaction_lm_scores.py
```

3. 将真实 LM scorer 的 top-1 / MRR / NDCG 与当前 PC-CNG reranker 结果对齐比较。
4. 优先使用 `chemformer_log_likelihood`，而不是只使用 top-k beam exact-match，因为后者候选命中率很低。
