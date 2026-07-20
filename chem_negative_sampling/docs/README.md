# 化学反应负样本框架

## 各阶段数据来源

### 第一阶段：引导负样本
1. **USPTO 数据库** (https://figshare.com/articles/dataset/USPTO_1976-Sep2016_/5104873)
- 包含 190 万个反应
- 用于启发式组合不相关的反应物/产物

**负样本示例**：
```
CC(=O)O.CCO>>CCOC(=O)C # 乙酸 + 乙醇在没有催化剂的情况下不会生成乙酸乙酯
C1=CC=CC=C1.C=O>>C1=CC=CC=C1C=O # 苯 + 甲醛（无效的直接偶联）
N#CCBr>>N#CCN # 氰基溴化物转化为二胺（无效的转化）
```

2. **Reaxys 数据库** (https://www.reaxys.com/)
- 包含反应产率的商业数据库
- 提取低产率反应 (<5-10%) 作为可能的阴性结果

3. **PubChem 反应** (https://pubchem.ncbi.nlm.nih.gov/)
- 包含可作为阴性结果的无效/标记反应

### 第二阶段：预训练阳性样本
1. **USPTO 有效反应**
- 筛选高置信度反应（专利示例）

**阳性样本示例**：
```
CCOC(=O)Cl.CN>>CCOC(=O)N(C)C # 氯甲酸乙酯 + 二甲胺
C1=CC=C(C=C1)C=O.CCN>>C1=CC=C(C=C1)C(=O)NCC # 苯甲醛 + 乙胺（还原胺化）
C=CCBr.C=CC>>C=CCC=CC # 烯丙基溴 + 丁二烯（偶联）
```

2. **开放反应数据库** (https://docs.open-reaction-database.org/)
- 社区精选的有效反应

3. **ChEMBL 反应** (https://www.ebi.ac.uk/chembl/)
- 生物活性分子反应

### 第三阶段：数据优化
1. **模型生成的候选产物**
- 来自第一阶段的阴性结果，且得分较高，为阳性结果

**硬阴性结果示例**：
```
CCOC(=O)Cl.CN>>CCOC(=O)NC # 几乎正确但错误的产物
C1=CC=CC=C1Br.CC=O>>C1=CC=CC=C1C(C)=O # 合理但无效的偶联
C=CC(=O)OCC.CN>>C=CC(=O)NCC # 酯基转化为酰胺，但机理错误
```

2. **VAE/LLM 生成的反应**
- 使用基于第 2 阶段数据训练的模型

### 第 4 阶段：大规模生成
1. **最终模型预测**
- 将训练好的模型应用于大规模生成的候选集

**生成的候选示例**:
```
CCOC(=O)Cl.NC(=O)C>>CCOC(=O)NC(=O)C # 有效的酰胺形成
C1=CC=C(C=C1)Br.C=CC>>C1=CC=C(C=C1)C=CC # 有效的偶联
C=CCN.CC(=O)Cl>>C=CCNC(=O)C # 有效的酰胺化
```

## 安装
```bash
git clone https://github.com/yourusername/chem_negative_sampling.git
cd chem_negative_sampling
pip install -r requirements.txt
```

## 使用方法

### 阶段 1：引导程序
```bash
python phase1_bootstrap/phase1_main.py \
--reaction_data_path data/uspto_reactions.csv \
--yield_data_path data/reactions_with_yields.csv
```

### 阶段 2：预训练
```bash
python phase2_pretrain/train.py \
--model_type gnn \
--train_data data/train_reactions.csv \
--val_data data/val_reactions.csv \
--epochs 100
```

### 阶段 3：迭代优化
```bash
python phase3_refinement/train.py \
--model_path checkpoints/best_model.pt \
--negative_pool data/phase1_combined_negatives.csv \
--epochs 50
```

### 阶段 4：生成式扩展
```bash
python phase4_expansion/generate.py \
--model_path checkpoints/final_model.pt \
--num_samples 1000000 \
--output data/final_dataset.csv
```

## Project Structure
```
chem_negative_sampling/
├── data/                   # Input/output data
├── phase1_bootstrap/       # Initial negative sample generation
│   ├── rule_based_generation.py
│   ├── heuristic_combinations.py
│   ├── low_yield_processor.py
│   └── phase1_main.py
├── phase2_pretrain/        # Encoder pretraining
│   ├── model.py
│   ├── gnn.py
│   ├── transformer.py
│   ├── data.py
│   └── train.py
├── phase3_refinement/      # Hard negative mining
├── phase4_expansion/       # Large-scale generation
├── evaluation/             # Evaluation protocols
├── utils/                  # Shared utilities
├── README.md
└── requirements.txt
```

## Requirements
See [requirements.txt](requirements.txt) for full list of dependencies.

## License
MIT
