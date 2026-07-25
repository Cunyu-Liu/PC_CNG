# Phase 0 非科学质量 Gate 失效声明

> **文档版本**：v1.0
> **创建日期**：2026-07-24
> **文档地位**：本文件声明 PC-CNG 项目中与科学质量无关的投稿 Gate 自即日起失效。这些 Gate 此前出现在 `顶刊论文核心思想与从0到1落地方案.md` 等历史文档中，但与科学质量无关，不符合 NMI 投稿标准。

---

## 一、失效的 Gate 定义

以下 Gate 自 2026-07-24 起正式失效，不再参与任何项目决策：

### 1. Manuscript 字节数 Gate

**原定义**（见 `顶刊论文核心思想与从0到1落地方案.md` 第 3618、3619、3633 行）：
- `GO: manuscript v4 ≥ 50KB AND 期刊定位升级至 Nature Machine Intelligence / JACS Au`
- `Partial GO: manuscript v4 ≥ 50KB AND 期刊定位保持 Chemical Science`
- `Manuscript v4 | ≥50KB | ≥50KB + 期刊升级决策`

**失效原因**：manuscript 字节数与科学质量、方法创新性、外部验证强度无关。期刊投稿标准基于科学贡献和证据强度，不基于文档大小。

### 2. P4 任务完成数 Gate

**原定义**（见 `顶刊论文核心思想与从0到1落地方案.md` 第 3618、3624、3665 行）：
- `P4 ≥ 5/9 tasks GO`
- `P4-01 ~ P4-09 全部完成（或至少 5/9 GO）`
- `科研成功：≥3/9 P4 任务 GO AND manuscript v4 ≥ 50KB`

**失效原因**：任务完成数量不等于科学质量。Phase 0 审计已证明部分 GO 判定（G6、G8-A）基于错误的方法或混杂实验。正确的做法是评估每个 claim 的证据强度，而非计数 GO 数量。

### 3. 九维评分决定期刊 Gate

**原定义**（见 P4_GOAL_20260721.md §1 第 7 条）：
- `不得用自定义九维评分决定期刊`

**状态**：此前已声明禁止，Phase 0 重申确认。

---

## 二、替代标准

自即日起，期刊投稿决策基于 `NMI_FINAL_GOAL.md` §五 Phase 5 NMI 投稿 Gate 定义的五类必达条件：

1. **方法**：Full learned PC-CNG 明确优于 rule PC-CNG
2. **外部效用**：≥2 独立数据集 + ≥2 backbone + paired cluster CI 全正 + OOD 成立
3. **可信性**：candidate-level FNR 校准 + known-positive stress test + collision analysis + expert review/prospective
4. **机制**：semi-hard utility 通过控制实验验证 + held-out 复现
5. **可复现性**：干净环境一键安装 + CI + 数据 provenance + split manifests 冻结 + 主表主图一键重建 + 永久归档

---

## 三、历史文档处理

`docs/00_当前有效文档/顶刊论文核心思想与从0到1落地方案.md` 等历史文档中的非科学 Gate 定义保留原位作为历史记录，但自即日起被本声明 supersede。项目决策以 `NMI_FINAL_GOAL.md` 为唯一权威。
