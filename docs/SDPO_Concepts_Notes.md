# SDPO 训练相关概念笔记

## 1. Tokenizer 的 `padding_side` 和 `truncation_side`

### 1.1 `padding_side`

**定义**: 当序列长度不足时，在哪一侧添加 `[PAD]` token。

| 值 | 含义 | 示例 |
|---|------|------|
| `"left"` | 左侧填充 | `[PAD][PAD][PAD] Hello World` |
| `"right"` | 右侧填充 | `Hello World [PAD][PAD][PAD]` |

**为什么 SDPO/LLM 训练用 `left`？**

```
场景：Batch 中有两个序列，需要对齐生成位置

left padding (推荐):
  序列1: [PAD][PAD] A B C [生成位置]
  序列2: X Y Z W Q [生成位置]
                    ↑ 生成位置对齐！

right padding (不推荐):
  序列1: A B C [PAD][PAD] [生成位置???]
  序列2: X Y Z W Q [生成位置]
                    ↑ 生成位置错位！
```

**关键点**: 
- LLM 是 **自回归生成**（从左到右），新 token 总是添加在**右侧**
- 使用 `left padding` 可以让所有序列的生成起点对齐
- 对于 **decoder-only** 模型（GPT、LLaMA、Qwen），几乎都用 `padding_side="left"`

---

### 1.2 `truncation_side`

**定义**: 当序列超过 `max_length` 时，从哪一侧截断。

| 值 | 含义 | 示例 |
|---|------|------|
| `"left"` | 截断左侧（保留最近内容） | `A B C D E` → `C D E` |
| `"right"` | 截断右侧（保留开头内容） | `A B C D E` → `A B C` |

**为什么我们遇到 `"error"` 报错？**

```python
# 原版 SDPO 代码
self.tokenizer.truncation_side = config.get("reprompt_truncation", "error")
```

原版设计意图：如果没配置 `reprompt_truncation`，就故意设成无效值 `"error"`，
让 transformers 报错提醒用户。

但 transformers 只接受：
- `"left"`
- `"right"`

所以当 YAML 中没有定义 `reprompt_truncation` 时，fallback `"error"` 导致崩溃。

**修复**: 改成合理的默认值 `"right"`

---

### 1.3 典型配置

```python
# 推荐的 LLM 训练配置
tokenizer.padding_side = "left"      # 生成位置对齐
tokenizer.truncation_side = "right"  # 保留 prompt 开头，截断过长的尾部
```

---

## 2. `use_remove_padding` (Remove Padding Optimization)

### 2.1 什么是 Remove Padding？

**问题**: Batch 训练时，不同序列长度不同，需要 padding 对齐：

```
普通 batching:
  序列1: [A B C PAD PAD PAD]  ← 50% 是无效计算
  序列2: [X Y Z W Q R]
  
  GPU 需要计算 PAD 位置，但这些计算是浪费的！
```

**Remove Padding 优化**: 把所有有效 token 拼接成一个长序列，消除 padding：

```
Remove padding batching:
  拼接序列: [A B C | X Y Z W Q R]
  位置 ID:  [0 1 2 | 0 1 2 3 4 5]  ← 用 position_id 区分不同样本
  
  无 PAD token，100% 都是有效计算！
```

### 2.2 为什么 SDPO 不能用 Remove Padding？

SDPO 的 full-logit distillation 需要：

```python
# 需要完整的 logits 矩阵来计算 KL divergence
student_logits: [batch_size, seq_len, vocab_size]
teacher_logits: [batch_size, seq_len, vocab_size]

# Remove padding 后：
packed_logits: [total_tokens, vocab_size]  # 无法对齐 student/teacher！
```

**问题**:
1. Remove padding 打破了 batch 维度结构
2. Student 和 Teacher 的 token 无法一一对应
3. KL divergence 计算需要相同位置的概率分布

**解决方案**: 在 `sdpo.yaml` 中禁用：
```yaml
actor_rollout_ref:
  actor:
    use_remove_padding: false  # SDPO 必须禁用
```

---

## 3. Teacher EMA (Exponential Moving Average)

### 3.1 什么是 EMA？

**EMA (指数移动平均)** 是一种平滑更新策略：

```
θ_teacher = (1 - α) × θ_teacher + α × θ_student

其中：
- θ_teacher: Teacher 模型权重
- θ_student: Student 模型权重 (正在训练的模型)
- α: update_rate (更新率)，通常很小，如 0.01
```

### 3.2 为什么用 EMA 而不是直接复制？

| 方法 | 公式 | 特点 |
|-----|------|------|
| 直接复制 | θ_teacher = θ_student | Teacher 完全跟随 Student，无稳定性 |
| EMA | θ_teacher = 0.99×θ_teacher + 0.01×θ_student | Teacher 缓慢跟随，保持稳定 |

**EMA 的好处**:

1. **平滑效果**: Teacher 是 Student 历史版本的加权平均
2. **稳定性**: 避免 Teacher 因为单次 bad update 而崩溃
3. **正则化**: 给 Student 提供一个"更保守"的目标

### 3.3 有效平均窗口计算

```
update_rate = α = 0.01

有效平均窗口 ≈ 1/α = 1/0.01 = 100 steps
```

**直觉理解**:

```
Step 1: θ_T = 0.99×θ_T + 0.01×θ_S1
Step 2: θ_T = 0.99×(0.99×θ_T + 0.01×θ_S1) + 0.01×θ_S2
       = 0.9801×θ_T + 0.0099×θ_S1 + 0.01×θ_S2
...
Step N: θ_T 包含了过去约 1/α = 100 个 step 的 Student 权重

权重衰减：
- Step N-1 的贡献: 0.01 × 0.99^1 ≈ 0.0099
- Step N-10 的贡献: 0.01 × 0.99^10 ≈ 0.009
- Step N-100 的贡献: 0.01 × 0.99^100 ≈ 0.0037 (约 37% 衰减)
- Step N-200 的贡献: 0.01 × 0.99^200 ≈ 0.0013 (约 87% 衰减)
```

### 3.4 SDPO 中的 EMA 参数

```yaml
self_distillation:
  teacher_update_rate: 0.01  # α = 0.01
```

**为什么选 0.01？**

| update_rate | 有效窗口 | 特点 |
|-------------|---------|------|
| 0.001 | ~1000 steps | 非常保守，Teacher 变化很慢 |
| 0.01 | ~100 steps | 平衡：Teacher 能跟上训练，但保持稳定 |
| 0.1 | ~10 steps | 激进：Teacher 快速跟随，可能不稳定 |
| 1.0 | 1 step | 直接复制，无 EMA 效果 |

SDPO 论文选择 0.01，经验上这个值在大多数情况下效果较好。

---

## 4. Self-Distillation 中的其他关键概念

### 4.1 `full_logit_distillation` vs Token-level KL

| 方法 | 计算 | 内存 |
|-----|------|------|
| Token-level | 只用选中 token 的 log prob | 低 |
| Full-logit | 用完整 vocab 的概率分布 | 高 (vocab_size ≈ 128K) |

```python
# Token-level KL (简化版)
kl = teacher_log_prob - student_log_prob  # 只看选中的 token

# Full-logit KL (完整版)
kl = sum(teacher_prob * (log(teacher_prob) - log(student_prob)))  # 整个分布
```

### 4.2 `distillation_topk`

**问题**: Full-logit 需要存储整个 vocab (128K) 的概率，内存爆炸！

**解决**: 只保留 Top-K 个 token 的概率：

```python
distillation_topk: 20  # 只用概率最高的 20 个 token

# Teacher 分布近似:
# Top-20 tokens 保留原概率
# 其余 tokens 合并成一个 "tail" bucket
```

### 4.3 `is_clip` (Importance Sampling Clipping)

**问题**: Self-distillation 中，Student 和 Teacher 的分布可能差异很大

**解决**: 用 importance sampling ratio 加权，并裁剪防止爆炸：

```python
is_clip: 2.0  # 裁剪 IS ratio 最大为 2.0

ratio = exp(student_log_prob - old_log_prob)
ratio = clamp(ratio, max=2.0)  # 防止 ratio 过大
loss = loss * ratio
```

---

## 5. 常见错误总结

| 错误 | 原因 | 修复 |
|-----|------|------|
| `ValueError: Unknown 'direction': 'error'` | `truncation_side` 设成无效值 | 改成 `"left"` 或 `"right"` |
| `NotImplementedError: use_remove_padding` | Remove padding 与 full-logit distillation 不兼容 | 设置 `use_remove_padding: false` |
| `AssertionError: Multi-modal inputs` | 代码检查了空的 multi_modal_inputs | 修改检查逻辑，忽略空容器 |
| `NameError: compute_position_id_with_mask` | 缺少 import | 添加 `from verl.utils.model import compute_position_id_with_mask` |
| `ConfigAttributeError: Key 'xxx' not in struct` | YAML 中缺少必要字段 | 在 sdpo.yaml 中添加缺失字段 |

---

## 6. 配置加载流程

```
Hydra/OmegaConf 配置加载:

1. 基础配置 (ppo_trainer.yaml)
   ↓ 合并
2. 用户配置 (sdpo.yaml)
   ↓ 合并
3. 命令行覆盖 (--config xxx=yyy)
   ↓
4. 最终配置对象

注意：
- .get("key", default) 的 default 只在 key 完全不存在时生效
- dataclass 默认值只在通过 dataclass 实例化时生效
- 直接访问 OmegaConf 的 YAML 字段不会触发 dataclass 默认值！
```

---

*最后更新: 2026-02-08*

