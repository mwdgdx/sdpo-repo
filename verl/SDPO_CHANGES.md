# SDPO 对 verl 的核心修改（均未提交，在 IDE Source Control 看 diff）

## 修改统计

```
verl/trainer/ppo/core_algos.py       | +120 行  # self-distillation loss 函数
verl/trainer/ppo/ray_trainer.py      | +183 行  # teacher batch 构建
verl/workers/actor/dp_actor.py       | +167 行  # actor 训练 SDPO 分支 + TrustRegionTeacher
verl/workers/config/actor.py         |  +76 行  # SelfDistillationConfig 配置类
verl/workers/fsdp_workers.py         |  +17 行  # teacher 模块初始化
verl/workers/reward_manager/naive.py |   +3 行  # truncated 检测

共 6 个文件修改，+538 行
```

---

## 修改的文件（6 个）

### 1. verl/workers/reward_manager/naive.py
**作用**：truncated 检测，供 feedback 判断响应是否被截断

```python
# 改动 1: return_dict 默认值
def __call__(self, data: DataProto, return_dict: bool = True) -> ...  # 原来是 False

# 改动 2: 新增 truncated 检测
extra_info["truncated"] = not (valid_response_ids == self.tokenizer.eos_token_id).any().item()
```

### 2. verl/trainer/ppo/core_algos.py
**作用**：SDPO 核心 loss 函数

- 新增 `import torch.nn.functional as F`
- 新增 `compute_self_distillation_loss()` 函数（~110 行）

```python
def compute_self_distillation_loss(
    student_log_probs, teacher_log_probs, response_mask,
    self_distillation_config, old_log_probs=None,
    student_all_log_probs=None, teacher_all_log_probs=None,
    student_topk_log_probs=None, teacher_topk_log_probs=None,
    self_distillation_mask=None, loss_agg_mode="token-mean",
    rollout_is_weights=None,
) -> tuple[torch.Tensor, dict]:
    # 支持:
    # - Full logit distillation / token-level distillation
    # - Top-k distillation with tail probability
    # - Generalized Jensen-Shannon Divergence (alpha 参数)
    # - Importance sampling clipping
```

### 3. verl/workers/config/actor.py
**作用**：SDPO 配置类

- 新增 `SelfDistillationConfig` dataclass（~80 行）
- `ActorConfig` 新增 `self_distillation` 字段
- `PolicyLossConfig.loss_mode` 增加 `'sdpo'` 选项

```python
@dataclass
class SelfDistillationConfig(BaseConfig):
    alpha: float = 1.0                    # KL 方向: 0=forward, 1=reverse, (0,1)=JS
    teacher_regularization: str = "ema"   # 'ema' 或 'trust-region'
    teacher_update_rate: float = 0.0      # EMA 更新率
    full_logit_distillation: bool = False # 是否用全 vocabulary
    distillation_topk: int | None = None  # top-k distillation
    include_environment_feedback: bool = True  # 是否使用环境反馈
    success_reward_threshold: float = 1.0      # 成功阈值
    solution_template: str = "..."        # 成功样本模板
    feedback_template: str = "..."        # 反馈模板
    reprompt_template: str = "..."        # 组合模板
```

### 4. verl/trainer/ppo/ray_trainer.py
**作用**：训练循环中构建 teacher batch

- 新增 `_collect_feedback()`: 收集环境反馈
- 新增 `_collect_solutions_by_uid()`: 收集成功样本
- 新增 `_get_solution()`: 获取成功 solution
- 新增 `_maybe_build_self_distillation_batch()`: **核心** - 构建 teacher prompt

```python
def _maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict):
    # 1. 检测 loss_mode == "sdpo"
    # 2. 收集成功样本的 solution
    # 3. 收集环境 feedback
    # 4. 构建 teacher prompt = original_prompt + solution/feedback + response
    # 5. 返回 teacher_input_ids, teacher_attention_mask, self_distillation_mask
```

- 训练循环中调用（在 `compute_advantage` 之后）

### 5. verl/workers/actor/dp_actor.py（**最关键**）
**作用**：Actor 训练时调用 `compute_self_distillation_loss`

**改动 1**: 新增 `TrustRegionTeacher` 类（trust-region 模式的 teacher）

```python
class TrustRegionTeacher(nn.Module):
    """Teacher = lerp(ref_logits, student_logits, mix_coef)"""
    def forward(self, *args, **kwargs):
        ref_logits = self.ref_module(...)
        student_logits = self.student_module(...)
        return lerp(ref_logits, student_logits, self.mix_coef)
```

**改动 2**: 新增 `self.teacher_module` 成员变量

**改动 3**: 新增 `_update_teacher()` 方法（EMA 更新）

**改动 4**: 修改 `_forward_micro_batch()` 支持 `module` 参数

**改动 5**: 修改 `update_policy()` 核心训练循环

```python
if self_distillation_enabled:
    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_outputs = self._forward_micro_batch(teacher_inputs, module=teacher_model)
    
    # SDPO Loss (KL divergence, not PPO clip)
    pg_loss, pg_metrics = compute_self_distillation_loss(
        student_log_probs=log_prob,
        teacher_log_probs=teacher_log_prob,
        ...
    )
else:
    # 原有 PPO loss
    pg_loss, pg_metrics = policy_loss_fn(advantages=advantages, ...)
```

### 6. verl/workers/fsdp_workers.py（**新增**）
**作用**：初始化 teacher 模块

```python
# 在 ref_policy 创建后，初始化 teacher
if self._is_actor:
    if loss_mode == "sdpo":
        if teacher_regularization == "trust-region":
            self.actor.teacher_module = TrustRegionTeacher(
                ref_module=self.ref_module_fsdp,
                student_module=self.actor_module_fsdp,
                mix_coef=teacher_update_rate,
            )
        else:  # EMA mode
            self.actor.teacher_module = self.ref_module_fsdp
```

---

## 新增的文件

### verl/trainer/config/sdpo.yaml
SDPO 算法配置示例

### verl/trainer/config/user.yaml
用户配置：数据路径、模型、`custom_reward_function.path`

### verl/utils/reward_score/feedback/ （整个目录）
Rich feedback 实现

---

## 完整调用链

```
ray_trainer.py: fit()
    ├── _maybe_build_self_distillation_batch()  # 构建 teacher batch
    │       └── 收集 solution/feedback → teacher_input_ids
    │
    └── actor_rollout_wg.update_actor(batch)
            ↓
        fsdp_workers.py: update_actor()
            ├── 初始化时: self.actor.teacher_module = ref_model 或 TrustRegionTeacher
            └── self.actor.update_policy(data)
                    ↓
                dp_actor.py: update_policy()
                    ├── Student forward → log_prob
                    ├── Teacher forward → teacher_log_prob
                    └── compute_self_distillation_loss(student, teacher, mask)
                            ↓
                        core_algos.py: compute_self_distillation_loss()
                            └── KL(student || teacher)
```

---

## 在 IDE 里看 diff

```bash
git diff verl/workers/fsdp_workers.py            # teacher 初始化
git diff verl/workers/actor/dp_actor.py          # 核心训练 + TrustRegionTeacher
git diff verl/trainer/ppo/core_algos.py          # self-distillation loss
git diff verl/trainer/ppo/ray_trainer.py         # teacher batch 构建
git diff verl/workers/config/actor.py            # SelfDistillationConfig
git diff verl/workers/reward_manager/naive.py    # truncated 检测
```
