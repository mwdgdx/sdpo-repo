# SDPO Rich Feedback (local repro)

复现 SDPO_original 的 rich_feedback 实验：小模型 + LoRA，结果可推到 Hugging Face。

## Dataset 准备（必做）

1. 准备 **train.json / test.json**（如用 tooluse：从 SDPO_original 拷 `datasets/tooluse` 到 `sdpo/datasets/tooluse`）。  
2. 转成 parquet（在 **sdpo** 根目录）：
   ```bash
   python data/preprocess.py --data_source datasets/tooluse
   ```
   详见 `sdpo/data/README.md`。

## Rich feedback 是否“真的执行程序”？

- **tooluse**：不做真实工具调用，只做**规则打分**（解析 Action / Action Input 与 ground_truth 比较）。不跑任何程序。  
- **code / livecodebench / humanevalplus / mbppplus**：会**在沙箱里执行**生成的代码（`code.py`），用 test cases 判对错 = **有 rich feedback**。  

原 SDPO 只给 code、livecodebench、humanevalplus 接了沙箱；**mbppplus 原版没有 rich feedback**，本 repo 已把 mbppplus 接到同一套 code 沙箱，所以 MBPP Plus 也有 rich feedback。

## 小数据集能 train 吗？原版怎么做的？

能。原版 rich_feedback 实验用的是 **lcb_v6**（LiveCodeBench v6，体量较大）；小数据集（如 HumanEval 164、MBPP 399）一样可以训：把 **train_batch_size** 调小（如 8），这样每个 epoch 有多轮 batch，多训几个 epoch 即可。

## 可选：Push to Hub

`sdpo/verl/verl/utils/push_to_hub.py` 里已有上传逻辑；若要改成“先合并 base+LoRA 再 push”，可在此文件里改。

## 数据路径

- 默认：`datasets/tooluse`（需先按上面步骤生成 `train.parquet` / `test.parquet`）。

## 运行

```bash
cd sdpo
export PROJECT_ROOT="$PWD"

# 可选：推送到你的 Hugging Face
export PUSH_TO_HUB_ID="YOUR_HF_USERNAME/sdpo-lora-lcb"

# 默认用 datasets/tooluse；也可传数据路径
./experiments/rich_feedback/run_sdpo_local.sh
# 或
./experiments/rich_feedback/run_sdpo_local.sh ../SDPO_original/datasets/lcb_v6
```

- 模型：默认 `Qwen/Qwen2.5-1.5B-Instruct`，LoRA rank=64。  
- 改模型：`MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct ./experiments/rich_feedback/run_sdpo_local.sh`  
- 改 HF 仓库：`PUSH_TO_HUB_ID=your_user/repo_name ./experiments/rich_feedback/run_sdpo_local.sh`

## 配置

- `sdpo/verl/verl/trainer/config/sdpo.yaml`：SDPO 算法与默认。  
- `sdpo/verl/verl/trainer/config/user_sdpo_rich_feedback.yaml`：小模型、LoRA、`push_to_hub_id` 等；可按需改 `lora_rank`、`push_to_hub_id`。
