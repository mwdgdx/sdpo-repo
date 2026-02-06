#!/bin/bash
# 完整 LCB v6 数据预处理。需在 sdpo 根目录执行。
# 使用项目 .venv：先 source .venv/bin/activate，或本脚本自动用 .venv/bin/python。

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
PYTHON="${PYTHON:-python}"

echo "[1/3] load_dataset: HF -> datasets/lcb_v6.json"
$PYTHON data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json

echo "[2/3] split_tests: train (50% tests) + test (full)"
$PYTHON data/split_tests.py --json_path datasets/lcb_v6.json --output_dir datasets/lcb_v6

echo "[3/3] preprocess: train/test.json -> train/test.parquet"
$PYTHON data/preprocess.py --data_source datasets/lcb_v6

echo "Done. Use: ./experiments/rich_feedback/run_sdpo_local.sh datasets/lcb_v6"
