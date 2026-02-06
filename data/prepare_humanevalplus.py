"""
Prepare HumanEval Plus (~164 samples, lightest code dataset) for SDPO.
Downloads from HF, formats to train.json / test.json, then run preprocess.py.

Usage (from sdpo repo root):
  export PYTHONPATH="$(pwd)/verl:$PYTHONPATH"
  python data/prepare_humanevalplus.py --output_dir datasets/humanevalplus
  python data/preprocess.py --data_source datasets/humanevalplus
"""
import argparse
import json
import os

CODE_PROMPT = """You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```.

{problem}"""

TIME_LIMIT = 5


def _prompt(prefix: str) -> str:
    return CODE_PROMPT.format(
        problem=f"Your task is to complete the following function. You are not allowed to modify the given code and should do the completion only. Here is the given code to complete: ```python\n{prefix}\n```"
    )


def _parse_description(problem: str, fn_name: str) -> str:
    text = problem.split(f"def {fn_name}")[1]
    if '"""' in text:
        text = text.split('"""')[1]
    elif "'''" in text:
        text = text.split("'''")[1]
    else:
        text = text.split(">>>")[0].strip() if ">>>" in text else text.strip()
    if '"""' in text:
        text = text.split('"""')[0].strip()
    if "'''" in text:
        text = text.split("'''")[0].strip()
    return text[:2000]


def load_and_format():
    from datasets import load_dataset

    ds = load_dataset("evalplus/humanevalplus", split="test")
    rows = []
    for i, ex in enumerate(ds):
        tests = {
            "inputs": [ex["test"] + "\n" + f'check({ex["entry_point"]})'],
            "outputs": [""],
            "testtype": "code",
            "fn_name": "",
            "time_limit": TIME_LIMIT,
        }
        try:
            desc = _parse_description(ex["prompt"], ex["entry_point"])
        except Exception:
            desc = ex["prompt"][:500]
        rows.append({
            "idx": i,
            "kind": "code",
            "dataset": "humanevalplus",
            "answer": "-",
            "elo": "-",
            "prompt": _prompt(ex["prompt"]),
            "description": desc,
            "tests": json.dumps(tests, ensure_ascii=False),
            "system": None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare HumanEval Plus for SDPO (train.json + test.json).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/humanevalplus",
        help="Directory to write train.json and test.json",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=24,
        help="Number of examples for test split (rest go to train). Default 24 so train=140.",
    )
    args = parser.parse_args()

    rows = load_and_format()
    n = len(rows)
    n_test = min(args.test_size, max(0, n - 1))
    n_train = n - n_test
    train_rows = rows[:n_train]
    test_rows = rows[n_train:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.json")
    test_path = os.path.join(args.output_dir, "test.json")

    def write_json_array(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)

    write_json_array(train_path, train_rows)
    write_json_array(test_path, test_rows)
    print(f"Wrote {train_path} ({n_train} samples) and {test_path} ({n_test} samples).")
    print("Next: python data/preprocess.py --data_source", args.output_dir)


if __name__ == "__main__":
    main()
