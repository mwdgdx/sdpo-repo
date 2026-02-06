"""
Prepare MBPP Plus (~399 samples) for SDPO with rich feedback (sandbox code execution).
Downloads from HF, formats to train.json / test.json, then run preprocess.py.

Usage (from sdpo repo root):
  pip install datasets
  python data/prepare_mbppplus.py --output_dir datasets/mbppplus
  python data/preprocess.py --data_source datasets/mbppplus
"""
import argparse
import json
import os

CODE_PROMPT = """You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```.

{problem}"""

TIME_LIMIT = 5


def _prompt(problem: str, fn_name: str) -> str:
    return CODE_PROMPT.format(problem=f"{problem} The function should be called `{fn_name}`.")


def _parse_fn_name(code: str) -> str:
    return code.split("def ")[1].split("(")[0].strip()


def load_and_format():
    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", split="test")
    rows = []
    for i, ex in enumerate(ds):
        fn_name = _parse_fn_name(ex["code"])
        test = ""
        for imp in ex.get("test_imports", []):
            test += f"{imp}\n"
        test += ex["test"]
        tests = {
            "inputs": [test],
            "outputs": [""],
            "testtype": "code",
            "fn_name": "",
            "time_limit": TIME_LIMIT,
        }
        rows.append({
            "idx": i,
            "kind": "code",
            "dataset": "mbppplus",
            "answer": "-",
            "elo": "-",
            "prompt": _prompt(ex["prompt"], fn_name),
            "description": (ex["prompt"] or "")[:2000],
            "tests": json.dumps(tests, ensure_ascii=False),
            "system": None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare MBPP Plus for SDPO (train.json + test.json, rich feedback).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/mbppplus",
        help="Directory to write train.json and test.json",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=40,
        help="Number of examples for test split (rest go to train). Default 40 so train≈359.",
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
