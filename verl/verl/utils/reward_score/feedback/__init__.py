def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """
    Compute reward score based on data source.
    Uses lazy imports to avoid loading unnecessary dependencies.
    """
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        from verl.utils.reward_score.feedback import code
        # Use dense rewards (pass rate) instead of sparse (all-or-nothing)
        # This gives finer-grained signal for improved SDPO's revision and imitation
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=False, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k"]:
        from verl.utils.reward_score.feedback import math
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        from verl.utils.reward_score.feedback import gpqa
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        from verl.utils.reward_score.feedback import mcq
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        from verl.utils.reward_score.feedback import tooluse
        results = tooluse.compute_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results
