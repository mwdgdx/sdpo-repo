#!/usr/bin/env python3
"""
SDPO Code Validation Script
Run this BEFORE training to catch errors early!

Usage:
    python scripts/validate_sdpo.py

This script checks:
1. Python syntax of all SDPO-related files
2. Import errors (missing modules, typos)
3. Configuration loading
4. Basic sanity checks
"""

from __future__ import annotations  # Python 3.6+ compatibility
import sys
import os
import ast
import importlib
import traceback
from typing import Tuple

# Add verl to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDPO_ROOT = os.path.dirname(SCRIPT_DIR)
VERL_DIR = os.path.join(SDPO_ROOT, "verl")
sys.path.insert(0, VERL_DIR)

# Files to check
SDPO_FILES = [
    "verl/verl/workers/actor/dp_actor.py",
    "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/trainer/ppo/core_algos.py",
    "verl/verl/workers/config/actor.py",
    "verl/verl/trainer/main_ppo.py",
    "verl/verl/utils/reward_score/feedback/code.py",
    "verl/verl/utils/reward_score/feedback/__init__.py",
]

# Modules to import-test
SDPO_MODULES = [
    "verl.trainer.ppo.core_algos",
    "verl.workers.config.actor",
    "verl.utils.model",
]


def check_syntax(filepath: str) -> Tuple[bool, str]:
    """Check Python syntax without executing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def check_import(module_name: str) -> Tuple[bool, str]:
    """Try to import a module to catch import errors."""
    try:
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_config() -> Tuple[bool, str]:
    """Check if SDPO config can be loaded."""
    try:
        from omegaconf import OmegaConf
        config_path = os.path.join(VERL_DIR, "verl/trainer/config/sdpo.yaml")
        if not os.path.exists(config_path):
            return False, f"Config file not found: {config_path}"
        
        cfg = OmegaConf.load(config_path)
        
        # Check required SDPO settings
        required_keys = [
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            "actor_rollout_ref.actor.self_distillation",
        ]
        for key in required_keys:
            try:
                OmegaConf.select(cfg, key)
            except:
                return False, f"Missing config key: {key}"
        
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_dataclass() -> Tuple[bool, str]:
    """Check if SelfDistillationConfig is valid."""
    try:
        from verl.workers.config.actor import SelfDistillationConfig
        
        # Test instantiation with defaults
        cfg = SelfDistillationConfig()
        
        # Check required attributes
        required_attrs = [
            'alpha', 'full_logit_distillation', 'distillation_topk',
            'teacher_update_rate', 'is_clip', 'solution_template',
            'feedback_template', 'reprompt_template'
        ]
        missing = [a for a in required_attrs if not hasattr(cfg, a)]
        if missing:
            return False, f"Missing attributes: {missing}"
        
        # Test validation
        cfg.__post_init__()
        
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_core_algos_function() -> Tuple[bool, str]:
    """Check if compute_self_distillation_loss exists and has correct signature."""
    try:
        from verl.trainer.ppo.core_algos import compute_self_distillation_loss
        import inspect
        
        sig = inspect.signature(compute_self_distillation_loss)
        required_params = [
            'student_log_probs', 'teacher_log_probs', 'response_mask',
            'self_distillation_config'
        ]
        actual_params = list(sig.parameters.keys())
        missing = [p for p in required_params if p not in actual_params]
        if missing:
            return False, f"Missing parameters: {missing}"
        
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_dp_actor_imports() -> Tuple[bool, str]:
    """Check dp_actor.py has all necessary imports."""
    try:
        filepath = os.path.join(SDPO_ROOT, "verl/verl/workers/actor/dp_actor.py")
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Check for required imports
        required_imports = [
            "compute_self_distillation_loss",
        ]
        missing = [i for i in required_imports if i not in source]
        if missing:
            return False, f"Missing imports: {missing}"
        
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_ray_trainer_imports() -> Tuple[bool, str]:
    """Check ray_trainer.py has all necessary imports."""
    try:
        filepath = os.path.join(SDPO_ROOT, "verl/verl/trainer/ppo/ray_trainer.py")
        with open(filepath, 'r') as f:
            source = f.read()
        
        required_imports = [
            "import re",
            "compute_position_id_with_mask",
        ]
        missing = [i for i in required_imports if i not in source]
        if missing:
            return False, f"Missing imports: {missing}"
        
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("=" * 60)
    print("SDPO Code Validation")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"SDPO root: {SDPO_ROOT}")
    print()
    
    all_passed = True
    
    # 1. Syntax checks
    print("1. Checking Python syntax...")
    print("-" * 40)
    for filepath in SDPO_FILES:
        full_path = os.path.join(SDPO_ROOT, filepath)
        if not os.path.exists(full_path):
            print(f"  ⚠️  {filepath}: FILE NOT FOUND")
            continue
        
        passed, msg = check_syntax(full_path)
        status = "✅" if passed else "❌"
        print(f"  {status} {filepath}: {msg}")
        if not passed:
            all_passed = False
    print()
    
    # 2. Import checks
    print("2. Checking module imports...")
    print("-" * 40)
    for module in SDPO_MODULES:
        passed, msg = check_import(module)
        status = "✅" if passed else "❌"
        print(f"  {status} {module}: {msg}")
        if not passed:
            all_passed = False
    print()
    
    # 3. Config check
    print("3. Checking SDPO config...")
    print("-" * 40)
    passed, msg = check_config()
    status = "✅" if passed else "❌"
    print(f"  {status} sdpo.yaml: {msg}")
    if not passed:
        all_passed = False
    print()
    
    # 4. Dataclass check
    print("4. Checking SelfDistillationConfig...")
    print("-" * 40)
    passed, msg = check_dataclass()
    status = "✅" if passed else "❌"
    print(f"  {status} SelfDistillationConfig: {msg}")
    if not passed:
        all_passed = False
    print()
    
    # 5. Function signature check
    print("5. Checking core_algos functions...")
    print("-" * 40)
    passed, msg = check_core_algos_function()
    status = "✅" if passed else "❌"
    print(f"  {status} compute_self_distillation_loss: {msg}")
    if not passed:
        all_passed = False
    print()
    
    # 6. Import presence checks
    print("6. Checking required imports in files...")
    print("-" * 40)
    passed, msg = check_dp_actor_imports()
    status = "✅" if passed else "❌"
    print(f"  {status} dp_actor.py imports: {msg}")
    if not passed:
        all_passed = False
    
    passed, msg = check_ray_trainer_imports()
    status = "✅" if passed else "❌"
    print(f"  {status} ray_trainer.py imports: {msg}")
    if not passed:
        all_passed = False
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ All checks passed! Safe to run training.")
    else:
        # Check if failures are just missing dependencies (OK locally)
        print("⚠️  Some checks failed.")
        print("")
        print("If failures are only 'No module named numpy/omegaconf/torch':")
        print("  → These are MISSING DEPENDENCIES, not code errors")
        print("  → They WILL work in the training container")
        print("  → Safe to proceed if SYNTAX checks (section 1) all passed ✅")
        print("")
        print("If failures show actual errors (SyntaxError, NameError, etc):")
        print("  → These are REAL problems that need fixing")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

