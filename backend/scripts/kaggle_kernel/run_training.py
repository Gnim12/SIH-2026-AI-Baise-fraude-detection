"""Kaggle kernel entry point for the 1.5.0 retrain (B1i).

Clones the repo branch, then runs scripts/kaggle_train.py: generate the 120k/12k/12k dataset with the
fixed renderer (seed 20260920, NOT 42), train 12 epochs with the severity curriculum and scale/offset
augmentation, export as 1.5.0, and publish the artifacts as a private Kaggle Dataset through the API.

Needs, in the kernel settings: GPU (T4), Internet on, and Add-ons > Secrets KAGGLE_USERNAME / KAGGLE_KEY
(so the final dataset upload can authenticate).

Push with:  kaggle kernels push -p scripts/kaggle_kernel
"""
import os
import subprocess
import sys

BRANCH = os.environ.get("MRZ_BRANCH", "worktree-b1g-mrz-weights")

try:  # expose the Kaggle secrets to the kaggle API client
    from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

    secrets = UserSecretsClient()
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            os.environ[name] = secrets.get_secret(name)
        except Exception as exc:  # noqa: BLE001
            print(f"secret {name} not available ({exc}); the final dataset upload will be skipped", flush=True)
except ImportError:
    pass

sys.exit(subprocess.call([sys.executable, "-u", "scripts/kaggle_train.py", "--branch", BRANCH, "--version", "1.5.0"]
                         if os.path.exists("scripts/kaggle_train.py") else
                         ["bash", "-lc",
                          f"git clone --depth 1 --branch {BRANCH} "
                          "https://github.com/Gnim12/SIH-2026-AI-Baise-fraude-detection.git /kaggle/working/repo && "
                          "cd /kaggle/working/repo/backend && "
                          f"python -u scripts/kaggle_train.py --branch {BRANCH} --version 1.5.0"]))
