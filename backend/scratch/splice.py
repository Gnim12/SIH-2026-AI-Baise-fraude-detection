from pathlib import Path

p = Path("app/ocr/mrz/train.py")
s = p.read_text(encoding="utf-8")
block = Path("scratch/disk_trainer_block.py").read_text(encoding="utf-8")

s = s.replace("import argparse\nimport time\nimport json\nimport os\n", "import argparse\nimport json\nimport os\nimport time\n")
s = s.replace("from torch.utils.data import DataLoader\n", "from torch.utils.data import DataLoader, Subset\n", 1)
s = s.replace("def _parse_args() -> argparse.Namespace:", block + "def _parse_args() -> argparse.Namespace:", 1)

s = s.replace(
    '    parser.add_argument("--self-test", action="store_true", help="run a tiny CPU smoke test instead of real training")',
    '    parser.add_argument("--self-test", action="store_true", help="run a tiny CPU smoke test instead of real training")\n'
    '    parser.add_argument("--disk", action="store_true", help="train from a pre-generated dataset (resumable, unattended)")\n'
    '    parser.add_argument("--train-dir", type=str, default="data/mrz/train")\n'
    '    parser.add_argument("--val-dir", type=str, default="data/mrz/val")\n'
    '    parser.add_argument("--workers", type=int, default=2)\n'
    '    parser.add_argument("--max-steps-per-epoch", type=int, default=None)\n'
    '    parser.add_argument("--time-budget-hours", type=float, default=None)',
)
s = s.replace(
    "    if args.self_test:",
    "    if args.disk:\n"
    "        train_disk_curriculum(\n"
    "            train_dir=args.train_dir, val_dir=args.val_dir, checkpoint_dir=args.checkpoint_dir,\n"
    "            epochs=args.epochs, batch_size=args.batch, lr=args.lr, line_len=args.line_len,\n"
    "            num_workers=args.workers, max_steps_per_epoch=args.max_steps_per_epoch,\n"
    "            time_budget_s=args.time_budget_hours * 3600 if args.time_budget_hours else None,\n"
    "        )\n"
    "    elif args.self_test:",
    1,
)
p.write_text(s, encoding="utf-8")
print("spliced; disk trainer present:", "train_disk_curriculum" in s)
