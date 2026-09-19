def _atomic_save(obj: dict[str, Any], path: Path) -> None:
    """Write-then-rename so a disconnect mid-save never leaves a truncated
    checkpoint that a resume would then choke on."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_disk_curriculum(
    *,
    train_dir: str | Path,
    val_dir: str | Path,
    checkpoint_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-4,
    line_len: int = 44,
    num_workers: int = 2,
    max_steps_per_epoch: int | None = None,
    time_budget_s: float | None = None,
    device: str | None = None,
) -> Path:
    """Unattended, resumable training over a pre-generated dataset.

    Severity curriculum: the manifest records each sample's severity, so epoch
    `e` trains only on samples whose severity is inside a window that widens
    from [0, 0.3] to [0, 1.0] -- easy shapes first, full print-scan damage by
    the end.

    Every epoch: `last.pt` (model + optimiser + progress) is rewritten
    atomically so a disconnect loses at most one epoch and a re-run resumes;
    `best.pt` is rewritten when validation per-character accuracy improves;
    and one line of metrics -- including the confusable-set accuracy and the
    per-character breakdown -- goes to stdout and `metrics.jsonl`.

    Returns the path of `best.pt`, the checkpoint export_mrz_onnx.py consumes.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    metrics_path = checkpoint_dir / "metrics.jsonl"

    model = MrzCRNN(line_len=line_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    train_ds = MrzDiskDataset(train_dir)
    val_ds = MrzDiskDataset(val_dir, jitter_frac=0.0)
    severities = [float(r["severity"]) for r in train_ds.rows]
    val_loader = DataLoader[MrzSample](
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers,
    )

    start_epoch, best_char_acc = 0, -1.0
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_char_acc = int(state["epoch"]), float(state["best_char_accuracy"])
        print(f"RESUMING from {last_path}: completed {start_epoch}/{epochs} epochs, "
              f"best char_acc so far {best_char_acc:.4f}", flush=True)

    run_start = time.time()
    for epoch in range(start_epoch, epochs):
        if time_budget_s is not None and time.time() - run_start > time_budget_s:
            print(f"time budget {time_budget_s:.0f}s exhausted before epoch {epoch + 1}; "
                  "stopping early so the export still happens", flush=True)
            break

        lo, hi = severity_for_step(epoch, max(epochs - 1, 1), start=(0.0, 0.3), end=(0.0, 1.0))
        indices = [i for i, sv in enumerate(severities) if lo <= sv <= hi]
        loader = DataLoader[MrzSample](
            Subset(train_ds, indices), batch_size=batch_size, shuffle=True, collate_fn=collate_batch,
            num_workers=num_workers, drop_last=True, pin_memory=device == "cuda",
        )

        model.train()
        epoch_start, loss_sum, n_batches = time.time(), 0.0, 0
        for step, batch in enumerate(loader):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            images, targets = batch["images"].to(device), batch["targets"].to(device)
            log_probs = model(images)
            loss = criterion(log_probs.reshape(-1, log_probs.shape[-1]), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

        val = evaluate(model, val_loader, device)
        improved = val.char_accuracy > best_char_acc
        seconds = round(time.time() - epoch_start, 1)
        record: dict[str, Any] = {
            "epoch": epoch + 1, "epochs": epochs, "severity_window": [round(lo, 3), round(hi, 3)],
            "train_samples": len(indices), "steps": n_batches,
            "loss": loss_sum / max(n_batches, 1),
            "char_acc": val.char_accuracy, "line_acc": val.line_accuracy,
            "confusable_acc": val.confusable_accuracy,
            "confusable_per_char": val.per_confusable_char,
            "best": improved, "seconds": seconds,
        }
        per_char = " ".join(f"{c}={a:.3f}" for c, a in val.per_confusable_char.items())
        marker = "*best*" if improved else ""
        print(
            f"epoch {epoch + 1}/{epochs}  loss={record['loss']:.4f}  window=[{lo:.2f},{hi:.2f}]  "
            f"char_acc={val.char_accuracy:.4f}  line_acc={val.line_accuracy:.4f}  "
            f"CONFUSABLE_ACC={val.confusable_accuracy:.4f}  {marker}  {seconds}s\n"
            f"    per-char: {per_char}",
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if improved:
            best_char_acc = val.char_accuracy
            _atomic_save(
                {"state_dict": model.state_dict(), "line_len": line_len, "epoch": epoch + 1,
                 "char_accuracy": val.char_accuracy, "line_accuracy": val.line_accuracy,
                 "confusable_accuracy": val.confusable_accuracy},
                best_path,
            )
        _atomic_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch + 1, "best_char_accuracy": best_char_acc},
            last_path,
        )

    if not best_path.exists():
        raise RuntimeError("training produced no checkpoint (zero epochs ran?)")
    return best_path


