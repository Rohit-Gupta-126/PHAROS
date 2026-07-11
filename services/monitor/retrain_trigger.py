"""Closed-loop retrain trigger: confirmed physics drift -> verified hot-swap.

Consumes ``alerts.drift`` and requires *sustained* drift -- ``--confirm``
consecutive alert-severity ``score_psi`` evaluations on the physics stream --
before acting (a single hot window never triggers). On confirmation it runs
the full Phase 0 -> Phase 2 pipeline against a NEW model directory:

1. retrain the VAE on Phase 0 background (demo-scale: reduced epochs/events,
   see --retrain-epochs / --retrain-events -- documented in design_log);
2. export ``encoder_mu.onnx`` (batch 1, opset 13);
3. ONNX parity gate (PyTorch vs onnxruntime, tol 1e-5) -- **if this fails,
   the pointer is NOT written and the old model keeps serving**;
4. re-derive the p99 threshold and eval AUC for the new model;
5. atomically publish ``models/physics_vae/current.json`` -- the running
   scorer (``--model-pointer``) hot-swaps on its next poll.

Before/after AUC + thresholds go to ``reports/phase3/retrain_log.json``.

Run: ``python -m services.monitor.retrain_trigger --max-retrains 1``
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.common.config import load_config, resolve_path
from services import common
from services.monitor.hot_swap import ModelPointer


def run_retrain_pipeline(phys_cfg: Dict[str, Any], out_dir: Path,
                         epochs: int, max_events: int,
                         percentile: float) -> Dict[str, Any]:
    """Train -> export -> parity gate -> threshold + AUC. Raises on failure."""
    # Imports here: torch etc. only load when a retrain actually fires.
    import torch
    from scripts import export_onnx, verify_onnx
    from scripts.derive_thresholds import derive_physics
    from scripts.eval_physics import run as eval_run
    from src.training.train_physics import run as train_run

    cfg = dict(phys_cfg)
    cfg.update({"out_dir": str(out_dir), "epochs": epochs,
                "max_events": max_events})
    print(f"[retrain] training -> {out_dir} (epochs={epochs}, "
          f"max_events={max_events})")
    train_run(cfg)

    export_onnx.main(["--model-dir", str(out_dir)])
    # Parity gate: SystemExit here propagates -> no pointer write.
    verify_onnx.main(["--model-dir", str(out_dir),
                      "--report", "reports/phase3/onnx_parity_retrain.txt"])

    thr = derive_physics({**cfg, "model_dir": str(out_dir)}, percentile,
                         int(phys_cfg.get("eval_events", 100_000)),
                         int(phys_cfg.get("seed", 1337)),
                         torch.device("cpu"))
    metrics = eval_run({**cfg, "model_dir": str(out_dir),
                        "reports_dir": "reports/phase3"})
    return {"threshold": thr["threshold"], "eval": metrics}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--physics-config", type=str,
                   default="configs/physics_vae.yaml")
    p.add_argument("--thresholds", type=str, default="configs/thresholds.json")
    p.add_argument("--pointer", type=str,
                   default="models/physics_vae/current.json")
    p.add_argument("--confirm", type=int, default=3,
                   help="consecutive alert-severity score_psi windows needed")
    p.add_argument("--max-retrains", type=int, default=1,
                   help="stop after this many retrains (demo cooldown)")
    p.add_argument("--retrain-epochs", type=int, default=6)
    p.add_argument("--retrain-events", type=int, default=500_000)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--reports-dir", type=str, default="reports/phase3")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--group", type=str, default=None)
    p.add_argument("--idle", type=float, default=120.0)
    args = p.parse_args(argv)

    phys_cfg = load_config(args.physics_config)
    old_thr = json.loads(resolve_path(args.thresholds).read_text())
    old = {"model_dir": phys_cfg["model_dir"],
           "threshold": old_thr["physics"]["threshold"],
           "auc": _read_old_auc(phys_cfg)}
    print(f"[retrain-trigger] armed: {args.confirm} consecutive physics "
          f"score_psi alerts -> retrain (max {args.max_retrains}); "
          f"serving {old['model_dir']} thr={old['threshold']:.6g}")

    group = args.group or common.fresh_group("pharos-retrain")
    consumer = common.make_consumer(common.TOPIC_DRIFT, group, args.bootstrap,
                                    from_beginning=bool(args.group))
    streak = 0
    retrains = []
    try:
        for ev in common.consume_json(consumer, idle_timeout_s=args.idle):
            if ev.get("stream") != "physics" or ev.get("metric") != "score_psi":
                continue
            if ev["severity"] == "alert":
                streak += 1
                print(f"[retrain-trigger] alert streak {streak}/{args.confirm} "
                      f"(psi={ev['value']:.3f})")
            else:
                streak = 0
            if streak < args.confirm:
                continue
            streak = 0
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = resolve_path(f"models/physics_vae_r{stamp}")
            entry: Dict[str, Any] = {
                "triggered_at": datetime.now(timezone.utc).isoformat(),
                "trigger_event": ev, "old": old,
                "new_model_dir": str(out_dir),
                "retrain_epochs": args.retrain_epochs,
                "retrain_events": args.retrain_events,
            }
            try:
                new = run_retrain_pipeline(phys_cfg, out_dir,
                                           args.retrain_epochs,
                                           args.retrain_events,
                                           args.percentile)
            except SystemExit as exc:
                entry.update({"swapped": False,
                              "reason": f"parity gate failed: {exc}"})
                print(f"[retrain-trigger] PARITY FAILED -- old model keeps "
                      f"serving ({exc})")
            except Exception as exc:  # noqa: BLE001
                entry.update({"swapped": False,
                              "reason": f"pipeline error: {exc}"})
                print(f"[retrain-trigger] pipeline error -- no swap: {exc}")
            else:
                ModelPointer.write(resolve_path(args.pointer), str(out_dir),
                                   new["threshold"])
                entry.update({
                    "swapped": True, "parity": "PASS",
                    "new_threshold": new["threshold"],
                    "new_auc": new["eval"].get("auc_latent_summu2"),
                    "old_auc": old["auc"],
                })
                print(f"[retrain-trigger] SWAPPED -> {out_dir} "
                      f"thr={new['threshold']:.6g} "
                      f"auc {old['auc']} -> {entry['new_auc']}")
            retrains.append(entry)
            _write_log(args.reports_dir, retrains)
            if len(retrains) >= args.max_retrains:
                print("[retrain-trigger] max retrains reached; exiting")
                break
    finally:
        consumer.close()
        _write_log(args.reports_dir, retrains)
        print(f"[retrain-trigger] done: {len(retrains)} retrain(s)")


def _read_old_auc(phys_cfg: Dict[str, Any]) -> float | None:
    try:
        m = json.loads((resolve_path(phys_cfg.get("reports_dir",
                                                  "reports/phase0"))
                        / "physics_metrics.json").read_text())
        return m.get("auc_latent_summu2")
    except OSError:
        return None


def _write_log(reports_dir: str, retrains) -> None:
    out = resolve_path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrain_log.json").write_text(
        json.dumps({"generated": datetime.now(timezone.utc).isoformat(),
                    "retrains": retrains}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
