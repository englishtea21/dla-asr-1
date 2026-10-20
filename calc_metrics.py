import argparse
import json
import statistics
from pathlib import Path
from src.metrics.utils import calc_cer, calc_wer


def from_predictions(pred_dir: Path):
    pred_file = pred_dir / "predictions.jsonl"
    if not pred_file.exists():
        candidates = list(pred_dir.rglob("predictions.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No predictions.jsonl in {pred_dir}")
        pred_file = candidates[0]
    rows = [
        json.loads(line)
        for line in pred_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cers = [r.get("cer", 0.0) for r in rows]
    wers = [r.get("wer", 0.0) for r in rows]
    print(f"num_samples: {len(rows)}")
    print(f"CER: {statistics.mean(cers) if cers else 0.0:.4f}")
    print(f"WER: {statistics.mean(wers) if wers else 0.0:.4f}")
    return rows


def from_txt_dirs(gt_dir: Path, pred_dir: Path):

    def read(p):
        return p.read_text(encoding="utf-8").strip()

    gt_files = {p.stem: p for p in gt_dir.glob("*.txt")}
    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}
    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        raise FileNotFoundError("No matching .txt files between gt and pred dirs")
    cers, wers = [], []
    for stem in common:
        t = read(gt_files[stem])
        p = read(pred_files[stem])

        cers.append(calc_cer(t, p))
        wers.append(calc_wer(t, p))
    print(f"num_samples: {len(common)}")
    print(f"CER: {statistics.mean(cers):.4f}")
    print(f"WER: {statistics.mean(wers):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--pred_file", type=str, default=None)
    args = parser.parse_args()

    if args.pred_file:
        rows = [
            json.loads(line)
            for line in Path(args.pred_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cers = [r.get("cer", 0.0) for r in rows]
        wers = [r.get("wer", 0.0) for r in rows]
        print(f"num_samples: {len(rows)}")
        print(f"CER: {statistics.mean(cers) if cers else 0.0:.4f}")
        print(f"WER: {statistics.mean(wers) if wers else 0.0:.4f}")
    elif args.pred_dir and args.gt_dir:
        from_txt_dirs(Path(args.gt_dir), Path(args.pred_dir))
    elif args.pred_dir:
        from_predictions(Path(args.pred_dir))
    else:
        raise SystemExit("Provide --pred_dir (and optionally --gt_dir) or --pred_file")


if __name__ == "__main__":
    main()
