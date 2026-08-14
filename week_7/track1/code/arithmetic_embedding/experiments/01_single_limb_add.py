"""Train/evaluate single-limb addition, including carry overflow."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from train_addition import main


if __name__ == "__main__":
    sys.argv.extend(["--min-limbs", "1", "--max-limbs", "1"])
    main()
