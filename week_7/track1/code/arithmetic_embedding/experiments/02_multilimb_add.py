"""Train/evaluate the shared recurrent adder on 1--4 limb numbers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from train_addition import main


if __name__ == "__main__":
    main()
