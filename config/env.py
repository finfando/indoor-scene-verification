from pathlib import Path
import os

TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")
NETVLAD_CHECKPOINT = os.getenv("NETVLAD_CHECKPOINT")
INDOORHACK_CHECKPOINT = os.getenv("INDOORHACK_CHECKPOINT")
SCAN_DATA_PATH = Path(os.getenv("SCAN_DATA_PATH"))
NTHREADS = int(os.getenv("NTHREADS", 1))
