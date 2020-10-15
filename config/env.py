from pathlib import Path
import os
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")
NETVLAD_CHECKPOINT = os.getenv("NETVLAD_CHECKPOINT")
INDOORHACK_CHECKPOINT = os.getenv("INDOORHACK_CHECKPOINT")
SCAN_DATA_PATH = Path(os.getenv("SCAN_DATA_PATH"))
NTHREADS = int(os.getenv("NTHREADS", 1))
