"""
TEKNOFEST resmi Google Drive 'Veri Seti' klasorunu HY/data/teknofest_drive altina indirir.

Kullanim (HY kok klasorunden):
  pip install gdown
  python pull_drive_data.py

Not: Klasor buyuk olabilir; indirme suresi ve disk alani gereksinimi yuksek olabilir.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Windows: Turkce dosya adlariyla gdown print Unicode hatasini onle
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

DRIVE_FOLDER_ID = "18_VqLBbyTubVSWAXG_CgmuJWGCx0mcBd"
OUT_DIR = Path(__file__).resolve().parent / "data" / "teknofest_drive"


def main() -> int:
    try:
        import gdown
    except ImportError:
        print("gdown gerekli: pip install gdown")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Hedef: {OUT_DIR}")
    print(f"Kaynak klasor ID: {DRIVE_FOLDER_ID}")
    print("Indiriliyor (bu islem uzun surebilir)...")

    gdown.download_folder(
        id=DRIVE_FOLDER_ID,
        output=str(OUT_DIR),
        quiet=False,
        use_cookies=False,
    )
    print("Tamam.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
