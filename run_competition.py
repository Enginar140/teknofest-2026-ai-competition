"""
TEKNOFEST Havacılıkta YZ — sunucu değerlendirme döngüsü.

Çalıştırma (HY kökünden):
  cd havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
  copy config\\example.env config\\.env   # Windows
  # .env içine TEAM_NAME, PASSWORD, EVALUATION_SERVER_URL, SESSION_NAME yazın
  python main.py

Bu betik, TAKIM_BAGLANTI_ARAYUZU içinde main.py'yi çalıştırır.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    takim = root / "havacilikta-yapay-zeka-yarismasi" / "TAKIM_BAGLANTI_ARAYUZU"
    main_py = takim / "main.py"
    if not main_py.is_file():
        print("Bulunamadı:", main_py)
        sys.exit(1)
    os.chdir(takim)
    sys.path.insert(0, str(takim))
    runpy.run_path(str(main_py), run_name="__main__")


if __name__ == "__main__":
    main()
