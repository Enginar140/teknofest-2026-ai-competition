"""
TEKNOFEST Drive (data/teknofest_drive) icindeki tum .zip dosyalarini acar.

Cikti yapisi:
  data/teknofest_drive/_extracted/<ust_klasor>/<zip_adi_uzantisiz>/

Ornek:
  .../TEKNOFEST HYZ 2025 Verileri/THYZ_2025_Oturum_1.zip
  -> .../_extracted/TEKNOFEST HYZ 2025 Verileri/THYZ_2025_Oturum_1/

Zaten basariyla acilmis klasorde `.extract_ok` varsa atlanir.
Yeniden acmak icin ilgili hedef klasoru veya `.extract_ok` dosyasini silin.

Kullanim (HY kokunden):
  python extract_teknofest_zips.py
  python extract_teknofest_zips.py --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRIVE = ROOT / "data" / "teknofest_drive"
OUT_BASE = DRIVE / "_extracted"
MARKER = ".extract_ok"


def _open_zip(path: Path) -> zipfile.ZipFile:
    """ZIP metadata kodlamasi (Turkce dosya adlari)."""
    try:
        return zipfile.ZipFile(path, "r", metadata_encoding="utf-8")
    except TypeError:
        return zipfile.ZipFile(path, "r")
    except Exception:
        return zipfile.ZipFile(path, "r", metadata_encoding="cp437")


def extract_one(zip_path: Path, dest: Path, force: bool) -> tuple[str, str]:
    """
    Tek zip ac. Donus: (durum, mesaj)
    durum: ok | skip | err
    """
    ok_file = dest / MARKER
    if dest.exists() and ok_file.exists() and not force:
        return "skip", f"Atlandi (zaten var): {dest.relative_to(OUT_BASE)}"

    if dest.exists() and force:
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    try:
        with _open_zip(zip_path) as zf:
            # Zip slip korumasi: hedef her zaman dest altinda kalsin
            for info in zf.infolist():
                name = info.filename
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"Guvenli olmayan zip girisi: {name!r}")
            zf.extractall(dest)
    except zipfile.BadZipFile as e:
        shutil.rmtree(dest, ignore_errors=True)
        return "err", f"Gecersiz/bozuk zip (parcali arsiv olabilir): {zip_path.name} — {e}"
    except Exception as e:
        shutil.rmtree(dest, ignore_errors=True)
        return "err", f"Hata {zip_path.name}: {e}"

    ok_file.write_text(zip_path.name + "\n", encoding="utf-8")
    return "ok", f"Acildi: {dest.relative_to(OUT_BASE)}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Teknofest Drive zip'lerini ac")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Mevcut cikti klasorlerini silip yeniden ac",
    )
    args = ap.parse_args()

    if not DRIVE.is_dir():
        print(f"Klasor yok: {DRIVE}", file=sys.stderr)
        return 1

    zips: list[Path] = []
    for p in DRIVE.rglob("*.zip"):
        if OUT_BASE in p.parents:
            continue
        zips.append(p)
    zips.sort()

    if not zips:
        print("Zip bulunamadi.")
        return 0

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "err": 0}
    for zp in zips:
        rel_parent = zp.parent.relative_to(DRIVE)
        dest = OUT_BASE / rel_parent / zp.stem
        status, msg = extract_one(zp, dest, args.force)
        stats[status] = stats.get(status, 0) + 1
        print(f"[{status.upper():4}] {msg}")

    print(
        f"\nOzet: basarili={stats['ok']} atlanan={stats['skip']} hata={stats['err']} | hedef={OUT_BASE}"
    )
    return 0 if stats["err"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
