"""Copy the source website into docs/ for GitHub Pages branch deployment.

Run from the repository root:
    python scripts/sync_website.py
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "website"
DESTINATION = ROOT / "docs"

if not SOURCE.is_dir():
    raise SystemExit(f"Website source folder not found: {SOURCE}")

if DESTINATION.exists():
    shutil.rmtree(DESTINATION)

shutil.copytree(SOURCE, DESTINATION)
print(f"Synced {SOURCE.relative_to(ROOT)} -> {DESTINATION.relative_to(ROOT)}")
