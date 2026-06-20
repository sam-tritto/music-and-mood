#!/usr/bin/env python3
"""
Download Spotify Charts Dataset from Kaggle
=============================================
Run this script after configuring your Kaggle API credentials:

1. Go to https://www.kaggle.com/settings → API → Create New Token
2. Move the downloaded file:
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
3. Run this script:
   uv run python scripts/download_kaggle_data.py

The script downloads the Spotify Charts dataset and places it in data/raw/.
"""

import os
import sys
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmi.config import DATA_RAW


def download_dataset():
    """Download the Spotify Charts dataset from Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # Primary dataset: Billboard Hot 100 charts with audio features
    dataset_slug = "thedevastator/billboard-hot-100-audio-features"
    download_dir = DATA_RAW

    print(f"📥 Downloading dataset: {dataset_slug}")
    print(f"   Target directory: {download_dir}")

    download_dir.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(
        dataset_slug,
        path=str(download_dir),
        unzip=True,
    )

    # List downloaded files
    files = list(download_dir.glob("*"))
    print(f"\n✅ Downloaded {len(files)} files:")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        print(f"   {f.name} ({size_mb:.1f} MB)")

    print("\n🎵 Next steps:")
    print("   1. Open the notebook: music_and_mood_trends.ipynb")
    print("   2. The loader will detect the CSV automatically")
    print("   3. If the dataset doesn't have audio features pre-joined,")
    print("      you may need to also download an audio features dataset.")


if __name__ == "__main__":
    # Check for kaggle credentials
    import os
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_credentials = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    
    if not kaggle_json.exists() and not has_env_credentials:
        print("❌ Kaggle credentials not found!")
        print()
        print("To set up Kaggle API access:")
        print("  Option 1: Add credentials to your .env file:")
        print("     KAGGLE_USERNAME=your_username")
        print("     KAGGLE_KEY=your_api_key")
        print()
        print("  Option 2: Place your kaggle.json:")
        print("     mkdir -p ~/.kaggle")
        print("     mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("     chmod 600 ~/.kaggle/kaggle.json")
        print()
        print("Then re-run this script.")
        sys.exit(1)

    download_dataset()
