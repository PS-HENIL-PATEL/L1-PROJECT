"""
Enterprise RAG OS — Setup Script
===================================

Purpose:
    First-run setup script that creates necessary directories,
    copies .env template, and validates the environment.

Usage:
    python scripts/setup.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    """Run initial project setup."""
    project_root = Path(__file__).resolve().parent.parent
    print("=" * 60)
    print("  Enterprise RAG OS — Initial Setup")
    print("=" * 60)

    # Create runtime directories
    dirs = ["data", "vector_store", "cache", "logs", "reports"]
    for d in dirs:
        dir_path = project_root / d
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        print(f"  ✓ Created directory: {d}/")

    # Copy .env if not exists
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("  ✓ Created .env from .env.example")
    elif env_file.exists():
        print("  ○ .env already exists (skipped)")

    print()
    print("  Setup complete! Next steps:")
    print("  1. Edit .env with your configuration")
    print("  2. Install dependencies: pip install -e \".[dev]\"")
    print("  3. Run the server: uvicorn app.main:app --reload")
    print("  4. Visit http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
