from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_models_compile import *  # noqa: F401,F403


if __name__ == "__main__":
    test_file = Path(__file__).with_name("test_models_compile.py")
    raise SystemExit(pytest.main([str(test_file), *sys.argv[1:]]))
