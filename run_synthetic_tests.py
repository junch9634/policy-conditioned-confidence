"""Run only bundled synthetic tests. Optional report contains no local paths."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest

import numpy
import scipy
import torch
import PIL

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), end="")
    sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ROOT.rglob("*.py")) if "__pycache__" not in path.parts
    }
    report = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped),
        "runtime": {"python": platform.python_version(), "numpy": numpy.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__,
                    "pillow": PIL.__version__, "device": "CPU"},
        "scope": "FICTITIOUS_IDS_AND_SYNTHETIC_ARRAYS_ONLY",
        "real_data_loaded": False, "optimizer_created": False, "optimizer_steps": 0,
        "vlm_loaded": False, "new_scientific_results": False,
        "code_sha256": sources,
        "test_log": stream.getvalue().splitlines(),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
