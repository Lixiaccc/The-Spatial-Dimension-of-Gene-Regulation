import logging
import sys

# -----------------------------------------------------------------------------
# Robust import of EpiFoundation
#   - If project is installed as a package: EpiFoundation.model.EpiFoundation
#   - If running from source in this repo:  .EpiFoundation
#   - If import fails, we just set EpiFoundation = None so code that only needs
#     logger still works (like dataset checks).
# -----------------------------------------------------------------------------
try:
    # case 1: installed as a package "EpiFoundation"
    from EpiFoundation.model.EpiFoundation import EpiFoundation  # type: ignore
except Exception:
    try:
        # case 2: running from the repo root
        from .EpiFoundation import EpiFoundation  # type: ignore
    except Exception:
        # case 3: we don't actually need the class (e.g. dataset checks)
        EpiFoundation = None  # type: ignore

# -----------------------------------------------------------------------------
# Logger (used by data/preprocess.py and others)
# -----------------------------------------------------------------------------
logger = logging.getLogger("scMultiomics")

# only initialize once
if not logger.hasHandlers():
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

