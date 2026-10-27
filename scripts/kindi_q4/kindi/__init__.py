"""Kindi - a custom inference engine for ternary language models on the NVIDIA Quadro K420.

Named after Abu Yusuf al-Kindi (c. 801-873), whose work on frequency analysis is the root of statistical
language modelling.
"""
import os
import subprocess

VERSION = "0.2.0-q4-experimental"


def _commit():
    """the source commit (12 hex digits) or None: kindi/COMMIT, which git archive fills in (export-subst), else git"""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = open(os.path.join(here, "COMMIT")).read().strip()
        if c and not c.startswith("$"):
            return c[:12]
    except OSError:
        pass
    try:
        r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, timeout=5)
        return r.stdout.strip()[:12] or None
    except (OSError, subprocess.SubprocessError):
        return None


COMMIT = _commit()
__version__ = VERSION + ("+" + COMMIT[:7] if COMMIT and "-" in VERSION else "")
MODEL_ID = "bonsai2-2b"
