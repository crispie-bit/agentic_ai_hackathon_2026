"""Puts lab/ on sys.path so these files can `from _common import ...`.

Python puts the SCRIPT's folder on sys.path, not the folder you ran it from —
so without this, section_3_bedrock/01_invoke_model.py cannot see lab/_common.py.
Import it first, before any _common import.
"""

import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent.parent
# APPEND, do not insert(0). The script's own folder is already sys.path[0],
# and prepending lab/ would let a file there shadow a same-named file in this
# section — e.g. lab/_agentcore.py silently winning over section_6's own copy.
if str(_LAB) not in sys.path:
    sys.path.append(str(_LAB))
