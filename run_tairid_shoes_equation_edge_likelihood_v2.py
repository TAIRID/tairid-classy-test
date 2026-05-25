#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder equation-edge likelihood v2.

Purpose:
The row-mask candidate-boundary likelihood v2 failed because blunt 277/77 row
membership masks were almost entirely absorbed by the original 47-parameter
SH0ES compact ladder model.

The equation-edge recovery v1 then found the sharper structure:

    277 rows bridge parameter 42 and parameter 46
    active columns = 42,46
    sign pattern = +,-

Parameter 46 is the H0-like parameter:
    H0_like = 10^(param46 / 5)

This test asks a stricter question:

Does a constrained deformation of the recovered 42<->46 equation edge improve
the compact SH0ES likelihood beyond the baseline 47-parameter model?

This is NOT a row offset test.
This is NOT a simple added row-group nuisance column.
This tests small coefficient deformations inside the equation edge itself.

Models tested:
1. Baseline compact SH0ES ladder:
   y = X beta, where X = L.T

2. Edge coefficient deformations on recovered 42<->46 bridge rows:
   - scale parameter 46 coefficient on bridge rows
   - scale parameter 42 coefficient on bridge rows
   - anti-phase deformation: 42 strengthens while 46 weakens
   - common-mode deformation: both 42 and 46 scale together

3. Controls:
   - all-param46 global scale control
   - random same-count row controls
   - contiguous same-count block controls

Decision checks:
- delta chi2
- effective AIC / BIC with one extra deformation parameter
- best lambda not simply at search boundary
- random/control p-values
- parameter 46 / H0-like shift audit
- full truth boundary

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only asks whether the recovered 42<->46 equation edge carries
independent likelihood pressure under a constrained deformation.
"""

import csv
import json
import math
import re
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_shoes_equation_edge_likelihood_v2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded")
