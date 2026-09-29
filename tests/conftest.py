# SPDX-License-Identifier: Apache-2.0
"""Session-wide setup that has to run before any test module is imported."""
from __future__ import annotations

import torch

# Importing sglang_omni maps a second Level-Zero loader, and on Intel hosts a loader
# that lands before the XPU runtime leaves the process unable to initialize a device
# at all. pytest imports this file before it collects test modules, which is the only
# point early enough to order the two.
if torch.xpu.is_available():
    try:
        torch.xpu.init()
    except RuntimeError:
        # A host without a usable XPU still has to collect and run the rest.
        pass
else:
    pass
