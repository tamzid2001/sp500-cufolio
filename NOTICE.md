# Upstream attribution

This repository includes unmodified copies of the following notebooks from
NVIDIA AI Blueprints' `cuFOLIO` repository:

- Source: https://github.com/NVIDIA-AI-Blueprints/cuFOLIO/tree/main/notebooks
- Pinned source commit: `dd7ca07db5e6c3624af80811f64562fc28480906`
- Files: `cvar_basic.ipynb`, `efficient_frontier.ipynb`, `launchable.ipynb`,
  `mean_variance_basic.ipynb`, and `rebalancing_strategies.ipynb`
- License: Apache License 2.0 (copied to `LICENSE`)

The copied notebooks live in `upstream_notebooks/` and remain GPU-oriented.
The CPU notebooks in `notebooks/` are new, deliberately smaller counterparts
that use a CPU solver. They are not endorsed by NVIDIA and do not provide
cuOpt, RAPIDS, or GPU performance characteristics.
