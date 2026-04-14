# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimesFM MCP Server — expose TimesFM forecasting as MCP tools.

Usage:
    # stdio transport (for Claude Desktop / Cursor / local agents)
    python server.py

    # streamable-http transport (for remote agents)
    python server.py --transport streamable-http --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("timesfm-mcp")

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_model = None
_model_compiled_config: dict[str, Any] | None = None


def _get_model(
  max_context: int = 1024,
  max_horizon: int = 256,
  per_core_batch_size: int = 32,
  normalize_inputs: bool = True,
  use_continuous_quantile_head: bool = True,
  force_flip_invariance: bool = True,
  infer_is_positive: bool = True,
  fix_quantile_crossing: bool = True,
  return_backcast: bool = False,
):
  """Return the TimesFM model singleton, recompiling only when config changes."""
  global _model, _model_compiled_config

  import timesfm  # deferred so server starts fast

  requested = dict(
    max_context=max_context,
    max_horizon=max_horizon,
    per_core_batch_size=per_core_batch_size,
    normalize_inputs=normalize_inputs,
    use_continuous_quantile_head=use_continuous_quantile_head,
    force_flip_invariance=force_flip_invariance,
    infer_is_positive=infer_is_positive,
    fix_quantile_crossing=fix_quantile_crossing,
    return_backcast=return_backcast,
  )

  if _model is None:
    logger.info("Loading TimesFM model from HuggingFace …")
    import torch

    torch.set_float32_matmul_precision("high")
    _model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-2.5-200m-pytorch")
    )

  if _model_compiled_config != requested:
    import timesfm

    logger.info("Compiling model with config: %s", requested)
    _model.compile(timesfm.ForecastConfig(**requested))
    _model_compiled_config = requested

  return _model


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
  "timesfm",
  instructions=(
    "TimesFM — zero-shot time-series forecasting. "
    "Call `forecast` for basic univariate forecasting, "
    "`forecast_with_covariates` when exogenous variables are available, "
    "and `check_system` to verify hardware before first use."
  ),
)


# ---------------------------------------------------------------------------
# Tool: forecast
# ---------------------------------------------------------------------------


@mcp.tool()
def forecast(
  inputs: list[list[float]],
  horizon: int,
  max_context: int = 1024,
  max_horizon: int = 256,
  per_core_batch_size: int = 32,
  normalize_inputs: bool = True,
  use_continuous_quantile_head: bool = True,
  force_flip_invariance: bool = True,
  infer_is_positive: bool = True,
  fix_quantile_crossing: bool = True,
) -> dict[str, Any]:
  """Forecast one or more univariate time series with TimesFM 2.5.

  Each element of *inputs* is a 1-D list of floats representing a single
  time series context (history).  The model returns *horizon* future
  time-steps for every series, plus calibrated quantile prediction
  intervals (10th–90th percentile).

  Returns a dict with keys:
    - point_forecast: list[list[float]]  (n_series × horizon)
    - quantile_forecast: list[list[list[float]]]  (n_series × horizon × 10)
      quantile indices: 0=mean, 1=q10, 2=q20, …, 5=q50 (median), …, 9=q90
  """
  if not inputs:
    return {"error": "inputs must be a non-empty list of time series"}
  if horizon < 1:
    return {"error": "horizon must be >= 1"}

  effective_max_horizon = max(max_horizon, horizon)

  model = _get_model(
    max_context=max_context,
    max_horizon=effective_max_horizon,
    per_core_batch_size=per_core_batch_size,
    normalize_inputs=normalize_inputs,
    use_continuous_quantile_head=use_continuous_quantile_head,
    force_flip_invariance=force_flip_invariance,
    infer_is_positive=infer_is_positive,
    fix_quantile_crossing=fix_quantile_crossing,
  )
  np_inputs = [np.array(s, dtype=np.float32) for s in inputs]
  point, quantiles = model.forecast(horizon=horizon, inputs=np_inputs)
  return {
    "point_forecast": point.tolist(),
    "quantile_forecast": quantiles.tolist(),
  }


# ---------------------------------------------------------------------------
# Tool: forecast_with_covariates
# ---------------------------------------------------------------------------


@mcp.tool()
def forecast_with_covariates(
  inputs: list[list[float]],
  horizon: int,
  dynamic_numerical_covariates: dict[str, list[list[float]]] | None = None,
  dynamic_categorical_covariates: (dict[str, list[list[str | int]]] | None) = None,
  static_numerical_covariates: dict[str, list[float]] | None = None,
  static_categorical_covariates: (dict[str, list[str | int]] | None) = None,
  xreg_mode: str = "xreg + timesfm",
  normalize_xreg_target_per_input: bool = True,
  ridge: float = 0.0,
  max_context: int = 1024,
  max_horizon: int = 256,
  per_core_batch_size: int = 32,
) -> dict[str, Any]:
  """Forecast with exogenous variables (covariates) using TimesFM 2.5 + XReg.

  Same as `forecast`, but accepts additional dynamic or static covariates
  (price, promotions, holidays, day-of-week, etc.).  Requires the `xreg`
  extra: ``pip install timesfm[xreg]``.

  Dynamic covariates must span **both** the context and forecast windows
  (length = context + horizon).

  Returns the same dict shape as `forecast`.
  """
  if not inputs:
    return {"error": "inputs must be a non-empty list of time series"}
  if horizon < 1:
    return {"error": "horizon must be >= 1"}

  has_covariates = any(
    [
      dynamic_numerical_covariates,
      dynamic_categorical_covariates,
      static_numerical_covariates,
      static_categorical_covariates,
    ]
  )
  if not has_covariates:
    return {
      "error": (
        "At least one covariate dict must be provided. "
        "Use the `forecast` tool for univariate forecasting."
      )
    }

  effective_max_horizon = max(max_horizon, horizon)

  model = _get_model(
    max_context=max_context,
    max_horizon=effective_max_horizon,
    per_core_batch_size=per_core_batch_size,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=True,
    fix_quantile_crossing=True,
    return_backcast=True,  # required for covariates
  )
  np_inputs = [np.array(s, dtype=np.float32) for s in inputs]

  point_list, quant_list = model.forecast_with_covariates(
    inputs=np_inputs,
    dynamic_numerical_covariates=dynamic_numerical_covariates,
    dynamic_categorical_covariates=dynamic_categorical_covariates,
    static_numerical_covariates=static_numerical_covariates,
    static_categorical_covariates=static_categorical_covariates,
    xreg_mode=xreg_mode,
    normalize_xreg_target_per_input=normalize_xreg_target_per_input,
    ridge=ridge,
  )
  return {
    "point_forecast": [p.tolist() for p in point_list],
    "quantile_forecast": [q.tolist() for q in quant_list],
  }


# ---------------------------------------------------------------------------
# Tool: check_system
# ---------------------------------------------------------------------------


@mcp.tool()
def check_system(
  num_series: int = 0,
  context_length: int = 1024,
  horizon: int = 256,
  batch_size: int = 32,
) -> dict[str, Any]:
  """Check whether the current machine can run TimesFM.

  Inspects RAM, GPU/VRAM, disk space, and Python version.
  Optionally estimates memory for a specific dataset shape.

  Returns a dict with:
    - passed: bool
    - mode: "gpu" | "cpu" | "mps"
    - checks: list of individual check results
    - memory_estimate (if num_series > 0): estimated RAM/VRAM in GB
  """
  import shutil
  import platform

  result: dict[str, Any] = {"checks": [], "passed": True, "mode": "cpu"}

  # --- RAM ---
  try:
    import psutil

    ram_gb = psutil.virtual_memory().total / (1024**3)
  except ImportError:
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
  ram_ok = ram_gb >= 2.0
  result["checks"].append(
    {
      "name": "ram",
      "value_gb": round(ram_gb, 2),
      "ok": ram_ok,
      "message": (
        f"RAM {ram_gb:.1f} GB" + (" (>=4 GB recommended)" if ram_gb < 4.0 else "")
      ),
    }
  )
  if not ram_ok:
    result["passed"] = False

  # --- GPU ---
  gpu_available = False
  vram_gb = 0.0
  try:
    import torch

    if torch.cuda.is_available():
      gpu_available = True
      vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
      result["mode"] = "gpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      gpu_available = True
      result["mode"] = "mps"
  except ImportError:
    pass
  result["checks"].append(
    {
      "name": "gpu",
      "available": gpu_available,
      "vram_gb": round(vram_gb, 2) if vram_gb else None,
      "ok": True,
      "message": (
        f"GPU ({result['mode']}, {vram_gb:.1f} GB VRAM)"
        if gpu_available
        else "CPU-only (slower but works)"
      ),
    }
  )

  # --- Disk ---
  disk_free_gb = shutil.disk_usage(os.path.expanduser("~")).free / (1024**3)
  disk_ok = disk_free_gb >= 2.0
  result["checks"].append(
    {
      "name": "disk",
      "free_gb": round(disk_free_gb, 2),
      "ok": disk_ok,
      "message": f"Disk free {disk_free_gb:.1f} GB",
    }
  )
  if not disk_ok:
    result["passed"] = False

  # --- Python ---
  py_version = platform.python_version()
  py_ok = sys.version_info >= (3, 10)
  result["checks"].append(
    {
      "name": "python",
      "version": py_version,
      "ok": py_ok,
      "message": f"Python {py_version}" + ("" if py_ok else " (3.10+ required)"),
    }
  )
  if not py_ok:
    result["passed"] = False

  # --- Memory estimate ---
  # Formula from SKILL.md:
  #   RAM ≈ 0.8 GB (model) + 0.5 GB (overhead)
  #       + (0.2 MB × num_series × context_length / 1000)
  # batch_gb accounts for per-batch activation memory.
  if num_series > 0:
    model_gb = 0.8
    overhead_gb = 0.5
    data_gb = 0.2e-3 * num_series * context_length / 1_000  # 0.2 MB per 1k points
    batch_gb = 0.1e-3 * batch_size * (context_length + horizon) / 1_000
    total_gb = model_gb + overhead_gb + data_gb + batch_gb
    fits = total_gb < ram_gb
    result["memory_estimate"] = {
      "total_gb": round(total_gb, 2),
      "model_gb": model_gb,
      "data_gb": round(data_gb, 2),
      "fits_in_ram": fits,
    }
    if not fits:
      result["passed"] = False

  return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(description="TimesFM MCP Server")
  parser.add_argument(
    "--transport",
    choices=["stdio", "streamable-http", "sse"],
    default="stdio",
    help="MCP transport (default: stdio)",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Port for HTTP transports (default: 8000)",
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
  )

  if args.transport == "stdio":
    mcp.run(transport="stdio")
  else:
    mcp.run(transport=args.transport, port=args.port)


if __name__ == "__main__":
  main()
