# TimesFM MCP Server

An [MCP](https://modelcontextprotocol.io) server that exposes
[TimesFM 2.5](https://github.com/google-research/timesfm) time-series
forecasting as agent-callable tools. Any MCP-compatible client (Claude
Desktop, Cursor, OpenAI Agents SDK, etc.) can call `forecast`,
`forecast_with_covariates`, and `check_system` without writing Python.

## Quick start

```bash
# Install
pip install timesfm[torch] "mcp>=1.0.0"

# Run (stdio — for Claude Desktop / Cursor)
python server.py

# Run (HTTP — for remote agents)
python server.py --transport streamable-http --port 8000
```

## Tools

| Tool | Description |
|------|-------------|
| `forecast` | Zero-shot univariate forecasting. Accepts one or more 1-D series and a horizon, returns point forecasts + calibrated quantile intervals (q10–q90). |
| `forecast_with_covariates` | Forecast with exogenous variables (price, holidays, etc.) via TimesFM XReg. Requires `pip install timesfm[xreg]`. |
| `check_system` | Hardware preflight — checks RAM, GPU/VRAM, disk, Python version. Optionally estimates memory for a given dataset shape. |

## Client configuration

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "timesfm": {
      "command": "python",
      "args": ["/absolute/path/to/timesfm-mcp-server/server.py"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "timesfm": {
      "command": "python",
      "args": ["/absolute/path/to/timesfm-mcp-server/server.py"]
    }
  }
}
```

### Remote (HTTP)

```bash
python server.py --transport streamable-http --port 8000
# Connect your client to http://localhost:8000/mcp
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMESFM_CHECKPOINT` | `google/timesfm-2.5-200m-pytorch` | HuggingFace model ID or local path |

## Architecture

```
LLM Agent ─── MCP protocol ──► server.py ──► TimesFM model (GPU/CPU)
                                  │
                                  ├─ forecast()
                                  ├─ forecast_with_covariates()
                                  └─ check_system()
```

The model is loaded lazily on first tool call and kept in memory. Config
changes trigger recompilation but do **not** reload weights.

## Example tool calls

### forecast

```json
{
  "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
  "horizon": 4
}
```

Response:

```json
{
  "point_forecast": [[9.01, 10.02, 11.01, 12.03]],
  "quantile_forecast": [[[...]]]
}
```

### check_system

```json
{
  "num_series": 1000,
  "context_length": 1024,
  "horizon": 24,
  "batch_size": 32
}
```

Response:

```json
{
  "passed": true,
  "mode": "gpu",
  "checks": [...],
  "memory_estimate": {
    "total_gb": 1.52,
    "model_gb": 0.8,
    "data_gb": 0.2,
    "fits_in_ram": true
  }
}
```

## Relationship to Agent Skills

This MCP server complements the existing
[Agent Skill](../timesfm-forecasting/SKILL.md) — the skill teaches coding
agents _how to write TimesFM code_, while this server lets any agent _call
TimesFM directly_ without generating code.

## License

Apache-2.0
