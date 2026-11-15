# Contributing to `masked_diffusion_lm`

Contributions, bug reports, and mathematical verification improvements are welcome.

## Development Setup

1. Clone the repository and sync the isolated `uv` environment:
   ```bash
   git clone https://github.com/HamzA-1729/masked_diffusion_lm.git
   cd masked_diffusion_lm
   uv sync
   ```
2. Run the automated test suite before submitting changes:
   ```bash
   uv run pytest -v
   ```
3. Format and lint with `ruff`:
   ```bash
   uv run ruff check src tests
   ```

## Standards
- Keep implementations pure PyTorch without high-level diffusion or trainer wrappers.
- Every new mathematical invariant or sampling property must include a deterministic unit test in `tests/`.
