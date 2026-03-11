# Data Format

The project expects a canonical `.npz` format for both SEED and DEAP adapters.

Required keys:
- `x`: EEG array
  - feature mode: `[N, T, C, Bf]`
  - raw mode: `[N, C, T]` or `[N, T, C]`
- `y`: labels `[N]` (or dataset-specific alternatives documented in README)
- `subject_id`: subject ids `[N]`

Optional keys:
- `session_id`: `[N]`
- `trial_id`: `[N]`
- `valence` / `arousal` for DEAP if `y` is not directly provided.

Use scripts in `scripts/` to preprocess raw EEG into cached DE features.
