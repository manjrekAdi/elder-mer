# Environment setup

Two machines, one shared requirements file. PyTorch differs by platform and is
installed by the per-machine scripts, not by `requirements-common.txt`.

| Machine | Script | PyTorch build | Role |
|---|---|---|---|
| M4 MacBook Pro | `setup_m4.sh` | MPS (Apple Silicon) | code, debug, small tests, Whisper inference, feature extraction |
| Atlas server | `setup_atlas.sh` | CUDA 12.8 (cu128, Blackwell sm_120) | real training runs |

## Usage

On each machine, from the repo root:

```bash
bash scripts/setup/setup_m4.sh      # on the Mac
# or
bash scripts/setup/setup_atlas.sh   # on Atlas
```

Both create/reuse a conda env named `elder-mer` (Python 3.11), install the
matching PyTorch, then the shared requirements, then `openai-whisper`. Each
script ends with a verification step (MPS check on the M4, a CUDA + sm_120
smoke test on Atlas).

## Notes / gotchas

- **Whisper is installed separately** with `--no-build-isolation`. Its build
  expects `setuptools`/`pkg_resources` to be present, which fails in an
  isolated build env on clean Python 3.12. Both scripts install
  `setuptools wheel` first to avoid this.
- **OpenFace is not a pip package.** It is a separate C++ toolkit for video
  action units: https://github.com/TadasBaltrusaitis/OpenFace . Install it
  outside conda; the scripts only print a reminder.
  **Installed on the M4 (2026-06-10)** at `~/tools/OpenFace`, binaries in
  `~/tools/OpenFace/build/bin` (`FeatureExtraction` is the one the pipeline
  uses). Building on macOS arm64 needed three fixes beyond
  `brew install cmake opencv dlib openblas wget`:
  1. `OpenBLAS_HOME=/opt/homebrew/opt/openblas` when running cmake —
     OpenFace's `FindOpenBLAS.cmake` only searches the Intel-Homebrew prefix.
  2. Wrap the `target_link_libraries(... stdc++fs)` lines in the three
     `lib/local/*/CMakeLists.txt` with `if(NOT APPLE)` — `stdc++fs` is a
     GCC-only library; libc++ has `std::filesystem` built in.
  3. In `stdafx_ut.h` / `stdafx_fa.h` / `stdafx.h`, reorder the filesystem
     `#if __has_include` chain to prefer `<filesystem>` over Boost —
     otherwise brew's boost headers get picked up at compile time without
     the matching libs at link time.
  The CEN patch-expert `.dat` models are not in git; the old Dropbox links
  in `download_models.sh` still work if you append `?dl=1`.
- **numpy is pinned <2.0** on purpose; several audio libraries are not yet
  numpy-2.x clean.
- **wandb**: run `wandb login` once before the first tracked experiment.
- On Atlas, set the cache env vars (printed by the script) so HF/Whisper
  downloads land on the 750 GB `~/data` volume, not your home quota.
- If the Atlas sm_120 smoke test warns, the cu128 wheel may lag the Blackwell
  GPU; check pytorch.org for a newer build and flag to Hamidreza.
