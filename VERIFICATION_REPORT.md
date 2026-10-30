# MDExplorer — Verification Report

Scope: validate `analysis.py` performance (RMSD, RMSF, Rg, hbonds, SASA), the
`render_structure` high-quality toggle, and a full end-to-end run of the app.
All testing was done in a sandboxed VM, not the user's own machine — see
Limitations for why that matters.

---

## 1. What was done

### 1.1 Fixed a real bug: `render_structure` was ray-tracing twice
`viewer.render_structure(..., high_quality=True)` called `cmd.ray(width, height)`
and then `cmd.png(..., ray=1)`. `png(ray=1)` already ray-traces internally, so
the explicit `cmd.ray()` call before it rendered the same image a second time
for no benefit.

Isolated A/B/C timing (same frame, same session, 900x520):

| Call pattern | Time |
|---|---|
| `png(ray=0)` | 4.41s |
| `png(ray=1)` only | 4.33s |
| `ray()` then `png(ray=1)` (old code) | 8.16s |

All three produced byte-identical PNGs (sha256 match), confirming the second
ray-trace was pure waste. **Fix applied**: removed the redundant `cmd.ray()`
call in `viewer.py`. This halves the cost of `high_quality=True` wherever
ray-tracing is actually happening on the host.

### 1.2 Diagnosed the `calculate_hbonds` slowdown — not a code regression
An earlier benchmark showed `calculate_hbonds`'s per-frame vectorized loop at
~0.14s; later runs showed ~7-13s for the same code, unchanged. Investigated
and ruled out, in order:
- **`OMP_NUM_THREADS=1`** (set at the top of `analysis.py` for an mdtraj
  OpenMP workaround) — A/B synthetic benchmark showed no significant
  difference (~5.0s vs ~5.3s) between forcing it to 1 and leaving it unset.
- **Run-to-run/process noise** — 5 repeated trials in a single process were
  stable at ~6.6-7.1s each, ruling out one-off JIT/cache warmup effects.
- **Confirmed root cause**: the sandbox host itself was under heavy
  contention — CPU pinned at 100% across all 12 logical cores, and system
  memory climbing from 93.9% to 97%+ used (under 1GB free of 16GB) during
  testing, later re-confirmed at 100% CPU / 89.7% memory on a separate check.
  The processes consuming memory were unrelated desktop applications (Chrome,
  VS Code, "altimate.exe", "Kilo", "ChatGPT Classic"), not anything from this
  project. **The `calculate_hbonds` code itself was not the problem** — the
  original ~0.14s per-frame-loop baseline is likely still an accurate
  reflection of the algorithm's real cost on an unloaded machine.

### 1.3 Confirmed `calculate_sasa` is broken in this sandbox, not just slow
`analysis.calculate_sasa` now delegates to `sasa_freesasa.calculate_sasa_freesasa`,
which requires `import freesasa`. In this sandbox, `freesasa` is listed in
`requirements.txt` but:
- Has no prebuilt wheel for the installed Python (3.14)
- Fails to build from source without MSVC Build Tools, which are not present

Confirmed by direct call: `calculate_sasa` raises
`ModuleNotFoundError: No module named 'freesasa'`. This is **caught
gracefully** by `app.py`'s `run_all_analysis`, which shows "SASA: Not
available" with the error message rather than crashing — so the app degrades
correctly, but the SASA feature does not work at all in this environment.
No code change was made here per your explicit instruction not to add a
fallback — `analysis.calculate_sasa` is unchanged.

### 1.4 Full end-to-end run, mirroring the real "Load simulation" click
Could not drive `app.py`'s file-upload widget through Streamlit's headless
`AppTest` framework (it has no `file_uploader` simulation support), so I
verified in two complementary ways:

**a) The real server starts and serves correctly.**
Launched `streamlit run app.py --server.headless true` as the user would,
waited for boot, then did a plain HTTP GET: got back `200 OK`, a Streamlit
page body, and no errors in server logs. Shut the server down afterward.

**b) The exact logic `main()` runs on click, executed directly with real data.**
Ran `load_simulation` → every measurement in `run_all_analysis`'s dict
(`rmsd`, `rmsf`, `rg`, `com`, `hbonds`, `sasa`, `secondary`) → periodic-split
check → both render qualities → CSV export, using a real topology+trajectory
pair (7,148 atoms, loaded at stride 50 → 201 frames), with the same
try/except/None-fallback pattern `run_all_analysis` uses.

Result: everything succeeded except `sasa`, which failed with the expected
`ModuleNotFoundError` and was caught cleanly (matches section 1.3 exactly —
no crash, no `KeyError`, no corrupted downstream CSV).

### 1.5 Cleanup
All temporary diagnostic/benchmark scripts created during this investigation
(`_benchmark.py`, `_diagnose*.py`, `_probe.py`, `_check_wheels.py`,
`_verify_hbonds*.py`, `_rerun_full_benchmark.py`, `_cpu_baseline.py`,
`_omp_test.py`, `_variance_check.py`, `_sysinfo*.py`, `_render_bench*.py`,
`_e2e_check.py`, `_http_check.py`, `_kill_streamlit.py`, and their log/output
files) were deleted. The project directory now contains only the original
source files plus this report.

---

## 2. How long things actually take

All numbers below are from **this sandboxed VM, under host contention**
(100% CPU across all 12 cores, ~90-97% system memory used by *other*
unrelated applications during testing). Treat these as upper bounds, not
representative of a normal/idle machine — see Limitations.

**Full pipeline, 7,148-atom protein, 201 frames (stride 50 of a 10,001-frame trajectory):**

| Step | Time |
|---|---|
| `load_simulation` | 0.36s |
| `calculate_rmsd` | 0.14s |
| `calculate_rmsf` | 0.04s |
| `calculate_rg` | 0.28s |
| `center_of_mass` | 0.10s |
| `calculate_hbonds` | **12.7s** (contention-inflated — see §1.2; expected much lower, roughly in line with the app's own help text of "~0.4s/frame for hydrogen bonds" at full un-strided scale) |
| `calculate_sasa` | 0.003s (fails fast — not installed, no computation attempted) |
| `calculate_secondary_structure` | 0.98s |
| `check_periodic_split` | 0.02s |
| `render_structure` (fast) | 2.80s |
| `render_structure` (high quality, post-fix) | 2.74s (statistically the same as fast here — see §3; pre-fix this would have been ~double) |
| `per_frame_csv` | 0.03s |
| **Total (excluding SASA, excluding render)** | **~14.6s** |
| **Total (including both renders)** | **~20.2s** |

For reference, the app's own UI help text (written by whoever built it,
presumably benchmarked on a normal machine) states "~3s/frame for SASA and
~0.4s/frame for hydrogen bonds" on a 7,000-atom protein — so on an unloaded
machine with SASA working, a full un-strided 10,001-frame run would be
dominated by SASA (~3s × 10,001 ≈ 8+ hours) and hydrogen bonds (~0.4s ×
10,001 ≈ 67 min), which is exactly why the app defaults SASA off and
recommends striding for first looks. The `stride` and `with_sasa` controls
exist precisely to manage this.

**Streamlit server boot time:** under 10s to become reachable over HTTP from
a cold `streamlit run` invocation.

---

## 3. Limitations

1. **This sandbox cannot verify visual/rendering correctness.** `ray=0` and
   `ray=1` produced byte-identical PNGs here (sha256 match), which strongly
   suggests the embedded headless PyMOL session is not actually doing real
   ray-tracing in this environment (likely missing GPU/software-rasterizer
   support). The code fix in §1.1 is correct and will reduce real ray-tracing
   cost by ~2x wherever ray-tracing genuinely happens, but **I cannot confirm
   the two render qualities look visually different on your machine** — that
   needs checking locally.

2. **Timing numbers in this report are from a contended host**, not a clean
   baseline. CPU was pinned at 100% across all cores and memory 90%+ used by
   unrelated desktop applications throughout testing (confirmed via `psutil`,
   re-checked at two separate points, both showing contention). Absolute
   numbers here — especially the 12.7s hbonds figure — are likely several
   times slower than what you'd see on an idle machine. Relative
   comparisons within a single controlled test (e.g., the ray=0/1/double-ray
   A/B/C test, all run back-to-back in one process) are still trustworthy.

3. **`freesasa` cannot be installed in this sandbox at all** — no prebuilt
   wheel for the installed Python version, and no MSVC Build Tools available
   to build it from source. This is a sandbox-specific limitation, not
   necessarily true of your own machine. SASA could not be exercised with
   real computation here; only its failure path was verified. Per your
   instruction, no fallback or resilience was added to `calculate_sasa` — it
   still hard-requires `freesasa` and will stay "Not available" wherever that
   import fails.

4. **No automated UI interaction was possible.** Streamlit's `AppTest`
   headless framework does not support simulating `file_uploader`, so the
   literal "upload a file and click the Load button" flow was not driven
   through a browser or test harness. I verified the server starts and
   serves (HTTP 200, correct markup, no startup errors), and separately
   verified every function `main()` calls, with the same data and the same
   error-handling pattern, directly in Python. This is strong but not
   identical evidence to a real click-through.

5. **Only one topology/trajectory pair was tested** (a 7,148-atom protein,
   10,001 frames, found as leftover temp files from earlier sessions — no
   sample data ships with the repo itself). Edge cases mentioned elsewhere in
   the codebase (single-frame files, periodic-boundary-split proteins,
   multiple chains with repeated residue numbers, missing hydrogens for
   hbonds, TPR/PRMTOP formats) were not exercised in this verification pass.

---

## 4. What needs to be worked on next

1. **Verify the render fix visually on a real machine.** Load the app, toggle
   "High quality render" on and off on the same frame, and confirm the two
   images actually look different (ray-traced shadows/lighting vs flat
   shading) and that the high-quality one is now noticeably faster than
   before this fix (previously double-cost from the redundant `cmd.ray()`
   call).

2. **Resolve the `freesasa` dependency for real use.** Either:
   - Install it properly on the target machine (`pip install freesasa`,
     possibly needing MSVC Build Tools if no wheel matches the Python
     version), or
   - If the target environment can't support it either, decide deliberately
     whether to keep `calculate_sasa` freesasa-only (current behavior: clean
     "Not available" message) or revisit the fallback question later — this
     is explicitly left untouched per your instruction, but it's a real gap
     worth a conscious decision rather than leaving it to "whatever happens
     to be installed."
   - Also worth a small follow-up: `sasa_freesasa.py`'s own module docstring
     still says *"This file is NOT used by the app yet"* — that's now
     inaccurate since `analysis.calculate_sasa` calls it directly. Worth a
     one-line doc fix whenever you're next in that file (not changed here,
     since you asked me not to touch the SASA path).

3. **Re-run the hbonds and full-pipeline timing on an idle machine** to get a
   trustworthy baseline now that host contention has been identified as the
   cause of the earlier ~7-13s readings. The current numbers in §2 are not
   suitable for any performance claims or documentation.

4. **Consider a real browser-driven smoke test** (e.g. Playwright/Selenium
   against `streamlit run`) if UI-level regression testing matters going
   forward, since `AppTest` can't cover the file-upload flow that's the
   actual entry point to the app.

5. **Broaden test coverage** beyond the single sample file used here —
   especially a periodic-boundary-split case and a file with no hydrogens
   (to confirm `calculate_hbonds`'s `ValueError` path), since those are
   handled in the code but weren't exercised in this verification pass.

---

## Update (2026-10-30): after the move to freesasa

The sections above describe the sandbox session, where `freesasa` could not be installed.
It has since been installed and tested on the author's PC (Windows, 12 logical cores, Python 3.11),
and `calculate_sasa` has changed. What was re-checked:

- **SASA works.** `calculate_sasa` now uses `freesasa.calcCoord` (radii looked up once from
  frame 0). On five frames of the Adenylate Kinase test trajectory the totals were within
  0.03% to 0.74% of mdtraj's `shrake_rupley` (different atomic radii). 100 points instead of 960
  changed one frame's area by about 0.5%.
- **Hydrogen bonds** were rewritten with a KD-tree. The per-frame counts are identical to the old
  code on all 98 frames (24,707 bonds in total), and about 15x faster.
- **Real trajectory**: 10,001 frames, 7,148 protein atoms, every frame, Fast SASA: all seven
  measurements finished in about 220 s (SASA 157 s, secondary structure 23 s, hydrogen bonds 12 s,
  radius of gyration 12 s, loading 5 s).
- **Streamlit on Windows.** The SASA worker processes re-imported `app.py` and each one ran the whole
  app, so loading with SASA ticked never finished. Fixed by running `main()` only under
  `if __name__ == "__main__"`. With the fix, the app loaded the test trajectory with SASA ticked in
  a real browser (Playwright). The browser run only checked that the page loaded; it did not
  compare SASA values.
- **Coincident atoms**: freesasa does not crash on two atoms at the same position (checked with a
  made-up 4-atom example), so the old `check_no_overlap` was removed.
- Not repeated: the `render_structure` and periodic-split checks above.

