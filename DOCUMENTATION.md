# MolNumPy — Function & Library Documentation

This document explains what every function in the project does, the algorithm/idea
behind it, and the official documentation it is based on. Use it as the reference
trail for how each piece was built.

Project files: [molecule.py](molecule.py) (file loading), [analysis.py](analysis.py)
(measurements), [viewer.py](viewer.py) (3D rendering), [app.py](app.py) (Streamlit UI).

---

## 1. molecule.py — reading simulation files

Uses **MDAnalysis** to parse the many possible topology/trajectory formats into one
plain object (`MolecularSystem`) that the rest of the app works with as NumPy arrays.

- MDAnalysis `Universe`: https://docs.mdanalysis.org/stable/documentation_pages/core/universe.html
- MDAnalysis `AtomGroup` attributes (`names`, `resids`, `resnames`, `masses`, `positions`):
  https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html
- MDAnalysis trajectory iteration (`for ts in universe.trajectory`):
  https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html
- Supported file formats reference: https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html#supported-coordinate-formats

### `MolecularSystem.__init__(self, universe, top_path, traj_path)`
Converts an MDAnalysis `Universe` into plain NumPy arrays: atom names, residue ids,
residue names, elements, masses, and a `(frames, atoms, 3)` coordinate array
(`float32` to halve memory vs `float64`). Iterates `universe.trajectory` once to
copy `positions` and `time` per frame (this is the standard MDAnalysis trajectory
read pattern, see the coordinates docs above). If a format has no mass records
(some PDBs), all atoms are given equal weight (mass = 1) rather than mass = 0,
since a total mass of zero would break center-of-mass and Rg later.

### `MolecularSystem.protein_mask(self)`
Boolean mask, `True` for non-solvent atoms. Solvent residue names (water/ion
codes such as `HOH`, `SOL`, `NA`, `CL`) are excluded from every downstream
measurement — this is standard MD-analysis practice since including solvent
would swamp RMSD/Rg/SASA with irrelevant bulk motion. Uses `numpy.isin`:
https://numpy.org/doc/stable/reference/generated/numpy.isin.html

### `MolecularSystem.select(self, atom_names)`
Returns indices of protein atoms whose name is in `atom_names` (e.g. `["CA"]`
for alpha carbons, or the `BACKBONE` list). Combines `protein_mask()` with
`numpy.isin` and `numpy.where`:
https://numpy.org/doc/stable/reference/generated/numpy.where.html

### `guess_element(name)`
Element guess from an atom name (PDB atom names sometimes prefix a digit, e.g.
`"1HB"` for a hydrogen) — strips leading digits and takes the first remaining
letter. This is the common heuristic used because PDB/PSF files don't always
carry an explicit element column (PDB format spec: https://www.wwpdb.org/documentation/file-format).

### `check_files(top_path, traj_path)`
Validates the uploaded files before attempting to parse them, so the user gets a
readable error instead of a low-level parser traceback. Checks: a topology file
was given, its extension is supported, and — since only PDB self-contains
coordinates — any non-PDB topology also has a trajectory file attached.

### `load_simulation(top_path, traj_path=None)`
Entry point: validates the files, builds an MDAnalysis `Universe` (single-file or
topology+trajectory form — both are documented constructor signatures:
https://docs.mdanalysis.org/stable/documentation_pages/core/universe.html#core-object-universe),
and wraps it in a `MolecularSystem`.

---

## 2. analysis.py — trajectory measurements

Per the file's own header comment: everything is written with plain Python loops
and lists on purpose (explainability), except two places that intentionally reuse
established, non-trivial algorithms rather than reinventing them:

- **`numpy.linalg.svd`** inside `superimpose()` — the Kabsch alignment algorithm.
- **mdtraj's `shrake_rupley`** (SASA) and **`compute_dssp`** (secondary structure)
  — both large, validated algorithms that are standard in the MD field rather than
  something to hand-roll.

References used:
- Kabsch algorithm (structural superposition): https://en.wikipedia.org/wiki/Kabsch_algorithm
- `numpy.linalg.svd`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
- RMSD / RMSF definitions (MDAnalysis' own explanation, used as the algorithmic
  reference even though the loops here are hand-written):
  https://docs.mdanalysis.org/stable/documentation_pages/analysis/rms.html
- Baker–Hubbard hydrogen bond geometric criterion (distance + angle cutoff):
  original paper — E. N. Baker, R. E. Hubbard, *Prog. Biophys. Mol. Biol.* 44 (1984);
  same criterion as implemented in MDAnalysis'
  `HydrogenBondAnalysis`: https://docs.mdanalysis.org/stable/documentation_pages/analysis/hydrogenbonds.html
- Radius of gyration definition: https://docs.mdanalysis.org/stable/documentation_pages/analysis/lineardensity.html (Rg is a standard mass-weighted RMS-distance-from-COM quantity, also defined in any MD textbook, e.g. Frenkel & Smit, *Understanding Molecular Simulation*)
- mdtraj `shrake_rupley` (SASA, Shrake–Rupley algorithm): https://www.mdtraj.org/1.9.8.dev0/api/generated/mdtraj.shrake_rupley.html
  (algorithm: A. Shrake, J.A. Rupley, *J. Mol. Biol.* 79 (1973))
- mdtraj `compute_dssp` (DSSP secondary structure assignment): https://www.mdtraj.org/1.9.8.dev0/api/generated/mdtraj.compute_dssp.html
  (algorithm: W. Kabsch, C. Sander, *Biopolymers* 22 (1983))

### `frame_positions(system, frame_index, atom_indices)`
Pulls out `[x, y, z]` for a set of atoms at one frame as plain Python lists
(rather than NumPy arrays), since the rest of the module deliberately works in
plain lists.

### `mean_center(points)`
Translates a point set so its centroid is at the origin — the first step of any
structural alignment (Kabsch algorithm reference above requires both point sets
pre-centered).

### `transpose3(m)`, `matmul3(a, b)`, `determinant3(m)`
Hand-written 3×3 matrix transpose, multiply, and determinant — the small amount
of linear algebra needed to apply a rotation matrix and to check its handedness,
written explicitly instead of pulling in a matrix library for 3×3 operations.
Standard formulas (see any linear algebra reference, e.g.
https://en.wikipedia.org/wiki/Determinant#3_%C3%97_3_matrices).

### `superimpose(mobile, reference)`
Implements the **Kabsch algorithm**: builds the 3×3 cross-covariance matrix
between two centered point sets, decomposes it with `numpy.linalg.svd`, corrects
for a reflection (negative determinant) by flipping the sign of the last
singular vector, and applies the resulting optimal rotation to `mobile`. This is
the textbook Kabsch procedure — see the Wikipedia reference above, which lays
out exactly this covariance → SVD → determinant-correction → rotation sequence.

### `calculate_rmsd(system)`
Root Mean Square Deviation of the protein backbone (N, CA, C, O) at every frame
against frame 0, after superimposing each frame onto the reference with
`superimpose()`. Standard RMSD definition (MDAnalysis rms docs, above).

### `calculate_rmsf(system)`
Root Mean Square Fluctuation of each residue's alpha carbon: aligns every frame
to frame 0 (same as RMSD, so overall drift/rotation isn't counted as
fluctuation), computes each atom's average position over all frames, then the
RMS distance of each frame's position from that average. Same definition class
as MDAnalysis' RMSF analysis (docs above).

### `center_of_mass(system)`
Mass-weighted centroid of the protein per frame: `sum(mass_i * position_i) / sum(mass_i)`.
Standard center-of-mass formula.

### `calculate_rg(system)`
Radius of gyration per frame: mass-weighted RMS distance of every protein atom
from that frame's center of mass — `sqrt(sum(mass_i * |r_i - r_com|^2) / total_mass)`.

### `calculate_hbonds(system)`
Implements the **Baker–Hubbard geometric criterion** for hydrogen bonds (same
criterion MDAnalysis' `HydrogenBondAnalysis` uses, docs above): a bond is counted
when the H···acceptor distance is under 2.5 Å (`HBOND_DIST`) and the
donor–H···acceptor angle is over 120° (`HBOND_ANGLE`). Donor hydrogens are first
matched to their covalently bonded heavy atom (N or O) using a simple nearest-neighbour
distance cutoff (`BOND_LENGTH` = 1.3 Å, a typical N–H/O–H bond length), computed
once from frame 0 since covalent bonding doesn't change during a simulation.

### `mdtraj_protein(system)`
Lazily reopens the same files with **mdtraj** (cached on the `system` object so
it only happens once), restricted to protein atoms via mdtraj's atom selection
language (`traj.top.select("protein")`):
https://www.mdtraj.org/1.9.8.dev0/atom_selection.html
This is needed because SASA and DSSP are not hand-rolled here — they're taken
directly from mdtraj's implementations.

### `calculate_sasa(system)`
Total solvent accessible surface area per frame, from mdtraj's
`shrake_rupley` (per-atom areas, in nm²), summed per frame and converted to Å²
(`× 100`, since 1 nm² = 100 Å²). Reference: mdtraj `shrake_rupley` docs above.

### `calculate_secondary_structure(system)`
Per-residue, per-frame DSSP letter codes from mdtraj's `compute_dssp`. Reference
above.

### `summarize(codes)`
Turns one frame's DSSP letters into helix / sheet / coil percentages, grouping
DSSP's eight codes into the three commonly reported categories:
`HELIX = ["H", "G", "I"]` (alpha helix, 3₁₀ helix, pi helix),
`SHEET = ["E", "B"]` (extended strand, beta bridge), everything else is coil.
This H/G/I vs E/B vs other grouping is the conventional simplification of the
DSSP alphabet (see the DSSP algorithm reference above). `"NA"` entries (residues
DSSP can't assign, usually chain termini) are excluded from the percentage base.

---

## 3. viewer.py — 3D visualization

Renders one frame of the protein by writing it out as PDB-format text and handing
that to **py3Dmol** (via the **stmol** Streamlit wrapper) for interactive 3D
rendering in the browser.

- PDB file format specification (ATOM record column layout used in
  `make_pdb_text`): https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
- py3Dmol: https://3dmol.csb.pitt.edu/ and https://pypi.org/project/py3Dmol/
- stmol (`showmol`, the Streamlit↔py3Dmol bridge): https://github.com/napoles-uach/stmol

### `make_pdb_text(system, frame)`
Writes out only the protein atoms (`protein_mask()`, so water/ions don't appear
and don't throw off the viewer's auto-zoom) of one frame as fixed-column PDB
`ATOM` records, following the PDB format's column spec linked above (atom
serial, atom name, residue name, chain, residue number, x/y/z, element).

### `show_structure(system, frame, width=900, height=520)`
Builds a py3Dmol `view`, loads the generated PDB text as a model
(`view.addModel(..., "pdb")` — py3Dmol model-loading API), applies a
`"cartoon"` style colored by `"spectrum"` (rainbow N→C terminus, a built-in
py3Dmol color scheme), zooms to fit (`view.zoomTo()`), and renders it into the
Streamlit page with `stmol.showmol`.

---

## 4. app.py — Streamlit UI

The web interface: file upload, running all analyses, and interactive charts.

- Streamlit `file_uploader`: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit `session_state` (persisting values across reruns — Streamlit reruns
  the whole script on every interaction, this is the documented way to keep
  state): https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- Streamlit `slider`, `radio`, `columns`, `metric`:
  https://docs.streamlit.io/develop/api-reference/widgets/st.slider ,
  https://docs.streamlit.io/develop/api-reference/widgets/st.radio ,
  https://docs.streamlit.io/develop/api-reference/layout/st.columns ,
  https://docs.streamlit.io/develop/api-reference/data/st.metric
- Streamlit `plotly_chart` with click/box-select events (`on_select="rerun"`,
  `selection_mode`): https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
- Plotly `graph_objects` (`Figure`, `Scatter`, `Bar`) and `make_subplots`:
  https://plotly.com/python/graph-objects/ , https://plotly.com/python/subplots/
- Python `tempfile.mkstemp` (Streamlit's uploader gives an in-memory buffer, but
  MDAnalysis/mdtraj need a real file path to open):
  https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp

### `save_upload(uploaded_file)`
Writes a Streamlit `UploadedFile`'s bytes to a real temporary file on disk
(`tempfile.mkstemp` + `os.fdopen`) and returns its path, since MDAnalysis/mdtraj
readers expect a filesystem path, not an in-memory buffer.

### `run_all_analysis(system)`
Runs every function in `analysis.py` against the loaded system, catching
exceptions per-measurement so one failing calculation (e.g. no hydrogens present
for hydrogen bonds) doesn't prevent the others from showing.

### `show_number(value, unit="")`
Formats a number to 2 decimal places with a unit, or `"Not available"` if the
value is `None` or non-finite (`numpy.isfinite`:
https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html).

### `value_at_frame(results, name, frame)`
Looks up one measurement's value at the currently displayed frame.

### `average_over_range(results, name, start, end, unit="")`
Averages a measurement over a user-selected frame range (from a box-select on
the chart).

### `show_center_of_mass(com)` / `show_secondary_structure(dssp, frame)`
Format the center-of-mass 3-vector and the helix/sheet/coil breakdown
(via `analysis.summarize`) for display.

### `make_plot(system, results, frame, metric)`
Builds the Plotly figure for whichever metric is selected: a bar chart for RMSF
(per-residue, not per-frame), a 3-row `make_subplots` figure for "All", or a
line+marker scatter plot for RMSD/Rg with a red vertical line marking the
current frame (`add_vline`) and a shaded rectangle for any selected frame range
(`add_vrect`).

### `handle_click(event, n_frames)`
Reads a Plotly click/box-select event from `st.plotly_chart`'s
`on_select` callback data. A single clicked point jumps the frame slider to that
frame; a multi-point (box) selection sets a frame range. Values are stashed in
`st.session_state` and `st.rerun()` is called because Streamlit widgets can't be
mutated directly after they've already been instantiated in the current run —
this is the documented pattern for widget state changes from non-widget events
(session_state docs above).

### `main()`
Page layout and control flow: sidebar file uploaders → load & run analysis on
button click → simulation summary metrics → 3D structure view + frame slider →
trajectory graph with click/range selection → per-frame numbers → per-range
averaged numbers. Called directly at module level (`main()`), which is how a
Streamlit script executes top-to-bottom on every rerun
(https://docs.streamlit.io/develop/concepts/architecture/run-your-app).

---

## Third-party libraries used, at a glance

| Library | Purpose | Docs |
|---|---|---|
| MDAnalysis | Parsing topology/trajectory files into a `Universe` | https://docs.mdanalysis.org/ |
| mdtraj | SASA (Shrake–Rupley) and DSSP secondary structure | https://www.mdtraj.org/ |
| NumPy | Arrays, `linalg.svd` for the Kabsch rotation, `isin`/`where`/`isfinite` | https://numpy.org/doc/stable/ |
| Streamlit | Web UI framework | https://docs.streamlit.io/ |
| Plotly | Interactive charts (RMSD/RMSF/Rg) | https://plotly.com/python/ |
| py3Dmol / stmol | Interactive 3D molecular rendering in the browser | https://3dmol.csb.pitt.edu/ , https://github.com/napoles-uach/stmol |
