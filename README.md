# MDExplorer — Interactive Molecular Dynamics Trajectory Explorer

A tool for exploring molecular dynamics (MD) simulations in the browser. You
upload the files from your simulation, and it shows the protein in 3D together
with graphs of the usual analysis measurements.

## The idea behind it

MD simulations are saved in a lot of different file formats, and every format
stores the same information in a different way. Normally that means writing
separate code for each one.

This project avoids that. It reads the file once, copies everything into plain
NumPy arrays inside a single `MolecularSystem` object, and then forgets where
the data came from. All the analysis code and the 3D viewer only ever talk to
that one object, so adding a new file format means changing only the reader.

```
   PDB                 |
   PSF + DCD           |
   GRO + XTC           |----> MolecularSystem ----> analysis + 3D viewer
   TPR + XTC           |      (NumPy arrays)
   PRMTOP + NC         |
```

A `MolecularSystem` holds:

* one array per atom property, which never changes: `names`, `resids`,
  `resnames`, `elements`, `masses`
* the coordinates, which do change, in one array of shape
  `(n_frames, n_atoms, 3)`
* the time of each frame

## What it measures

Water molecules and ions are removed before every calculation, otherwise the
solvent box around the protein would dominate the results.

| Measurement | What it tells you | How it is done |
| --- | --- | --- |
| RMSD | how far the shape has moved from the starting structure | our own Python code, using the Kabsch algorithm to remove the tumbling of the molecule (NumPy does the SVD step of the alignment) |
| RMSF | which residues are flexible and which are rigid | our own Python code, one value per alpha carbon, aligned with the same Kabsch step as RMSD |
| Radius of gyration | how compact the folded protein is | our own Python code, mass weighted |
| Centre of mass | whether the protein is drifting through the box | our own Python code |
| Hydrogen bonds | whether the folded structure is holding together | our own Python code, the Baker-Hubbard geometric rule: H...acceptor under 2.5 A and a donor-H...acceptor angle over 120 degrees |
| SASA | how much surface is exposed to water | mdtraj (Shrake-Rupley) |
| Secondary structure | the helix / sheet / coil percentages | mdtraj (DSSP, simplified 3-class output: H, E, C) |

The last two use a library because the algorithms behind them are much bigger
than the rest. Everything else is calculated directly from the NumPy arrays.

## What the page does

* the protein is drawn in 3D with PyMOL (rendered as an image of the current
  frame), and a slider moves through the frames
* the graph can be clicked to jump the 3D view to that frame
* dragging a box across the graph selects a range of frames, and the averages
  over that range are shown underneath
* if one measurement cannot be calculated, the page says which one and shows
  the reason, and the other measurements still appear

## Periodic-boundary warning

A simulation box repeats itself in space, so MD programs wrap every atom back
into the box. One protein can then end up as pieces on opposite sides of the
box. The files still load and every number is still calculated, but RMSD, RMSF,
radius of gyration, centre of mass and the 3D picture describe the broken
pieces, not the protein. In our test the same GROMACS trajectory gave an RMSD
up to 20.8 A when split and up to 2.0 A once it was made whole.

After loading, MDExplorer therefore checks the distance between neighbouring
alpha carbons (residue n and n+1 of the same chain, which are always about 3.8 A
apart) in every frame and shows one of three answers:

* **Potential split detected**: a pair is more than 6 A apart. A warning names
  the frame and residues. Nothing is blocked and no coordinate is changed.
* **No obvious split detected**: no such pair was found.
* **Unable to assess reliably**: there was nothing to compare (no alpha carbons,
  only one residue per chain, gaps in the residue numbers, or coordinates that
  are not numbers).

What the check cannot do:

* "No obvious split" is **not** a guarantee. It only looks at alpha carbons, so
  it can miss a split in a chain with missing residues, a split between two
  chains, a split that only moves side chains, or a molecule that jumps by less
  than 6 A. It never uses the box size, so a file without a box is not treated
  as safe.
* It can give a false alarm if two chains share one chain label and continue the
  residue numbering (for example chain A ends at residue 214 and chain B starts
  at 215), because the end of one chain then looks like a broken bond.
* The threshold (`CA_JUMP_LIMIT` in `analysis.py`) comes from protein geometry:
  neighbouring alpha carbons are 3.8 A apart (4.1 A at most in our real test
  data), while a split jumps by tens of A.

MDExplorer does not unwrap the coordinates for you. Unwrapping needs the bonds
of the molecule, which GRO and PDB files do not contain, and a wrong guess would
silently change every result. Fix the trajectory with the program that made it
and load the corrected files. For GROMACS, an example is:

```
gmx trjconv -s topol.tpr -f trajectory.xtc -o trajectory_whole.xtc -pbc mol
```

Notes on this command:

* Use your own run-input file (`topol.tpr`) and trajectory; they must belong to
  the same simulation.
* GROMACS asks which group to write out; choose the protein group.
* `-pbc mol` makes each molecule whole. That is not the same as centring the
  protein or removing every jump of the molecule between frames, and some
  simulations or analyses need extra steps (for example `-center`).
* Always check the corrected trajectory (the MDExplorer check, the 3D view and
  the RMSD curve) before trusting the results.

To run the tests of this check (they use tiny built-in proteins, no files):

```
python test_periodic_check.py
python test_frames.py     (one-frame and many-frame pages)
```

## Known limitations

* Periodic boundaries are only checked, not corrected; see the section above.
* SASA and secondary structure are read by mdtraj, which cannot open `.tpr`
  topologies. With a TPR file those two show "Not available" and the page
  explains why; the other five measurements still work.
* The RMSF graph places each bar at its residue number, so it assumes the
  residue numbers are unique. If a structure has several chains that reuse the
  same residue numbers, their bars can overlap.
* Loading time depends on the number of frames, the number of atoms and your
  computer. SASA is the slowest measurement because it is deliberately run on
  one thread, one frame at a time: mdtraj's multi-threaded version sometimes
  returned a wrong value for a frame when the computer was busy, and it made
  every frame after the first about 0.1% too large. Example from one machine (a
  Windows PC with 12 logical cores, Adenylate Kinase with 3,341 protein atoms):
  SASA took about 0.6 s per frame (98 frames in about 60 s) and hydrogen bonds
  about 0.1 s per frame. These are examples, not guarantees; a long trajectory
  can take several minutes to load.

## Libraries used

| Library | Used for |
| --- | --- |
| MDAnalysis | opening the simulation files |
| NumPy | all the arrays and most of the maths |
| mdtraj | SASA and DSSP |
| Streamlit | the web page |
| Plotly | the graphs |
| PyMOL (pymol-open-source-whl) | the 3D view |

## Files

There are only four of them:

```
molecule.py    the MolecularSystem class, and reading a file into one
analysis.py    all seven measurements
viewer.py      writes a frame as PDB text and draws it in 3D
app.py         the Streamlit page, this is what you run
```

## Running it

Install the requirements once:

```
pip install -r requirements.txt
```

Then start the app:

```
streamlit run app.py
```

It opens at `http://localhost:8501`.

Upload the topology file first (PDB, PSF, GRO, TPR or PRMTOP), then the
trajectory file (DCD, XTC, TRR or NC), and click **Load simulation**. A plain
`.pdb` file already contains coordinates, so for that one no trajectory file is
needed.
