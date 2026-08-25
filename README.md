# MolNumPy

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
| RMSD | how far the shape has moved from the starting structure | our own NumPy code, using the Kabsch algorithm to remove the tumbling of the molecule |
| RMSF | which residues are flexible and which are rigid | our own NumPy code, one value per alpha carbon |
| Radius of gyration | how compact the folded protein is | our own NumPy code, mass weighted |
| Centre of mass | whether the protein is drifting through the box | our own NumPy code |
| Hydrogen bonds | whether the folded structure is holding together | our own NumPy code, the Baker-Hubbard geometric rule: H...acceptor under 2.5 A and a donor-H...acceptor angle over 120 degrees |
| SASA | how much surface is exposed to water | mdtraj (Shrake-Rupley) |
| Secondary structure | the helix / sheet / coil percentages | mdtraj (DSSP) |

The last two use a library because the algorithms behind them are much bigger
than the rest. Everything else is calculated directly from the NumPy arrays.

## What the page does

* the protein is drawn in 3D with py3Dmol, and a slider moves through the frames
* the graph can be clicked to jump the 3D view to that frame
* dragging a box across the graph selects a range of frames, and the averages
  over that range are shown underneath
* a selected range can be played back like a short video

## Libraries used

| Library | Used for |
| --- | --- |
| MDAnalysis | opening the simulation files |
| NumPy | all the arrays and most of the maths |
| SciPy | distances between atoms, for the hydrogen bonds |
| mdtraj | SASA and DSSP |
| Streamlit | the web page |
| Plotly | the graphs |
| py3Dmol, stmol | the 3D view |

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
