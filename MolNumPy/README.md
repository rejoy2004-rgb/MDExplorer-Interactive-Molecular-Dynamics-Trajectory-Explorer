# MolNumPy

MolNumPy converts molecular dynamics (MD) files of different formats into **one
common NumPy-based representation**, and lets you view the molecular trajectory
in 3D.

## What is MolNumPy?

Every MD program saves its files in its own format. MolNumPy hides that
difference: no matter which format you upload, you get the same
`MolecularSystem` object with a `Topology` and a `Trajectory`, stored as plain
NumPy arrays.

The trajectory is always a single array:

```python
coordinates.shape == (number_of_frames, number_of_atoms, 3)
```

## Supported formats

| Topology file | Trajectory file | Name |
|---|---|---|
| `.pdb` | *(none needed)* | PDB |
| `.psf` | `.dcd` | NAMD |
| `.gro` | `.xtc` or `.trr` | GROMACS |
| `.tpr` | `.xtc` or `.trr` | GROMACS |
| `.prmtop` / `.parm7` | `.nc` | AMBER |

A single `.pdb` file is turned into a one-frame trajectory, so it uses exactly
the same interface as every other format.

## How it works

```
Input file
   ↓
Reader (format detected from the file extension)
   ↓
Common MolecularSystem (topology + trajectory)
   ↓
NumPy trajectory (frames, atoms, 3)
   ↓
3D visualization (py3Dmol)
```

## Project structure

```
MolNumPy/
│
├── app.py                       # Streamlit interface
│
├── src/molnumpy/
│   ├── topology.py              # common topology (atoms, residues, bonds)
│   ├── trajectory.py            # common trajectory (coordinates, box, time)
│   ├── system.py                # MolecularSystem = topology + trajectory
│   │
│   ├── readers/
│   │   ├── factory.py           # extension -> reader -> MolecularSystem
│   │   └── reader.py            # MDAnalysis -> NumPy conversion
│   │
│   └── visualization/
│       └── viewer.py            # py3Dmol 3D viewer
│
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Then:

1. Upload your molecular file(s).
2. Click **Load Molecule**.
3. Read the system information (atoms, frames, residues, format).
4. View the molecule in 3D and move through frames with the slider.

## Example

If you upload `protein.psf` and `protein.dcd`, MolNumPy reads both files and
converts them into the common trajectory representation:

```python
system = load_molecule("protein.psf", "protein.dcd")

system.n_atoms              # 12445
system.n_frames             # 500
system.coordinates.shape    # (500, 12445, 3)
```

The same `MolecularSystem` is returned for every input format, so the 3D viewer
and everything after the reader never cares about the original format.

## Tests

```bash
python -m pytest tests/ -v
```
