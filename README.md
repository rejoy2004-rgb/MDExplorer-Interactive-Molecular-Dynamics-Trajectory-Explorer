# 🧬 MDExplorer: Interactive Molecular Dynamics Trajectory Explorer

MDExplorer converts molecular dynamics (MD) simulation files of different formats into **one common NumPy-based representation**, allowing you to view and interact with the molecular trajectory in 3D while analyzing key biophysical metrics in real time.

---

## 🌟 Key Features

*   **Format Agnostic**: Automatically parses and converts files of different formats (PDB, PSF+DCD, GRO+XTC, TPR+XTC, PRMTOP+NC) into a single unified `MolecularSystem` object containing pure NumPy arrays.
*   **3D Trajectory Viewer**: Renders the protein structure in 3D using `py3Dmol` cartoon/stick styles with dynamic frame navigation.
*   **Interactive Digital Cockpit**: Featuring Plotly charts linked directly to the 3D viewer. Clicking any point on the chart snaps the 3D model to that frame.
*   **Scientifically Validated Trajectory Metrics**:
    *   **Backbone RMSD**: Computes structural displacement over time relative to the reference frame. Uses the **Kabsch algorithm** to superimpose structures and remove translational/rotational diffusion.
    *   **Radius of Gyration ($R_g$)**: Measures overall shape compactness by computing the root-mean-square distance of atoms from their center of geometry.
    *   **Hydrogen Bonds**: Computes intramolecular polar-pair contacts (donor/acceptor N, O) within a $3.5\text{ Å}$ distance cutoff using SciPy's vectorized `cdist`.
    *   **SASA (Solvent Accessible Surface Area)**: Calculates exposed surface area using MDTraj's Shrake-Rupley algorithm.
    *   **RMSF (Root Mean Square Fluctuation)**: Calculates the fluctuation profile per residue across the trajectory to identify highly mobile loops.
    *   **DSSP Secondary Structure**: Computes Helix, Sheet, and Coil percentages per frame using MDTraj's DSSP module.
*   **Solvent & Ion Filtering**: All metric calculations automatically exclude water and ion molecules (e.g. `HOH`, `WAT`, `NA`, `CL`), ensuring data is biologically accurate.

---

## 🛠️ Technology Stack

1.  **Frontend/GUI**: Streamlit
2.  **Interactive Plots**: Plotly
3.  **Core Math & Distance Computations**: NumPy & SciPy
4.  **Simulation Loading**: MDAnalysis
5.  **Biophysical Analytics**: MDTraj (DSSP & SASA) & MDAnalysis (RMSD & RMSF)
6.  **3D Rendering**: py3Dmol & stmol

---

## 📂 Project Directory Structure

```text
MolNumPy/
│
├── app.py                     # Streamlit Main Dashboard Entrypoint
├── requirements.txt           # Package Dependencies
├── README.md                  # Project Documentation
│
├── src/molnumpy/              # Core Unified Data Representations
│   ├── system.py              # MolecularSystem wrapper (Topology + Trajectory)
│   ├── topology.py            # Static attributes (Atoms, residue indices, bonds)
│   ├── trajectory.py          # Dynamic coordinates (n_frames, n_atoms, 3)
│   │
│   ├── readers/
│   │   ├── factory.py         # Format detection and loader mappings
│   │   └── reader.py          # low-level converters (MDAnalysis -> NumPy)
│   │
│   └── visualization/
│       └── viewer.py          # 3D py3Dmol viewer string building
│
├── analysis/                  # Custom Trajectory Analyzers
│   ├── com.py                 # Center of Mass Calculator
│   ├── hbonds.py              # Hydrogen Bond calculator (polar name selection)
│   ├── rg.py                  # Radius of Gyration calculator
│   ├── rmsd.py                # Backbone RMSD (Kabsch aligned)
│   ├── rmsf.py                # Per-residue fluctuation
│   ├── sasa.py                # Solvent Accessible Surface Area (Shrake-Rupley)
│   └── secondary.py           # DSSP Secondary Structure assignment
│
└── utils/
    └── loader.py              # Wrappers and file loader helpers
```

---

## 🚀 Installation & Running

### 1. Prerequisites
Open your terminal and navigate to the project directory:
```bash
cd "c:\Users\rejoy\Desktop\BTP_1\MolNumPy"
```

### 2. Activate Virtual Environment
Activate the pre-configured virtual environment:
```bash
# Windows PowerShell
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
Streamlit will launch a local server and open `http://localhost:8501` in your browser.

---

## 📊 Trajectory Upload Guide

1.  **Topology File**: Upload your structure file (PDB, PSF, GRO, TPR, PRMTOP).
2.  **Trajectory File**: Upload your coordinate file (DCD, XTC, TRR, NC). Note: Single `.pdb` files contain both structure and single-frame coordinates, and do not require a separate trajectory file.
3.  Click **Load Simulation** to process the data and open the interactive digital cockpit!
