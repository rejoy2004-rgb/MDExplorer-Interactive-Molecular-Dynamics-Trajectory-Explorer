"""MolNumPy - Molecular Dynamics Explorer.

A simple Streamlit app that loads an MD simulation (topology + trajectory),
computes the analyses once, and lets the user explore frames and ranges
through the interactive RMSD plot and the 3D viewer.
"""

import os
import tempfile
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analysis.com import center_of_mass
from analysis.hbonds import calculate_hbonds
from analysis.rg import calculate_rg
from analysis.rmsd import calculate_rmsd
from analysis.rmsf import calculate_rmsf
from analysis.sasa import calculate_sasa
from analysis.secondary import calculate_secondary_structure, summarize_secondary_structure
from molnumpy.visualization.viewer import render_molecular_viewer
from utils.loader import load_simulation

st.set_page_config(page_title="MolNumPy", page_icon="🧬", layout="wide")

TOP_EXTENSIONS = [".pdb", ".psf", ".gro", ".tpr", ".prmtop", ".parm7"]
TRAJ_EXTENSIONS = [".dcd", ".xtc", ".trr", ".nc"]


def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to a temporary path so readers can open it."""
    ext = os.path.splitext(uploaded_file.name)[1]
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


@st.cache_resource
def load_and_analyze_v2(top_path, traj_path):
    """Load the files and compute all trajectory-wide analyses once."""
    system = load_simulation(top_path, traj_path)

    results = {}
    for name, fn in [
        ("rmsd", calculate_rmsd),
        ("rg", calculate_rg),
        ("sasa", calculate_sasa),
        ("hbonds", calculate_hbonds),
        ("rmsf", calculate_rmsf),
        ("secondary", calculate_secondary_structure),
        ("com", center_of_mass),
    ]:
        try:
            results[name] = fn(system)
        except Exception:
            results[name] = None  # shown as "Not available" in the UI
    return system, results


def frame_value(results, name, frame):
    """Single value from a cached analysis, or None if not available."""
    arr = results.get(name)
    if arr is None:
        return None
    return arr[frame]


def format_value(value, unit=""):
    if value is None or not np.isfinite(value):
        return "Not available"
    return f"{value:.2f} {unit}".strip()


def format_com(com):
    if com is None:
        return "Not available"
    x, y, z = com
    return f"({x:.1f}, {y:.1f}, {z:.1f}) Å"


def format_secondary(dssp, frame):
    if dssp is None:
        return "Not available"
    helix, sheet, coil = summarize_secondary_structure(dssp[frame])
    return f"H {helix:.0f}%  S {sheet:.0f}%  C {coil:.0f}%"


def range_value(results, name, start, end, stat, unit=""):
    """Mean/min/max of a cached analysis over the frames start..end."""
    arr = results.get(name)
    if arr is None:
        return "Not available"
    vals = arr[start:end + 1]
    return format_value(getattr(np, stat)(vals), unit)


def build_figure(system, results, frame, metric):
    """Plot of the selected metric; RMSD and Rg are per frame, RMSF per residue."""
    if metric == "RMSF":
        rmsf = results["rmsf"]
        fig = go.Figure(go.Bar(
            x=np.arange(len(rmsf)),
            y=rmsf,
            hovertemplate="Residue %{x}<br>RMSF %{y:.2f} Å<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Residue index", yaxis_title="RMSF (Å)", height=350)
        return fig

    if metric == "All":
        fig = make_subplots(rows=3, subplot_titles=("RMSD", "RMSF", "Rg"))
        if results.get("rmsd") is not None:
            fig.append_trace(go.Scatter(
                x=np.arange(system.n_frames), y=results["rmsd"], mode="lines", name="RMSD"
            ), 1, 1)
        if results.get("rmsf") is not None:
            fig.append_trace(go.Bar(
                x=np.arange(len(results["rmsf"])), y=results["rmsf"], name="RMSF"
            ), 2, 1)
        if results.get("rg") is not None:
            fig.append_trace(go.Scatter(
                x=np.arange(system.n_frames), y=results["rg"], mode="lines", name="Rg"
            ), 3, 1)
        fig.update_layout(height=700)
        return fig

    is_rmsd = metric == "RMSD"
    name = "RMSD" if is_rmsd else "Rg"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(system.n_frames),
        y=results["rmsd"] if is_rmsd else results["rg"],
        mode="lines",
        name=name,
        customdata=np.stack([system.time], axis=1),
        hovertemplate="Frame %{x}<br>Time %{customdata[0]:.1f} ps<br>%{y:.2f} Å<extra></extra>",
    ))
    fig.add_vline(x=frame, line_color="red", line_dash="dash", opacity=0.6)
    if st.session_state.get("selected_range"):
        start, end = st.session_state["selected_range"]
        fig.add_vrect(x0=start, x1=end, fillcolor="lightblue", opacity=0.2, line_width=0)
    fig.update_layout(xaxis_title="Frame", yaxis_title=f"{name} (Å)", height=350)
    return fig


def handle_selection(event, n_frames):
    """React to clicks / box selections on the trajectory graph."""
    if event is None:
        return
    points = (event.selection or {}).get("points", [])
    if not points:
        return

    indices = sorted({int(p.get("point_index", p.get("x", 0))) for p in points})
    if len(indices) == 1:
        st.session_state["frame"] = min(indices[0], n_frames - 1)
        st.session_state["selected_range"] = None
    else:
        st.session_state["selected_range"] = (indices[0], indices[-1])
        st.session_state["frame"] = indices[0]
    st.rerun()


def main():
    st.title("🧬 MolNumPy")
    st.caption("Molecular Dynamics Explorer")

    with st.sidebar:
        st.header("Files")
        top_file = st.file_uploader(
            "Topology file (PDB, PSF, GRO, TPR, PRMTOP)",
            type=[e.lstrip('.') for e in TOP_EXTENSIONS],
        )
        traj_file = st.file_uploader(
            "Trajectory file (DCD, XTC, TRR, NC)",
            type=[e.lstrip('.') for e in TRAJ_EXTENSIONS],
        )
        load_clicked = st.button("Load Simulation", width="stretch")

    if load_clicked:
        top_path = save_uploaded_file(top_file) if top_file else None
        traj_path = save_uploaded_file(traj_file) if traj_file else None
        try:
            with st.spinner("Loading files and computing analyses..."):
                system, results = load_and_analyze_v2(top_path, traj_path)
            st.session_state["system"] = system
            st.session_state["results"] = results
            st.session_state["frame"] = 0
            st.session_state["selected_range"] = None
            st.session_state["playing"] = False
        except ValueError as e:
            st.error(str(e))

    if "system" not in st.session_state:
        st.info("Upload a topology (and trajectory) file, then click **Load Simulation**.")
        return

    system = st.session_state["system"]
    results = st.session_state["results"]
    n_frames = system.n_frames

    # Frame playback: advance the frame and re-render the page
    if st.session_state.get("playing") and st.session_state.get("selected_range"):
        start, end = st.session_state["selected_range"]
        next_frame = st.session_state["frame"] + 1
        if next_frame > end or next_frame < start:
            next_frame = start
        st.session_state["frame"] = next_frame
        time.sleep(0.1)
        st.rerun()

    frame = st.session_state["frame"]

    st.subheader("Simulation information")
    n_residues = len(np.unique(system.topology.atoms["resid"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of atoms", f"{system.n_atoms:,}")
    c2.metric("Number of residues", f"{n_residues:,}")
    c3.metric("Number of frames", f"{n_frames:,}")
    c4.metric("Simulation time", f"{system.time[-1]:.1f} ps")

    st.subheader("3D protein structure")
    st.caption(f"Frame {frame} | time {system.time[frame]:.1f} ps")
    render_molecular_viewer(system, frame, width=900, height=520)
    st.slider("Frame", min_value=0, max_value=n_frames - 1, key="frame")

    st.subheader("Trajectory analysis")
    metrics = {"RMSD": results.get("rmsd"), "RMSF": results.get("rmsf"), "Rg": results.get("rg")}
    options = [name for name, arr in metrics.items() if arr is not None]
    if not options:
        st.info("RMSD, RMSF and Rg are not available for this system.")
    else:
        options.append("All")
        metric = st.radio("Metric", options, horizontal=True)
        fig = build_figure(system, results, frame, metric)
        if metric in ("RMSD", "Rg"):
            event = st.plotly_chart(
                fig,
                on_select="rerun",
                selection_mode=["points", "box", "lasso"],
                key="main_plot",
            )
            handle_selection(event, n_frames)
        else:
            st.plotly_chart(fig, key="main_plot")

    st.subheader("Selected frame")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frame", frame)
    m1.metric("Time", f"{system.time[frame]:.1f} ps")
    m2.metric("RMSD", format_value(frame_value(results, "rmsd", frame), "Å"))
    m2.metric("Radius of gyration", format_value(frame_value(results, "rg", frame), "Å"))
    m3.metric("SASA", format_value(frame_value(results, "sasa", frame), "Å²"))
    m3.metric("H-bonds", format_value(frame_value(results, "hbonds", frame)))
    m4.metric("Center of mass", format_com(frame_value(results, "com", frame)))
    m4.metric("Secondary structure", format_secondary(results.get("secondary"), frame))

    st.subheader("Selected frame range")
    if st.session_state.get("selected_range"):
        start, end = st.session_state["selected_range"]
        st.caption(f"Frames {start} - {end}  ({end - start + 1} frames)")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Average RMSD", range_value(results, "rmsd", start, end, "mean", "Å"))
        r2.metric("Average Rg", range_value(results, "rg", start, end, "mean", "Å"))
        r3.metric("Average SASA", range_value(results, "sasa", start, end, "mean", "Å²"))
        r4.metric("Average H-bonds", range_value(results, "hbonds", start, end, "mean"))
        r1.metric("Minimum RMSD", range_value(results, "rmsd", start, end, "min", "Å"))
        r2.metric("Maximum RMSD", range_value(results, "rmsd", start, end, "max", "Å"))

        if st.button("▶ Play selected frames"):
            st.session_state["playing"] = True
        if st.session_state.get("playing"):
            if st.button("⏹ Stop"):
                st.session_state["playing"] = False
    else:
        st.info("Draw a box on the RMSD graph to select a range of frames.")


if __name__ == "__main__":
    main()
