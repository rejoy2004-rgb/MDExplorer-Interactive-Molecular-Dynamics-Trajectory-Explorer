"""MolNumPy - a small tool for looking at molecular dynamics simulations.

Start it from the terminal with:   streamlit run app.py
"""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from molecule import load_simulation, TOPOLOGY_FORMATS, TRAJECTORY_FORMATS
from viewer import render_structure
import analysis

st.set_page_config(page_title="MolNumPy", layout="wide")


def save_upload(uploaded_file):
    """Save an upload to a temp file and return its path (readers need a real file)."""
    extension = os.path.splitext(uploaded_file.name)[1]
    handle, path = tempfile.mkstemp(suffix=extension)
    with os.fdopen(handle, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def run_all_analysis(system):
    """Run every measurement; store None (shown as "Not available") if one fails."""
    to_run = {
        "rmsd": analysis.calculate_rmsd,
        "rmsf": analysis.calculate_rmsf,
        "rg": analysis.calculate_rg,
        "com": analysis.center_of_mass,
        "hbonds": analysis.calculate_hbonds,
        "sasa": analysis.calculate_sasa,
        "secondary": analysis.calculate_secondary_structure,
    }

    results = {}
    for name in to_run:
        try:
            results[name] = to_run[name](system)
        except Exception:
            results[name] = None

    return results


def show_number(value, unit=""):
    """Format one number for the page, or say so if we do not have it."""
    if value is None or not np.isfinite(value):
        return "Not available"
    return "%.2f %s" % (value, unit)


def value_at_frame(results, name, frame):
    """One value of one measurement at one frame."""
    if results[name] is None:
        return None
    return results[name][frame]


def average_over_range(results, name, start, end, unit=""):
    """Average of one measurement over the frames the user selected."""
    if results[name] is None:
        return "Not available"
    values = results[name][start:end + 1]
    return show_number(sum(values) / len(values), unit)


def show_center_of_mass(com):
    if com is None:
        return "Not available"
    return "(%.1f, %.1f, %.1f) A" % (com[0], com[1], com[2])


def show_secondary_structure(dssp, frame):
    if dssp is None:
        return "Not available"
    helix, sheet, coil = analysis.summarize(dssp[frame])
    return "- Helix: %.0f%%\n- Sheet: %.0f%%\n- Coil: %.0f%%" % (helix, sheet, coil)


def make_plot(system, results, frame, metric):
    """Build the Plotly graph for whichever metric the user picked."""
    frames = np.arange(system.n_frames)

    # RMSF is different from the others, it is per residue and not per frame
    if metric == "RMSF":
        figure = go.Figure(go.Bar(x=np.arange(len(results["rmsf"])), y=results["rmsf"]))
        figure.update_layout(xaxis_title="Residue", yaxis_title="RMSF (A)", height=350)
        return figure

    # all three metrics under each other, for comparing them
    if metric == "All":
        figure = make_subplots(rows=3, subplot_titles=("RMSD", "RMSF", "Rg"))
        if results["rmsd"] is not None:
            figure.add_trace(go.Scatter(x=frames, y=results["rmsd"], name="RMSD"), row=1, col=1)
        if results["rmsf"] is not None:
            figure.add_trace(go.Bar(x=np.arange(len(results["rmsf"])), y=results["rmsf"], name="RMSF"), row=2, col=1)
        if results["rg"] is not None:
            figure.add_trace(go.Scatter(x=frames, y=results["rg"], name="Rg"), row=3, col=1)
        figure.update_layout(height=700)
        return figure

    if metric == "RMSD":
        values = results["rmsd"]
    else:
        values = results["rg"]

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=frames, y=values, mode="lines+markers", name=metric,
        # markers give clicks something to hit; a plain line has no target
        marker=dict(size=5, opacity=0.15),
    ))

    # a red line showing which frame is on screen in the 3D viewer
    figure.add_vline(x=frame, line_color="red", line_dash="dash", opacity=0.6)

    # a shaded band if the user has selected a range of frames
    if st.session_state["selected_range"] is not None:
        start, end = st.session_state["selected_range"]
        figure.add_vrect(x0=start, x1=end, fillcolor="lightblue", opacity=0.2, line_width=0)

    figure.update_layout(xaxis_title="Frame", yaxis_title=metric + " (A)", height=350)
    return figure


def handle_click(event, n_frames):
    """Jump to the clicked frame, or select a dragged range."""
    if event is None:
        return

    points = event.selection["points"]
    if len(points) == 0:
        return

    # the frame number is the x value of the point that was clicked
    clicked = sorted(set(int(point["x"]) for point in points))

    # slider's already drawn this run, so stash the value for next run
    if len(clicked) == 1:
        st.session_state["new_frame"] = min(clicked[0], n_frames - 1)
        st.session_state["selected_range"] = None
    else:
        st.session_state["selected_range"] = (clicked[0], clicked[-1])
        st.session_state["new_frame"] = clicked[0]

    st.rerun()


def main():
    st.title("MolNumPy")
    st.caption("Molecular dynamics trajectory explorer")

    # --- the sidebar, where the files are uploaded ---
    with st.sidebar:
        st.header("Files")
        top_file = st.file_uploader(
            "Topology file (PDB, PSF, GRO, TPR, PRMTOP)",
            type=[extension[1:] for extension in TOPOLOGY_FORMATS],
        )
        traj_file = st.file_uploader(
            "Trajectory file (DCD, XTC, TRR, NC)",
            type=[extension[1:] for extension in TRAJECTORY_FORMATS],
        )
        load_clicked = st.button("Load simulation")

    if load_clicked:
        top_path = None
        traj_path = None
        if top_file is not None:
            top_path = save_upload(top_file)
        if traj_file is not None:
            traj_path = save_upload(traj_file)

        try:
            with st.spinner("Reading the files and running the analysis..."):
                system = load_simulation(top_path, traj_path)
                results = run_all_analysis(system)

            # session_state survives reruns, so we don't reload on every click
            st.session_state["system"] = system
            st.session_state["results"] = results
            st.session_state["frame"] = 0
            st.session_state["selected_range"] = None
        except ValueError as error:
            st.error(str(error))

    # nothing loaded yet, so there is nothing else to draw
    if "system" not in st.session_state:
        st.info("Upload a topology file (and a trajectory file) and click **Load simulation**.")
        return

    system = st.session_state["system"]
    results = st.session_state["results"]

    # apply a frame picked last run, before the slider is drawn
    if "new_frame" in st.session_state:
        st.session_state["frame"] = st.session_state.pop("new_frame")

    frame = st.session_state["frame"]

    # --- what was loaded ---
    st.subheader("Simulation information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Atoms", "{:,}".format(system.n_atoms))
    col2.metric("Residues", "{:,}".format(system.n_residues))
    col3.metric("Frames", "{:,}".format(system.n_frames))
    col4.metric("Length", "%.1f ps" % system.time[-1])

    # --- the 3D view ---
    st.subheader("3D structure")
    st.caption("Frame %d, at %.1f ps" % (frame, system.time[frame]))
    st.image(render_structure(system, frame))
    st.slider("Frame", min_value=0, max_value=system.n_frames - 1, key="frame")

    # --- the graph ---
    st.subheader("Trajectory analysis")
    choices = []
    for name, label in [("rmsd", "RMSD"), ("rmsf", "RMSF"), ("rg", "Rg")]:
        if results[name] is not None:
            choices.append(label)

    if len(choices) == 0:
        st.info("None of the graphs could be calculated for this file.")
    else:
        choices.append("All")
        metric = st.radio("Metric", choices, horizontal=True)
        figure = make_plot(system, results, frame, metric)

        # the RMSD and Rg graphs can be clicked to change the frame
        if metric == "RMSD" or metric == "Rg":
            st.caption("Click a point to jump to that frame. To select a range, "
                       "pick the box select tool (dashed square) in the chart's "
                       "toolbar first, then drag across the graph.")
            event = st.plotly_chart(
                figure,
                on_select="rerun",
                selection_mode=["points", "box", "lasso"],
                key="graph",
            )
            handle_click(event, system.n_frames)
        else:
            st.plotly_chart(figure, key="graph")

    # --- the numbers for the frame that is on screen ---
    st.subheader("This frame")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frame", frame)
    col1.metric("Time", "%.1f ps" % system.time[frame])
    col2.metric("RMSD", show_number(value_at_frame(results, "rmsd", frame), "A"))
    col2.metric("Radius of gyration", show_number(value_at_frame(results, "rg", frame), "A"))
    col3.metric("SASA", show_number(value_at_frame(results, "sasa", frame), "A2"))
    col3.metric("Hydrogen bonds", show_number(value_at_frame(results, "hbonds", frame)))
    col4.metric("Centre of mass", show_center_of_mass(value_at_frame(results, "com", frame)))
    col4.markdown("**Secondary structure**")
    col4.markdown(show_secondary_structure(results["secondary"], frame))

    # --- the numbers for a whole range of frames ---
    st.subheader("Selected frames")
    if st.session_state["selected_range"] is None:
        st.info("Drag a box across the RMSD or Rg graph to select a range of frames.")
        return

    start, end = st.session_state["selected_range"]
    st.caption("Frames %d to %d (%d frames)" % (start, end, end - start + 1))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average RMSD", average_over_range(results, "rmsd", start, end, "A"))
    col2.metric("Average Rg", average_over_range(results, "rg", start, end, "A"))
    col3.metric("Average SASA", average_over_range(results, "sasa", start, end, "A2"))
    col4.metric("Average hydrogen bonds", average_over_range(results, "hbonds", start, end))

    if results["rmsd"] is not None:
        rmsd_values = results["rmsd"][start:end + 1]
        col1.metric("Lowest RMSD", show_number(min(rmsd_values), "A"))
        col2.metric("Highest RMSD", show_number(max(rmsd_values), "A"))


main()
