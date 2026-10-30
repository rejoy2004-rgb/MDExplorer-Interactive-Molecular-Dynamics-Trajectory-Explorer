"""MDExplorer - a small tool for looking at molecular dynamics simulations.

Start it from the terminal with:   streamlit run app.py
"""

import gc
import os
import tempfile

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from molecule import load_simulation, TOPOLOGY_FORMATS, TRAJECTORY_FORMATS
from viewer import render_structure
import analysis

st.set_page_config(page_title="MDExplorer", layout="wide")


def save_upload(uploaded_file):
    """Save an upload to a temp file and return its path (readers need a real file)."""
    extension = os.path.splitext(uploaded_file.name)[1]
    handle, path = tempfile.mkstemp(suffix=extension)
    with os.fdopen(handle, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def run_all_analysis(system, with_sasa=True, sasa_points=960):
    """Run every measurement. If one fails, its result is None (shown as "Not
    available") and the reason is kept in errors so the page can explain it.
    SASA is by far the slowest, so the user can leave it out."""
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
    errors = {}
    for name in to_run:
        if name == "sasa" and not with_sasa:
            results[name] = None
            errors[name] = "Skipped: 'Calculate SASA' was not ticked (it is the slowest measurement)."
            continue
        try:
            if name == "sasa":
                results[name] = analysis.calculate_sasa(system, sasa_points)
            else:
                results[name] = to_run[name](system)
        except Exception as error:
            results[name] = None
            errors[name] = type(error).__name__ + ": " + str(error)

    return results, errors


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


def make_plot(system, results, frame, metric, x_values, gromacs_units, rmsd_reference):
    """Build the Plotly graph for whichever metric the user picked.

    x_values is what the frame axis shows: the frame numbers, or the time of each frame in ns.
    gromacs_units shows distances in nm (1 nm = 10 A) like GROMACS does, instead of A."""
    scale = 0.1 if gromacs_units else 1.0
    unit = "nm" if gromacs_units else "A"
    x_title = "Time (ns)" if gromacs_units else "Frame"

    # RMSF has one value per alpha carbon, so label its bars with that residue's number
    if results["rmsf"] is not None:
        numbers = system.resids[system.select(["CA"])]
        if len(numbers) != len(results["rmsf"]):
            raise ValueError("Got %d residue numbers but %d RMSF values."
                             % (len(numbers), len(results["rmsf"])))

        if len(set(numbers.tolist())) == len(numbers):
            residues = numbers
            residue_axis = "Residue number"
        else:
            # chains that reuse residue numbers would stack their bars, so plot in file order
            residues = np.arange(1, len(numbers) + 1)
            residue_axis = "Residue position in the file (residue numbers repeat across chains)"
        rmsf = np.array(results["rmsf"]) * scale
        rmsf_bars = dict(x=residues, y=rmsf, customdata=numbers,
                         hovertemplate="residue number %{customdata}<br>RMSF %{y:.3f} " + unit + "<extra></extra>")

    # RMSF is different from the others, it is per residue and not per frame
    if metric == "RMSF":
        figure = go.Figure(go.Bar(**rmsf_bars))
        figure.update_layout(xaxis_title=residue_axis, yaxis_title="RMSF (" + unit + ")", height=350)
        return figure

    # all three metrics under each other, for comparing them
    if metric == "All":
        figure = make_subplots(rows=3, subplot_titles=("RMSD", "RMSF", "Rg"))
        if results["rmsd"] is not None:
            figure.add_trace(go.Scatter(x=x_values, y=np.array(results["rmsd"]) * scale, name="RMSD"), row=1, col=1)
        if results["rmsf"] is not None:
            figure.add_trace(go.Bar(name="RMSF", **rmsf_bars), row=2, col=1)
        if results["rg"] is not None:
            figure.add_trace(go.Scatter(x=x_values, y=np.array(results["rg"]) * scale, name="Rg"), row=3, col=1)
        figure.update_layout(height=700)
        return figure

    if metric == "RMSD":
        values = np.array(results["rmsd"]) * scale
        y_title = "RMSD vs frame %d (%s)" % (rmsd_reference, unit)
    else:
        values = np.array(results["rg"]) * scale
        y_title = "Rg (" + unit + ")"

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=x_values, y=values, mode="lines+markers", name=metric,
        # markers give clicks something to hit; a plain line has no target
        marker=dict(size=5, opacity=0.15),
    ))

    # a red line showing which frame is on screen in the 3D viewer
    figure.add_vline(x=x_values[frame], line_color="red", line_dash="dash", opacity=0.6)

    # a shaded band if the user has selected a range of frames
    if st.session_state["selected_range"] is not None:
        start, end = st.session_state["selected_range"]
        figure.add_vrect(x0=x_values[start], x1=x_values[end], fillcolor="lightblue", opacity=0.2, line_width=0)

    figure.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=350)
    return figure


def handle_click(event, x_values):
    """Jump to the clicked frame, or select a dragged range."""
    if event is None:
        return

    points = event.selection["points"]
    if len(points) == 0:
        return

    # a point's x is a frame number or a time; find which frame it belongs to
    clicked = sorted(set(int(np.argmin(np.abs(x_values - point["x"]))) for point in points))

    # slider's already drawn this run, so stash the value for next run
    if len(clicked) == 1:
        st.session_state["new_frame"] = clicked[0]
        st.session_state["selected_range"] = None
    else:
        st.session_state["selected_range"] = (clicked[0], clicked[-1])
        st.session_state["new_frame"] = clicked[0]

    st.rerun()


def main():
    st.title("MDExplorer — Interactive Molecular Dynamics Trajectory Explorer")

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
        stride = st.number_input(
            "Use every Nth frame", min_value=1, value=1, step=1,
            help="Long trajectories are slow: on one test PC a 7,000-atom protein needed about 3 s per "
                 "frame for SASA and 0.4 s for hydrogen bonds. 1 = every frame, 100 = every 100th frame.",
        )
        with_sasa = st.checkbox(
            "Calculate SASA (slow)", value=False,
            help="Leave it off for a quick first look, then reload with it ticked and a larger "
                 "'every Nth frame'.",
        )
        fast_sasa = st.checkbox(
            "Fast SASA (100 points per atom)", value=True,
            help="About 5x faster than the full 960 points; the area comes out about 0.5% different. "
                 "Untick it for the most precise values.",
        )
        hq_render = st.checkbox("High quality render", value=False, help="Ray-traced render (slow).")
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
                system = load_simulation(top_path, traj_path, int(stride))
                results, errors = run_all_analysis(system, with_sasa, 100 if fast_sasa else 960)

            # session_state survives reruns, so we don't reload on every click
            st.session_state["system"] = system
            st.session_state["results"] = results
            st.session_state["errors"] = errors
            st.session_state["rmsd_reference"] = 0          # the frame the stored RMSD is measured against
            st.session_state["rmsd_reference_input"] = 0    # what the reference box shows
            st.session_state["periodic"] = analysis.check_periodic_split(system)
            st.session_state["frame"] = 0
            st.session_state["selected_range"] = None
        except ValueError as error:
            # our own checks in molecule.py, already worded for the user
            st.error(str(error))
        except Exception as error:
            # anything else, e.g. a corrupt file that MDAnalysis cannot parse
            st.error("Could not read these files. " + type(error).__name__ + ": " + str(error))
        finally:
            # the temp copies are not needed any more: a good load holds everything in
            # memory and a bad one is over. A failed MDAnalysis reader is only closed when
            # Python's garbage collector runs, and Windows will not delete an open file.
            gc.collect()
            for path in (top_path, traj_path):
                if path is not None:
                    try:
                        os.remove(path)
                    except OSError:
                        pass  # still locked (for example by a virus scanner); leave it

    # nothing loaded yet, so there is nothing else to draw
    if "system" not in st.session_state:
        st.info("Upload a topology file (and a trajectory file) and click **Load simulation**.")
        return

    system = st.session_state["system"]
    results = st.session_state["results"]
    errors = st.session_state.get("errors", {})

    # apply a frame picked last run, before the slider is drawn
    if "new_frame" in st.session_state:
        st.session_state["frame"] = st.session_state.pop("new_frame")

    # with one frame there is no slider, so Streamlit may have dropped the "frame" key
    frame = st.session_state.get("frame", 0)

    # --- what was loaded ---
    st.subheader("Simulation information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Atoms", "{:,}".format(system.n_atoms))
    col2.metric("Residues", "{:,}".format(system.n_residues))
    col3.metric("Frames", "{:,}".format(system.n_frames))
    col4.metric("Length", "%.1f ps" % system.time[-1])
    if system.stride > 1:
        st.caption("Frames were skipped: 1 frame in every %d of the file was loaded (frames 0, %d, %d, ...); "
                   "times are those of the original file." % (system.stride, system.stride, 2 * system.stride))

    # warn before the numbers are read if the protein may be cut by the box edge
    if "periodic" in st.session_state:
        answer, details = st.session_state["periodic"]
        if answer == analysis.SPLIT:
            st.warning("**The protein may cross a periodic boundary.** " + details + "\n\n"
                       "RMSD, RMSF, radius of gyration, centre of mass and the 3D structure may be "
                       "misleading. MDExplorer has not changed your coordinates. Make the "
                       "trajectory whole with an external tool (the README shows a GROMACS "
                       "example), check it, and load the corrected files.")
        elif answer == analysis.UNKNOWN:
            st.info("MDExplorer could not reliably check whether the protein crosses a "
                    "periodic boundary. " + details)
        else:
            st.caption("Periodic-boundary check: " + answer.lower() + ". " + details +
                       " This is a limited check and does not prove the trajectory is whole.")

    # say why any measurement is "Not available"
    if len(errors) > 0:
        lines = ["Some measurements could not be calculated:"]
        for name in errors:
            lines.append("- **" + name + "**: " + errors[name])
        st.warning("\n".join(lines))

    # --- the 3D view ---
    st.subheader("3D structure")
    st.caption("Frame %d, at %.1f ps" % (frame, system.time[frame]))
    st.image(render_structure(system, frame, high_quality=hq_render))
    if system.n_frames > 1:
        st.slider("Frame", min_value=0, max_value=system.n_frames - 1, key="frame")
    else:
        # Streamlit does not allow a slider whose minimum equals its maximum
        st.caption("This file has only one frame, so only frame 0 is available.")

    # --- the graph ---
    st.subheader("Trajectory analysis")

    # RMSD is measured against frame 0 unless the user picks another frame
    if results["rmsd"] is not None and system.n_frames > 1:
        reference = int(st.number_input(
            "RMSD reference frame", min_value=0, max_value=system.n_frames - 1, step=1,
            key="rmsd_reference_input",
            help="Every frame is compared with this one, so its RMSD is 0."))
        if reference != st.session_state.get("rmsd_reference", 0):
            with st.spinner("Recalculating RMSD..."):
                results["rmsd"] = analysis.calculate_rmsd(system, reference)
            st.session_state["rmsd_reference"] = reference
    rmsd_reference = st.session_state.get("rmsd_reference", 0)

    # GROMACS plots time in ns and distances in nm; the numbers below the graph stay in ps and A
    gromacs_units = st.checkbox("GROMACS-style graphs (time in ns, distances in nm)")
    if gromacs_units:
        x_values = system.time / 1000
    else:
        x_values = np.arange(system.n_frames)

    choices = []
    for name, label in [("rmsd", "RMSD"), ("rmsf", "RMSF"), ("rg", "Rg")]:
        if results[name] is not None:
            choices.append(label)

    if len(choices) == 0:
        st.info("None of the graphs could be calculated for this file.")
    else:
        choices.append("All")
        metric = st.radio("Metric", choices, horizontal=True)
        figure = make_plot(system, results, frame, metric, x_values, gromacs_units, rmsd_reference)

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
            handle_click(event, x_values)
        else:
            st.plotly_chart(figure, key="graph")

    # --- the results as files, for plotting in other programs ---
    st.subheader("Download the results")
    col1, col2 = st.columns(2)
    col1.download_button("Per-frame values (CSV)", analysis.per_frame_csv(system, results),
                         file_name="mdexplorer_per_frame.csv", mime="text/csv")
    col2.download_button("RMSF of each residue (CSV)", analysis.rmsf_csv(system, results),
                         file_name="mdexplorer_rmsf.csv", mime="text/csv", disabled=results["rmsf"] is None)

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


# SASA worker processes re-import this file on Windows; without this guard each would run the whole app
if __name__ == "__main__":
    main()
