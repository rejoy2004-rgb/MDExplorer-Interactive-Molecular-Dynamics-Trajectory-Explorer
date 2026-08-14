import os
import tempfile

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

from molnumpy.readers.factory import load_molecule, detect_format, FORMATS
from molnumpy.visualization.viewer import render_molecular_viewer

st.set_page_config(page_title="MolNumPy - Interactive Dashboard", page_icon="🧬", layout="wide")

TOP_EXTENSIONS = ['.pdb', '.psf', '.gro', '.tpr', '.prmtop', '.parm7']
TRAJ_EXTENSIONS = ['.dcd', '.xtc', '.trr', '.nc']


def save_uploaded_file(uploaded_file) -> str:
    """Write an uploaded file to a temp path so MDAnalysis can read it."""
    ext = os.path.splitext(uploaded_file.name)[1]
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def kabsch_rmsd(ref: np.ndarray, target: np.ndarray) -> float:
    """Calculate the RMSD between ref and target after Kabsch alignment.

    ref and target have shape (N, 3).
    """
    # 1. Translate centroids to origin
    ref_centroid = np.mean(ref, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    ref_c = ref - ref_centroid
    target_c = target - target_centroid
    
    # 2. Compute covariance matrix
    h = target_c.T @ ref_c
    
    # 3. SVD decomposition
    u, s, vt = np.linalg.svd(h)
    
    # 4. Correct reflection if det < 0
    d = np.linalg.det(u @ vt)
    if d < 0:
        u[:, -1] = -u[:, -1]
    r = u @ vt
    
    # 5. Rotate target coordinate system
    target_aligned = target_c @ r
    
    # 6. Calculate root mean square deviation
    diff = ref_c - target_aligned
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


@st.cache_data
def compute_trajectory_metrics(coordinates: np.ndarray, atom_names: np.ndarray, resnames: np.ndarray):
    """Precompute physical trajectory analysis metrics (Rg, RMSD, HBonds, SASA).

    All math runs in compiled vector operations via NumPy and SciPy.
    - Excludes solvent/water molecules for accurate biological measurements.
    - Uses Kabsch alignment to compute structural RMSD.
    """
    n_frames, n_atoms, _ = coordinates.shape
    
    # Filter out solvent molecules (water, ions)
    _WATER_NAMES = {'HOH', 'WAT', 'SOL', 'TIP3', 'SPC', 'CL', 'NA', 'K'}
    is_not_water = np.array([str(res).strip().upper() not in _WATER_NAMES for res in resnames])
    
    non_water_idx = np.where(is_not_water)[0]
    if len(non_water_idx) == 0:
        non_water_idx = np.arange(n_atoms)
    
    # 1. Radius of Gyration (Rg) - computed on non-water atoms
    coordinates_nw = coordinates[:, non_water_idx, :]
    centers = np.mean(coordinates_nw, axis=1, keepdims=True)
    squared_distances = np.sum((coordinates_nw - centers) ** 2, axis=2)
    rg_values = np.sqrt(np.mean(squared_distances, axis=1))
    
    # 2. RMSD with Kabsch Alignment (computed on protein C-alpha backbone atoms)
    is_ca = np.array([str(name).strip().upper() == 'CA' for name in atom_names])
    ca_idx = np.where(is_ca & is_not_water)[0]
    # Fallback to sub-sampled non-water slice if no C-alphas found
    if len(ca_idx) < 5:
        ca_idx = non_water_idx[::max(1, len(non_water_idx) // 100)]
        
    ref_ca = coordinates[0, ca_idx, :]
    rmsd_values = []
    for f in range(n_frames):
        target_ca = coordinates[f, ca_idx, :]
        rmsd_values.append(kabsch_rmsd(ref_ca, target_ca))
    rmsd_values = np.array(rmsd_values)
    
    # 3. Hydrogen Bonds (Geometric polar-pair contacts < 3.5 Å on non-water atoms)
    is_polar = np.array([str(name).strip().upper().startswith(('N', 'O')) for name in atom_names])
    polar_idx = np.where(is_polar & is_not_water)[0]
    
    h_bonds = []
    if len(polar_idx) > 1:
        for f in range(n_frames):
            coords_f = coordinates[f, polar_idx, :]
            dists = cdist(coords_f, coords_f)
            # Exclude self-interactions (>0.1)
            count = np.sum((dists > 0.1) & (dists < 3.5)) / 2
            h_bonds.append(int(count))
    else:
        h_bonds = [0] * n_frames
        
    # 4. SASA (Solvent Accessible Surface Area) approximation on C-alpha atoms
    sasa_approx = []
    if len(ca_idx) > 1:
        for f in range(n_frames):
            coords_f = coordinates[f, ca_idx, :]
            dists = cdist(coords_f, coords_f)
            neighbors = np.sum(dists < 8.0, axis=1) - 1
            exposure = np.sum(np.maximum(0, 12 - neighbors)) * 14.5
            sasa_approx.append(round(exposure, 1))
    else:
        sasa_approx = [0.0] * n_frames
        
    return {
        "Rg": rg_values,
        "RMSD": rmsd_values,
        "HBonds": h_bonds,
        "SASA": sasa_approx
    }


def main():
    st.title("🧬 MolNumPy Interactive Digital Dashboard")
    st.caption(
        "Upload molecular dynamics trajectories to interact with 3D views and real-time analytical plots "
        "including RMSD, Hydrogen Bonds, and SASA."
    )

    # ---------------------------------------------------------------- upload
    st.sidebar.header("1. Upload files")
    top_file = st.sidebar.file_uploader(
        "Topology / structure file",
        type=[e.lstrip('.') for e in TOP_EXTENSIONS],
        help="PDB (alone) or the topology of PSF/GRO/TPR/PRMTOP.",
    )
    traj_file = st.sidebar.file_uploader(
        "Trajectory file (if required)",
        type=[e.lstrip('.') for e in TRAJ_EXTENSIONS],
        help="DCD, XTC, TRR or NC. Not needed for a plain PDB.",
    )

    if top_file is not None:
        info = FORMATS.get(os.path.splitext(top_file.name)[1].lower())
        if info is not None:
            st.sidebar.info(info['hint'])
        else:
            st.sidebar.error("Unsupported file format.")

    # ---------------------------------------------------------------- load
    st.sidebar.header("2. Load")
    if st.sidebar.button("Load Molecule", use_container_width=True):
        try:
            top_path = save_uploaded_file(top_file) if top_file else None
            traj_path = save_uploaded_file(traj_file) if traj_file else None

            info = detect_format(top_path, traj_path)
            system = load_molecule(top_path, traj_path)
            system.format = info['name']

            # Calculate and store trajectory metrics
            metrics = compute_trajectory_metrics(
                system.coordinates,
                system.topology.atoms['name'],
                system.topology.atoms['resname']
            )

            st.session_state['system'] = system
            st.session_state['metrics'] = metrics
            st.session_state['frame'] = 0
            st.session_state['load_error'] = None
        except ValueError as e:
            st.session_state['system'] = None
            st.session_state['metrics'] = None
            st.session_state['load_error'] = str(e)

    if st.session_state.get('load_error'):
        st.error(st.session_state['load_error'])

    # ---------------------------------------------------------------- display
    system = st.session_state.get('system')
    metrics = st.session_state.get('metrics')
    
    if system is None or metrics is None:
        st.info("Please upload your molecular topology and trajectory files, then click **Load Molecule**.")
        return

    # Check Plotly graph clicks before instantiating the slider widget
    if 'trajectory_plot' in st.session_state and st.session_state['trajectory_plot'] is not None:
        plot_event = st.session_state['trajectory_plot']
        if 'selection' in plot_event and 'points' in plot_event['selection']:
            points_data = plot_event['selection']['points']
            if len(points_data) > 0:
                clicked_frame = int(points_data[0]['x'])
                st.session_state['frame'] = clicked_frame

    n_residues = len(np.unique(system.topology.atoms['resid']))
    fmt = getattr(system, 'format', 'unknown')

    # Top Metrics Bar
    st.subheader("System Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File format", fmt)
    c2.metric("Number of atoms", f"{system.n_atoms:,}")
    c3.metric("Number of frames", f"{system.n_frames:,}")
    c4.metric("Number of residues", f"{n_residues:,}")

    # Layout: Split 3D View and Plotly Graph side by side
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.header("3D Trajectory Viewer")
        frame_idx = st.slider(
            "Select Trajectory Frame",
            min_value=0,
            max_value=system.n_frames - 1,
            value=st.session_state.get('frame', 0),
            key='frame',
        )
        st.caption(
            f"Frame {frame_idx} of {system.n_frames - 1} | "
            f"time = {system.time[frame_idx]:.1f} (if available)"
        )
        render_molecular_viewer(system, frame_idx, width=650, height=450)

    with col_right:
        st.header("Interactive Graph")
        
        # Mapping metrics to data & aesthetics
        metric_map = {
            "Radius of Gyration (Rg)": (metrics["Rg"], "Radius of Gyration (Å)", "#00b4d8"),
            "RMSD": (metrics["RMSD"], "RMSD (Å)", "#ffb703"),
            "Hydrogen Bonds": (metrics["HBonds"], "H-Bond Count", "#fb8500"),
            "SASA (Approximate)": (metrics["SASA"], "SASA (Å²)", "#2a9d8f")
        }
        
        selected_metric_name = st.selectbox(
            "Select Metric to Plot:",
            list(metric_map.keys()),
            index=0
        )
        y_data, y_label, line_color = metric_map[selected_metric_name]
        
        # Build tooltips displaying all data points for any hovered frame
        hovertext = [
            f"<b>Frame {f}</b><br>"
            f"Radius of Gyration (Rg): {rg:.3f} Å<br>"
            f"RMSD: {rmsd:.3f} Å<br>"
            f"Hydrogen Bonds: {hb}<br>"
            f"SASA (Approx): {sasa:,.1f} Å²"
            for f, (rg, rmsd, hb, sasa) in enumerate(zip(
                metrics["Rg"], metrics["RMSD"], metrics["HBonds"], metrics["SASA"]
            ))
        ]
        
        # Build Plotly interactive line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(y_data)),
            y=y_data,
            mode='lines+markers',
            name=selected_metric_name,
            line=dict(color=line_color, width=3),
            marker=dict(size=5, opacity=0.7),
            hovertext=hovertext,
            hoverinfo="text"
        ))
        
        # Dotted line highlighting the active frame
        fig.add_vline(
            x=frame_idx,
            line_width=2.5,
            line_dash="dash",
            line_color="#ff4b4b"
        )
        
        fig.update_layout(
            clickmode='event+select',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            xaxis=dict(
                title="Frame Index",
                gridcolor="#2d2d2d",
                zerolinecolor="#2d2d2d",
                tickfont=dict(color="#888")
            ),
            yaxis=dict(
                title=y_label,
                gridcolor="#2d2d2d",
                zerolinecolor="#2d2d2d",
                tickfont=dict(color="#888")
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
            showlegend=False
        )
        
        # Render Plotly graph with click selection capabilities
        selected_points = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            key="trajectory_plot"
        )
        
        st.caption("💡 *Click any point on the graph above to jump directly to that frame in 3D!*")

    # Bottom Dashboard Panel
    st.write("---")
    st.header("Active Frame Analytics")
    
    rg_val = metrics["Rg"][frame_idx]
    rmsd_val = metrics["RMSD"][frame_idx]
    h_val = metrics["HBonds"][frame_idx]
    sasa_val = metrics["SASA"][frame_idx]
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    
    # Custom Styled cards
    mc1.markdown(
        f"<div style='background-color:#1e293b; padding:15px; border-radius:10px; border-left: 5px solid #00b4d8;'>"
        f"<span style='color:#94a3b8; font-size:12px; font-weight:bold;'>RADIUS OF GYRATION (Rg)</span>"
        f"<h2 style='margin:5px 0 0 0; color:#f8fafc;'>{rg_val:.3f} Å</h2>"
        f"<small style='color:#64748b;'>Compactness / size score</small>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    mc2.markdown(
        f"<div style='background-color:#1e293b; padding:15px; border-radius:10px; border-left: 5px solid #ffb703;'>"
        f"<span style='color:#94a3b8; font-size:12px; font-weight:bold;'>RMSD (FROM FRAME 0)</span>"
        f"<h2 style='margin:5px 0 0 0; color:#f8fafc;'>{rmsd_val:.3f} Å</h2>"
        f"<small style='color:#64748b;'>Structural displacement</small>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    mc3.markdown(
        f"<div style='background-color:#1e293b; padding:15px; border-radius:10px; border-left: 5px solid #fb8500;'>"
        f"<span style='color:#94a3b8; font-size:12px; font-weight:bold;'>HYDROGEN BONDS</span>"
        f"<h2 style='margin:5px 0 0 0; color:#f8fafc;'>{h_val}</h2>"
        f"<small style='color:#64748b;'>Polar contact stability</small>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    mc4.markdown(
        f"<div style='background-color:#1e293b; padding:15px; border-radius:10px; border-left: 5px solid #2a9d8f;'>"
        f"<span style='color:#94a3b8; font-size:12px; font-weight:bold;'>SASA (ESTIMATED)</span>"
        f"<h2 style='margin:5px 0 0 0; color:#f8fafc;'>{sasa_val:,.1f} Å²</h2>"
        f"<small style='color:#64748b;'>Solvent exposure area</small>"
        f"</div>",
        unsafe_allow_html=True
    )




if __name__ == "__main__":
    main()
