"""Tests for how the page handles one frame, many frames and zero frames.

Uses Streamlit's headless AppTest (not a real browser) and the tiny in-memory
proteins from test_periodic_check.py. Run with:   python test_frames.py
(or:  python -m pytest test_frames.py)
"""

import os
import tempfile

from streamlit.testing.v1 import AppTest

import analysis
import molecule
from molecule import load_simulation
from test_periodic_check import make_system

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def open_page(n_frames, keep_frame_key=True):
    """Run app.py as if a simulation with this many frames had just been loaded."""
    system = make_system(list(range(1, 11)), frames=n_frames)
    results = {"rmsd": analysis.calculate_rmsd(system), "rmsf": analysis.calculate_rmsf(system),
               "rg": analysis.calculate_rg(system), "com": analysis.center_of_mass(system),
               "hbonds": None, "sasa": None, "secondary": None}

    page = AppTest.from_file(APP, default_timeout=120)
    page.session_state["system"] = system
    page.session_state["results"] = results
    page.session_state["errors"] = {}
    page.session_state["selected_range"] = None
    if keep_frame_key:
        page.session_state["frame"] = 0
    return page.run()


def test_one_frame_page_has_no_slider_and_still_draws_everything():
    page = open_page(1)
    assert len(page.exception) == 0, [e.value for e in page.exception]
    assert len(page.slider) == 0
    assert any("only one frame" in c.value and "frame 0" in c.value for c in page.caption)
    assert len(page.get("imgs")) == 1                                   # the 3D picture is drawn
    assert [m.value for m in page.metric if m.label == "Frames"] == ["1"]
    assert "0.00 A" in [m.value for m in page.metric if m.label == "RMSD"]   # the numbers for frame 0


def test_one_frame_page_works_when_streamlit_dropped_the_frame_key():
    page = open_page(1, keep_frame_key=False)
    assert len(page.exception) == 0, [e.value for e in page.exception]
    assert len(page.slider) == 0


def test_many_frames_keep_the_slider():
    page = open_page(5)
    assert len(page.exception) == 0, [e.value for e in page.exception]
    assert len(page.slider) == 1
    slider = page.slider[0]
    assert (slider.min, slider.max, slider.value) == (0, 4, 0)
    assert not any("only one frame" in c.value for c in page.caption)


def test_many_frames_slider_moves_the_frame():
    page = open_page(5)
    page.slider[0].set_value(3).run()
    assert len(page.exception) == 0, [e.value for e in page.exception]
    assert any(c.value.startswith("Frame 3, at 3.0 ps") for c in page.caption)
    assert [m.value for m in page.metric if m.label == "Time"] == ["3.0 ps"]


def test_two_frames_is_the_smallest_case_with_a_slider():
    page = open_page(2)
    assert len(page.exception) == 0, [e.value for e in page.exception]
    assert (page.slider[0].min, page.slider[0].max) == (0, 1)


def test_empty_trajectory_file_is_refused_with_an_error():
    # MDAnalysis already refuses a trajectory with no frames; the app shows that error
    folder = tempfile.mkdtemp()
    top = os.path.join(folder, "one_atom.pdb")
    traj = os.path.join(folder, "empty.dcd")
    with open(top, "w") as f:
        f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    open(traj, "wb").close()

    # the topology alone is fine, so any error comes from the empty trajectory
    assert load_simulation(top).n_frames == 1
    try:
        load_simulation(top, traj)
        assert False, "an empty trajectory was accepted"
    except OSError as error:
        assert "DCD" in str(error) or "empty" in str(error)


def test_system_with_zero_frames_is_refused_by_load_simulation():
    # no real reader produces zero frames, so a stand-in system checks our own guard
    class NoFrames:
        n_frames = 0

    real_universe, real_system = molecule.mda.Universe, molecule.MolecularSystem
    molecule.mda.Universe = lambda *arguments: None
    molecule.MolecularSystem = lambda *arguments: NoFrames()
    try:
        load_simulation(os.path.abspath(__file__).replace("test_frames.py", "x.pdb"))
        assert False, "load_simulation accepted a system with no frames"
    except ValueError as error:
        assert "no frames" in str(error)
    finally:
        molecule.mda.Universe, molecule.MolecularSystem = real_universe, real_system


if __name__ == "__main__":
    tests = [(name, function) for name, function in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, function in tests:
        try:
            function()
            print("PASS", name)
        except Exception as error:
            failed += 1
            print("FAIL", name, type(error).__name__, error)
    print("%d tests, %d failed" % (len(tests), failed))
    raise SystemExit(1 if failed else 0)
