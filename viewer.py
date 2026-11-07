"""Draws one frame in 3D by writing it out as PDB text and rendering it with PyMOL.

Water and ions are excluded, or the camera would zoom out to fit them.
"""

import json
import threading

import numpy as np
import pymol2

# The classic `pymol.finish_launching()` API expects to run on the process's
# main thread (it sets up a windowing/GL context), but Streamlit runs the
# app script on a worker thread, so it hangs forever there. pymol2's PyMOL()
# is the embeddable session API built for exactly this: no GUI, no
# main-thread requirement.
_session = pymol2.PyMOL()
_session.start()
cmd = _session.cmd

# The session is still one shared object. Streamlit reruns each interaction
# (e.g. every slider tick) on its own thread, so without this lock two
# overlapping renders can corrupt PyMOL's state and crash the whole server,
# dropping every connected browser.
# RLock: pymol_3d_html holds it while calling render_structure, which takes it again
_pymol_lock = threading.RLock()


def make_pdb_text(system, frame):
    """Write the protein atoms of one frame as PDB text. Each chain (segid) gets its own
    chain letter A, B, C, ... so PyMOL can colour by chain."""
    lines = []
    chain_letters = {}
    atoms = np.where(system.protein_mask())[0]
    coords = system.coordinates[frame][atoms]

    for k, i in enumerate(atoms):
        if system.segids[i] not in chain_letters:
            chain_letters[system.segids[i]] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(chain_letters) % 26]
        x, y, z = coords[k]
        name = str(system.names[i])
        resname = str(system.resnames[i])

        # PDB uses fixed columns; pad short names with a leading space
        if len(name) < 4:
            name = " " + name.ljust(3)

        line = "ATOM  " + str(i + 1).rjust(5) + " " + name[:4] + " " + resname[:3].rjust(3)
        line += " " + chain_letters[system.segids[i]] + str(int(system.resids[i])).rjust(4) + "    "
        line += "%8.3f%8.3f%8.3f  1.00  0.00          %2s" % (x, y, z, system.elements[i])
        lines.append(line)

    lines.append("END")
    return "\n".join(lines)


# PyMOL's chain colours, in the order its "colour by chain" uses for the first chains
CHAIN_COLORS = ["#33ff33", "#ffff80", "#ff3333", "#33ffff", "#ff33ff", "#ff9933", "#9999ff", "#bfbfbf"]

# residues drawn as cartoon; anything else in the protein selection (a ligand) is drawn as sticks
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
               "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
               "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "ASH", "GLH", "LYN",
               "MSE", "CSO", "SEP", "TPO", "PTR"]   # common modified amino acids

# the same list as a PyMOL selection. PyMOL's own "polymer.protein" does not know force-field
# names such as HSD, and would draw those residues as a ligand with pocket dots around them.
PROTEIN = "resn " + "+".join(AMINO_ACIDS)

COLOR_SCHEMES = ["Secondary structure", "Rainbow", "Chain"]


def render_structure(system, frame, width=900, height=520, high_quality=False, color_by="Secondary structure"):
    """Render the protein PyMOL-style (cartoon, ligands as sticks, pocket dots). color_by is one of
    COLOR_SCHEMES: helices red, strands yellow and loops green; a rainbow from the first residue
    (blue) to the last (red); or one colour per chain.
    Returns (PNG bytes, pixels): pixels[k] is the (x, y) position in the picture of atom
    click_targets(system)[k], so clicks on the picture can be matched to atoms."""
    with _pymol_lock:
        cmd.reinitialize()
        cmd.read_pdbstr(make_pdb_text(system, frame), "frame")

        cmd.hide("everything")
        cmd.show("cartoon")
        # named selections: PyMOL misreads the long residue list when it is written inside "within"
        cmd.select("protein_atoms", PROTEIN)
        cmd.select("ligand", "not protein_atoms")
        cmd.show("sticks", "ligand")
        # chain colours first: they stay on the ligands' carbons in every scheme
        for k, chain in enumerate(cmd.get_chains("frame")):
            cmd.color(CHAIN_COLORS[k % len(CHAIN_COLORS)].replace("#", "0x"), "chain " + chain)
        if color_by == "Secondary structure":
            # PyMOL works out the helices and strands itself when it reads the frame
            cmd.color("green", "protein_atoms")
            cmd.color("red", "protein_atoms and ss H")
            cmd.color("yellow", "protein_atoms and ss S")
        elif color_by == "Rainbow":
            cmd.spectrum("count", "rainbow", "protein_atoms and name CA")
        cmd.color("atomic", "ligand and not elem C")
        # magenta dots on the protein residues within 5 A of a ligand (the binding pocket)
        cmd.show("dots", "byres (protein_atoms within 5 of ligand)")
        cmd.delete("protein_atoms")   # only the selections, not their atoms
        cmd.delete("ligand")
        cmd.set("dot_color", "magenta")
        cmd.set("dot_density", 0)   # sparse dots, so the ligand stays visible
        cmd.bg_color("black")
        cmd.set("ray_opaque_background", 1)   # otherwise the PNG background is transparent, not black
        cmd.orient()

        # cmd.png(..., ray=1) already ray-traces internally, so a separate cmd.ray()
        # call before it would ray-trace the same image twice for no benefit.
        png = cmd.png(None, width=width, height=height, dpi=150, ray=1 if high_quality else 0)

        # where each clickable atom lands in the picture, through PyMOL's own camera
        view = cmd.get_view()
        points = system.coordinates[frame][click_targets(system)].astype(float)
        camera = (points - np.array(view[12:15])) @ np.array(view[:9]).reshape(3, 3) + np.array(view[9:12])
        scale = np.tan(np.radians(float(cmd.get("field_of_view"))) / 2) * -camera[:, 2]
        pixels = np.column_stack([(camera[:, 0] / (scale * width / height) + 1) / 2 * width,
                                  (1 - camera[:, 1] / scale) / 2 * height])
        return png, pixels


# The free-rotation view: PyMOL's own 3D shapes (cartoon, sticks, dots, exported as VRML)
# drawn in the browser by three.js, so it looks like the PyMOL picture but turns with the mouse.
# Clicking finds the clickable atom nearest the clicked point; two clicks give a distance.
PYMOL_3D_HTML = """
<style>body { margin: 0; background: black; }</style>
<div id="box" style="position:relative;width:WIDTHpx;height:HEIGHTpx;font-family:sans-serif"></div>
<script type="importmap">{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {VRMLLoader} from "three/addons/loaders/VRMLLoader.js";
import {CSS2DRenderer, CSS2DObject} from "three/addons/renderers/CSS2DRenderer.js";

const DATA = DATA_JSON;
const box = document.getElementById("box");
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(DATA.width, DATA.height);
box.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(DATA.width, DATA.height);
Object.assign(labels.domElement.style, {position: "absolute", top: "0px", pointerEvents: "none"});
box.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(DATA.fov, DATA.width / DATA.height, 1, 5000);
camera.position.set(0, 0, DATA.distance);
scene.add(camera);

// PyMOL's lighting, with its own settings: ambient 0.14, a light from the camera ("direct" 0.45),
// a light from the upper right ("reflect" 0.45, light -0.4 -0.4 -1) that makes the white highlights
// (specular 1, shininess 55), and depth cueing: parts further away fade into the black background.
// Colours are used as they are, like PyMOL, without three.js's colour-space conversion.
THREE.ColorManagement.enabled = false;
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.useLegacyLights = true;   // light intensities as plain numbers, as PyMOL uses them
scene.add(new THREE.AmbientLight(0xffffff, DATA.ambient));
const direct = new THREE.DirectionalLight(0xffffff, DATA.direct);
direct.position.set(0, 0, 1);
camera.add(direct);
const reflect = new THREE.DirectionalLight(0xffffff, DATA.reflect);
reflect.position.set(-DATA.light[0], -DATA.light[1], -DATA.light[2]);
camera.add(reflect);
scene.fog = new THREE.Fog(0x000000, DATA.fog_start, DATA.fog_end);

const model = new VRMLLoader().parse(DATA.vrml, "");
const extras = [];
model.traverse(function (o) {
  if (o.isLight || o.isCamera) extras.push(o);
  if (o.isMesh) {
    const old = o.material;
    o.material = new THREE.MeshPhongMaterial({color: old.color, vertexColors: old.vertexColors,
      specular: new THREE.Color(DATA.specular, DATA.specular, DATA.specular), shininess: DATA.shininess,
      side: THREE.DoubleSide});
  }
});
extras.forEach(function (o) { o.removeFromParent(); });
scene.add(model);

const controls = new OrbitControls(camera, renderer.domElement);
(function draw() {
  requestAnimationFrame(draw);
  controls.update();
  // zooming moves the camera; move the depth cueing with it, as PyMOL does
  const shift = camera.position.length() - DATA.distance;
  scene.fog.near = DATA.fog_start + shift;
  scene.fog.far = DATA.fog_end + shift;
  renderer.render(scene, camera);
  labels.render(scene, camera);
})();

function label(text, position, color, background) {
  const div = document.createElement("div");
  div.textContent = text;
  Object.assign(div.style, {color: color, background: background, fontSize: "12px", padding: "1px 4px",
                            borderRadius: "3px", transform: "translateY(-14px)"});
  const object = new CSS2DObject(div);
  object.position.copy(position);
  scene.add(object);
  return object;
}

const added = [];       // labels and lines, so they can be cleared
const picked = [];      // clicked atom numbers
let first = null;       // the first atom of an unfinished distance
function pick(k) {
  if (picked.includes(k)) {   // clicking a labelled atom again clears everything
    added.forEach(function (o) { o.removeFromParent(); });
    added.length = 0; picked.length = 0; first = null;
    return;
  }
  const p = new THREE.Vector3(...DATA.targets[k]);
  picked.push(k);
  added.push(label(DATA.names[k], p, "white", "black"));
  if (first === null) { first = p; return; }
  // the dashed line as short yellow cylinders (browsers draw real lines only 1 pixel wide)
  const along = p.clone().sub(first);
  const turn = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), along.clone().normalize());
  const dash = new THREE.CylinderGeometry(0.15, 0.15, 0.8);
  const yellow = new THREE.MeshBasicMaterial({color: 0xffff00, depthTest: false});   // always on top
  for (let s = 0.4; s < along.length(); s += 1.3) {
    const piece = new THREE.Mesh(dash, yellow);
    piece.position.copy(first).addScaledVector(along, s / along.length());
    piece.quaternion.copy(turn);
    piece.renderOrder = 1;
    scene.add(piece);
    added.push(piece);
  }
  const middle = first.clone().add(p).multiplyScalar(0.5);
  added.push(label(first.distanceTo(p).toFixed(1), middle, "yellow", "transparent"));
  first = null;
}

// a click (not a drag) picks the clickable atom nearest the point of the model that was hit
const raycaster = new THREE.Raycaster();
let down = null;
renderer.domElement.addEventListener("pointerdown", function (e) { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", function (e) {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  const r = renderer.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1,
                                            -(e.clientY - r.top) / r.height * 2 + 1), camera);
  const hits = raycaster.intersectObject(model, true);
  if (hits.length === 0) return;
  let best = -1, bestDistance = 4.0;   // A; further than this from any clickable atom counts as a miss
  DATA.targets.forEach(function (t, k) {
    const d = hits[0].point.distanceTo(new THREE.Vector3(...t));
    if (d < bestDistance) { best = k; bestDistance = d; }
  });
  if (best >= 0) pick(best);
});
</script>
"""


def pymol_3d_html(system, frame, width=900, height=520, color_by="Secondary structure"):
    """The free-rotation view of one frame as HTML: PyMOL's own 3D shapes, turned with the mouse.
    Needs internet in the browser, because three.js is loaded from its website."""
    with _pymol_lock:   # one lock round both, so no other render can change the scene in between
        render_structure(system, frame, color_by=color_by)   # sets up PyMOL's scene for this frame
        vrml = cmd.get_vrml()
        view = cmd.get_view()
        fov = float(cmd.get("field_of_view"))
        lighting = {name: float(cmd.get(name)) for name in ["ambient", "direct", "reflect", "specular",
                                                             "shininess", "fog_start"]}
        lighting["light"] = [float(x) for x in cmd.get("light").strip("[] ").split(",")]

    # the exported shapes sit in PyMOL's camera space: rotated as on screen, centred on the origin
    # of rotation; the clickable atoms are turned the same way
    targets = click_targets(system)
    points = (system.coordinates[frame][targets].astype(float) - np.array(view[12:15])) @ np.array(view[:9]).reshape(3, 3)
    names = ["%s %s %d : %s" % (system.segids[i], system.resnames[i], system.resids[i], system.names[i])
             for i in targets]

    # PyMOL's own camera: its distance and its (vertical) field of view, so the first view is the picture
    # depth cueing runs from fog_start of the way between PyMOL's front and back clipping planes to the back
    front, back = view[15], view[16]
    # The browser's lights are a little stronger than PyMOL's and its highlights much stronger, so they
    # are scaled down (0.9 and 0.15): those factors gave the smallest pixel difference from PyMOL's own
    # picture of the same frame.
    data = dict(width=width, height=height, vrml=vrml, targets=points.round(3).tolist(), names=names,
                distance=-view[11], fov=fov, ambient=0.9 * lighting["ambient"], direct=0.9 * lighting["direct"],
                reflect=0.9 * lighting["reflect"], light=lighting["light"], specular=0.15 * lighting["specular"],
                shininess=lighting["shininess"], fog_start=front + lighting["fog_start"] * (back - front),
                fog_end=back)
    return (PYMOL_3D_HTML.replace("WIDTH", str(width)).replace("HEIGHT", str(height))
            .replace("DATA_JSON", json.dumps(data)))


def click_targets(system):
    """The atoms that can be clicked in the PyMOL picture: the alpha carbon of every amino acid
    (the cartoon runs through them) and every atom of a ligand (drawn as sticks)."""
    amino = np.isin(system.resnames, AMINO_ACIDS)
    return np.where(system.protein_mask() & ((amino & (system.names == "CA")) | ~amino))[0]
