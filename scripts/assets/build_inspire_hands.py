"""Build the Inspire RH56 DFQ hand assets that ``supermimic.robots`` attaches to mjlab's G1.

Source: unitreerobotics/unitree_ros ``robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf``
(BSD-3-Clause). Writes ``src/supermimic/assets/inspire_hands/``:

  * ``inspire_hands.xml``: the two hand subtrees (``L_hand_base_link``, ``R_hand_base_link``) with the URDF
    inertials, joint ranges and ``<mimic>`` couplings (MuJoCo joint equalities), body poses relative to
    ``*_wrist_yaw_link``;
  * collision: one primitive per link, fitted by MuJoCo to the URDF collision mesh (box for the palm and the
    thumb rotor, capsule for every phalanx), named ``<link>_collision`` so mjlab's collision configs match them;
  * ``meshes/*.stl``: visual meshes decimated to ``--faces`` triangles.

It also prints the URDF ``*_wrist_yaw_link`` inertial (without hand), which replaces mjlab's (that one includes
the rubber hand).

  uv run --with fast-simplification --with trimesh python scripts/assets/build_inspire_hands.py \\
      --g1-description ~/g1-sweep-retarget/third_party/unitree_ros/robots/g1_description
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import tyro

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/supermimic/assets/inspire_hands"
URDF_NAME = "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
BOX_LINKS = ("hand_base_link", "thumb_proximal_base")


def _load_urdf(desc: Path) -> tuple[mujoco.MjSpec, str]:
    text = (desc / URDF_NAME).read_text()
    text, n = re.subn(r"<compiler[^>]*/>", f'<compiler meshdir="{desc}" discardvisual="false" fusestatic="false"/>', text)
    assert n == 1
    spec = mujoco.MjSpec.from_string(text)
    spec.compiler.degree = False
    return spec, (desc / URDF_NAME).read_text()


def _hand_bodies(spec: mujoco.MjSpec) -> list[mujoco.MjsBody]:
    out = []
    for side in "LR":
        stack = [spec.body(f"{side}_hand_base_link")]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(b.bodies)
    return out


def main(g1_description: Path, faces: int = 1500) -> None:
    import fast_simplification
    import trimesh

    spec, urdf_text = _load_urdf(g1_description)
    mimics = {}
    for j in ET.fromstring(urdf_text).findall("joint"):
        m = j.find("mimic")
        if m is not None:
            mimics[j.get("name")] = (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))
    for eq in spec.equalities:
        driver, mult, off = mimics[eq.name1]
        assert eq.name2 == driver and abs(eq.data[0] - off) < 1e-9 and abs(eq.data[1] - mult) < 1e-9, eq.name1
        eq.name = f"mimic_{eq.name1}"

    # Fit one primitive per hand link to its collision mesh (MuJoCo fits a primitive geom that names a mesh).
    spec.compiler.fitaabb = True
    bodies = _hand_bodies(spec)
    fitted: list[mujoco.MjsGeom] = []
    for b in bodies:
        coll = [g for g in b.geoms if g.contype != 0 or g.conaffinity != 0]
        assert len(coll) == 1, (b.name, len(coll))
        g = coll[0]
        g.name = f"{b.name}_collision"
        g.type = mujoco.mjtGeom.mjGEOM_BOX if b.name.endswith(BOX_LINKS) else mujoco.mjtGeom.mjGEOM_CAPSULE
        fitted.append(g)
    model = spec.compile()
    for g in fitted:
        mg = model.geom(g.name)
        g.size = mg.size.copy()
        g.pos = mg.pos.copy()
        g.quat = mg.quat.copy()
        g.meshname = ""
        g.group = 3
    for b in bodies:
        for g in b.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                g.group = 2

    for side in "LR":
        w = model.body(f"{'left' if side == 'L' else 'right'}_wrist_yaw_link")
        print(f"{side} wrist_yaw_link (URDF, no hand): mass {w.mass[0]:.7g} ipos {np.round(w.ipos, 8).tolist()} "
              f"iquat {np.round(w.iquat, 8).tolist()} inertia {np.round(w.inertia, 10).tolist()}")
        hb = model.body(f"{side}_hand_base_link")
        print(f"{side} hand: pos {hb.pos.tolist()} quat {np.round(hb.quat, 10).tolist()} subtree mass {model.body_subtreemass[hb.id]:.5f}")

    hand = mujoco.MjSpec()
    hand.modelname = "inspire_rh56dfq_hands"
    hand.compiler.degree = False
    hand.meshdir = "meshes"
    for side in "LR":
        frame = hand.worldbody.add_frame()
        frame.attach_body(spec.body(f"{side}_hand_base_link"), "", "")
    used = {g.meshname for g in hand.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for m in list(hand.meshes):
        if m.name not in used:
            hand.delete(m)
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    for m in hand.meshes:
        src = Path(m.file) if Path(m.file).is_absolute() else g1_description / "meshes" / Path(m.file).name
        mesh = trimesh.load(src, force="mesh")
        if len(mesh.faces) > faces:
            v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=1.0 - faces / len(mesh.faces))
            mesh = trimesh.Trimesh(v, f)
        dst = OUT / "meshes" / f"{m.name}.stl"
        mesh.export(dst)
        m.file = str(dst.resolve())
    # Attached meshes keep resolving against the URDF's meshdir, so write absolute paths and relativise them.
    xml = hand.to_xml().replace(f"{(OUT / 'meshes').resolve()}/", "")
    xml = re.sub(r"\n  <default>(\s*<default/>)*\s*</default>\n", "\n", xml)
    (OUT / "inspire_hands.xml").write_text(xml)
    out_model = mujoco.MjModel.from_xml_path(str(OUT / "inspire_hands.xml"))
    print(f"wrote {OUT}: {out_model.nbody - 1} bodies, {out_model.njnt} joints, {out_model.neq} equalities, "
          f"{out_model.ngeom} geoms, {out_model.nmesh} meshes")


if __name__ == "__main__":
    tyro.cli(main)
