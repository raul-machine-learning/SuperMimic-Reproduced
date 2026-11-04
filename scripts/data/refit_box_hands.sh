#!/bin/bash
# Box hands: fit the taxel blocks and the distal thumb box's fixed roll angle, rebuild the hands at that angle and
# their clips, and repeat until the angle settles (the fit holds the contact forces of the built hands fixed).
#   bash scripts/data/refit_box_hands.sh [rounds] [start angle, deg]
set -e
cd "$(dirname "$0")/../.."
export SM_HAND_SHAPE=box SM_ROBOT=capsule_hands_props
R="env -u PYTHONPATH uv run python"
rebuild() {
  $R scripts/assets/build_capsule_hands.py "$@" | grep "thumb roll"
  $R scripts/data/register_hands_to_props.py | grep shift
  $R scripts/data/close_capsule_fingers.py --clip data/hands_box/sweep_task023_hands_registered.npz | tail -1
  SM_ROBOT=capsule_hands $R scripts/data/make_rig_clip.py 2>&1 | grep wrote
}
[ -n "$2" ] && rebuild --thumb-roll-deg "$2"
for i in $(seq 1 "${1:-4}"); do
  $R scripts/data/fit_glove_blocks.py 2>&1 | grep -E "held_out|thumb_roll"
  moved=$($R -c "
import mujoco, numpy as np
from supermimic.robots import HANDS_XML
from supermimic.data.glove_taxels import built_thumb_roll, fitted_thumb_roll
print(abs(np.degrees(fitted_thumb_roll() - built_thumb_roll(mujoco.MjModel.from_xml_path(str(HANDS_XML))))) > 2.0)")
  rebuild
  [ "$moved" = "False" ] && { echo "round $i: settled"; break; }
  echo "round $i: angle moved, refitting"
done
