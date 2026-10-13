# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generates gel_pad.stl: a flat gel surface in the z = 0 plane with +z normals.

The mesh is a single-layer grid so that every vertex sits on the rendered gel
surface, which is the convention used by MuJoCo tactile sensors. STL is used
instead of OBJ because MuJoCo does not rotate STL meshes into a canonical
orientation.
"""

from __future__ import annotations

import struct
from pathlib import Path

HALF_X = 0.010
HALF_Y = 0.008
NX = 25  # vertices along x
NY = 21  # vertices along y


def main():
  verts = []
  for j in range(NY):
    y = -HALF_Y + 2.0 * HALF_Y * j / (NY - 1)
    for i in range(NX):
      x = -HALF_X + 2.0 * HALF_X * i / (NX - 1)
      verts.append((x, y, 0.0))

  def vid(i, j):
    return j * NX + i

  tris = []
  for j in range(NY - 1):
    for i in range(NX - 1):
      tris.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
      tris.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))

  out = Path(__file__).parent / "gel_pad.stl"
  with open(out, "wb") as f:
    f.write(b"\0" * 80)
    f.write(struct.pack("<I", len(tris)))
    for tri in tris:
      # zero face normal: MuJoCo recomputes normals from the winding
      f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
      for idx in tri:
        f.write(struct.pack("<3f", *verts[idx]))
      f.write(struct.pack("<H", 0))
  print(f"wrote {out} ({NX}x{NY} vertices, {len(tris)} triangles)")


if __name__ == "__main__":
  main()
