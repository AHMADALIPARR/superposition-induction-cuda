#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright Ahmad Ali Parr
# Regenerates docs/img/*.png. Needs graphviz, gnuplot and ImageMagick.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out="$here/.."
dot -Tpng -Gdpi=130 "$here/sid_flow.dot" -o "$out/sid_flow.png"
dot -Tpng -Gdpi=130 "$here/amm_flow.dot" -o "$out/amm_flow.png"
gnuplot -e "outfile='$out/smooth_leaky.png'" "$here/smooth_leaky.gp"
gnuplot -e "outfile='$out/constrained_softmax.png'" "$here/csoftmax.gp"
for f in sid_flow amm_flow smooth_leaky constrained_softmax; do
  magick "$out/$f.png" -trim +repage -bordercolor white -border 16 -strip "$out/$f.png"
done
echo "figures written to $out"
