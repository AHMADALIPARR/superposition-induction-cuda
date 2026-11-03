# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright Ahmad Ali Parr
# SmoothLeaky f(x) = a x + (1-a)/b (softplus(b x) - ln 2) and f'(x), a = 0.01, b = 1.
set terminal pngcairo size 900,420 font "Helvetica,11"
set output outfile
set multiplot layout 1,2
a = 0.01; b = 1.0
sp(t) = t > 20 ? t : log(1 + exp(t))
f(x) = a*x + (1-a)/b*(sp(b*x) - log(2))
g(x) = a + (1-a)/(1 + exp(-b*x))
set grid lc rgb "#dddddd"
set key top left
set xrange [-6:6]
set title "SmoothLeaky"
plot f(x) lw 2 lc rgb "#4a6fa5" title "f(x)", (x > 0 ? x : 0) dt 2 lc rgb "#999999" title "ReLU"
set title "derivative, inside (a, 1)"
set yrange [-0.05:1.05]
plot g(x) lw 2 lc rgb "#3d8b4f" title "f'(x)", a dt 3 lc rgb "#b04848" title "a = 0.01"
unset multiplot
