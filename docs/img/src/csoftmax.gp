# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright Ahmad Ali Parr
# Entropy of softmax(l / tau) against tau for one router row (K = 4); the
# bisection picks the tau where it meets the bound 0.2.
set terminal pngcairo size 640,400 font "Helvetica,11"
set output outfile
l1 = 0.9; l2 = 0.3; l3 = -0.2; l4 = -0.6
e(t,l) = exp((l - l1)/t)
Z(t) = e(t,l1) + e(t,l2) + e(t,l3) + e(t,l4)
p(t,l) = e(t,l)/Z(t)
h1(t,l) = -p(t,l)*log(p(t,l) + 1e-10)
H(t) = h1(t,l1) + h1(t,l2) + h1(t,l3) + h1(t,l4)
set samples 400
set grid lc rgb "#dddddd"
set xrange [0.01:1]
set xlabel "temperature tau"
set ylabel "entropy H(softmax(l / tau))"
set key top left
plot H(x) lw 2 lc rgb "#4a6fa5" title "logits (0.9, 0.3, -0.2, -0.6)", 0.2 dt 2 lw 2 lc rgb "#b04848" title "bound 0.2"
