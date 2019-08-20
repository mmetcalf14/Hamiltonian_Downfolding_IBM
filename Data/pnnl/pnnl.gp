set terminal pngcairo font "FreeSans" size 1024,768
set output "pnnl.png"

set format x "%.1f"
set format y "%.2f"
set grid
set style data linespoints

set key at graph 0.98,0.21 Left reverse samplen 1.5 spacing 1.25 height 1 width 1 box

set xrange [1.640:13.865]
set yrange [-14.90:-14.84]

set title "Li_2 potential energy curve"
set xlabel "Internuclear separation (Å)"
set ylabel "Energy (Ha)"

plot "cc-pvtz-vqe.dat" t "cc-pVTZ VQE",\
     "cc-pvtz-ccsd.dat" t "cc-pVTZ CSSD",\
     "downfolded-vqe.dat" t "Downfolded VQE",\
     "downfolded-ccsd.dat" t "Downfolded CCSD"
