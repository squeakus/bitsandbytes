# Gnuplot script file for plotting data in file "ave128v2.dat"
# This file is called gnuPlotRun128v2Test.p
unset log
set palette rgbformulae 30,31,32
plot "graph.txt" w l pal
set term png
set output "graph.png"
replot
