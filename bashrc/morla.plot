#Y35W3dd
#set term postscript
#
set term png
set yrange [0:2500]
set title "Morla's weight"
set xlabel "Date"
set ylabel "Weight (grams)"
set output "graph.png"
unset key

set xdata time
set timefmt "%Y-%m-%d"
plot "morlaFmt.dat" using 1:2,\
     "morlaFmt.dat" using 1:2 smooth bezier
#     "morlaFmt.dat" using 1:2 smooth csplines,\
