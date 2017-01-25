set terminal postscript color
set title "population size over time" 
set xlabel "generation"
set ylabel "population size"
set output "graph.ps"
set key 0.01,100

plot "results.dat" with points pointsize 0.5 smooth csplines 
