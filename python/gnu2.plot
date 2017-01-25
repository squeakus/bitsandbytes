set terminal postscript color
set title "population size over time" 
set xlabel "generation"
set ylabel "population size"
set output "graph.ps"
set yrange [0:1]
set key bottom right
set key box

plot "rate3.dat" with points pointsize 0.5 smooth csplines
#"rate1.dat" with points pointsize 0.5 smooth csplines,\
#"rate2.dat" with points pointsize 0.5 smooth csplines,\
#"rate3.dat" with points pointsize 0.5 smooth csplines,\
#"rate4.dat" with points pointsize 0.5 smooth csplines
 
