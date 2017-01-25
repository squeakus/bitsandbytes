set terminal postscript color

set output "front.ps"
set title " Evolved Beam Structures" 
set xlabel "Average Stress (kN)"
set ylabel "Beams"

plot "gen000.dat" with points pointsize 0.5 title 'graph',\
    "gen001.dat" with points pointsize 0.5,\
    "gen002.dat" with points pointsize 0.5,\
    "gen003.dat" with points pointsize 0.5,\
    "gen004.dat" with points pointsize 0.5 title 'moo',\
    "gen005.dat" with points pointsize 0.5,\
    "gen006.dat" with points pointsize 0.5,\
    "gen007.dat" with points pointsize 0.5,\
    "gen008.dat" with points pointsize 0.5,\
    "gen009.dat" with points pointsize 0.5,\
    "gen010.dat" with points pointsize 0.5,\
    "gen011.dat" with points pointsize 0.5,\
    "gen012.dat" with points pointsize 0.5,\
    "gen013.dat" with points pointsize 0.5,\
    "gen014.dat" with points pointsize 0.5
