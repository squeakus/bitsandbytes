/*  This program creates a file called "gplot" which has all the
    specifications and parameters needed to view the data generated
    by my non-linear codes.  The data is to see whether convergence
    has been reached.  You first run "gnuplot" then type in:

       load "gplot"

    at the prompt(including the quotes).

                     Updated 12/1/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 2008  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gnuplot2.h"

#define SMALL      1.0e-20
#define BIG        1.0e20

int GnuplotWriter2( char *gplot_dat_name, FORCE_DAT max, double max_x )
{
	int i, j, k;
	FILE *gplot;

	double max_y = -BIG, min_y = 0.0;

	gplot = fopen( "gplot","w" );

/* Write out the gnuplot file. */

	if( max_y < max.force ) max_y = max.force;
	if( max_y < max.P_global ) max_y = max.P_global;
	if( max_y < max.P_global2 ) max_y = max.P_global2;
	if( max_y < max.P_global_CG ) max_y = max.P_global_CG;
	if( max_y < max.P_global_CG2 ) max_y = max.P_global_CG2;
	if( min_y > max.force ) min_y = max.force;
	if( min_y > max.P_global ) min_y = max.P_global;
	if( min_y > max.P_global2 ) min_y = max.P_global2;
	if( min_y > max.P_global_CG ) min_y = max.P_global_CG;
	if( min_y > max.P_global_CG2 ) min_y = max.P_global_CG2;
	fprintf( gplot, "#set term postscript\n");
	fprintf( gplot, "#set out 'test.ps'\n");
	fprintf( gplot, "#set term png\n");
	fprintf( gplot, "#set out 'test.png'\n");
	fprintf( gplot, "set nokey\n");
	fprintf( gplot, "set yrange [%12.6e:%12.6e]\n", 1.1*min_y, 1.1*max_y);
	fprintf( gplot, "set xrange [%12.6e:%12.6e]\n", 0.0, max_x);
	fprintf( gplot, "plot  \"%s\" using  2: 3 with points  1, \\\n", gplot_dat_name);
	fprintf( gplot, "      \"%s\" using  2: 4 with points  2, \\\n", gplot_dat_name);
	fprintf( gplot, "      \"%s\" using  2: 5 with points  0, \\\n", gplot_dat_name);
	fprintf( gplot, "      \"%s\" using  2: 6 with points  0\n", gplot_dat_name);
	fprintf( gplot, "#set out\n");

	fclose(gplot);

	return 1;
}


