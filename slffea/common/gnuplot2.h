/*  This file contains the structure for storing force, P_global,
    and P_global_CG at certain DOFs.  I use these quantities in
    plots of P_global or P_global_CG vs. time to see whether convergence
    has been reached in the non-linear codes.  The data generated
    is formatted for plotting with "gnuplot".

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

typedef struct {
        double force, force2;
        double P_global, P_global2;
        double P_global_CG, P_global_CG2;
        int dof_force, dof_force2;
} FORCE_DAT;

