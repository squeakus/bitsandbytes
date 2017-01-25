/*
    This program calculates the shape functions for a beam
    element.

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/


#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "bmshape_struct.h"


int bmshape(SHAPE *sh, double s, double L, double Lsq)
{
/* Maps from L2 space to coordinate space and
   calculates the shape functions 

            Updated 1/7/05
*/

	s = L*(1. + s)/2.;

	sh->Nhat[0].dx0 = 1.0 - s/L;
	sh->Nhat[1].dx0 = s/L;
	sh->Nhat[0].dx1 = - 1.0/L;
	sh->Nhat[1].dx1 = 1.0/L;

	sh->N[0].dx0 = 1.0 - 3.0*s*s/(Lsq) + 2.0*s*s*s/(L*Lsq);
	sh->N[1].dx0 = s - 2.0*s*s/L + s*s*s/Lsq;
	sh->N[2].dx0 = 3.0*s*s/(Lsq) - 2.0*s*s*s/(L*Lsq);
	sh->N[3].dx0 =  - s*s/L + s*s*s/Lsq;

	sh->N[0].dx1 = 6.0*s*( - 1.0 + s/L)/(Lsq);
	sh->N[1].dx1 = 1.0 - 4*s/L + 3.0*s*s/Lsq;
	sh->N[2].dx1 = 6.0*s*(1.0 - s/L)/(Lsq);
	sh->N[3].dx1 = s*( - 2.0 + 3.0*s/L)/L;

	sh->N[0].dx2 = 6.0*( - 1.0 + 2.0*s/L)/(Lsq);
	sh->N[1].dx2 = 2.0*( - 2.0 + 3.0*s/L)/L;
	sh->N[2].dx2 = 6.0*(1.0 - 2.0*s/L)/(Lsq);
	sh->N[3].dx2 = 2.0*( - 1.0 + 3.0*s/L)/L;

	sh->N[0].dx3 = 12.0/(L*Lsq);
	sh->N[1].dx3 = 6.0/Lsq;
	sh->N[2].dx3 = -12.0/(L*Lsq);
	sh->N[3].dx3 = 6.0/Lsq;

	return 1;
}

int bmshape_mass(SHAPE *sh, double s, double L, double Lsq)
{
/* Maps from L2 space to coordinate space and
   calculates the shape functions for mass matrices 

                Updated 9/11/00
*/

	s = L*(1. + s)/2.;

	sh->Nhat[0].dx0 = 1.0 - s/L;
	sh->Nhat[1].dx0 = s/L;

	sh->N[0].dx0 = 1.0 - 3.0*s*s/(Lsq) + 2.0*s*s*s/(L*Lsq);
	sh->N[1].dx0 = s - 2.0*s*s/L + s*s*s/Lsq;
	sh->N[2].dx0 = 3.0*s*s/(Lsq) - 2.0*s*s*s/(L*Lsq);
	sh->N[3].dx0 =  - s*s/L + s*s*s/Lsq;

	sh->N[0].dx1 = 6.0*s*( - 1.0 + s/L)/(Lsq);
	sh->N[1].dx1 = 1.0 - 4*s/L + 3.0*s*s/Lsq;
	sh->N[2].dx1 = 6.0*s*(1.0 - s/L)/(Lsq);
	sh->N[3].dx1 = s*( - 2.0 + 3.0*s/L)/L;

	return 1;
}

