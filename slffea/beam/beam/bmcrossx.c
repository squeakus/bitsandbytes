/*
    This utility function normalizes vectors as well as 
    calculates normalized crossproducts.  It is based on the
    subroutines normalize and normcrossprod given in the book
    "Open GL Programming Guide" by Jackie Neider, Tom Davis, and
    Mason Woo, page 58. 

    I am modifying it for calculation of the beam element local axis
    vectors.

                Updated 9/10/99 

    SLFFEA source file
    Version:  1.5

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SMALL               1.0e-20

int bmnormcrossX(double v1[3], double v2[3], double out[3])
{
	int i, j, check;
	double d;

	out[0] = v1[1]*v2[2] - v1[2]*v2[1];
	out[1] = v1[2]*v2[0] - v1[0]*v2[2];
	out[2] = v1[0]*v2[1] - v1[1]*v2[0];

/* normalize out */

	d = sqrt(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]);
	if (d < SMALL)
	{
		return 0;
	}
	out[0] /= d; out[1] /= d; out[2] /= d;

	return 1;
}

