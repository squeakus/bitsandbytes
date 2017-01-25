/*
    This file contains the structures of the triangle FEM code.

	Updated 9/20/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "trconst.h"

typedef struct {
	double xx,yy,xy,I,II;
} SDIM;

typedef struct {
	double x, y, z;
} XYZF;

typedef struct {
	int x, y, z;
} XYZI;

typedef struct {
	XYZI *num_fix;
	int *num_force;
	XYZI *fix;
	int *force;
} BOUND;

typedef struct {
	double E;
	double nu;
	double rho;
	double thick;
	int extrathick;
} MATL;

