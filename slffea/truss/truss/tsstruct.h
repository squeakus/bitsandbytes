/*
    This file contains the structures of the linear truss
    FEM code.
		Udated 5/18/99

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tsconst.h"

typedef struct {
	double xx;
} SDIM;

typedef struct {
	double x, y, z;
} XYZF;

typedef struct {
	int x, y, z;
} XYZI;

typedef struct {
	XYZI *num_fix;
	int  *num_force;
	XYZI *fix;
	int  *force;
} BOUND;

typedef struct {
	double E;
	double rho;
	double area;
} MATL;

