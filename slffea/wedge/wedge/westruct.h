/*
    This file contains the structures of the wedge FEM code.

	Updated 5/22/00

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
#include "weconst.h"

typedef struct {
	double xx,yy,zz,xy,zx,yz,I,II,III;
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
	double nu;
	double rho;
} MATL;

typedef struct {
	SDIM pt[num_int];
} STRESS;

typedef struct {
	SDIM pt[num_int];
} STRAIN;
