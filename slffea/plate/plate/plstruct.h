/*
    This file contains the structures of the plate FEM code.

	Updated 9/22/06

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
#include "plconst.h"

typedef struct {
	double xx,yy,xy,I,II;
} MDIM;

typedef struct {
	double xx,yy,xy,zx,yz,I,II,III;
} SDIM;

typedef struct {
	double x, y, z, phix, phiy, phiz;
} XYZPhiF;

typedef struct {
	int x, y, z, phix, phiy, phiz;
} XYZPhiI;

typedef struct {
	XYZPhiI *num_fix;
	int   *num_force;
	XYZPhiI *fix;
	int   *force;
} BOUND;

typedef struct {
	double E;
	double nu;
	double rho;
	double thick;
	double shear;
} MATL;

typedef struct {
	double *bend;
	double *shear;
} SH;

typedef struct {
	MDIM pt[num_int];
} MOMENT;

typedef struct {
	SDIM pt[num_int];
} STRESS;

typedef struct {
	MDIM pt[num_int];
} CURVATURE;

typedef struct {
	SDIM pt[num_int];
} STRAIN;
