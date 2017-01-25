/*
    This file contains the structures of the thermal brick FEM code.

	Updated 4/6/00

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../brick/brconst.h"

typedef struct {
	double xy,yx,zx,xz,yz,zy;
} ORTHO;

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
	int *num_force;
	int *num_heat_el;
	int *num_heat_node;
	int *num_Q;
	int *num_T;
	int *num_TB;
	int *num_TS;
	XYZI *fix;
	int *force;
	int *heat_el;
	int *heat_node;
	int *Q;
	int *T;
	int *TB;
	int *TS;
} BOUND;

typedef struct {
	double film;
	XYZF E;
	ORTHO nu;
	XYZF thrml_cond;
	XYZF thrml_expn;
} MATL;

typedef struct {
	SDIM pt[num_int];
} STRESS;

typedef struct {
	SDIM pt[num_int];
} STRAIN;
