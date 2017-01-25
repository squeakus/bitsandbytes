/*
    This file contains the structures of the quad FEM code.
    It is for eletromagnetism.

	Updated 11/6/00

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "qd3const.h"

typedef struct {
        double xx,yy,xy,I,II;
} SDIM;

typedef struct {
        double x, y;
} XYF;

typedef struct {
        int x, y;
} XYI;

typedef struct {
	XYI *num_fix;
	int *num_edge;
	int *num_force;
	XYI *fix;
	int *edge;
	int *force;
} BOUND;

typedef struct {
        double eta;
        double nu;
        double ko;
} MATL;

typedef struct {
        SDIM pt[num_int];
} STRESS;

typedef struct {
        SDIM pt[num_int];
} STRAIN;
