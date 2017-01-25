/*
    This file contains the structures of the linear beam
    FEM code.
		Udated 1/7/05

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
	double dx0, dx1, dx2, dx3;
} dNdx;

typedef struct {
	dNdx Nhat[2];
	dNdx N[4];
} SHAPE;

