/*
    This file contains the structures of the graphics program
    for tetrahedral elements.

                  Last Update 8/16/06

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
#include "../tetra/teconst.h"

typedef struct {
	int xx,yy,zz,xy,zx,yz,I,II,III;
} ISDIM;

/* The structure below is a repeat of XYZF found in ../tetra/testruct.h.
   I cannot simply include brstruct.h in here because brstruct.h is
   already included in other modules which testrcgr.h is included in
   and this causes a redundancy which is not allowed. */

typedef struct {
	double x, y, z;
} XYZF_GR;

typedef struct {
	XYZF_GR face[4];
} NORM;
