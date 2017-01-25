/*
    This file contains the structures of the graphics program
    for shell elements.

	Updated 9/22/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../shell/shconst.h"

typedef struct {
	int xx,yy,xy,zx,yz,I,II,III;
} ISDIM;

typedef struct {
	ISDIM pt[num_int8];
} ISTRESS;

typedef struct {
	ISDIM pt[num_int8];
} ISTRAIN;

/* The structure below is a repeat of XYZF found in ../shell/shstruct.h.
   I cannot simply include shstruct.h in here because shstruct.h is
   already included in other modules which shstrcgr.h is included in
   and this causes a redundancy which is not allowed. */

typedef struct {
	double x, y, z, phix, phiy;
} XYZPhiF_GR;

typedef struct {
	double x, y, z;
} XYZF_GR;

typedef struct {
	XYZF_GR face[12];
} NORM;
