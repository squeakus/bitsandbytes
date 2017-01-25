/*
    This file contains the structures of the graphics program
    for wedge elements.

	Updated 3/19/01

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
#include "../wedge/weconst.h"

typedef struct {
	int xx,yy,zz,xy,zx,yz,I,II,III;
} ISDIM;

typedef struct {
	ISDIM pt[num_int];
} ISTRESS;

typedef struct {
	ISDIM pt[num_int];
} ISTRAIN;

/* The structure below is a repeat of XYZF found in ../wedge/westruct.h.
   I cannot simply include brstruct.h in here because brstruct.h is
   already included in other modules which westrcgr.h is included in
   and this causes a redundancy which is not allowed. */

typedef struct {
	double x, y, z;
} XYZF_GR;

typedef struct {
	XYZF_GR face[8];
} NORM;
