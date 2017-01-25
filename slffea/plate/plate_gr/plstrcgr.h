/*
    This file contains the structures of the graphics program
    for plate elements.

	Updated 9/22/06

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
#include "../plate/plconst.h"

typedef struct {
	int xx,yy,xy,I,II;
} IMDIM;

typedef struct {
	int xx,yy,xy,zx,yz,I,II,III;
} ISDIM;

typedef struct {
	IMDIM pt[num_int];
} IMOMENT;

typedef struct {
	ISDIM pt[num_int];
} ISTRESS;

typedef struct {
	IMDIM pt[num_int];
} ICURVATURE;

typedef struct {
	ISDIM pt[num_int];
} ISTRAIN;

/* The structure below is a repeat of XYZPhiF found in ../plate/plstruct.h.
   I cannot simply include plstruct.h in here because plstruct.h is
   already included in other modules which plstrcgr.h is included in
   and this causes a redundancy which is not allowed. */

typedef struct {
	double x, y, z;
} XYZPhiF_GR;

typedef struct {
	double x, y, z;
} XYZF_GR;

typedef struct {
	XYZF_GR face[2];
} NORM;

