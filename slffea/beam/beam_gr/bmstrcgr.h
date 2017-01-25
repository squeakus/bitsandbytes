/*
    This file contains the structures of the graphics program
    for beam elements.

	Updated 1/12/05

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
#include "../beam/bmconst.h"

typedef struct {
	int xx,yy,zz;
} IMDIM;

typedef struct {
	int xx, xy, zx;
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

/* The structure below is a repeat of XYZPhiF found in ../beam/bmstruct.h.
   I cannot simply include bmstruct.h in here because bmstruct.h is
   already included in other modules which bmstrcgr.h is included in
   and this causes a redundancy which is not allowed. */

typedef struct {
	double x, y, z, phix, phiy, phiz;
} XYZPhiF_GR;

typedef struct {
	double x, y, z;
} XYZF_GR;

