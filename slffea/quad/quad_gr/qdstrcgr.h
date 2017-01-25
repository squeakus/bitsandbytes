/*
    This file contains the structures of the graphics program
    for quad elements.

	Updated 8/16/06

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
#include "../quad/qdconst.h"

typedef struct {
	int xx,yy,xy,I,II;
} ISDIM;

typedef struct {
	ISDIM pt[num_int];
} ISTRESS;

typedef struct {
	ISDIM pt[num_int];
} ISTRAIN;

/* The structure below is a repeat of XYZF found in ../quad/qdstruct.h.
   I cannot simply include qdstruct.h in here because qdstruct.h is
   already included in other modules which qdstrcgr.h is included in
   and this causes a redundancy which is not allowed. */

typedef struct {
	double x, y, z;
} XYZF_GR;

typedef struct {
	XYZF_GR face[2];
} NORM;


