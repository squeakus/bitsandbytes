/*
    This file contains the structures of the graphics program
    for truss elements.

	Updated 1/22/06

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
#include "../truss/tsconst.h"

typedef struct {
	int xx;
} ISDIM;

/* The structure below is a repeat of XYZPhiF found in ../truss/tsstruct.h.
   I cannot simply include tsstruct.h in here because tsstruct.h is
   already included in other modules which tsstrcgr.h is included in
   and this causes a redundancy which is not allowed. */

typedef struct {
	double x, y, z;
} XYZF_GR;

