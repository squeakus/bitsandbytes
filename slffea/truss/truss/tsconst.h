/*
    This is the include file "tsconst.h" for the finite element progam 
    which uses 3D truss elements.

                Updated 6/27/00

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

#define pt5           .5
#define pt25          .25
#define pt1667        .166666666667
#define sq3          1.732050808
#define sq3pt33      1.825741858
#define sqpt3        0.547722557
#define ONE          0.99999

#define nsd       3                     /* number of spatial dimensions per node */
#define ndof      3                     /* degrees of freedom per node */
#define npel      2                     /* nodes per element */
#define neqel     npel*ndof             /* degrees of freedom per element */
#define num_int   1                     /* number of integration points */
#define nsdsq          nsd*nsd            /* nsd squared */
#define neqlsq    neqel*neqel           /* neqel squared */
#define npelsq    npel*npel             /* npel squared */
#define sdim      1                     /* stress dimensions per element */
#define soB       sdim*npel             /* size of B matrix */
#define KB            1024.0                      /* number of bytes in kilobyte */
#define MB            1.0486e+06                  /* number of bytes in megabyte */

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

/*
  title               problem title
  numel               number of elements
  numnp               number of nodal points 
  nmat                number of materials
  dof                 total number of degrees of freedom 
  coord               *(coord + 0) x coordinate of node
                      *(coord + 1) y coordinate of node
                      *(coord + 2) z coordinate of node
  connect             (0-1) connectivity array 
  matl                material structure
  Emod                young's modulus
  area                Area
  force               *(force + 0) x component of applied load
                      *(force + 1) y component of applied load
                      *(force + 2) z component of applied load
  analysis_flag       1 calculate unknown displacemnts
                      2 calculate reaction forces
  LU_decomp_flag      0 if numel <= 750 elements, use LU Decomposition for
                        displacements
                      1 if numel > 750 elements, use conjugate
                        gradient method for displacements
*/

