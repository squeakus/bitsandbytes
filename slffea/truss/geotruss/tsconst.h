/*
    This is the include file "tsconst.h" for the finite element progam 
    which uses 3D truss elements.

                Updated 12/28/98

    SLFFEA source file
    Version:  1.0
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define nsd       3                     /* number of spatial dimentions per node */
#define ndof      3                     /* degrees of freedom per node */
#define npel      2                     /* nodes per element */
#define neqel     npel*ndof             /* degrees of freedom per element */
#define num_int   1                     /* number of integration points */
#define neqlsq    neqel*neqel           /* neqel squared */
#define npelsq    npel*npel             /* npel squared */
#define sdim      1                     /* stress dimentions per element */
#define soB       sdim*npel             /* size of B matrix */

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

/*
  title   problem title
  numel   number of elements
  numnp   number of nodal points 
  nmat    number of materials
  dof     total number of degrees of freedom 
  coord   *(coord + 0) x coordinate of node
          *(coord + 1) y coordinate of node
          *(coord + 2) z coordinate of node
  connect (0-1) connectivity array 
  matl    material structure
  Emod    young's modulus
  area    Area
  force   *(force + 0) x component of applied load
          *(force + 1) y component of applied load
          *(force + 2) z component of applied load
  analysis_flag  1 calculate unknown displacemnts
                 2 calculate reaction forces
*/

