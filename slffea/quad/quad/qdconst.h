/*
    This is the include file "qdconst.h" for the finite element progam 
    which uses quadrialteral elements.

                Updated 9/16/06

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

#define pt5     .5 
#define pt25    .25
#define pt1667  .166666666667
#define sq3    1.732050808 

#define nsd           3                       /* spatial dimensions per node */
#define nsdsq         9                       /* nsd squared */
#define nsd2          2                       /* spatial dimensions per node in local coordinates*/
#define ndof          3                       /* degrees of freedom per node */
#define ndof2         2                       /* degrees of freedom per node in local coordinates*/
#define npel          4                       /* nodes per element */
#define npel4         4                       /* nodes per element */
#define neqel         npel*ndof               /* degrees of freedom per element */
#define neqel8        npel*ndof2              /* degrees of freedom per element in local coordinates*/
#define num_int       4                       /* number of integration points */
#define neqlsq        neqel*neqel             /* neqel squared */
#define neqlsq64      neqel8*neqel8           /* neqel8 squared = 64*/
#define sdim          3                       /* stress dimensions per element */
#define soB           sdim*neqel8             /* size of B matrix */
#define MsoB          nsd2*neqel8             /* size of B_mass matrix */
#define sosh          (nsd2 +1)*npel*num_int  /* size of shl and shg matrix */
#define sosh_node2    nsd2*2*num_int          /* size of shl_node2 */
#define KB            1024.0                  /* number of bytes in kilobyte */
#define MB            1.0486e+06              /* number of bytes in megabyte */

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
  connect             (0-3) connectivity array 
  matl                material structure
  Emod                young's modulus
  nu                  poisson's ratio
  force               *(force + 0) x component of applied load
                      *(force + 1) y component of applied load
  plane_stress_flag   0 flag indicating whether plane stress
                      1 or strain theory is used 
  analysis_flag       1 calculate unknown displacemnts
                      2 calculate reaction forces
  LU_decomp_flag      0 if numel <= 750 elements, use LU Decomposition for
                        displacements
                      1 if numel > 750 elements, use conjugate
                        gradient method for displacements
*/
