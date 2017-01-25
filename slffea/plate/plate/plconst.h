/*
    This is the include file "plconst.h" for the finite element progam 
    which uses plate elements.

                Updated 10/23/06

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

#define nsd             3                          /* spatial dimensions per node */
#define nsdsq           9                          /* nsd squared */
#define nsd2            2                          /* spatial dimensions per node in local coordinates*/
#define ndof            5                          /* degrees of freedom per node */
#define ndof2           2                          /* membrane degrees of freedom per node in local coordinates*/
#define ndof3           3                          /* bending degrees of freedom per node in local coordinates*/
#define ndof5           5                          /* degrees of freedom per node */
#define ndof6           6                          /* pseudo degrees of freedom per node in global coordinates*/
#define npel            4                          /* nodes per plate element */
#define npel4           4                          /* nodes per plate element */
#define npel3           3                          /* nodes per triangle element */
#define neqel           npel*ndof                  /* degrees of freedom per quad element */
#define neqel8          npel4*ndof2                /* membrane degrees of freedom per quad element in local coordinates*/
#define neqel12         npel4*ndof3                /* bending degrees of freedom per quad element in local coordinates*/
#define neqel20         npel4*ndof5                /* degrees of freedom per quad element */
#define neqel24         npel4*ndof6                /* degrees of freedom per quad element in global coordinates*/
#define neqel6          npel3*ndof2                /* membrane degrees of freedom per triangle element in local coordinates*/
#define neqel9          npel3*ndof3                /* bending degrees of freedom per triangle element in local coordinates*/
#define neqel15         npel3*ndof5                /* degrees of freedom per triangle element */
#define neqel18         npel3*ndof6                /* degrees of freedom per triangle element in global coordinates*/
#define num_int         4                          /* number of integration points for 4 point integration */
#define num_int4        4                          /* number of integration points for 4 point integration */
#define num_int3        3                          /* number of integration points for 3 point integration */
#define num_int1        1                          /* number of integration points for 1 point integration */
#define neqlsq          neqel*neqel                /* neqel squared */
#define neqlsq64        neqel8*neqel8              /* neqel8 squared for plates*/
#define neqlsq144       neqel12*neqel12            /* neqel12 squared for plates*/
#define neqlsq576       neqel24*neqel24            /* neqel24 squared for plates*/
#define neqlsq36        neqel6*neqel6              /* neqel6 squared for triangles */
#define neqlsq81        neqel9*neqel9              /* neqel9 squared for triangles */
#define neqlsq324       neqel18*neqel18            /* neqel24 squared for triangles */
#define lenb            3*neqel*num_int            /*  */
#define sdim3           3                          /* membrane stress dimensions per element */
#define sdim5           5                          /* bending stress dimensions per element */
#define soB             sdim5*neqel12              /* size of B matrix for plate */
#define soBmem          sdim3*neqel8               /* size of membrane B matrix for plate */
#define MsoB            (nsd2 + 1)*neqel12         /* size of B_mass matrix for plate */
#define MsoBmem         (nsd2 + 1)*neqel8          /* size of membrane B_mass matrix for plate */
#define soBtr           sdim5*neqel9               /* size of B matrix for triangle */
#define soBmemtr        sdim3*neqel6               /* size of membrane B matrix for triangle */
#define MsoBtr          (nsd2 + 1)*neqel9          /* size of B_mass matrix for triangle */
#define MsoBmemtr       (nsd2 + 1)*neqel6          /* size of membrane B_mass matrix for triangle */
#define soshb           (nsd2 + 1)*npel*(num_int)  /* size of shl and shg matrix
                                                      for 2X2 PT. Gauss for bending */
#define soshs           (nsd2 +1)*npel*(1)         /* size of shl and shg matrix
                                                      for 1X1 PT. Gauss for shear */
#define sosh_node2      nsd*2*num_int              /* size of shl_node2 */
#define soshtr          (nsd2+1)*npel3*num_int3    /* size of shl and shg matrix */
#define soshtr_node2    nsd2*2*num_int3            /* size of shl_node2 */
#define KB              1024.0                     /* number of bytes in kilobyte */
#define MB              1.0486e+06                 /* number of bytes in megabyte */

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
  thick               thickness of plate
  shear               shear correction factor
  force               *(force + 0) z component of applied load
                      *(force + 1) phi1 component of applied load
                      *(force + 2) phi2 component of applied load
  analysis_flag       1 calculate unknown displacemnts
                      2 calculate reaction forces
  LU_decomp_flag      0 if numel <= 750 elements, use LU Decomposition for
                        displacements
                      1 if numel > 750 elements, use conjugate
                        gradient method for displacements
*/

