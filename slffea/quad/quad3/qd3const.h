/*
    This is the include file "qdconst.h" for the finite element progam 
    which uses quadrialteral elements.

                Updated 11/2/00

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

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

#define nsd           2                      /* spatial dimensions per node */
#define ndof          2                      /* degrees of freedom per node */
#define npel          4                      /* nodes per element */
#define neqel         npel*ndof              /* degrees of freedom per element */
#define num_int       4                      /* number of integration points */
#define nsdsq         nsd*nsd                /* nsd squared */
#define neqlsq        neqel*neqel            /* neqel squared */
#define sdim          3                      /* stress dimensions per element */
#define soB           sdim*neqel             /* size of B matrix */
#define MsoB          nsd*neqel              /* size of B_mass matrix */
#define sosh          (nsd +1)*npel*num_int  /* size of shl and shg matrix */
#define sosh_node2    nsd*2*num_int          /* size of shl_node2 */
#define KB            1024.0                 /* number of bytes in kilobyte */
#define MB            1.0486e+06             /* number of bytes in megabyte */

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

/*    Below is the additional list of constants for the brick element br2
      which has thermal properties.
*/


#define edof          1                      /* degrees of freedom per edge */
#define epel          4                      /* edges per element */
#define nped          2                      /* nodes per edge */
#define EMneqel       epel*edof              /* degrees of freedom per element for electromagnetics */
#define EMneqlsq      EMneqel*EMneqel        /* EMeqel squared */
#define EMsoB         nsd*npel                 /* size of B matrix for electromagnetics */

/*
  title               problem title
  numed               number of edges
  numel               number of elements
  numnp               number of nodal points 
  nmat                number of materials
  dof                 total number of degrees of freedom 
  EMdof                total number of degrees of freedom for electromagnetic element
  coord               *(coord + 0) x coordinate of node
                      *(coord + 1) y coordinate of node
  connect             (0-3) connectivity array 
  matl                material structure
  EMod                young's modulus
  nu                  poisson's ratio
  force               *(force + 0) x component of applied load
                      *(force + 1) y component of applied load
  plane_stress_flag   0 flag indicating whether plane stress
                      1 or strain theory is used 
  analysis_flag       1 calculate unknown displacemnts
                      2 calculate reaction forces
  LU_decomp_flag      0 if numel <= 1125 elements, use LU Decomposition for
                        displacements
                      1 if numel > 1125 elements, use conjugate
                        gradient method for displacements
*/
