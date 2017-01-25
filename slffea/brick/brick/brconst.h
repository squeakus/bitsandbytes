/*
    This is the include file "brconst.h" for the finite element progam 
    which uses brick elements.  It is set up for all the different
    brick elements including thermal. 

                Updated 9/31/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

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

#define nsd           3                /* number of spatial dimensions per node */
#define ndof          3                /* degrees of freedom per node */
#define npel          8                /* nodes per element */
#define npel8         8                /* nodes per 8 node element */
#define neqel         npel*ndof        /* degrees of freedom per element */
#define num_int       8                /* number of integration points */
#define neqlsq        neqel*neqel      /* neqel squared */
#define sdim          6                /* stress dimensions per element */
#define soB           sdim*neqel                  /* size of B matrix */
#define MsoB          nsd*neqel                   /* size of B_mass matrix */
#define sosh          (nsd +1)*npel*num_int       /* size of shl and shg matrix */
#define sosh_node2    nsd*2*num_int               /* size of shl_node2 */
#define KB            1024.0                      /* number of bytes in kilobyte */
#define MB            1.0486e+06                  /* number of bytes in megabyte */

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

/*    Below is the additional list of constants for the brick element br2
      which has thermal properties.  
*/

#define Tndof         1                /* Temperature degrees of freedom per node */
#define npel_film     4                /* nodes per convection surface */
#define Tneqel        npel*Tndof       /* Temperature degrees of freedom per element */
#define TBneqel       npel_film*Tndof  /* Temperature degrees of freedom per film surfaces */
#define num_int_film  4                /* number of integration points for film surfaces */
#define Tneqlsq       Tneqel*Tneqel    /* Tneqel squared */
#define TBneqlsq      TBneqel*TBneqel  /* Tneqel squared */
#define Tdim          3                /* Temperature stress dimensions per element */
#define TsoB          Tdim*Tneqel                 /* size of B_T matrix */
#define TBsoB         Tdim*TBneqel                /* size of B_TB matrix */
#define sosh_film     nsd*npel_film*num_int_film  /* size of shl_film matrix */

/*
      Below is a list of variables for the brick elements br and nbr.  

  title               problem title
  numel               number of elements
  numel_K             number of elements whose element stiffness can be stored 
  numel_P             number of elements whose element stiffness are not stored.  They
                      only come in the calculation through P_el
  numnp               number of nodal points 
  nmat                number of materials
  nmode               number of modes  nmode = 0 means standard analysis
  dof                 total number of degrees of freedom 
  coord               *(coord + 0) x coordinate of node
                      *(coord + 1) y coordinate of node
                      *(coord + 2) z coordinate of node
  connect      (0-7)  connectivity array for 3-D brick elements
  matl                material structure
     Emod             young's modulus
     nu               poisson's ratio
     K                bulk modulus
     G                shear modulus 
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

/*
      Below is the additional list of variables for the brick element br2
      which has thermal properties.  

  connect_film (0-3)  connectivity array for surface elements

      The material can handle orthotropic cases so you can specify
      the values of different material properties depending on
      their direction relative to the global coordinate system.

  matl                material structure
     thrml_cond.x     thermal conductivity x
     thrml_cond.y     thermal conductivity y
     thrml_cond.z     thermal conductivity z
     thrml_expn.x     thermal expansion x
     thrml_expn.y     thermal expansion x
     thrml_expn.z     thermal expansion x
     film             film constant used in convection
     E.x              young's modulus x
     E.y              young's modulus y
     E.z              young's modulus z
     nu.xy            poisson's ratio xy
     nu.xz            poisson's ratio xz
     nu.yz            poisson's ratio yz

  heat                heat generation 
  temp_analysis_flag  1 calculate unknown temperatues
                      2 calculate equivalant nodal heat flow
  TLU_decomp_flag     0 if numel <= 2250 elements, use LU Decomposition for
                        temperatures
                      1 if numel > 2250 elements, use conjugate
                        gradient method for temperatures
  disp_analysis       1 calculate displacements, stress and strain 
                      0 skip calculation of displacement
  thermal_analysis    1 calculate thermal properties
                      0 skip calculation of temperature
*/
