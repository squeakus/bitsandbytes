/*
    This is the include file "shconst.h" for the finite element progam 
    which uses shell elements.

                Updated 9/29/08

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
#define pt6667  2.0/3.0
#define pt3333  1.0/3.0
#define sqpt5   .707106781
#define sq3     1.732050808 

#define nsd         3                                    /* spatial dimensions per node */
#define nsdl        2                                    /* isoparametric dimensions in lamina */
#define nsdf        1                                    /* isoparametric dimensions in fiber */
#define ndof        5                                    /* degrees of freedom per node */
#define npelf       2                                    /* nodes per element fiber */
#define npell       4                                    /* nodes per element lamina */
#define npell3      3                                    /* nodes per element lamina */
#define npell4      4                                    /* nodes per element lamina */
#define npel        npelf*npell                          /* nodes per element */
#define npel6       npelf*npell3                         /* nodes per 3 node element */
#define npel8       npelf*npell4                         /* nodes per 4 node element */
#define neqel       npell*ndof                           /* degrees of freedom per element */
#define neqel15     npell3*ndof                          /* degrees of freedom per 3 node element */
#define neqel20     npell4*ndof                          /* degrees of freedom per 4 node element */
#define num_ints    2                                    /* number of integ. points on fiber*/
#define num_intb    4                                    /* number of integ. points on lamina */
#define num_intb3   3                                    /* number of integ. points on lamina 3 node element */
#define num_intb4   4                                    /* number of integ. points on lamina 4 node element */
#define num_int     num_intb*num_ints                    /* number of integ. points on lamina */
#define num_int6    num_intb3*num_ints                   /* number of integ. points on lamina 3 node element */
#define num_int8    num_intb4*num_ints                   /* number of integ. points on lamina 4 node element */
#define neqlsq      neqel*neqel                          /* neqel squared */
#define neqlsq225   neqel15*neqel15                      /* neqel15 squared 3 node element */
#define neqlsq400   neqel20*neqel20                      /* neqel20 squared 4 node element */
#define nsdsq       nsd*nsd                              /* nsd squared */
#define lenb        3*neqel*num_intb                     /*  */
#define lenb135     3*neqel15*num_intb3                  /* 3 node element */
#define lenb240     3*neqel20*num_intb4                  /* 4 node element */
#define nremat      5                                    /* number of material properties */
#define nrconnect1      8                                /* number of possible disp. bound. cond. */
#define nrconnect2      4                                /* number of possible rotat. bound. cond. */
#define nrconnect1_3    6                                /* number of possible disp. bound. cond. 3(total) node element */
#define nrconnect2_3    3                                /* number of possible rotat. bound. cond. 6(total) node element */
#define nrconnect1_4    8                                /* number of possible disp. bound. cond. 4(total) node element */
#define nrconnect2_4    4                                /* number of possible rotat. bound. cond. 8(total) node element */
#define sdim        5                                    /* stress dimensions per element */
#define soB         sdim*neqel                           /* size of B matrix for shell */
#define soB75       sdim*neqel15                         /* size of B matrix for shell 3 node element */
#define soB100      sdim*neqel20                         /* size of B matrix for shell 4 node element */
#define MsoB        nsd*neqel                            /* size of B_mass matrix */
#define MsoB45      nsd*neqel15                          /* size of B_mass matrix 3 node element */
#define MsoB60      nsd*neqel20                          /* size of B_mass matrix 4 node element */
#define soxsbar     nsd*(nsd-1)                          /* size of xs.bar */
#define soxshat     nsd*nsd                              /* size of xs.hat */

#define soshlb      (nsdl+1)*npell*(num_intb)            /* size of shl matrix
                                                            for 2X2 PT. Gauss for bending */
#define soshlb27    (nsdl+1)*npell3*(num_intb3)          /* size of shl matrix
                                                            for 3 PT. Gauss for bending 3 node element */
#define soshlb48    (nsdl+1)*npell4*(num_intb4)          /* size of shl matrix
                                                            for 2X2 PT. Gauss for bending 4 node element */
#define soshls      (nsdl+1)*npell*(1)                   /* size of shl matrix
                                                            for 1X1 PT. Gauss for shear */
#define soshls9     (nsdl+1)*npell3*(1)                  /* size of shl matrix
                                                            for 1X1 PT. Gauss for shear 3 node element */
#define soshls12    (nsdl+1)*npell4*(1)                  /* size of shl matrix
                                                            for 1X1 PT. Gauss for shear 4 node element */
#define soshl_zb    (nsdf+1)*npelf*num_ints              /* size of shl_z matrix
                                                            for 2X1 PT. Gauss for bending */
#define soshl_zs    (nsdf+1)*npelf                       /* size of shl_z matrix
                                                            for 1X1 PT. Gauss for shear */
#define soshgb      (nsd+1)*npell*(num_intb)*num_ints    /* size of shg matrix (+1 SRI)
                                                            for 2X2 PT. Gauss for bending */
#define soshgb72    (nsd+1)*npell3*(num_intb3)*num_ints  /* size of shg matrix (+1 SRI)
                                                            for 3 PT. Gauss for bending 3 node element */
#define soshgb128   (nsd+1)*npell4*(num_intb4)*num_ints  /* size of shg matrix (+1 SRI)
                                                            for 2X2 PT. Gauss for bending 4 node element */
#define soshgs      (nsd+1)*npell*(1)*num_ints           /* size of shg matrix (+1 SRI)
                                                            for 1X1 PT. Gauss for shear */
#define soshgs24    (nsd+1)*npell3*(1)*num_ints          /* size of shg matrix (+1 SRI)
                                                            for 1X1 PT. Gauss for shear 3 node element */
#define soshgs32    (nsd+1)*npell4*(1)*num_ints          /* size of shg matrix (+1 SRI)
                                                            for 1X1 PT. Gauss for shear 4 node element */
#define soshg_zb    (nsd)*npell*(num_intb)*num_ints      /* size of shg_z matrix
                                                            for 2X2 PT. Gauss for bending */
#define soshg_zb54  (nsd)*npell3*(num_intb3)*num_ints    /* size of shg_z matrix
                                                            for 2X2 PT. Gauss for bending 3 node element */
#define soshg_zb96  (nsd)*npell4*(num_intb4)*num_ints    /* size of shg_z matrix
                                                            for 2X2 PT. Gauss for bending 4 node element */
#define soshg_zs    (nsd)*npell*(1)*num_ints             /* size of shg_z matrix
                                                            for 1X1 PT. Gauss for shear */
#define soshg_zs18  (nsd)*npell3*(1)*num_ints            /* size of shg_z matrix
                                                            for 1X1 PT. Gauss for shear 3 node element */
#define soshg_zs24  (nsd)*npell4*(1)*num_ints            /* size of shg_z matrix
                                                            for 1X1 PT. Gauss for shear 4 node element */
#define sorlb       nsdsq*num_int                        /* size of lamina rotat. matrix bending */
#define sorlb54     nsdsq*num_int6                       /* size of lamina rotat. matrix bending 3 node element */
#define sorlb72     nsdsq*num_int8                       /* size of lamina rotat. matrix bending 4 node element */
#define sorls       nsdsq*num_ints                       /* size of lamina rotat. matrix shear */
#define sorfs       nsdsq*npell                          /* size of fiber rotat. matrix shear */
#define sorfs27     nsdsq*npell3                         /* size of fiber rotat. matrix shear 3 node element */
#define sorfs36     nsdsq*npell4                         /* size of fiber rotat. matrix shear 4 node element */
#define zeta        0.00                                 /* reference point on lamina */
#define sosh_node2  nsd*2*num_ints                       /* size of shl_node2 */
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
  connect             (0-3) connectivity array 
  matl                material structure
  Emod                young's modulus
  nu                  poisson's ratio
  shear               shear correction factor
  force               *(force + 0) x component of applied load
                      *(force + 1) y component of applied load
                      *(force + 2) z component of applied load
                      *(force + 3) phi1 component of applied load
                      *(force + 4) phi2 component of applied load
  integ_flag          0 Reduced integration of membrane shear
                      1 Reduced integration of transverse shear
                      2 Reduced integration of membrane and 
                        transverse shear
  analysis_flag       1 calculate unknown displacemnts
                      2 calculate reaction forces
  LU_decomp_flag      0 if numel <= 450 elements, use LU Decomposition for
                        displacements
                      1 if numel > 450 elements, use conjugate
                        gradient method for displacements
*/
