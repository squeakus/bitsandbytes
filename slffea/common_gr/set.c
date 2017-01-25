/*
    This program sets viewing and analysis values based on the parameters
    determined in the subroutine "parameters" for the FEM GUI for 3-D
    non-constant strain elements(an example of a constant strain element is the
    tetrahedron)  with displacements Ux, Uy, Uz and stresses xx, yy, zz, xy, zx, yz,
    I, II, III.

   			Last Update 1/22/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if BRICK1
#include "../brick/brick/brconst.h"
#include "../brick/brick/brstruct.h"
#include "../brick/brick_gr/brstrcgr.h"
#endif
#if BRICK2
#include "../brick/brick/brconst.h"
#include "../brick/brick2/br2struct.h"
#include "../brick/brick_gr/brstrcgr.h"
#endif
#if WEDGE1
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge/westruct.h"
#include "../wedge/wedge_gr/westrcgr.h"
#endif
#include "control.h"


#define init_far0      -2.0

extern int nmat, numnp, numel, dof;
extern double step_sizex, step_sizey, step_sizez;
extern double left, right, top, bottom, near, far, fscale;
extern int control_height, control_width, mesh_height, mesh_width;
extern double ortho_left, ortho_right, ortho_top, ortho_bottom;
extern double left_right, up_down, in_out, left_right0, up_down0, in_out0;
extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z,
	IAxisMin_x, IAxisMin_y, IAxisMin_z;
extern double AxisLength_x, AxisLength_y, AxisLength_z,
	AxisLength_max;
extern double AxisPoint_step;
extern double amplify_step0;

extern double Ux_div[boxnumber+1], Uy_div[boxnumber+1], Uz_div[boxnumber+1];
extern SDIM stress_div[boxnumber+1], strain_div[boxnumber+1];
extern double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
extern SDIM del_stress, del_strain, max_stress, min_stress,
	max_strain, min_strain;
extern double max_Ux, min_Ux, del_Ux, max_Uy, min_Uy, del_Uy,
	max_Uz, min_Uz, del_Uz, absolute_max_U;
#if BRICK2
extern double T_div[boxnumber+1], Q_div[boxnumber+1];
extern double max_T, min_T, del_T, max_Q, min_Q, del_Q;
#endif

#if BRICK1 || WEDGE1
int set( BOUND bc, int *connecter, double *force, XYZF *force_vec0,
	SDIM *strain_node, ISTRAIN *strain_color, SDIM *stress_node,
	ISTRESS *stress_color, double *U, int *U_color )
#endif
#if BRICK2
int set( BOUND bc, int *connecter, double *force, XYZF *force_vec0,
	double *Q, int *Q_color, SDIM *strain_node, ISTRAIN *strain_color,
	SDIM *stress_node, ISTRESS *stress_color, double *T, int *T_color,
	double *U, int *U_color )
#endif
{
	int i, j, check;
	double force_vec_length;
	int node;

/* Determine displacement color scheme */

	del_Ux = (max_Ux - min_Ux + SMALL)/(double)(boxnumber);
	del_Uy = (max_Uy - min_Uy + SMALL)/(double)(boxnumber);
	del_Uz = (max_Uz - min_Uz + SMALL)/(double)(boxnumber);
	Ux_div[0] = min_Ux;
	Uy_div[0] = min_Uy;
	Uz_div[0] = min_Uz;

	for( i = 0; i < boxnumber; ++i )
	{
		Ux_div[i+1] = Ux_div[i] + del_Ux;
		Uy_div[i+1] = Uy_div[i] + del_Uy;
		Uz_div[i+1] = Uz_div[i] + del_Uz;
		/*printf(" U div x y z %10.5e %10.5e %10.5e\n",
			Ux_div[i], Uy_div[i], Uz_div[i]);*/
	}

#if BRICK2
/* Determine Temperature and heat Q color scheme */

	del_T = (max_T - min_T + SMALL)/(double)(boxnumber);
	del_Q = (max_Q - min_Q + SMALL)/(double)(boxnumber);
	T_div[0] = min_T;
	Q_div[0] = min_Q;

	for( i = 0; i < boxnumber; ++i )
	{
		T_div[i+1] = T_div[i] + del_T;
		Q_div[i+1] = Q_div[i] + del_Q;
		/*printf(" T div Q div %10.5e %10.5e\n",
			T_div[i], Q_div[i]);*/
	}
#endif

/* Determine strain and stress color scheme */

/* For strains */

	del_strain.xx =
		(max_strain.xx - min_strain.xx + SMALL)/(double)(boxnumber);
	del_strain.yy =
		(max_strain.yy - min_strain.yy + SMALL)/(double)(boxnumber);
	del_strain.zz =
		(max_strain.zz - min_strain.zz + SMALL)/(double)(boxnumber);
	del_strain.xy =
		(max_strain.xy - min_strain.xy + SMALL)/(double)(boxnumber);
	del_strain.zx =
		(max_strain.zx - min_strain.zx + SMALL)/(double)(boxnumber);
	del_strain.yz =
		(max_strain.yz - min_strain.yz + SMALL)/(double)(boxnumber);
	del_strain.I =
		(max_strain.I - min_strain.I + SMALL)/(double)(boxnumber);
	del_strain.II =
		(max_strain.II - min_strain.II + SMALL)/(double)(boxnumber);
	del_strain.III =
		(max_strain.III - min_strain.III + SMALL)/(double)(boxnumber);
	strain_div[0].xx = min_strain.xx;
	strain_div[0].yy = min_strain.yy;
	strain_div[0].zz = min_strain.zz;
	strain_div[0].xy = min_strain.xy;
	strain_div[0].zx = min_strain.zx;
	strain_div[0].yz = min_strain.yz;
	strain_div[0].I = min_strain.I;
	strain_div[0].II = min_strain.II;
	strain_div[0].III = min_strain.III;
	/*printf(" max min strain xx %10.5e %10.5e \n", max_strain.xx, min_strain.xx);
	printf(" strain div xx %10.5e \n", strain_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		strain_div[i+1].xx = strain_div[i].xx + del_strain.xx;
		strain_div[i+1].yy = strain_div[i].yy + del_strain.yy;
		strain_div[i+1].zz = strain_div[i].zz + del_strain.zz;
		strain_div[i+1].xy = strain_div[i].xy + del_strain.xy;
		strain_div[i+1].zx = strain_div[i].zx + del_strain.zx;
		strain_div[i+1].yz = strain_div[i].yz + del_strain.yz;
		strain_div[i+1].I = strain_div[i].I + del_strain.I;
		strain_div[i+1].II = strain_div[i].II + del_strain.II;
		strain_div[i+1].III = strain_div[i].III + del_strain.III;
		/*printf(" strain div xx %10.5e \n", strain_div[i+1].xx);*/
	}


/* For stresses */

	del_stress.xx =
		(max_stress.xx - min_stress.xx + SMALL)/(double)(boxnumber);
	del_stress.yy =
		(max_stress.yy - min_stress.yy + SMALL)/(double)(boxnumber);
	del_stress.zz =
		(max_stress.zz - min_stress.zz + SMALL)/(double)(boxnumber);
	del_stress.xy =
		(max_stress.xy - min_stress.xy + SMALL)/(double)(boxnumber);
	del_stress.zx =
		(max_stress.zx - min_stress.zx + SMALL)/(double)(boxnumber);
	del_stress.yz =
		(max_stress.yz - min_stress.yz + SMALL)/(double)(boxnumber);
	del_stress.I =
		(max_stress.I - min_stress.I + SMALL)/(double)(boxnumber);
	del_stress.II =
		(max_stress.II - min_stress.II + SMALL)/(double)(boxnumber);
	del_stress.III =
		(max_stress.III - min_stress.III + SMALL)/(double)(boxnumber);
	stress_div[0].xx = min_stress.xx;
	stress_div[0].yy = min_stress.yy;
	stress_div[0].zz = min_stress.zz;
	stress_div[0].xy = min_stress.xy;
	stress_div[0].zx = min_stress.zx;
	stress_div[0].yz = min_stress.yz;
	stress_div[0].I = min_stress.I;
	stress_div[0].II = min_stress.II;
	stress_div[0].III = min_stress.III;
	/*printf(" max min stress xx %10.5e %10.5e \n", max_stress.xx, min_stress.xx);
	printf(" stress div xx %10.5e \n", stress_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		stress_div[i+1].xx = stress_div[i].xx + del_stress.xx;
		stress_div[i+1].yy = stress_div[i].yy + del_stress.yy;
		stress_div[i+1].zz = stress_div[i].zz + del_stress.zz;
		stress_div[i+1].xy = stress_div[i].xy + del_stress.xy;
		stress_div[i+1].zx = stress_div[i].zx + del_stress.zx;
		stress_div[i+1].yz = stress_div[i].yz + del_stress.yz;
		stress_div[i+1].I = stress_div[i].I + del_stress.I;
		stress_div[i+1].II = stress_div[i].II + del_stress.II;
		stress_div[i+1].III = stress_div[i].III + del_stress.III;
		/*printf(" stress div xx %10.5e \n", stress_div[i+1].xx);*/
	}

/* Assign Colors for displacement, Temperature, heat Q, strains, and stresses */

	for( i = 0; i < numnp; ++i )
	{
/* Assign colors for Ux */
	       *(U_color + ndof*i) = 0;
	       if(  *(U + ndof*i) > Ux_div[1] )
	       {
		  *(U_color + ndof*i) = 1;
		  if(  *(U + ndof*i) > Ux_div[2] )
		  {
		     *(U_color + ndof*i) = 2;
		     if(  *(U + ndof*i) > Ux_div[3] )
		     {
			*(U_color + ndof*i) = 3;
			if(  *(U + ndof*i) > Ux_div[4] )
			{
			   *(U_color + ndof*i) = 4;
			   if(  *(U + ndof*i) > Ux_div[5] )
			   {
			      *(U_color + ndof*i) = 5;
			      if(  *(U + ndof*i) > Ux_div[6] )
			      {
				 *(U_color + ndof*i) = 6;
				 if(  *(U + ndof*i) > Ux_div[7] )
				 {
				    *(U_color + ndof*i) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Ux %d %10.5e %d \n", i,
			*(U+ndof*i), *(U_color+ndof*i));*/
/* Assign colors for Uy */
	       *(U_color + ndof*i + 1) = 0;
	       if(  *(U + ndof*i + 1) > Uy_div[1] )
	       {
		  *(U_color + ndof*i + 1) = 1;
		  if(  *(U + ndof*i + 1) > Uy_div[2] )
		  {
		     *(U_color + ndof*i + 1) = 2;
		     if(  *(U + ndof*i + 1) > Uy_div[3] )
		     {
			*(U_color + ndof*i + 1) = 3;
			if(  *(U + ndof*i + 1) > Uy_div[4] )
			{
			   *(U_color + ndof*i + 1) = 4;
			   if(  *(U + ndof*i + 1) > Uy_div[5] )
			   {
			      *(U_color + ndof*i + 1) = 5;
			      if(  *(U + ndof*i + 1) > Uy_div[6] )
			      {
				 *(U_color + ndof*i + 1) = 6;
				 if(  *(U + ndof*i + 1) > Uy_div[7] )
				 {
				    *(U_color + ndof*i + 1) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uy %d %10.5e %d \n", i,
			*(U+ndof*i + 1), *(U_color+ndof*i + 1));*/
/* Assign colors for Uz */
	       *(U_color + ndof*i + 2) = 0;
	       if(  *(U + ndof*i + 2) > Uz_div[1] )
	       {
		  *(U_color + ndof*i + 2) = 1;
		  if(  *(U + ndof*i + 2) > Uz_div[2] )
		  {
		     *(U_color + ndof*i + 2) = 2;
		     if(  *(U + ndof*i + 2) > Uz_div[3] )
		     {
			*(U_color + ndof*i + 2) = 3;
			if(  *(U + ndof*i + 2) > Uz_div[4] )
			{
			   *(U_color + ndof*i + 2) = 4;
			   if(  *(U + ndof*i + 2) > Uz_div[5] )
			   {
			      *(U_color + ndof*i + 2) = 5;
			      if(  *(U + ndof*i + 2) > Uz_div[6] )
			      {
				 *(U_color + ndof*i + 2) = 6;
				 if(  *(U + ndof*i + 2) > Uz_div[7] )
				 {
				    *(U_color + ndof*i + 2) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uz %d %10.5e %d \n", i,
			*(U+ndof*i + 2), *(U_color+ndof*i + 2));*/
#if BRICK2
/* Assign Colors for Temperature and heat Q for thermal elements */

/* Assign colors for T */
	       *(T_color + Tndof*i) = 0;
	       if(  *(T + Tndof*i) > T_div[1] )
	       {
		  *(T_color + Tndof*i) = 1;
		  if(  *(T + Tndof*i) > T_div[2] )
		  {
		     *(T_color + Tndof*i) = 2;
		     if(  *(T + Tndof*i) > T_div[3] )
		     {
			*(T_color + Tndof*i) = 3;
			if(  *(T + Tndof*i) > T_div[4] )
			{
			   *(T_color + Tndof*i) = 4;
			   if(  *(T + Tndof*i) > T_div[5] )
			   {
			      *(T_color + Tndof*i) = 5;
			      if(  *(T + Tndof*i) > T_div[6] )
			      {
				 *(T_color + Tndof*i) = 6;
				 if(  *(T + Tndof*i) > T_div[7] )
				 {
				    *(T_color + Tndof*i) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" T %d %10.5e %d \n", i, *(T+Tndof*i), *(T_color+Tndof*i));*/
/* Assign colors for Q */
	       *(Q_color + Tndof*i) = 0;
	       if(  *(Q + Tndof*i) > Q_div[1] )
	       {
		  *(Q_color + Tndof*i) = 1;
		  if(  *(Q + Tndof*i) > Q_div[2] )
		  {
		     *(Q_color + Tndof*i) = 2;
		     if(  *(Q + Tndof*i) > Q_div[3] )
		     {
			*(Q_color + Tndof*i) = 3;
			if(  *(Q + Tndof*i) > Q_div[4] )
			{
			   *(Q_color + Tndof*i) = 4;
			   if(  *(Q + Tndof*i) > Q_div[5] )
			   {
			      *(Q_color + Tndof*i) = 5;
			      if(  *(Q + Tndof*i) > Q_div[6] )
			      {
				 *(Q_color + Tndof*i) = 6;
				 if(  *(Q + Tndof*i) > Q_div[7] )
				 {
				    *(Q_color + Tndof*i) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Q %d %10.5e %d \n", i,
			*(Q+Tndof*i), *(Q_color+Tndof*i));*/
#endif
	}


/* Assign colors to strains and stresses */

	for( i = 0; i < numel; ++i )
	{
	    for( j = 0; j < num_int; ++j )
	    {

	       node = *(connecter+npel*i+j);

/* Assign colors for strain xx */
	       strain_color[i].pt[j].xx = 0;
	       if(  strain_node[node].xx > strain_div[1].xx )
	       {
		  strain_color[i].pt[j].xx = 1;
		  if(  strain_node[node].xx > strain_div[2].xx )
		  {
		     strain_color[i].pt[j].xx = 2;
		     if(  strain_node[node].xx > strain_div[3].xx )
		     {
			strain_color[i].pt[j].xx = 3;
			if(  strain_node[node].xx > strain_div[4].xx )
			{
			   strain_color[i].pt[j].xx = 4;
			   if(  strain_node[node].xx > strain_div[5].xx )
			   {
			      strain_color[i].pt[j].xx = 5;
			      if(  strain_node[node].xx > strain_div[6].xx )
			      {
				 strain_color[i].pt[j].xx = 6;
				 if(  strain_node[node].xx > strain_div[7].xx )
				 {
				    strain_color[i].pt[j].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain xx %d %d %10.5e %d \n", i, j,
			strain_node[node].xx, strain_color[i].pt[j].xx);*/
/* Assign colors for strain yy */
	       strain_color[i].pt[j].yy = 0;
	       if(  strain_node[node].yy > strain_div[1].yy )
	       {
		  strain_color[i].pt[j].yy = 1;
		  if(  strain_node[node].yy > strain_div[2].yy )
		  {
		     strain_color[i].pt[j].yy = 2;
		     if(  strain_node[node].yy > strain_div[3].yy )
		     {
			strain_color[i].pt[j].yy = 3;
			if(  strain_node[node].yy > strain_div[4].yy )
			{
			   strain_color[i].pt[j].yy = 4;
			   if(  strain_node[node].yy > strain_div[5].yy )
			   {
			      strain_color[i].pt[j].yy = 5;
			      if(  strain_node[node].yy > strain_div[6].yy )
			      {
				 strain_color[i].pt[j].yy = 6;
				 if(  strain_node[node].yy > strain_div[7].yy )
				 {
				    strain_color[i].pt[j].yy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain yy %d %d %10.5e %d \n", i, j,
			strain_node[node].yy, strain_color[i].pt[j].yy);*/
/* Assign colors for strain zz */
	       strain_color[i].pt[j].zz = 0;
	       if(  strain_node[node].zz > strain_div[1].zz )
	       {
		  strain_color[i].pt[j].zz = 1;
		  if(  strain_node[node].zz > strain_div[2].zz )
		  {
		     strain_color[i].pt[j].zz = 2;
		     if(  strain_node[node].zz > strain_div[3].zz )
		     {
			strain_color[i].pt[j].zz = 3;
			if(  strain_node[node].zz > strain_div[4].zz )
			{
			   strain_color[i].pt[j].zz = 4;
			   if(  strain_node[node].zz > strain_div[5].zz )
			   {
			      strain_color[i].pt[j].zz = 5;
			      if(  strain_node[node].zz > strain_div[6].zz )
			      {
				 strain_color[i].pt[j].zz = 6;
				 if(  strain_node[node].zz > strain_div[7].zz )
				 {
				    strain_color[i].pt[j].zz = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain zz %d %d %10.5e %d \n", i, j,
			strain_node[node].zz, strain_color[i].pt[j].zz);*/
/* Assign colors for strain xy */
	       strain_color[i].pt[j].xy = 0;
	       if(  strain_node[node].xy > strain_div[1].xy )
	       {
		  strain_color[i].pt[j].xy = 1;
		  if(  strain_node[node].xy > strain_div[2].xy )
		  {
		     strain_color[i].pt[j].xy = 2;
		     if(  strain_node[node].xy > strain_div[3].xy )
		     {
			strain_color[i].pt[j].xy = 3;
			if(  strain_node[node].xy > strain_div[4].xy )
			{
			   strain_color[i].pt[j].xy = 4;
			   if(  strain_node[node].xy > strain_div[5].xy )
			   {
			      strain_color[i].pt[j].xy = 5;
			      if(  strain_node[node].xy > strain_div[6].xy )
			      {
				 strain_color[i].pt[j].xy = 6;
				 if(  strain_node[node].xy > strain_div[7].xy )
				 {
				    strain_color[i].pt[j].xy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain xy %d %d %10.5e %d \n", i, j,
			strain_node[node].xy, strain_color[i].pt[j].xy);*/
/* Assign colors for strain zx */
	       strain_color[i].pt[j].zx = 0;
	       if(  strain_node[node].zx > strain_div[1].zx )
	       {
		  strain_color[i].pt[j].zx = 1;
		  if(  strain_node[node].zx > strain_div[2].zx )
		  {
		     strain_color[i].pt[j].zx = 2;
		     if(  strain_node[node].zx > strain_div[3].zx )
		     {
			strain_color[i].pt[j].zx = 3;
			if(  strain_node[node].zx > strain_div[4].zx )
			{
			   strain_color[i].pt[j].zx = 4;
			   if(  strain_node[node].zx > strain_div[5].zx )
			   {
			      strain_color[i].pt[j].zx = 5;
			      if(  strain_node[node].zx > strain_div[6].zx )
			      {
				 strain_color[i].pt[j].zx = 6;
				 if(  strain_node[node].zx > strain_div[7].zx )
				 {
				    strain_color[i].pt[j].zx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain zx %d %d %10.5e %d \n", i, j,
			strain_node[node].zx, strain_color[i].pt[j].zx);*/
/* Assign colors for strain yz */
	       strain_color[i].pt[j].yz = 0;
	       if(  strain_node[node].yz > strain_div[1].yz )
	       {
		  strain_color[i].pt[j].yz = 1;
		  if(  strain_node[node].yz > strain_div[2].yz )
		  {
		     strain_color[i].pt[j].yz = 2;
		     if(  strain_node[node].yz > strain_div[3].yz )
		     {
			strain_color[i].pt[j].yz = 3;
			if(  strain_node[node].yz > strain_div[4].yz )
			{
			   strain_color[i].pt[j].yz = 4;
			   if(  strain_node[node].yz > strain_div[5].yz )
			   {
			      strain_color[i].pt[j].yz = 5;
			      if(  strain_node[node].yz > strain_div[6].yz )
			      {
				 strain_color[i].pt[j].yz = 6;
				 if(  strain_node[node].yz > strain_div[7].yz )
				 {
				    strain_color[i].pt[j].yz = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain yz %d %d %10.5e %d \n", i, j,
			strain_node[node].yz, strain_color[i].pt[j].yz);*/
/* Assign colors for strain I */
	       strain_color[i].pt[j].I = 0;
	       if(  strain_node[node].I > strain_div[1].I )
	       {
		  strain_color[i].pt[j].I = 1;
		  if(  strain_node[node].I > strain_div[2].I )
		  {
		     strain_color[i].pt[j].I = 2;
		     if(  strain_node[node].I > strain_div[3].I )
		     {
			strain_color[i].pt[j].I = 3;
			if(  strain_node[node].I > strain_div[4].I )
			{
			   strain_color[i].pt[j].I = 4;
			   if(  strain_node[node].I > strain_div[5].I )
			   {
			      strain_color[i].pt[j].I = 5;
			      if(  strain_node[node].I > strain_div[6].I )
			      {
				 strain_color[i].pt[j].I = 6;
				 if(  strain_node[node].I > strain_div[7].I )
				 {
				    strain_color[i].pt[j].I = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain I %d %d %10.5e %d \n", i, j,
			strain_node[node].I, strain_color[i].pt[j].I);*/
/* Assign colors for strain II */
	       strain_color[i].pt[j].II = 0;
	       if(  strain_node[node].II > strain_div[1].II )
	       {
		  strain_color[i].pt[j].II = 1;
		  if(  strain_node[node].II > strain_div[2].II )
		  {
		     strain_color[i].pt[j].II = 2;
		     if(  strain_node[node].II > strain_div[3].II )
		     {
			strain_color[i].pt[j].II = 3;
			if(  strain_node[node].II > strain_div[4].II )
			{
			   strain_color[i].pt[j].II = 4;
			   if(  strain_node[node].II > strain_div[5].II )
			   {
			      strain_color[i].pt[j].II = 5;
			      if(  strain_node[node].II > strain_div[6].II )
			      {
				 strain_color[i].pt[j].II = 6;
				 if(  strain_node[node].II > strain_div[7].II )
				 {
				    strain_color[i].pt[j].II = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain II %d %d %10.5e %d \n", i, j,
			strain_node[node].II, strain_color[i].pt[j].II);*/
/* Assign colors for strain III */
	       strain_color[i].pt[j].III = 0;
	       if(  strain_node[node].III > strain_div[1].III )
	       {
		  strain_color[i].pt[j].III = 1;
		  if(  strain_node[node].III > strain_div[2].III )
		  {
		     strain_color[i].pt[j].III = 2;
		     if(  strain_node[node].III > strain_div[3].III )
		     {
			strain_color[i].pt[j].III = 3;
			if(  strain_node[node].III > strain_div[4].III )
			{
			   strain_color[i].pt[j].III = 4;
			   if(  strain_node[node].III > strain_div[5].III )
			   {
			      strain_color[i].pt[j].III = 5;
			      if(  strain_node[node].III > strain_div[6].III )
			      {
				 strain_color[i].pt[j].III = 6;
				 if(  strain_node[node].III > strain_div[7].III )
				 {
				    strain_color[i].pt[j].III = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain III %d %d %10.5e %d \n", i, j,
			strain_node[node].III, strain_color[i].pt[j].III);*/
/* Assign colors for stress xx */
	       stress_color[i].pt[j].xx = 0;
	       if(  stress_node[node].xx > stress_div[1].xx )
	       {
		  stress_color[i].pt[j].xx = 1;
		  if(  stress_node[node].xx > stress_div[2].xx )
		  {
		     stress_color[i].pt[j].xx = 2;
		     if(  stress_node[node].xx > stress_div[3].xx )
		     {
			stress_color[i].pt[j].xx = 3;
			if(  stress_node[node].xx > stress_div[4].xx )
			{
			   stress_color[i].pt[j].xx = 4;
			   if(  stress_node[node].xx > stress_div[5].xx )
			   {
			      stress_color[i].pt[j].xx = 5;
			      if(  stress_node[node].xx > stress_div[6].xx )
			      {
				 stress_color[i].pt[j].xx = 6;
				 if(  stress_node[node].xx > stress_div[7].xx )
				 {
				    stress_color[i].pt[j].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress xx %d %d %10.5e %d \n", i, j,
			stress_node[node].xx, stress_color[i].pt[j].xx);*/
/* Assign colors for stress yy */
	       stress_color[i].pt[j].yy = 0;
	       if(  stress_node[node].yy > stress_div[1].yy )
	       {
		  stress_color[i].pt[j].yy = 1;
		  if(  stress_node[node].yy > stress_div[2].yy )
		  {
		     stress_color[i].pt[j].yy = 2;
		     if(  stress_node[node].yy > stress_div[3].yy )
		     {
			stress_color[i].pt[j].yy = 3;
			if(  stress_node[node].yy > stress_div[4].yy )
			{
			   stress_color[i].pt[j].yy = 4;
			   if(  stress_node[node].yy > stress_div[5].yy )
			   {
			      stress_color[i].pt[j].yy = 5;
			      if(  stress_node[node].yy > stress_div[6].yy )
			      {
				 stress_color[i].pt[j].yy = 6;
				 if(  stress_node[node].yy > stress_div[7].yy )
				 {
				    stress_color[i].pt[j].yy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress yy %d %d %10.5e %d \n", i, j,
			stress_node[node].yy, stress_color[i].pt[j].yy);*/
/* Assign colors for stress zz */
	       stress_color[i].pt[j].zz = 0;
	       if(  stress_node[node].zz > stress_div[1].zz )
	       {
		  stress_color[i].pt[j].zz = 1;
		  if(  stress_node[node].zz > stress_div[2].zz )
		  {
		     stress_color[i].pt[j].zz = 2;
		     if(  stress_node[node].zz > stress_div[3].zz )
		     {
			stress_color[i].pt[j].zz = 3;
			if(  stress_node[node].zz > stress_div[4].zz )
			{
			   stress_color[i].pt[j].zz = 4;
			   if(  stress_node[node].zz > stress_div[5].zz )
			   {
			      stress_color[i].pt[j].zz = 5;
			      if(  stress_node[node].zz > stress_div[6].zz )
			      {
				 stress_color[i].pt[j].zz = 6;
				 if(  stress_node[node].zz > stress_div[7].zz )
				 {
				    stress_color[i].pt[j].zz = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress zz %d %d %10.5e %d \n", i, j,
			stress_node[node].zz, stress_color[i].pt[j].zz);*/
/* Assign colors for stress xy */
	       stress_color[i].pt[j].xy = 0;
	       if(  stress_node[node].xy > stress_div[1].xy )
	       {
		  stress_color[i].pt[j].xy = 1;
		  if(  stress_node[node].xy > stress_div[2].xy )
		  {
		     stress_color[i].pt[j].xy = 2;
		     if(  stress_node[node].xy > stress_div[3].xy )
		     {
			stress_color[i].pt[j].xy = 3;
			if(  stress_node[node].xy > stress_div[4].xy )
			{
			   stress_color[i].pt[j].xy = 4;
			   if(  stress_node[node].xy > stress_div[5].xy )
			   {
			      stress_color[i].pt[j].xy = 5;
			      if(  stress_node[node].xy > stress_div[6].xy )
			      {
				 stress_color[i].pt[j].xy = 6;
				 if(  stress_node[node].xy > stress_div[7].xy )
				 {
				    stress_color[i].pt[j].xy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress xy %d %d %10.5e %d \n", i, j,
			stress_node[node].xy, stress_color[i].pt[j].xy);*/
/* Assign colors for stress zx */
	       stress_color[i].pt[j].zx = 0;
	       if(  stress_node[node].zx > stress_div[1].zx )
	       {
		  stress_color[i].pt[j].zx = 1;
		  if(  stress_node[node].zx > stress_div[2].zx )
		  {
		     stress_color[i].pt[j].zx = 2;
		     if(  stress_node[node].zx > stress_div[3].zx )
		     {
			stress_color[i].pt[j].zx = 3;
			if(  stress_node[node].zx > stress_div[4].zx )
			{
			   stress_color[i].pt[j].zx = 4;
			   if(  stress_node[node].zx > stress_div[5].zx )
			   {
			      stress_color[i].pt[j].zx = 5;
			      if(  stress_node[node].zx > stress_div[6].zx )
			      {
				 stress_color[i].pt[j].zx = 6;
				 if(  stress_node[node].zx > stress_div[7].zx )
				 {
				    stress_color[i].pt[j].zx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress zx %d %d %10.5e %d \n", i, j,
			stress_node[node].zx, stress_color[i].pt[j].zx);*/
/* Assign colors for stress yz */
	       stress_color[i].pt[j].yz = 0;
	       if(  stress_node[node].yz > stress_div[1].yz )
	       {
		  stress_color[i].pt[j].yz = 1;
		  if(  stress_node[node].yz > stress_div[2].yz )
		  {
		     stress_color[i].pt[j].yz = 2;
		     if(  stress_node[node].yz > stress_div[3].yz )
		     {
			stress_color[i].pt[j].yz = 3;
			if(  stress_node[node].yz > stress_div[4].yz )
			{
			   stress_color[i].pt[j].yz = 4;
			   if(  stress_node[node].yz > stress_div[5].yz )
			   {
			      stress_color[i].pt[j].yz = 5;
			      if(  stress_node[node].yz > stress_div[6].yz )
			      {
				 stress_color[i].pt[j].yz = 6;
				 if(  stress_node[node].yz > stress_div[7].yz )
				 {
				    stress_color[i].pt[j].yz = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress yz %d %d %10.5e %d \n", i, j,
			stress_node[node].yz, stress_color[i].pt[j].yz);*/
/* Assign colors for stress I */
	       stress_color[i].pt[j].I = 0;
	       if(  stress_node[node].I > stress_div[1].I )
	       {
		  stress_color[i].pt[j].I = 1;
		  if(  stress_node[node].I > stress_div[2].I )
		  {
		     stress_color[i].pt[j].I = 2;
		     if(  stress_node[node].I > stress_div[3].I )
		     {
			stress_color[i].pt[j].I = 3;
			if(  stress_node[node].I > stress_div[4].I )
			{
			   stress_color[i].pt[j].I = 4;
			   if(  stress_node[node].I > stress_div[5].I )
			   {
			      stress_color[i].pt[j].I = 5;
			      if(  stress_node[node].I > stress_div[6].I )
			      {
				 stress_color[i].pt[j].I = 6;
				 if(  stress_node[node].I > stress_div[7].I )
				 {
				    stress_color[i].pt[j].I = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress I %d %d %10.5e %d \n", i, j,
			stress_node[node].I, stress_color[i].pt[j].I);*/
/* Assign colors for stress II */
	       stress_color[i].pt[j].II = 0;
	       if(  stress_node[node].II > stress_div[1].II )
	       {
		  stress_color[i].pt[j].II = 1;
		  if(  stress_node[node].II > stress_div[2].II )
		  {
		     stress_color[i].pt[j].II = 2;
		     if(  stress_node[node].II > stress_div[3].II )
		     {
			stress_color[i].pt[j].II = 3;
			if(  stress_node[node].II > stress_div[4].II )
			{
			   stress_color[i].pt[j].II = 4;
			   if(  stress_node[node].II > stress_div[5].II )
			   {
			      stress_color[i].pt[j].II = 5;
			      if(  stress_node[node].II > stress_div[6].II )
			      {
				 stress_color[i].pt[j].II = 6;
				 if(  stress_node[node].II > stress_div[7].II )
				 {
				    stress_color[i].pt[j].II = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress II %d %d %10.5e %d \n", i, j,
			stress_node[node].II, stress_color[i].pt[j].II);*/
/* Assign colors for stress III */
	       stress_color[i].pt[j].III = 0;
	       if(  stress_node[node].III > stress_div[1].III )
	       {
		  stress_color[i].pt[j].III = 1;
		  if(  stress_node[node].III > stress_div[2].III )
		  {
		     stress_color[i].pt[j].III = 2;
		     if(  stress_node[node].III > stress_div[3].III )
		     {
			stress_color[i].pt[j].III = 3;
			if(  stress_node[node].III > stress_div[4].III )
			{
			   stress_color[i].pt[j].III = 4;
			   if(  stress_node[node].III > stress_div[5].III )
			   {
			      stress_color[i].pt[j].III = 5;
			      if(  stress_node[node].III > stress_div[6].III )
			      {
				 stress_color[i].pt[j].III = 6;
				 if(  stress_node[node].III > stress_div[7].III )
				 {
				    stress_color[i].pt[j].III = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress III %d %d %10.5e %d \n", i, j,
			stress_node[node].III, stress_color[i].pt[j].III);*/
	    }
	}

/* Calculate the graphical force vectors for display */

	for( i = 0; i < bc.num_force[0]; ++i )
	{
		
		force_vec_length =
		   *(force + ndof*bc.force[i])*(*(force + ndof*bc.force[i])) +
		   *(force + ndof*bc.force[i] +1)*(*(force + ndof*bc.force[i] +1)) +
		   *(force + ndof*bc.force[i] +2)*(*(force + ndof*bc.force[i] +2)) +
		   SMALL;
		force_vec_length = sqrt(force_vec_length);
		force_vec0[i].x =
		   *(force + ndof*bc.force[i])*3*AxisLength_max/force_vec_length;
		force_vec0[i].y =
		   *(force + ndof*bc.force[i] + 1)*3*AxisLength_max/force_vec_length;
		force_vec0[i].z =
		   *(force + ndof*bc.force[i] + 2)*3*AxisLength_max/force_vec_length;
		/*printf(" force %d %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e\n", i, 
		   *(force + ndof*bc.force[i]), *(force + ndof*bc.force[i] + 1),
		   *(force + ndof*bc.force[i] + 2), force_vec0[i].x, force_vec0[i].y,
		   force_vec0[i].z);*/
	}

	return 1;    /* ANSI C requires main to return int. */
}
