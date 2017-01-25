/*
    This program sets viewing and analysis values based on the parameters 
    determined in shparameters for the FEM GUI for shell elements.
  
   			Last Update 10/3/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le

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
#include "../shell/shconst.h"
#include "../shell/shstruct.h"
#include "shstrcgr.h"
#include "../../common_gr/control.h"

#define init_far0      -2.0

extern int nmat, numnp, numel, dof, flag_quad_element;
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

extern double Ux_div[boxnumber+1], Uy_div[boxnumber+1], Uz_div[boxnumber+1],
	Uphi_x_div[boxnumber+1], Uphi_y_div[boxnumber+1];
extern SDIM stress_div[boxnumber+1];
extern SDIM strain_div[boxnumber+1];
extern double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
extern SDIM del_stress, del_strain, max_stress, min_stress,
	max_strain, min_strain;
extern double max_Uphi_x, min_Uphi_x, del_Uphi_x, max_Uphi_y, min_Uphi_y, del_Uphi_y,
	max_Ux, min_Ux, del_Ux, max_Uy, min_Uy, del_Uy,
	max_Uz, min_Uz, del_Uz, absolute_max_U;

int shset( BOUND bc, int *connecter, double *force, XYZPhiF *force_vec0,
	SDIM *strain_node, ISTRAIN *strain_color, SDIM *stress_node,
	ISTRESS *stress_color, double *U, int *U_color )
{
	int i, i4, i5, j, check;
	double force_vec_length, moment_vec_length;
	int node, npell_, num_intb_;

/* Determine displacement and angle color scheme */

	del_Ux = (max_Ux - min_Ux + SMALL)/(double)(boxnumber);
	del_Uy = (max_Uy - min_Uy + SMALL)/(double)(boxnumber);
	del_Uz = (max_Uz - min_Uz + SMALL)/(double)(boxnumber);
	del_Uphi_x = (max_Uphi_x - min_Uphi_x + SMALL)/(double)(boxnumber);
	del_Uphi_y = (max_Uphi_y - min_Uphi_y + SMALL)/(double)(boxnumber);
	Ux_div[0] = min_Ux;
	Uy_div[0] = min_Uy;
	Uz_div[0] = min_Uz;
	Uphi_x_div[0] = min_Uphi_x;
	Uphi_y_div[0] = min_Uphi_y;

	for( i = 0; i < boxnumber; ++i )
	{
		Ux_div[i+1] = Ux_div[i] + del_Ux;
		Uy_div[i+1] = Uy_div[i] + del_Uy;
		Uz_div[i+1] = Uz_div[i] + del_Uz;
		Uphi_x_div[i+1] = Uphi_x_div[i] + del_Uphi_x;
		Uphi_y_div[i+1] = Uphi_y_div[i] + del_Uphi_y;
		/*printf(" U div x y z %10.5e %10.5e %10.5e\n",
			Uphi_x_div[i], Uphi_y_div[i]);*/
	}

/* Determine strain and stress color scheme */

/* For strains */

	del_strain.xx =
		(max_strain.xx - min_strain.xx + SMALL)/(double)(boxnumber);
	del_strain.yy =
		(max_strain.yy - min_strain.yy + SMALL)/(double)(boxnumber);
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
		stress_div[i+1].xy = stress_div[i].xy + del_stress.xy;
		stress_div[i+1].zx = stress_div[i].zx + del_stress.zx;
		stress_div[i+1].yz = stress_div[i].yz + del_stress.yz;
		stress_div[i+1].I = stress_div[i].I + del_stress.I;
		stress_div[i+1].II = stress_div[i].II + del_stress.II;
		stress_div[i+1].III = stress_div[i].III + del_stress.III;
		/*printf(" stress div xx %10.5e \n", stress_div[i+1].xx);*/
	}

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
/* Assign colors for Uphi_x */
	       *(U_color + ndof*i + 3) = 0;
	       if(  *(U + ndof*i + 3) > Uphi_x_div[1] )
	       {
		  *(U_color + ndof*i + 3) = 1;
		  if(  *(U + ndof*i + 3) > Uphi_x_div[2] )
		  {
		     *(U_color + ndof*i + 3) = 2;
		     if(  *(U + ndof*i + 3) > Uphi_x_div[3] )
		     {
			*(U_color + ndof*i + 3) = 3;
			if(  *(U + ndof*i + 3) > Uphi_x_div[4] )
			{
			   *(U_color + ndof*i + 3) = 4;
			   if(  *(U + ndof*i + 3) > Uphi_x_div[5] )
			   {
			      *(U_color + ndof*i + 3) = 5;
			      if(  *(U + ndof*i + 3) > Uphi_x_div[6] )
			      {
				 *(U_color + ndof*i + 3) = 6;
				 if(  *(U + ndof*i + 3) > Uphi_x_div[7] )
				 {
				    *(U_color + ndof*i + 3) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uphi_x %d %10.5e %d \n", i,
			*(U+ndof*i + 4), *(U_color+ndof*i + 3));*/
/* Assign colors for Uphi_y */
	       *(U_color + ndof*i + 4) = 0;
	       if(  *(U + ndof*i + 4) > Uphi_y_div[1] )
	       {
		  *(U_color + ndof*i + 4) = 1;
		  if(  *(U + ndof*i + 4) > Uphi_y_div[2] )
		  {
		     *(U_color + ndof*i + 4) = 2;
		     if(  *(U + ndof*i + 4) > Uphi_y_div[3] )
		     {
			*(U_color + ndof*i + 4) = 3;
			if(  *(U + ndof*i + 4) > Uphi_y_div[4] )
			{
			   *(U_color + ndof*i + 4) = 4;
			   if(  *(U + ndof*i + 4) > Uphi_y_div[5] )
			   {
			      *(U_color + ndof*i + 4) = 5;
			      if(  *(U + ndof*i + 4) > Uphi_y_div[6] )
			      {
				 *(U_color + ndof*i + 4) = 6;
				 if(  *(U + ndof*i + 4) > Uphi_y_div[7] )
				 {
				    *(U_color + ndof*i + 4) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uphi_y %d %10.5e %d \n", i,
			*(U+ndof*i + 4), *(U_color+ndof*i + 4));*/
	}

/* Assign colors to strains and stresses */

	num_intb_ = num_intb4;
	npell_ = npell4;
	if(!flag_quad_element)
	{
		npell_ = npell3;
		num_intb_ = num_intb3;
	}

	for( i = 0; i < numel; ++i )
	{
	   for( i4 = 0; i4 < num_ints; ++i4 )
	   {
	     for( j = 0; j < num_intb_; ++j )
	     {

	       node = *(connecter+npell_*i+j) + i4*numnp;
	       i5 = num_intb_*i4+j;

/* Assign colors for strain xx */
	       strain_color[i].pt[i5].xx = 0;
	       if(  strain_node[node].xx > strain_div[1].xx )
	       {
		  strain_color[i].pt[i5].xx = 1;
		  if(  strain_node[node].xx > strain_div[2].xx )
		  {
		     strain_color[i].pt[i5].xx = 2;
		     if(  strain_node[node].xx > strain_div[3].xx )
		     {
			strain_color[i].pt[i5].xx = 3;
			if(  strain_node[node].xx > strain_div[4].xx )
			{
			   strain_color[i].pt[i5].xx = 4;
			   if(  strain_node[node].xx > strain_div[5].xx )
			   {
			      strain_color[i].pt[i5].xx = 5;
			      if(  strain_node[node].xx > strain_div[6].xx )
			      {
				 strain_color[i].pt[i5].xx = 6;
				 if(  strain_node[node].xx > strain_div[7].xx )
				 {
				    strain_color[i].pt[i5].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain xx %d %d %10.5e %d \n", i, j,
			strain_node[node].xx, strain_color[i].pt[i5].xx);*/
/* Assign colors for strain yy */
	       strain_color[i].pt[i5].yy = 0;
	       if(  strain_node[node].yy > strain_div[1].yy )
	       {
		  strain_color[i].pt[i5].yy = 1;
		  if(  strain_node[node].yy > strain_div[2].yy )
		  {
		     strain_color[i].pt[i5].yy = 2;
		     if(  strain_node[node].yy > strain_div[3].yy )
		     {
			strain_color[i].pt[i5].yy = 3;
			if(  strain_node[node].yy > strain_div[4].yy )
			{
			   strain_color[i].pt[i5].yy = 4;
			   if(  strain_node[node].yy > strain_div[5].yy )
			   {
			      strain_color[i].pt[i5].yy = 5;
			      if(  strain_node[node].yy > strain_div[6].yy )
			      {
				 strain_color[i].pt[i5].yy = 6;
				 if(  strain_node[node].yy > strain_div[7].yy )
				 {
				    strain_color[i].pt[i5].yy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain yy %d %d %10.5e %d \n", i, j,
			strain_node[node].yy, strain_color[i].pt[i5].yy);*/
/* Assign colors for strain xy */
	       strain_color[i].pt[i5].xy = 0;
	       if(  strain_node[node].xy > strain_div[1].xy )
	       {
		  strain_color[i].pt[i5].xy = 1;
		  if(  strain_node[node].xy > strain_div[2].xy )
		  {
		     strain_color[i].pt[i5].xy = 2;
		     if(  strain_node[node].xy > strain_div[3].xy )
		     {
			strain_color[i].pt[i5].xy = 3;
			if(  strain_node[node].xy > strain_div[4].xy )
			{
			   strain_color[i].pt[i5].xy = 4;
			   if(  strain_node[node].xy > strain_div[5].xy )
			   {
			      strain_color[i].pt[i5].xy = 5;
			      if(  strain_node[node].xy > strain_div[6].xy )
			      {
				 strain_color[i].pt[i5].xy = 6;
				 if(  strain_node[node].xy > strain_div[7].xy )
				 {
				    strain_color[i].pt[i5].xy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain xy %d %d %10.5e %d \n", i, j,
			strain_node[node].xy, strain_color[i].pt[i5].xy);*/
/* Assign colors for strain zx */
	       strain_color[i].pt[i5].zx = 0;
	       if(  strain_node[node].zx > strain_div[1].zx )
	       {
		  strain_color[i].pt[i5].zx = 1;
		  if(  strain_node[node].zx > strain_div[2].zx )
		  {
		     strain_color[i].pt[i5].zx = 2;
		     if(  strain_node[node].zx > strain_div[3].zx )
		     {
			strain_color[i].pt[i5].zx = 3;
			if(  strain_node[node].zx > strain_div[4].zx )
			{
			   strain_color[i].pt[i5].zx = 4;
			   if(  strain_node[node].zx > strain_div[5].zx )
			   {
			      strain_color[i].pt[i5].zx = 5;
			      if(  strain_node[node].zx > strain_div[6].zx )
			      {
				 strain_color[i].pt[i5].zx = 6;
				 if(  strain_node[node].zx > strain_div[7].zx )
				 {
				    strain_color[i].pt[i5].zx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain zx %d %d %10.5e %d \n", i, j,
			strain_node[node].zx, strain_color[i].pt[i5].zx);*/
/* Assign colors for strain yz */
	       strain_color[i].pt[i5].yz = 0;
	       if(  strain_node[node].yz > strain_div[1].yz )
	       {
		  strain_color[i].pt[i5].yz = 1;
		  if(  strain_node[node].yz > strain_div[2].yz )
		  {
		     strain_color[i].pt[i5].yz = 2;
		     if(  strain_node[node].yz > strain_div[3].yz )
		     {
			strain_color[i].pt[i5].yz = 3;
			if(  strain_node[node].yz > strain_div[4].yz )
			{
			   strain_color[i].pt[i5].yz = 4;
			   if(  strain_node[node].yz > strain_div[5].yz )
			   {
			      strain_color[i].pt[i5].yz = 5;
			      if(  strain_node[node].yz > strain_div[6].yz )
			      {
				 strain_color[i].pt[i5].yz = 6;
				 if(  strain_node[node].yz > strain_div[7].yz )
				 {
				    strain_color[i].pt[i5].yz = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain yz %d %d %10.5e %d \n", i, j,
			strain_node[node].yz, strain_color[i].pt[i5].yz);*/
/* Assign colors for strain I */
	       strain_color[i].pt[i5].I = 0;
	       if(  strain_node[node].I > strain_div[1].I )
	       {
		  strain_color[i].pt[i5].I = 1;
		  if(  strain_node[node].I > strain_div[2].I )
		  {
		     strain_color[i].pt[i5].I = 2;
		     if(  strain_node[node].I > strain_div[3].I )
		     {
			strain_color[i].pt[i5].I = 3;
			if(  strain_node[node].I > strain_div[4].I )
			{
			   strain_color[i].pt[i5].I = 4;
			   if(  strain_node[node].I > strain_div[5].I )
			   {
			      strain_color[i].pt[i5].I = 5;
			      if(  strain_node[node].I > strain_div[6].I )
			      {
				 strain_color[i].pt[i5].I = 6;
				 if(  strain_node[node].I > strain_div[7].I )
				 {
				    strain_color[i].pt[i5].I = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain I %d %d %10.5e %d \n", i, j,
			strain_node[node].I, strain_color[i].pt[i5].I);*/
/* Assign colors for strain II */
	       strain_color[i].pt[i5].II = 0;
	       if(  strain_node[node].II > strain_div[1].II )
	       {
		  strain_color[i].pt[i5].II = 1;
		  if(  strain_node[node].II > strain_div[2].II )
		  {
		     strain_color[i].pt[i5].II = 2;
		     if(  strain_node[node].II > strain_div[3].II )
		     {
			strain_color[i].pt[i5].II = 3;
			if(  strain_node[node].II > strain_div[4].II )
			{
			   strain_color[i].pt[i5].II = 4;
			   if(  strain_node[node].II > strain_div[5].II )
			   {
			      strain_color[i].pt[i5].II = 5;
			      if(  strain_node[node].II > strain_div[6].II )
			      {
				 strain_color[i].pt[i5].II = 6;
				 if(  strain_node[node].II > strain_div[7].II )
				 {
				    strain_color[i].pt[i5].II = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain II %d %d %10.5e %d \n", i, j,
			strain_node[node].II, strain_color[i].pt[i5].II);*/
/* Assign colors for strain III */
	       strain_color[i].pt[i5].III = 0;
	       if(  strain_node[node].III > strain_div[1].III )
	       {
		  strain_color[i].pt[i5].III = 1;
		  if(  strain_node[node].III > strain_div[2].III )
		  {
		     strain_color[i].pt[i5].III = 2;
		     if(  strain_node[node].III > strain_div[3].III )
		     {
			strain_color[i].pt[i5].III = 3;
			if(  strain_node[node].III > strain_div[4].III )
			{
			   strain_color[i].pt[i5].III = 4;
			   if(  strain_node[node].III > strain_div[5].III )
			   {
			      strain_color[i].pt[i5].III = 5;
			      if(  strain_node[node].III > strain_div[6].III )
			      {
				 strain_color[i].pt[i5].III = 6;
				 if(  strain_node[node].III > strain_div[7].III )
				 {
				    strain_color[i].pt[i5].III = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain III %d %d %10.5e %d \n", i, j,
			strain_node[node].III, strain_color[i].pt[i5].III);*/
/* Assign colors for stress xx */
	       stress_color[i].pt[i5].xx = 0;
	       if(  stress_node[node].xx > stress_div[1].xx )
	       {
		  stress_color[i].pt[i5].xx = 1;
		  if(  stress_node[node].xx > stress_div[2].xx )
		  {
		     stress_color[i].pt[i5].xx = 2;
		     if(  stress_node[node].xx > stress_div[3].xx )
		     {
			stress_color[i].pt[i5].xx = 3;
			if(  stress_node[node].xx > stress_div[4].xx )
			{
			   stress_color[i].pt[i5].xx = 4;
			   if(  stress_node[node].xx > stress_div[5].xx )
			   {
			      stress_color[i].pt[i5].xx = 5;
			      if(  stress_node[node].xx > stress_div[6].xx )
			      {
				 stress_color[i].pt[i5].xx = 6;
				 if(  stress_node[node].xx > stress_div[7].xx )
				 {
				    stress_color[i].pt[i5].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress xx %d %d %10.5e %d \n", i, j,
			stress_node[node].xx, stress_color[i].pt[i5].xx);*/
/* Assign colors for stress yy */
	       stress_color[i].pt[i5].yy = 0;
	       if(  stress_node[node].yy > stress_div[1].yy )
	       {
		  stress_color[i].pt[i5].yy = 1;
		  if(  stress_node[node].yy > stress_div[2].yy )
		  {
		     stress_color[i].pt[i5].yy = 2;
		     if(  stress_node[node].yy > stress_div[3].yy )
		     {
			stress_color[i].pt[i5].yy = 3;
			if(  stress_node[node].yy > stress_div[4].yy )
			{
			   stress_color[i].pt[i5].yy = 4;
			   if(  stress_node[node].yy > stress_div[5].yy )
			   {
			      stress_color[i].pt[i5].yy = 5;
			      if(  stress_node[node].yy > stress_div[6].yy )
			      {
				 stress_color[i].pt[i5].yy = 6;
				 if(  stress_node[node].yy > stress_div[7].yy )
				 {
				    stress_color[i].pt[i5].yy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress yy %d %d %10.5e %d \n", i, j,
			stress_node[node].yy, stress_color[i].pt[i5].yy);*/
/* Assign colors for stress xy */
	       stress_color[i].pt[i5].xy = 0;
	       if(  stress_node[node].xy > stress_div[1].xy )
	       {
		  stress_color[i].pt[i5].xy = 1;
		  if(  stress_node[node].xy > stress_div[2].xy )
		  {
		     stress_color[i].pt[i5].xy = 2;
		     if(  stress_node[node].xy > stress_div[3].xy )
		     {
			stress_color[i].pt[i5].xy = 3;
			if(  stress_node[node].xy > stress_div[4].xy )
			{
			   stress_color[i].pt[i5].xy = 4;
			   if(  stress_node[node].xy > stress_div[5].xy )
			   {
			      stress_color[i].pt[i5].xy = 5;
			      if(  stress_node[node].xy > stress_div[6].xy )
			      {
				 stress_color[i].pt[i5].xy = 6;
				 if(  stress_node[node].xy > stress_div[7].xy )
				 {
				    stress_color[i].pt[i5].xy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress xy %d %d %10.5e %d \n", i, j,
			stress_node[node].xy, stress_color[i].pt[i5].xy);*/
/* Assign colors for stress zx */
	       stress_color[i].pt[i5].zx = 0;
	       if(  stress_node[node].zx > stress_div[1].zx )
	       {
		  stress_color[i].pt[i5].zx = 1;
		  if(  stress_node[node].zx > stress_div[2].zx )
		  {
		     stress_color[i].pt[i5].zx = 2;
		     if(  stress_node[node].zx > stress_div[3].zx )
		     {
			stress_color[i].pt[i5].zx = 3;
			if(  stress_node[node].zx > stress_div[4].zx )
			{
			   stress_color[i].pt[i5].zx = 4;
			   if(  stress_node[node].zx > stress_div[5].zx )
			   {
			      stress_color[i].pt[i5].zx = 5;
			      if(  stress_node[node].zx > stress_div[6].zx )
			      {
				 stress_color[i].pt[i5].zx = 6;
				 if(  stress_node[node].zx > stress_div[7].zx )
				 {
				    stress_color[i].pt[i5].zx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress zx %d %d %10.5e %d \n", i, j,
			stress_node[node].zx, stress_color[i].pt[i5].zx);*/
/* Assign colors for stress yz */
	       stress_color[i].pt[i5].yz = 0;
	       if(  stress_node[node].yz > stress_div[1].yz )
	       {
		  stress_color[i].pt[i5].yz = 1;
		  if(  stress_node[node].yz > stress_div[2].yz )
		  {
		     stress_color[i].pt[i5].yz = 2;
		     if(  stress_node[node].yz > stress_div[3].yz )
		     {
			stress_color[i].pt[i5].yz = 3;
			if(  stress_node[node].yz > stress_div[4].yz )
			{
			   stress_color[i].pt[i5].yz = 4;
			   if(  stress_node[node].yz > stress_div[5].yz )
			   {
			      stress_color[i].pt[i5].yz = 5;
			      if(  stress_node[node].yz > stress_div[6].yz )
			      {
				 stress_color[i].pt[i5].yz = 6;
				 if(  stress_node[node].yz > stress_div[7].yz )
				 {
				    stress_color[i].pt[i5].yz = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress yz %d %d %10.5e %d \n", i, j,
			stress_node[node].yz, stress_color[i].pt[i5].yz);*/
/* Assign colors for stress I */
	       stress_color[i].pt[i5].I = 0;
	       if(  stress_node[node].I > stress_div[1].I )
	       {
		  stress_color[i].pt[i5].I = 1;
		  if(  stress_node[node].I > stress_div[2].I )
		  {
		     stress_color[i].pt[i5].I = 2;
		     if(  stress_node[node].I > stress_div[3].I )
		     {
			stress_color[i].pt[i5].I = 3;
			if(  stress_node[node].I > stress_div[4].I )
			{
			   stress_color[i].pt[i5].I = 4;
			   if(  stress_node[node].I > stress_div[5].I )
			   {
			      stress_color[i].pt[i5].I = 5;
			      if(  stress_node[node].I > stress_div[6].I )
			      {
				 stress_color[i].pt[i5].I = 6;
				 if(  stress_node[node].I > stress_div[7].I )
				 {
				    stress_color[i].pt[i5].I = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress I %d %d %10.5e %d \n", i, j,
			stress_node[node].I, stress_color[i].pt[i5].I);*/
/* Assign colors for stress II */
	       stress_color[i].pt[i5].II = 0;
	       if(  stress_node[node].II > stress_div[1].II )
	       {
		  stress_color[i].pt[i5].II = 1;
		  if(  stress_node[node].II > stress_div[2].II )
		  {
		     stress_color[i].pt[i5].II = 2;
		     if(  stress_node[node].II > stress_div[3].II )
		     {
			stress_color[i].pt[i5].II = 3;
			if(  stress_node[node].II > stress_div[4].II )
			{
			   stress_color[i].pt[i5].II = 4;
			   if(  stress_node[node].II > stress_div[5].II )
			   {
			      stress_color[i].pt[i5].II = 5;
			      if(  stress_node[node].II > stress_div[6].II )
			      {
				 stress_color[i].pt[i5].II = 6;
				 if(  stress_node[node].II > stress_div[7].II )
				 {
				    stress_color[i].pt[i5].II = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress II %d %d %10.5e %d \n", i, j,
			stress_node[node].II, stress_color[i].pt[i5].II);*/
/* Assign colors for stress III */
	       stress_color[i].pt[i5].III = 0;
	       if(  stress_node[node].III > stress_div[1].III )
	       {
		  stress_color[i].pt[i5].III = 1;
		  if(  stress_node[node].III > stress_div[2].III )
		  {
		     stress_color[i].pt[i5].III = 2;
		     if(  stress_node[node].III > stress_div[3].III )
		     {
			stress_color[i].pt[i5].III = 3;
			if(  stress_node[node].III > stress_div[4].III )
			{
			   stress_color[i].pt[i5].III = 4;
			   if(  stress_node[node].III > stress_div[5].III )
			   {
			      stress_color[i].pt[i5].III = 5;
			      if(  stress_node[node].III > stress_div[6].III )
			      {
				 stress_color[i].pt[i5].III = 6;
				 if(  stress_node[node].III > stress_div[7].III )
				 {
				    stress_color[i].pt[i5].III = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress III %d %d %10.5e %d \n", i, j,
			stress_node[node].III, stress_color[i].pt[i5].III);*/
	     }
	   }
	}

/* Calculate the graphical force vectors for display */

	for( i = 0; i < bc.num_force[0]; ++i )
	{
		
		force_vec_length =
		   *(force + ndof*bc.force[i])*(*(force + ndof*bc.force[i])) +
		   *(force + ndof*bc.force[i] + 1)*(*(force + ndof*bc.force[i] + 1)) +
		   *(force + ndof*bc.force[i] + 2)*(*(force + ndof*bc.force[i] + 2)) +
		   SMALL;
		moment_vec_length =
		   *(force + ndof*bc.force[i] +3)*(*(force + ndof*bc.force[i] +3)) +
		   *(force + ndof*bc.force[i] +4)*(*(force + ndof*bc.force[i] +4)) +
		   SMALL;
		force_vec_length = sqrt(force_vec_length);
		moment_vec_length = sqrt(moment_vec_length);
		force_vec0[i].x =
		   *(force + ndof*bc.force[i])*3*AxisLength_max/force_vec_length;
		force_vec0[i].y =
		   *(force + ndof*bc.force[i] + 1)*3*AxisLength_max/force_vec_length;
		force_vec0[i].z =
		   *(force + ndof*bc.force[i] + 2)*3*AxisLength_max/force_vec_length;
		force_vec0[i].phix =
		   *(force + ndof*bc.force[i] + 3)*2.0*AxisLength_max/moment_vec_length;
		force_vec0[i].phiy =
		   *(force + ndof*bc.force[i] + 4)*2.0*AxisLength_max/moment_vec_length;
  /*printf(" force %d %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e\n",
		   i, *(force + ndof*bc.force[i]), *(force + ndof*bc.force[i] + 1),
		   *(force + ndof*bc.force[i] + 2), *(force + ndof*bc.force[i] + 3),
		   *(force + ndof*bc.force[i] + 4),
		   force_vec0[i].x, force_vec0[i].y, force_vec0[i].z,
		   force_vec0[i].phix, force_vec0[i].phiy);*/
	}

	return 1;    /* ANSI C requires main to return int. */
}


