/*
    This program sets viewing and analysis values based on the parameters 
    determined in plparameters for the FEM GUI for plate elements.
  
   			Last Update 9/24/06

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
#include "../plate/plconst.h"
#include "../plate/plstruct.h"
#include "plstrcgr.h"
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
	Uphi_x_div[boxnumber+1], Uphi_y_div[boxnumber+1], Uphi_z_div[boxnumber+1];
extern MDIM moment_div[boxnumber+1];
extern MDIM curve_div[boxnumber+1];
extern SDIM stress_div[boxnumber+1];
extern SDIM strain_div[boxnumber+1];
extern double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
extern MDIM del_moment, del_curve, max_moment, min_moment,
	max_curve, min_curve;
extern SDIM del_stress, del_strain, max_stress, min_stress,
	max_strain, min_strain;
extern double max_Uphi_x, min_Uphi_x, del_Uphi_x, max_Uphi_y, min_Uphi_y, del_Uphi_y,
	max_Uphi_z, min_Uphi_z, del_Uphi_z,
	max_Ux, min_Ux, del_Ux, max_Uy, min_Uy, del_Uy,
	max_Uz, min_Uz, del_Uz, absolute_max_U;

int plset( BOUND bc, int *connecter, MDIM *curve_node, ICURVATURE *curve_color,
	double *force, XYZPhiF *force_vec0, MDIM *moment_node, IMOMENT *moment_color,
	SDIM *strain_node, ISTRAIN *strain_color, SDIM *stress_node,
	ISTRESS *stress_color, double *U, int *U_color )
{
	int i, j, check;
	int node, npel_dum, num_int_dum;

	double force_vec_length, moment_vec_length;

/* Determine displacement and angle color scheme */

	del_Ux = (max_Ux - min_Ux + SMALL)/(double)(boxnumber);
	del_Uy = (max_Uy - min_Uy + SMALL)/(double)(boxnumber);
	del_Uz = (max_Uz - min_Uz + SMALL)/(double)(boxnumber);
	del_Uphi_x = (max_Uphi_x - min_Uphi_x + SMALL)/(double)(boxnumber);
	del_Uphi_y = (max_Uphi_y - min_Uphi_y + SMALL)/(double)(boxnumber);
	del_Uphi_z = (max_Uphi_z - min_Uphi_z + SMALL)/(double)(boxnumber);
	Ux_div[0] = min_Ux;
	Uy_div[0] = min_Uy;
	Uz_div[0] = min_Uz;
	Uphi_x_div[0] = min_Uphi_x;
	Uphi_y_div[0] = min_Uphi_y;
	Uphi_z_div[0] = min_Uphi_z;

	for( i = 0; i < boxnumber; ++i )
	{
		Ux_div[i+1] = Ux_div[i] + del_Ux;
		Uy_div[i+1] = Uy_div[i] + del_Uy;
		Uz_div[i+1] = Uz_div[i] + del_Uz;
		Uphi_x_div[i+1] = Uphi_x_div[i] + del_Uphi_x;
		Uphi_y_div[i+1] = Uphi_y_div[i] + del_Uphi_y;
		Uphi_z_div[i+1] = Uphi_z_div[i] + del_Uphi_z;
		/*printf(" U div x y z %10.5e %10.5e %10.5e\n",
			Uphi_x_div[i], Uphi_y_div[i], Uphi_z_div[i]);*/
	}

/* Determine color schemes */

/* For curvatures and strains */

	del_curve.xx =
		(max_curve.xx - min_curve.xx + SMALL)/(double)(boxnumber);
	del_curve.yy =
		(max_curve.yy - min_curve.yy + SMALL)/(double)(boxnumber);
	del_curve.xy =
		(max_curve.xy - min_curve.xy + SMALL)/(double)(boxnumber);
	del_curve.I =
		(max_curve.I - min_curve.I + SMALL)/(double)(boxnumber);
	del_curve.II =
		(max_curve.II - min_curve.II + SMALL)/(double)(boxnumber);
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
	curve_div[0].xx = min_curve.xx;
	curve_div[0].yy = min_curve.yy;
	curve_div[0].xy = min_curve.xy;
	curve_div[0].I = min_curve.I;
	curve_div[0].II = min_curve.II;
	strain_div[0].xx = min_strain.xx;
	strain_div[0].yy = min_strain.yy;
	strain_div[0].xy = min_strain.xy;
	strain_div[0].zx = min_strain.zx;
	strain_div[0].yz = min_strain.yz;
	strain_div[0].I = min_strain.I;
	strain_div[0].II = min_strain.II;
	strain_div[0].III = min_strain.III;
	/*printf(" max min curve xx %10.5e %10.5e \n", max_curve.xx, min_curve.xx);
	printf(" curve div xx %10.5e \n", curve_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		curve_div[i+1].xx = curve_div[i].xx + del_curve.xx;
		curve_div[i+1].yy = curve_div[i].yy + del_curve.yy;
		curve_div[i+1].xy = curve_div[i].xy + del_curve.xy;
		curve_div[i+1].I = curve_div[i].I + del_curve.I;
		curve_div[i+1].II = curve_div[i].II + del_curve.II;
		strain_div[i+1].xx = strain_div[i].xx + del_strain.xx;
		strain_div[i+1].yy = strain_div[i].yy + del_strain.yy;
		strain_div[i+1].xy = strain_div[i].xy + del_strain.xy;
		strain_div[i+1].zx = strain_div[i].zx + del_strain.zx;
		strain_div[i+1].yz = strain_div[i].yz + del_strain.yz;
		strain_div[i+1].I = strain_div[i].I + del_strain.I;
		strain_div[i+1].II = strain_div[i].II + del_strain.II;
		strain_div[i+1].III = strain_div[i].III + del_strain.III;
		/*printf(" curve div xx %10.5e \n", curve_div[i+1].xx);*/
	}


/* For moments and stresses */

	del_moment.xx =
		(max_moment.xx - min_moment.xx + SMALL)/(double)(boxnumber);
	del_moment.yy =
		(max_moment.yy - min_moment.yy + SMALL)/(double)(boxnumber);
	del_moment.xy =
		(max_moment.xy - min_moment.xy + SMALL)/(double)(boxnumber);
	del_moment.I =
		(max_moment.I - min_moment.I + SMALL)/(double)(boxnumber);
	del_moment.II =
		(max_moment.II - min_moment.II + SMALL)/(double)(boxnumber);
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
	moment_div[0].xx = min_moment.xx;
	moment_div[0].yy = min_moment.yy;
	moment_div[0].xy = min_moment.xy;
	moment_div[0].I = min_moment.I;
	moment_div[0].II = min_moment.II;
	stress_div[0].xx = min_stress.xx;
	stress_div[0].yy = min_stress.yy;
	stress_div[0].xy = min_stress.xy;
	stress_div[0].zx = min_stress.zx;
	stress_div[0].yz = min_stress.yz;
	stress_div[0].I = min_stress.I;
	stress_div[0].II = min_stress.II;
	stress_div[0].III = min_stress.III;
	/*printf(" max min moment xx %10.5e %10.5e \n", max_moment.xx, min_moment.xx);
	printf(" moment div xx %10.5e \n", moment_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		moment_div[i+1].xx = moment_div[i].xx + del_moment.xx;
		moment_div[i+1].yy = moment_div[i].yy + del_moment.yy;
		moment_div[i+1].xy = moment_div[i].xy + del_moment.xy;
		moment_div[i+1].I = moment_div[i].I + del_moment.I;
		moment_div[i+1].II = moment_div[i].II + del_moment.II;
		stress_div[i+1].xx = stress_div[i].xx + del_stress.xx;
		stress_div[i+1].yy = stress_div[i].yy + del_stress.yy;
		stress_div[i+1].xy = stress_div[i].xy + del_stress.xy;
		stress_div[i+1].zx = stress_div[i].zx + del_stress.zx;
		stress_div[i+1].yz = stress_div[i].yz + del_stress.yz;
		stress_div[i+1].I = stress_div[i].I + del_stress.I;
		stress_div[i+1].II = stress_div[i].II + del_stress.II;
		stress_div[i+1].III = stress_div[i].III + del_stress.III;
		/*printf(" moment div xx %10.5e \n", moment_div[i+1].xx);*/
	}

/* Assign Colors for displacement, angle, curvature, strains, moments, and stresses */

	for( i = 0; i < numnp; ++i )
	{
/* Assign colors for Ux */
	       *(U_color + ndof6*i) = 0;
	       if(  *(U + ndof6*i) > Ux_div[1] )
	       {
		  *(U_color + ndof6*i) = 1;
		  if(  *(U + ndof6*i) > Ux_div[2] )
		  {
		     *(U_color + ndof6*i) = 2;
		     if(  *(U + ndof6*i) > Ux_div[3] )
		     {
			*(U_color + ndof6*i) = 3;
			if(  *(U + ndof6*i) > Ux_div[4] )
			{
			   *(U_color + ndof6*i) = 4;
			   if(  *(U + ndof6*i) > Ux_div[5] )
			   {
			      *(U_color + ndof6*i) = 5;
			      if(  *(U + ndof6*i) > Ux_div[6] )
			      {
				 *(U_color + ndof6*i) = 6;
				 if(  *(U + ndof6*i) > Ux_div[7] )
				 {
				    *(U_color + ndof6*i) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Ux %d %10.5e %d \n", i,
			*(U+ndof6*i), *(U_color+ndof6*i));*/
/* Assign colors for Uy */
	       *(U_color + ndof6*i + 1) = 0;
	       if(  *(U + ndof6*i + 1) > Uy_div[1] )
	       {
		  *(U_color + ndof6*i + 1) = 1;
		  if(  *(U + ndof6*i + 1) > Uy_div[2] )
		  {
		     *(U_color + ndof6*i + 1) = 2;
		     if(  *(U + ndof6*i + 1) > Uy_div[3] )
		     {
			*(U_color + ndof6*i + 1) = 3;
			if(  *(U + ndof6*i + 1) > Uy_div[4] )
			{
			   *(U_color + ndof6*i + 1) = 4;
			   if(  *(U + ndof6*i + 1) > Uy_div[5] )
			   {
			      *(U_color + ndof6*i + 1) = 5;
			      if(  *(U + ndof6*i + 1) > Uy_div[6] )
			      {
				 *(U_color + ndof6*i + 1) = 6;
				 if(  *(U + ndof6*i + 1) > Uy_div[7] )
				 {
				    *(U_color + ndof6*i + 1) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uy %d %10.5e %d \n", i,
			*(U+ndof6*i + 1), *(U_color+ndof6*i + 1));*/
/* Assign colors for Uz */
	       *(U_color + ndof6*i + 2) = 0;
	       if(  *(U + ndof6*i + 2) > Uz_div[1] )
	       {
		  *(U_color + ndof6*i + 2) = 1;
		  if(  *(U + ndof6*i + 2) > Uz_div[2] )
		  {
		     *(U_color + ndof6*i + 2) = 2;
		     if(  *(U + ndof6*i + 2) > Uz_div[3] )
		     {
			*(U_color + ndof6*i + 2) = 3;
			if(  *(U + ndof6*i + 2) > Uz_div[4] )
			{
			   *(U_color + ndof6*i + 2) = 4;
			   if(  *(U + ndof6*i + 2) > Uz_div[5] )
			   {
			      *(U_color + ndof6*i + 2) = 5;
			      if(  *(U + ndof6*i + 2) > Uz_div[6] )
			      {
				 *(U_color + ndof6*i + 2) = 6;
				 if(  *(U + ndof6*i + 2) > Uz_div[7] )
				 {
				    *(U_color + ndof6*i + 2) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uz %d %10.5e %d \n", i,
			*(U+ndof6*i + 2), *(U_color+ndof6*i + 2)) + 2;*/
/* Assign colors for Uphi_x */
	       *(U_color + ndof6*i + 3) = 0;
	       if(  *(U + ndof6*i + 3) > Uphi_x_div[1] )
	       {
		  *(U_color + ndof6*i + 3) = 1;
		  if(  *(U + ndof6*i + 3) > Uphi_x_div[2] )
		  {
		     *(U_color + ndof6*i + 3) = 2;
		     if(  *(U + ndof6*i + 3) > Uphi_x_div[3] )
		     {
			*(U_color + ndof6*i + 3) = 3;
			if(  *(U + ndof6*i + 3) > Uphi_x_div[4] )
			{
			   *(U_color + ndof6*i + 3) = 4;
			   if(  *(U + ndof6*i + 3) > Uphi_x_div[5] )
			   {
			      *(U_color + ndof6*i + 3) = 5;
			      if(  *(U + ndof6*i + 3) > Uphi_x_div[6] )
			      {
				 *(U_color + ndof6*i + 3) = 6;
				 if(  *(U + ndof6*i + 3) > Uphi_x_div[7] )
				 {
				    *(U_color + ndof6*i + 3) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uphi_x %d %10.5e %d \n", i,
			*(U+ndof6*i + 3), *(U_color+ndof6*i + 3));*/
/* Assign colors for Uphi_y */
	       *(U_color + ndof6*i + 4) = 0;
	       if(  *(U + ndof6*i + 4) > Uphi_y_div[1] )
	       {
		  *(U_color + ndof6*i + 4) = 1;
		  if(  *(U + ndof6*i + 4) > Uphi_y_div[2] )
		  {
		     *(U_color + ndof6*i + 4) = 2;
		     if(  *(U + ndof6*i + 4) > Uphi_y_div[3] )
		     {
			*(U_color + ndof6*i + 4) = 3;
			if(  *(U + ndof6*i + 4) > Uphi_y_div[4] )
			{
			   *(U_color + ndof6*i + 4) = 4;
			   if(  *(U + ndof6*i + 4) > Uphi_y_div[5] )
			   {
			      *(U_color + ndof6*i + 4) = 5;
			      if(  *(U + ndof6*i + 4) > Uphi_y_div[6] )
			      {
				 *(U_color + ndof6*i + 4) = 6;
				 if(  *(U + ndof6*i + 4) > Uphi_y_div[7] )
				 {
				    *(U_color + ndof6*i + 4) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uphi_y %d %10.5e %d \n", i,
			*(U+ndof6*i + 4), *(U_color+ndof6*i + 4));*/
/* Assign colors for Uphi_z */
	       *(U_color + ndof6*i + 5) = 0;
	       if(  *(U + ndof6*i + 5) > Uphi_z_div[1] )
	       {
		  *(U_color + ndof6*i + 5) = 1;
		  if(  *(U + ndof6*i + 5) > Uphi_z_div[2] )
		  {
		     *(U_color + ndof6*i + 5) = 2;
		     if(  *(U + ndof6*i + 5) > Uphi_z_div[3] )
		     {
			*(U_color + ndof6*i + 5) = 3;
			if(  *(U + ndof6*i + 5) > Uphi_z_div[4] )
			{
			   *(U_color + ndof6*i + 5) = 4;
			   if(  *(U + ndof6*i + 5) > Uphi_z_div[5] )
			   {
			      *(U_color + ndof6*i + 5) = 5;
			      if(  *(U + ndof6*i + 5) > Uphi_z_div[6] )
			      {
				 *(U_color + ndof6*i + 5) = 6;
				 if(  *(U + ndof6*i + 5) > Uphi_z_div[7] )
				 {
				    *(U_color + ndof6*i + 5) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uphi_z %d %10.5e %d \n", i,
			*(U+ndof6*i + 5), *(U_color+ndof6*i + 5));*/
	}

/* Assign colors to curves, strains, moments and stresses */

	num_int_dum = num_int4;
	npel_dum = npel4;
	if(!flag_quad_element)
	{
		npel_dum = npel3;
		num_int_dum = num_int3;
	}

	for( i = 0; i < numel; ++i )
	{
	    for( j = 0; j < num_int_dum; ++j )
	    {

	       node = *(connecter+npel_dum*i+j);

/* Assign colors for curve xx */
	       curve_color[i].pt[j].xx = 0;
	       if(  curve_node[node].xx > curve_div[1].xx )
	       {
		  curve_color[i].pt[j].xx = 1;
		  if(  curve_node[node].xx > curve_div[2].xx )
		  {
		     curve_color[i].pt[j].xx = 2;
		     if(  curve_node[node].xx > curve_div[3].xx )
		     {
			curve_color[i].pt[j].xx = 3;
			if(  curve_node[node].xx > curve_div[4].xx )
			{
			   curve_color[i].pt[j].xx = 4;
			   if(  curve_node[node].xx > curve_div[5].xx )
			   {
			      curve_color[i].pt[j].xx = 5;
			      if(  curve_node[node].xx > curve_div[6].xx )
			      {
				 curve_color[i].pt[j].xx = 6;
				 if(  curve_node[node].xx > curve_div[7].xx )
				 {
				    curve_color[i].pt[j].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" curve xx %d %d %10.5e %d \n", i, j,
			curve_node[node].xx, curve_color[i].pt[j].xx);*/
/* Assign colors for curve yy */
	       curve_color[i].pt[j].yy = 0;
	       if(  curve_node[node].yy > curve_div[1].yy )
	       {
		  curve_color[i].pt[j].yy = 1;
		  if(  curve_node[node].yy > curve_div[2].yy )
		  {
		     curve_color[i].pt[j].yy = 2;
		     if(  curve_node[node].yy > curve_div[3].yy )
		     {
			curve_color[i].pt[j].yy = 3;
			if(  curve_node[node].yy > curve_div[4].yy )
			{
			   curve_color[i].pt[j].yy = 4;
			   if(  curve_node[node].yy > curve_div[5].yy )
			   {
			      curve_color[i].pt[j].yy = 5;
			      if(  curve_node[node].yy > curve_div[6].yy )
			      {
				 curve_color[i].pt[j].yy = 6;
				 if(  curve_node[node].yy > curve_div[7].yy )
				 {
				    curve_color[i].pt[j].yy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" curve yy %d %d %10.5e %d \n", i, j,
			curve_node[node].yy, curve_color[i].pt[j].yy);*/
/* Assign colors for curve xy */
	       curve_color[i].pt[j].xy = 0;
	       if(  curve_node[node].xy > curve_div[1].xy )
	       {
		  curve_color[i].pt[j].xy = 1;
		  if(  curve_node[node].xy > curve_div[2].xy )
		  {
		     curve_color[i].pt[j].xy = 2;
		     if(  curve_node[node].xy > curve_div[3].xy )
		     {
			curve_color[i].pt[j].xy = 3;
			if(  curve_node[node].xy > curve_div[4].xy )
			{
			   curve_color[i].pt[j].xy = 4;
			   if(  curve_node[node].xy > curve_div[5].xy )
			   {
			      curve_color[i].pt[j].xy = 5;
			      if(  curve_node[node].xy > curve_div[6].xy )
			      {
				 curve_color[i].pt[j].xy = 6;
				 if(  curve_node[node].xy > curve_div[7].xy )
				 {
				    curve_color[i].pt[j].xy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" curve xy %d %d %10.5e %d \n", i, j,
			curve_node[node].xy, curve_color[i].pt[j].xy);*/
/* Assign colors for curve I */
	       curve_color[i].pt[j].I = 0;
	       if(  curve_node[node].I > curve_div[1].I )
	       {
		  curve_color[i].pt[j].I = 1;
		  if(  curve_node[node].I > curve_div[2].I )
		  {
		     curve_color[i].pt[j].I = 2;
		     if(  curve_node[node].I > curve_div[3].I )
		     {
			curve_color[i].pt[j].I = 3;
			if(  curve_node[node].I > curve_div[4].I )
			{
			   curve_color[i].pt[j].I = 4;
			   if(  curve_node[node].I > curve_div[5].I )
			   {
			      curve_color[i].pt[j].I = 5;
			      if(  curve_node[node].I > curve_div[6].I )
			      {
				 curve_color[i].pt[j].I = 6;
				 if(  curve_node[node].I > curve_div[7].I )
				 {
				    curve_color[i].pt[j].I = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" curve I %d %d %10.5e %d \n", i, j,
			curve_node[node].I, curve_color[i].pt[j].I);*/
/* Assign colors for curve II */
	       curve_color[i].pt[j].II = 0;
	       if(  curve_node[node].II > curve_div[1].II )
	       {
		  curve_color[i].pt[j].II = 1;
		  if(  curve_node[node].II > curve_div[2].II )
		  {
		     curve_color[i].pt[j].II = 2;
		     if(  curve_node[node].II > curve_div[3].II )
		     {
			curve_color[i].pt[j].II = 3;
			if(  curve_node[node].II > curve_div[4].II )
			{
			   curve_color[i].pt[j].II = 4;
			   if(  curve_node[node].II > curve_div[5].II )
			   {
			      curve_color[i].pt[j].II = 5;
			      if(  curve_node[node].II > curve_div[6].II )
			      {
				 curve_color[i].pt[j].II = 6;
				 if(  curve_node[node].II > curve_div[7].II )
				 {
				    curve_color[i].pt[j].II = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" curve II %d %d %10.5e %d \n", i, j,
			curve_node[node].II, curve_color[i].pt[j].II);*/
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
/* Assign colors for moment xx */
	       moment_color[i].pt[j].xx = 0;
	       if(  moment_node[node].xx > moment_div[1].xx )
	       {
		  moment_color[i].pt[j].xx = 1;
		  if(  moment_node[node].xx > moment_div[2].xx )
		  {
		     moment_color[i].pt[j].xx = 2;
		     if(  moment_node[node].xx > moment_div[3].xx )
		     {
			moment_color[i].pt[j].xx = 3;
			if(  moment_node[node].xx > moment_div[4].xx )
			{
			   moment_color[i].pt[j].xx = 4;
			   if(  moment_node[node].xx > moment_div[5].xx )
			   {
			      moment_color[i].pt[j].xx = 5;
			      if(  moment_node[node].xx > moment_div[6].xx )
			      {
				 moment_color[i].pt[j].xx = 6;
				 if(  moment_node[node].xx > moment_div[7].xx )
				 {
				    moment_color[i].pt[j].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" moment xx %d %d %10.5e %d \n", i, j,
			moment_node[node].xx, moment_color[i].pt[j].xx);*/
/* Assign colors for moment yy */
	       moment_color[i].pt[j].yy = 0;
	       if(  moment_node[node].yy > moment_div[1].yy )
	       {
		  moment_color[i].pt[j].yy = 1;
		  if(  moment_node[node].yy > moment_div[2].yy )
		  {
		     moment_color[i].pt[j].yy = 2;
		     if(  moment_node[node].yy > moment_div[3].yy )
		     {
			moment_color[i].pt[j].yy = 3;
			if(  moment_node[node].yy > moment_div[4].yy )
			{
			   moment_color[i].pt[j].yy = 4;
			   if(  moment_node[node].yy > moment_div[5].yy )
			   {
			      moment_color[i].pt[j].yy = 5;
			      if(  moment_node[node].yy > moment_div[6].yy )
			      {
				 moment_color[i].pt[j].yy = 6;
				 if(  moment_node[node].yy > moment_div[7].yy )
				 {
				    moment_color[i].pt[j].yy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" moment yy %d %d %10.5e %d \n", i, j,
			moment_node[node].yy, moment_color[i].pt[j].yy);*/
/* Assign colors for moment xy */
	       moment_color[i].pt[j].xy = 0;
	       if(  moment_node[node].xy > moment_div[1].xy )
	       {
		  moment_color[i].pt[j].xy = 1;
		  if(  moment_node[node].xy > moment_div[2].xy )
		  {
		     moment_color[i].pt[j].xy = 2;
		     if(  moment_node[node].xy > moment_div[3].xy )
		     {
			moment_color[i].pt[j].xy = 3;
			if(  moment_node[node].xy > moment_div[4].xy )
			{
			   moment_color[i].pt[j].xy = 4;
			   if(  moment_node[node].xy > moment_div[5].xy )
			   {
			      moment_color[i].pt[j].xy = 5;
			      if(  moment_node[node].xy > moment_div[6].xy )
			      {
				 moment_color[i].pt[j].xy = 6;
				 if(  moment_node[node].xy > moment_div[7].xy )
				 {
				    moment_color[i].pt[j].xy = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" moment xy %d %d %10.5e %d \n", i, j,
			moment_node[node].xy, moment_color[i].pt[j].xy);*/
/* Assign colors for moment I */
	       moment_color[i].pt[j].I = 0;
	       if(  moment_node[node].I > moment_div[1].I )
	       {
		  moment_color[i].pt[j].I = 1;
		  if(  moment_node[node].I > moment_div[2].I )
		  {
		     moment_color[i].pt[j].I = 2;
		     if(  moment_node[node].I > moment_div[3].I )
		     {
			moment_color[i].pt[j].I = 3;
			if(  moment_node[node].I > moment_div[4].I )
			{
			   moment_color[i].pt[j].I = 4;
			   if(  moment_node[node].I > moment_div[5].I )
			   {
			      moment_color[i].pt[j].I = 5;
			      if(  moment_node[node].I > moment_div[6].I )
			      {
				 moment_color[i].pt[j].I = 6;
				 if(  moment_node[node].I > moment_div[7].I )
				 {
				    moment_color[i].pt[j].I = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" moment I %d %d %10.5e %d \n", i, j,
			moment_node[node].I, moment_color[i].pt[j].I);*/
/* Assign colors for moment II */
	       moment_color[i].pt[j].II = 0;
	       if(  moment_node[node].II > moment_div[1].II )
	       {
		  moment_color[i].pt[j].II = 1;
		  if(  moment_node[node].II > moment_div[2].II )
		  {
		     moment_color[i].pt[j].II = 2;
		     if(  moment_node[node].II > moment_div[3].II )
		     {
			moment_color[i].pt[j].II = 3;
			if(  moment_node[node].II > moment_div[4].II )
			{
			   moment_color[i].pt[j].II = 4;
			   if(  moment_node[node].II > moment_div[5].II )
			   {
			      moment_color[i].pt[j].II = 5;
			      if(  moment_node[node].II > moment_div[6].II )
			      {
				 moment_color[i].pt[j].II = 6;
				 if(  moment_node[node].II > moment_div[7].II )
				 {
				    moment_color[i].pt[j].II = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" moment II %d %d %10.5e %d \n", i, j,
			moment_node[node].II, moment_color[i].pt[j].II);*/
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
		   *(force + ndof6*bc.force[i])*(*(force + ndof6*bc.force[i])) +
		   *(force + ndof6*bc.force[i] + 1)*(*(force + ndof6*bc.force[i] + 1)) +
		   *(force + ndof6*bc.force[i] + 2)*(*(force + ndof6*bc.force[i] + 2)) +
		   SMALL;
		force_vec_length = sqrt(force_vec_length);
		force_vec0[i].x =
		   *(force + ndof6*bc.force[i])*3*AxisLength_max/force_vec_length;
		force_vec0[i].y =
		   *(force + ndof6*bc.force[i] + 1)*3*AxisLength_max/force_vec_length;
		force_vec0[i].z =
		   *(force + ndof6*bc.force[i] + 2)*3*AxisLength_max/force_vec_length;

/* Calculate the moment part of force_vec */

		moment_vec_length =
		   *(force + ndof6*bc.force[i] + 3)*(*(force + ndof6*bc.force[i] +3)) +
		   *(force + ndof6*bc.force[i] + 4)*(*(force + ndof6*bc.force[i] +4)) +
		   *(force + ndof6*bc.force[i] + 5)*(*(force + ndof6*bc.force[i] +5));
		   SMALL;

		moment_vec_length = sqrt(moment_vec_length);
		force_vec0[i].phix =
		   *(force + ndof6*bc.force[i] + 3)*2.0*AxisLength_max/moment_vec_length;
		force_vec0[i].phiy =
		   *(force + ndof6*bc.force[i] + 4)*2.0*AxisLength_max/moment_vec_length;
		force_vec0[i].phiz =
		   *(force + ndof6*bc.force[i] + 5)*2.0*AxisLength_max/moment_vec_length;

 		/*printf(" force %d %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e\n",
		   i, *(force + ndof*bc.force[i]), *(force + ndof*bc.force[i] + 1),
		   *(force + ndof*bc.force[i] + 2), *(force + ndof*bc.force[i] + 3),
		   *(force + ndof*bc.force[i] + 5),
		   force_vec0[i].x, force_vec0[i].y,
		   force_vec0[i].phix, force_vec0[i].phiy, force_vec0[i].phiz);*/
	}

	return 1;    /* ANSI C requires main to return int. */
}

