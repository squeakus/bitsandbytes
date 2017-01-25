/*
    This program sets viewing and analysis values based on the parameters 
    determined in bmparameters for the FEM GUI for beam elements.
  
   			Last Update 1/20/06

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
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"
#include "bmstrcgr.h"
#include "../../common_gr/control.h"

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

int bmset( BOUND bc, CURVATURE *curve, ICURVATURE *curve_color, double *dist_load,
	QYQZ *dist_load_vec0, int *el_type, double *force , XYZPhiF *force_vec0,
	MOMENT *moment, IMOMENT *moment_color, STRAIN *strain, ISTRAIN *strain_color,
	STRESS *stress, ISTRESS *stress_color, double *U, int *U_color )
{
	int i, j, check; 
	double force_vec_length, moment_vec_length, dist_load_vec_length;
	int el_num, type_num;

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
	del_curve.zz =
		(max_curve.zz - min_curve.zz + SMALL)/(double)(boxnumber);
	del_strain.xx =
		(max_strain.xx - min_strain.xx + SMALL)/(double)(boxnumber);
	del_strain.xy =
		(max_strain.xy - min_strain.xy + SMALL)/(double)(boxnumber);
	del_strain.zx =
		(max_strain.zx - min_strain.zx + SMALL)/(double)(boxnumber);
	curve_div[0].xx = min_curve.xx;
	curve_div[0].yy = min_curve.yy;
	curve_div[0].zz = min_curve.zz;
	strain_div[0].xx = min_strain.xx;
	strain_div[0].xy = min_strain.xy;
	strain_div[0].zx = min_strain.zx;
	/*printf(" max min curve xx %10.5e %10.5e \n", max_curve.xx, min_curve.xx);
	printf(" curve div xx %10.5e \n", curve_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		curve_div[i+1].xx = curve_div[i].xx + del_curve.xx;
		curve_div[i+1].yy = curve_div[i].yy + del_curve.yy;
		curve_div[i+1].zz = curve_div[i].zz + del_curve.zz;
		strain_div[i+1].xx = strain_div[i].xx + del_strain.xx;
		strain_div[i+1].xy = strain_div[i].xy + del_strain.xy;
		strain_div[i+1].zx = strain_div[i].zx + del_strain.zx;
		/*printf(" curve div xx %10.5e \n", curve_div[i+1].xx);*/
	}

/* For moments and stresses */

	del_moment.xx =
		(max_moment.xx - min_moment.xx + SMALL)/(double)(boxnumber);
	del_moment.yy =
		(max_moment.yy - min_moment.yy + SMALL)/(double)(boxnumber);
	del_moment.zz =
		(max_moment.zz - min_moment.zz + SMALL)/(double)(boxnumber);
	del_stress.xx =
		(max_stress.xx - min_stress.xx + SMALL)/(double)(boxnumber);
	del_stress.xy =
		(max_stress.xy - min_stress.xy + SMALL)/(double)(boxnumber);
	del_stress.zx =
		(max_stress.zx - min_stress.zx + SMALL)/(double)(boxnumber);
	moment_div[0].xx = min_moment.xx;
	moment_div[0].yy = min_moment.yy;
	moment_div[0].zz = min_moment.zz;
	stress_div[0].xx = min_stress.xx;
	stress_div[0].xy = min_stress.xy;
	stress_div[0].zx = min_stress.zx;
	/*printf(" max min moment xx %10.5e %10.5e \n", max_moment.xx, min_moment.xx);
	printf(" moment div xx %10.5e \n", moment_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		moment_div[i+1].xx = moment_div[i].xx + del_moment.xx;
		moment_div[i+1].yy = moment_div[i].yy + del_moment.yy;
		moment_div[i+1].zz = moment_div[i].zz + del_moment.zz;
		stress_div[i+1].xx = stress_div[i].xx + del_stress.xx;
		stress_div[i+1].xy = stress_div[i].xy + del_stress.xy;
		stress_div[i+1].zx = stress_div[i].zx + del_stress.zx;
		/*printf(" moment div xx %10.5e \n", moment_div[i+1].xx);*/
	}

/* Assign Colors for displacement, angle, curvature, strains, moments, and stresses */

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
/* Assign colors for Uphi_z */
	       *(U_color + ndof*i + 5) = 0;
	       if(  *(U + ndof*i + 5) > Uphi_z_div[1] )
	       {
		  *(U_color + ndof*i + 5) = 1;
		  if(  *(U + ndof*i + 5) > Uphi_z_div[2] )
		  {
		     *(U_color + ndof*i + 5) = 2;
		     if(  *(U + ndof*i + 5) > Uphi_z_div[3] )
		     {
			*(U_color + ndof*i + 5) = 3;
			if(  *(U + ndof*i + 5) > Uphi_z_div[4] )
			{
			   *(U_color + ndof*i + 5) = 4;
			   if(  *(U + ndof*i + 5) > Uphi_z_div[5] )
			   {
			      *(U_color + ndof*i + 5) = 5;
			      if(  *(U + ndof*i + 5) > Uphi_z_div[6] )
			      {
				 *(U_color + ndof*i + 5) = 6;
				 if(  *(U + ndof*i + 5) > Uphi_z_div[7] )
				 {
				    *(U_color + ndof*i + 5) = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" Uphi_z %d %10.5e %d \n", i,
			*(U+ndof*i + 5), *(U_color+ndof*i + 5));*/
	}
/* Assign colors to curves, strains, moments and stresses.

   Trusses and hinged elements do not have curvature or moments
   and trusses do not have shear.  This means that:

   if el_type == 1, 5, 7 only assign color for stress.xx/strain.xx
   if el_type == 4, only assign color for stress.xx/strain.xx
		    stress.xy/strain.xy, and stress.zx/strain.zx

*/

	for( i = 0; i < numel; ++i )
	{
	    type_num = *(el_type + i);
	    for( j = 0; j < num_int; ++j )
	    {
	       if( type_num != 1 && type_num != 4 && type_num != 5 && type_num != 7 )
	       {
/* Assign colors for curve xx */
		   curve_color[i].pt[j].xx = 0;
		   if(  curve[i].pt[j].xx > curve_div[1].xx )
		   {
		      curve_color[i].pt[j].xx = 1;
		      if(  curve[i].pt[j].xx > curve_div[2].xx )
		      {
			 curve_color[i].pt[j].xx = 2;
			 if(  curve[i].pt[j].xx > curve_div[3].xx )
			 {
			    curve_color[i].pt[j].xx = 3;
			    if(  curve[i].pt[j].xx > curve_div[4].xx )
			    {
			       curve_color[i].pt[j].xx = 4;
			       if(  curve[i].pt[j].xx > curve_div[5].xx )
			       {
				  curve_color[i].pt[j].xx = 5;
				  if(  curve[i].pt[j].xx > curve_div[6].xx )
				  {
				     curve_color[i].pt[j].xx = 6;
				     if(  curve[i].pt[j].xx > curve_div[7].xx )
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
			    curve[i].pt[j].xx, curve_color[i].pt[j].xx);*/
/* Assign colors for curve yy */
		   curve_color[i].pt[j].yy = 0;
		   if(  curve[i].pt[j].yy > curve_div[1].yy )
		   {
		      curve_color[i].pt[j].yy = 1;
		      if(  curve[i].pt[j].yy > curve_div[2].yy )
		      {
			 curve_color[i].pt[j].yy = 2;
			 if(  curve[i].pt[j].yy > curve_div[3].yy )
			 {
			    curve_color[i].pt[j].yy = 3;
			    if(  curve[i].pt[j].yy > curve_div[4].yy )
			    {
			       curve_color[i].pt[j].yy = 4;
			       if(  curve[i].pt[j].yy > curve_div[5].yy )
			       {
				  curve_color[i].pt[j].yy = 5;
				  if(  curve[i].pt[j].yy > curve_div[6].yy )
				  {
				     curve_color[i].pt[j].yy = 6;
				     if(  curve[i].pt[j].yy > curve_div[7].yy )
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
			    curve[i].pt[j].yy, curve_color[i].pt[j].yy);*/
/* Assign colors for curve zz */
		   curve_color[i].pt[j].zz = 0;
		   if(  curve[i].pt[j].zz > curve_div[1].zz )
		   {
		      curve_color[i].pt[j].zz = 1;
		      if(  curve[i].pt[j].zz > curve_div[2].zz )
		      {
			 curve_color[i].pt[j].zz = 2;
			 if(  curve[i].pt[j].zz > curve_div[3].zz )
			 {
			    curve_color[i].pt[j].zz = 3;
			    if(  curve[i].pt[j].zz > curve_div[4].zz )
			    {
			       curve_color[i].pt[j].zz = 4;
			       if(  curve[i].pt[j].zz > curve_div[5].zz )
			       {
				  curve_color[i].pt[j].zz = 5;
				  if(  curve[i].pt[j].zz > curve_div[6].zz )
				  {
				     curve_color[i].pt[j].zz = 6;
				     if(  curve[i].pt[j].zz > curve_div[7].zz )
				     {
					curve_color[i].pt[j].zz = 7;
				     }
				  }
			       }
			    }
			 }
		      }
		   }
		   /*printf(" curve zz %d %d %10.5e %d \n", i, j,
			    curve[i].pt[j].zz, curve_color[i].pt[j].zz);*/
	       }
	       else
	       {
		   curve_color[i].pt[j].xx = 12;
		   curve_color[i].pt[j].yy = 12;
		   curve_color[i].pt[j].zz = 12;
	       }
/* Assign colors for strain xx */
	       strain_color[i].pt[j].xx = 0;
	       if(  strain[i].pt[j].xx > strain_div[1].xx )
	       {
		  strain_color[i].pt[j].xx = 1;
		  if(  strain[i].pt[j].xx > strain_div[2].xx )
		  {
		     strain_color[i].pt[j].xx = 2;
		     if(  strain[i].pt[j].xx > strain_div[3].xx )
		     {
			strain_color[i].pt[j].xx = 3;
			if(  strain[i].pt[j].xx > strain_div[4].xx )
			{
			   strain_color[i].pt[j].xx = 4;
			   if(  strain[i].pt[j].xx > strain_div[5].xx )
			   {
			      strain_color[i].pt[j].xx = 5;
			      if(  strain[i].pt[j].xx > strain_div[6].xx )
			      {
				 strain_color[i].pt[j].xx = 6;
				 if(  strain[i].pt[j].xx > strain_div[7].xx )
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
			strain[i].pt[j].xx, strain_color[i].pt[j].xx);*/
	       if( type_num != 1 && type_num != 5 && type_num != 7 )
	       {
/*     Assign colors for strain xy */
		   strain_color[i].pt[j].xy = 0;
		   if(  strain[i].pt[j].xy > strain_div[1].xy )
		   {
		      strain_color[i].pt[j].xy = 1;
		      if(  strain[i].pt[j].xy > strain_div[2].xy )
		      {
			 strain_color[i].pt[j].xy = 2;
			 if(  strain[i].pt[j].xy > strain_div[3].xy )
			 {
			    strain_color[i].pt[j].xy = 3;
			    if(  strain[i].pt[j].xy > strain_div[4].xy )
			    {
			       strain_color[i].pt[j].xy = 4;
			       if(  strain[i].pt[j].xy > strain_div[5].xy )
			       {
				  strain_color[i].pt[j].xy = 5;
				  if(  strain[i].pt[j].xy > strain_div[6].xy )
				  {
				     strain_color[i].pt[j].xy = 6;
				     if(  strain[i].pt[j].xy > strain_div[7].xy )
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
			    strain[i].pt[j].xy, strain_color[i].pt[j].xy);*/
/*     Assign colors for strain zx */
		   strain_color[i].pt[j].zx = 0;
		   if(  strain[i].pt[j].zx > strain_div[1].zx )
		   {
		      strain_color[i].pt[j].zx = 1;
		      if(  strain[i].pt[j].zx > strain_div[2].zx )
		      {
			 strain_color[i].pt[j].zx = 2;
			 if(  strain[i].pt[j].zx > strain_div[3].zx )
			 {
			    strain_color[i].pt[j].zx = 3;
			    if(  strain[i].pt[j].zx > strain_div[4].zx )
			    {
			       strain_color[i].pt[j].zx = 4;
			       if(  strain[i].pt[j].zx > strain_div[5].zx )
			       {
				  strain_color[i].pt[j].zx = 5;
				  if(  strain[i].pt[j].zx > strain_div[6].zx )
				  {
				     strain_color[i].pt[j].zx = 6;
				     if(  strain[i].pt[j].zx > strain_div[7].zx )
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
			    strain[i].pt[j].zx, strain_color[i].pt[j].zx);*/
	       }
	       else
	       {
		   strain_color[i].pt[j].xy = 12;
		   strain_color[i].pt[j].zx = 12;
	       }
	       if( type_num != 1 && type_num != 4 && type_num != 5 && type_num != 7 )
	       {
/* Assign colors for moment xx */
		   moment_color[i].pt[j].xx = 0;
		   if(  moment[i].pt[j].xx > moment_div[1].xx )
		   {
		      moment_color[i].pt[j].xx = 1;
		      if(  moment[i].pt[j].xx > moment_div[2].xx )
		      {
			 moment_color[i].pt[j].xx = 2;
			 if(  moment[i].pt[j].xx > moment_div[3].xx )
			 {
			    moment_color[i].pt[j].xx = 3;
			    if(  moment[i].pt[j].xx > moment_div[4].xx )
			    {
			       moment_color[i].pt[j].xx = 4;
			       if(  moment[i].pt[j].xx > moment_div[5].xx )
			       {
				  moment_color[i].pt[j].xx = 5;
				  if(  moment[i].pt[j].xx > moment_div[6].xx )
				  {
				     moment_color[i].pt[j].xx = 6;
				     if(  moment[i].pt[j].xx > moment_div[7].xx )
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
			    moment[i].pt[j].xx, moment_color[i].pt[j].xx);*/
/* Assign colors for moment zz */
/* Assign colors for moment yy */
		   moment_color[i].pt[j].yy = 0;
		   if(  moment[i].pt[j].yy > moment_div[1].yy )
		   {
		      moment_color[i].pt[j].yy = 1;
		      if(  moment[i].pt[j].yy > moment_div[2].yy )
		      {
			 moment_color[i].pt[j].yy = 2;
			 if(  moment[i].pt[j].yy > moment_div[3].yy )
			 {
			    moment_color[i].pt[j].yy = 3;
			    if(  moment[i].pt[j].yy > moment_div[4].yy )
			    {
			       moment_color[i].pt[j].yy = 4;
			       if(  moment[i].pt[j].yy > moment_div[5].yy )
			       {
				  moment_color[i].pt[j].yy = 5;
				  if(  moment[i].pt[j].yy > moment_div[6].yy )
				  {
				     moment_color[i].pt[j].yy = 6;
				     if(  moment[i].pt[j].yy > moment_div[7].yy )
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
			    moment[i].pt[j].yy, moment_color[i].pt[j].yy);*/
/* Assign colors for moment zz */
		   moment_color[i].pt[j].zz = 0;
		   if(  moment[i].pt[j].zz > moment_div[1].zz )
		   {
		      moment_color[i].pt[j].zz = 1;
		      if(  moment[i].pt[j].zz > moment_div[2].zz )
		      {
			 moment_color[i].pt[j].zz = 2;
			 if(  moment[i].pt[j].zz > moment_div[3].zz )
			 {
			    moment_color[i].pt[j].zz = 3;
			    if(  moment[i].pt[j].zz > moment_div[4].zz )
			    {
			       moment_color[i].pt[j].zz = 4;
			       if(  moment[i].pt[j].zz > moment_div[5].zz )
			       {
				  moment_color[i].pt[j].zz = 5;
				  if(  moment[i].pt[j].zz > moment_div[6].zz )
				  {
				     moment_color[i].pt[j].zz = 6;
				     if(  moment[i].pt[j].zz > moment_div[7].zz )
				     {
					moment_color[i].pt[j].zz = 7;
				     }
				  }
			       }
			    }
			 }
		      }
		   }
		   /*printf(" moment zz %d %d %10.5e %d \n", i, j,
			    moment[i].pt[j].zz, moment_color[i].pt[j].zz);*/
	       }
	       else
	       {
		   moment_color[i].pt[j].xx = 12;
		   moment_color[i].pt[j].yy = 12;
		   moment_color[i].pt[j].zz = 12;
	       }
/* Assign colors for stress xx */
	       stress_color[i].pt[j].xx = 0;
	       if(  stress[i].pt[j].xx > stress_div[1].xx )
	       {
		  stress_color[i].pt[j].xx = 1;
		  if(  stress[i].pt[j].xx > stress_div[2].xx )
		  {
		     stress_color[i].pt[j].xx = 2;
		     if(  stress[i].pt[j].xx > stress_div[3].xx )
		     {
			stress_color[i].pt[j].xx = 3;
			if(  stress[i].pt[j].xx > stress_div[4].xx )
			{
			   stress_color[i].pt[j].xx = 4;
			   if(  stress[i].pt[j].xx > stress_div[5].xx )
			   {
			      stress_color[i].pt[j].xx = 5;
			      if(  stress[i].pt[j].xx > stress_div[6].xx )
			      {
				 stress_color[i].pt[j].xx = 6;
				 if(  stress[i].pt[j].xx > stress_div[7].xx )
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
			stress[i].pt[j].xx, stress_color[i].pt[j].xx);*/
	       if( type_num != 1 && type_num != 5 && type_num != 7 )
	       {
/* Assign colors for stress xy */
		   stress_color[i].pt[j].xy = 0;
		   if(  stress[i].pt[j].xy > stress_div[1].xy )
		   {
		      stress_color[i].pt[j].xy = 1;
		      if(  stress[i].pt[j].xy > stress_div[2].xy )
		      {
			 stress_color[i].pt[j].xy = 2;
			 if(  stress[i].pt[j].xy > stress_div[3].xy )
			 {
			    stress_color[i].pt[j].xy = 3;
			    if(  stress[i].pt[j].xy > stress_div[4].xy )
			    {
			       stress_color[i].pt[j].xy = 4;
			       if(  stress[i].pt[j].xy > stress_div[5].xy )
			       {
				  stress_color[i].pt[j].xy = 5;
				  if(  stress[i].pt[j].xy > stress_div[6].xy )
				  {
				     stress_color[i].pt[j].xy = 6;
				     if(  stress[i].pt[j].xy > stress_div[7].xy )
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
			    stress[i].pt[j].xy, stress_color[i].pt[j].xy);*/
/* Assign colors for stress zx */
		   stress_color[i].pt[j].zx = 0;
		   if(  stress[i].pt[j].zx > stress_div[1].zx )
		   {
		      stress_color[i].pt[j].zx = 1;
		      if(  stress[i].pt[j].zx > stress_div[2].zx )
		      {
			 stress_color[i].pt[j].zx = 2;
			 if(  stress[i].pt[j].zx > stress_div[3].zx )
			 {
			    stress_color[i].pt[j].zx = 3;
			    if(  stress[i].pt[j].zx > stress_div[4].zx )
			    {
			       stress_color[i].pt[j].zx = 4;
			       if(  stress[i].pt[j].zx > stress_div[5].zx )
			       {
				  stress_color[i].pt[j].zx = 5;
				  if(  stress[i].pt[j].zx > stress_div[6].zx )
				  {
				     stress_color[i].pt[j].zx = 6;
				     if(  stress[i].pt[j].zx > stress_div[7].zx )
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
			    stress[i].pt[j].zx, stress_color[i].pt[j].zx);*/
	       }
	       else
	       {
		   stress_color[i].pt[j].xy = 12;
		   stress_color[i].pt[j].zx = 12;
	       }
	    }
	}

/* Calculate the graphical force vectors for display */

	for( i = 0; i < bc.num_force[0]; ++i )
	{
/* Calculate the force part of force_vec */
		
		force_vec_length =
		   *(force + ndof*bc.force[i])*(*(force + ndof*bc.force[i])) +
		   *(force + ndof*bc.force[i] + 1)*(*(force + ndof*bc.force[i] + 1)) +
		   *(force + ndof*bc.force[i] + 2)*(*(force + ndof*bc.force[i] + 2)) +
		   SMALL;
		force_vec_length = sqrt(force_vec_length);
		force_vec0[i].x =
		   *(force + ndof*bc.force[i])*3*AxisLength_max/force_vec_length;
		force_vec0[i].y =
		   *(force + ndof*bc.force[i] + 1)*3*AxisLength_max/force_vec_length;
		force_vec0[i].z =
		   *(force + ndof*bc.force[i] + 2)*3*AxisLength_max/force_vec_length;

/* Calculate the moment part of force_vec */

		moment_vec_length =
		   *(force + ndof*bc.force[i] +3)*(*(force + ndof*bc.force[i] +3)) +
		   *(force + ndof*bc.force[i] +4)*(*(force + ndof*bc.force[i] +4)) +
		   *(force + ndof*bc.force[i] +5)*(*(force + ndof*bc.force[i] +5)) +
		   SMALL;
		moment_vec_length = sqrt(moment_vec_length);
		force_vec0[i].phix =
		   *(force + ndof*bc.force[i] + 3)*2.0*AxisLength_max/moment_vec_length;
		force_vec0[i].phiy =
		   *(force + ndof*bc.force[i] + 4)*2.0*AxisLength_max/moment_vec_length;
		force_vec0[i].phiz =
		   *(force + ndof*bc.force[i] + 5)*2.0*AxisLength_max/moment_vec_length;

  /*printf(" force %d %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e\n",
		   i, *(force + ndof*bc.force[i]), *(force + ndof*bc.force[i] + 1),
		   *(force + ndof*bc.force[i] + 2), *(force + ndof*bc.force[i] + 3),
		   *(force + ndof*bc.force[i] + 5),
		   force_vec0[i].x, force_vec0[i].y,
		   force_vec0[i].phix, force_vec0[i].phiy, force_vec0[i].phiz);*/
	}

/* Calculate the graphical distributed load vectors for display */

	for( i = 0; i < bc.num_dist_load[0]; ++i )
	{
/* Calculate dist_load_vec */

		el_num = bc.dist_load[i];

		dist_load_vec_length =
		   *(dist_load + 2*el_num )*(*(dist_load + 2*el_num )) +
		   *(dist_load + 2*el_num + 1)*(*(dist_load + 2*el_num + 1)) +
		   SMALL;
		dist_load_vec_length = sqrt(dist_load_vec_length);
		dist_load_vec0[i].qy = -
		   *(dist_load + 2*el_num )*.5/dist_load_vec_length;
		dist_load_vec0[i].qz = -
		   *(dist_load + 2*el_num + 1)*.5/dist_load_vec_length;

  /*printf(" dist_load %d %10.5e %10.5e %10.5e %10.5e %10.5e\n",
		   i, dist_load_vec_length, *(dist_load + 2*el_num ), *(dist_load + 2*el_num + 1),
		   dist_load_vec0[i].qy, dist_load_vec0[i].qz);*/
	}

	return 1;    /* ANSI C requires main to return int. */
}


