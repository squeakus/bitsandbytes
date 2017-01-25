/*
    This program sets viewing and analysis values based on the parameters 
    determined in tsparameters for the FEM GUI for truss elements.
  
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
#include "../truss/tsconst.h"
#include "../truss/tsstruct.h"
#include "tsstrcgr.h"
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

extern double Ux_div[boxnumber+1], Uy_div[boxnumber+1], Uz_div[boxnumber+1];
extern SDIM stress_div[boxnumber+1];
extern SDIM strain_div[boxnumber+1];
extern double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
extern SDIM del_stress, max_stress, min_stress;
extern SDIM del_strain, max_strain, min_strain;
extern double max_Ux, min_Ux, del_Ux, max_Uy, min_Uy, del_Uy,
	max_Uz, min_Uz, del_Uz, absolute_max_U;

int tsset( BOUND bc, double *force, XYZF *force_vec0, SDIM *strain,
	ISDIM *strain_color, SDIM *stress, ISDIM *stress_color,
	double *U, int *U_color)
{
	int i, j, check;
	double force_vec_length;

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
			Ux_div[i], Uy_div[i]), Uy_div[i]);*/
	}

/* For strains */

	del_strain.xx =
		(max_strain.xx - min_strain.xx + SMALL)/(double)(boxnumber);
	strain_div[0].xx = min_strain.xx;
	/*printf(" max min strain xx %10.5e %10.5e \n", max_strain.xx, min_strain.xx);
	printf(" strain div xx %10.5e \n", strain_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		strain_div[i+1].xx = strain_div[i].xx + del_strain.xx;
		/*printf(" strain div xx %10.5e \n", strain_div[i+1].xx);*/
	}


/* For stresses */

	del_stress.xx =
		(max_stress.xx - min_stress.xx + SMALL)/(double)(boxnumber);
	stress_div[0].xx = min_stress.xx;
	/*printf(" max min stress xx %10.5e \n", max_stress.xx);
	printf(" stress div xx %10.5e \n", stress_div[0].xx);*/
	for( i = 0; i < boxnumber; ++i )
	{
		stress_div[i+1].xx = stress_div[i].xx + del_stress.xx;
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
	}


/* Assign colors to strains and stresses */

	for( i = 0; i < numel; ++i )
	{
/* Assign colors for strain xx */
	       strain_color[i].xx = 0;
	       if(  strain[i].xx > strain_div[1].xx )
	       {
		  strain_color[i].xx = 1;
		  if(  strain[i].xx > strain_div[2].xx )
		  {
		     strain_color[i].xx = 2;
		     if(  strain[i].xx > strain_div[3].xx )
		     {
			strain_color[i].xx = 3;
			if(  strain[i].xx > strain_div[4].xx )
			{
			   strain_color[i].xx = 4;
			   if(  strain[i].xx > strain_div[5].xx )
			   {
			      strain_color[i].xx = 5;
			      if(  strain[i].xx > strain_div[6].xx )
			      {
				 strain_color[i].xx = 6;
				 if(  strain[i].xx > strain_div[7].xx )
				 {
				    strain_color[i].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" strain xx %d %10.5e %d \n", i,
			strain[i].xx, strain_color[i].xx);*/
/* Assign colors for stress xx */
	       stress_color[i].xx = 0;
	       if(  stress[i].xx > stress_div[1].xx )
	       {
		  stress_color[i].xx = 1;
		  if(  stress[i].xx > stress_div[2].xx )
		  {
		     stress_color[i].xx = 2;
		     if(  stress[i].xx > stress_div[3].xx )
		     {
			stress_color[i].xx = 3;
			if(  stress[i].xx > stress_div[4].xx )
			{
			   stress_color[i].xx = 4;
			   if(  stress[i].xx > stress_div[5].xx )
			   {
			      stress_color[i].xx = 5;
			      if(  stress[i].xx > stress_div[6].xx )
			      {
				 stress_color[i].xx = 6;
				 if(  stress[i].xx > stress_div[7].xx )
				 {
				    stress_color[i].xx = 7;
				 }
			      }
			   }
			}
		     }
		  }
	       }
	       /*printf(" stress xx %d %10.5e %d \n", i,
			stress[i].xx, stress_color[i].xx);*/
	}

/* Calculate the graphical force vectors for display */

	for( i = 0; i < bc.num_force[0]; ++i )
	{
		
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
  /*printf(" force %d %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e\n",
		   i, *(force + ndof*bc.force[i]), *(force + ndof*bc.force[i] + 1),
		   *(force + ndof*bc.force[i] + 2), *(force + ndof*bc.force[i] + 3),
		   *(force + ndof*bc.force[i] + 4),
		   force_vec0[i].x, force_vec0[i].y, force_vec0[i].z);*/
	}

	return 1;    /* ANSI C requires main to return int. */
}


