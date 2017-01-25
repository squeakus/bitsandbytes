/*
    This program calculates and writes the parameters for
    the FEM GUI for triangle elements.
  
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
#if QUAD1
#include "../quad/quad/qdconst.h"
#include "../quad/quad/qdstruct.h"
#include "../quad/quad_gr/qdstrcgr.h"
#endif
#if TRI1
#include "../tri/tri/trconst.h"
#include "../tri/tri/trstruct.h"
#include "../tri/tri_gr/trstrcgr.h"
#endif
#include "control.h"

#define init_far0      -2.0

extern int nmat, numnp, numel, dof;
extern double step_sizex, step_sizey, step_sizez;
extern double left, right, top, bottom, near, far, fscale, coord_rescale;
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out,
	cross_sec_left_right0, cross_sec_up_down0, cross_sec_in_out0;
extern int control_height, control_width, mesh_height, mesh_width;
extern double ortho_left, ortho_right, ortho_top, ortho_bottom,
	ortho_left0, ortho_right0, ortho_top0, ortho_bottom0;
extern double left_right, up_down, in_out, left_right0, up_down0, in_out0;
extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z,
	IAxisMin_x, IAxisMin_y, IAxisMin_z;
extern double AxisLength_x, AxisLength_y, AxisLength_z,
	AxisLength_max;
extern double AxisPoint_step;
extern double amplify_step0;

extern double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
extern SDIM del_stress, del_strain, max_stress, min_stress,
	max_strain, min_strain;
extern double max_Ux, min_Ux, del_Ux, max_Uy, min_Uy, del_Uy,
	max_Uz, min_Uz, del_Uz, absolute_max_U, absolute_max_coord;

int parameter2( double *coord, SDIM *strain_node, SDIM *stress_node, double *U )
{
	int i, j, check;
	int node_Ux_max, node_Ux_min, node_Uy_max, node_Uy_min, node_Uz_max, node_Uz_min;
	ISDIM node_stress_max, node_stress_min, node_strain_max, node_strain_min;
	FILE *view_data;
	
/*   view_data contains all the parameters and extreme values
*/
#if QUAD1
	view_data = fopen( "qdview.dat","w" );
#endif
#if TRI1
	view_data = fopen( "trview.dat","w" );
#endif

/* Initialize parameters */

	step_sizex = .1; step_sizey = .1; step_sizez = .1;
	init_right = - BIG; init_left = BIG;
	init_top = - BIG; init_bottom = BIG;
	init_near = - BIG; init_far = init_far0; true_far = BIG;
	max_Ux = - BIG; min_Ux = BIG;
	max_Uy = - BIG; min_Uy = BIG;
	max_Uz = - BIG; min_Uz = BIG;

	node_Ux_max = 0; node_Ux_min = 0;
	node_Uy_max = 0; node_Uy_min = 0;
	node_Uz_max = 0; node_Uz_min = 0;

	node_strain_max.xx = 0; node_strain_min.xx = 0;
	node_strain_max.yy = 0; node_strain_min.yy = 0;
	node_strain_max.xy = 0; node_strain_min.xy = 0;
	node_strain_max.I = 0; node_strain_min.I = 0;
	node_strain_max.II = 0; node_strain_min.II = 0;

	node_stress_max.xx = 0; node_stress_min.xx = 0;
	node_stress_max.yy = 0; node_stress_min.yy = 0;
	node_stress_max.xy = 0; node_stress_min.xy = 0;
	node_stress_max.I = 0; node_stress_min.I = 0;
	node_stress_max.II = 0; node_stress_min.II = 0;

/* Initialize largest and smallest strains */

	max_strain.xx = - BIG; min_strain.xx = BIG;
	max_strain.yy = - BIG; min_strain.yy = BIG;
	max_strain.xy = - BIG; min_strain.xy = BIG;
	max_strain.I = - BIG; min_strain.I = BIG;
	max_strain.II = - BIG; min_strain.II = BIG;

/* Initialize largest and smallest stresses */

	max_stress.xx = - BIG; min_stress.xx = BIG;
	max_stress.yy = - BIG; min_stress.yy = BIG;
	max_stress.xy = - BIG; min_stress.xy = BIG;
	max_stress.I = - BIG; min_stress.I = BIG;
	max_stress.II = - BIG; min_stress.II = BIG;

/* Search for extreme values */
 
/* Search for extreme values of displacement and nodal point */

	for( i = 0; i < numnp; ++i )
	{

/* Search for extreme nodal coordinates for parameters when object is
   viewed orthographically */

		if( init_right < *(coord+nsd*i))
			init_right=*(coord+nsd*i);

		if( init_left > *(coord+nsd*i))
			init_left=*(coord+nsd*i);

		if( init_top < *(coord+nsd*i+1))
			init_top=*(coord+nsd*i+1);

		if( init_bottom > *(coord+nsd*i+1))
			init_bottom=*(coord+nsd*i+1);

		if( init_near < *(coord+nsd*i+2))
			init_near=*(coord+nsd*i+2);

		if( true_far > *(coord+nsd*i+2))
			true_far=*(coord+nsd*i+2);

/* Search for extreme nodal displacements */

		if( max_Ux < *(U+ndof*i))
		{
			max_Ux=*(U+ndof*i);
			node_Ux_max = i;
		}
		if( min_Ux > *(U+ndof*i))
		{
			min_Ux=*(U+ndof*i);
			node_Ux_min = i;
		}
		if( max_Uy < *(U+ndof*i+1))
		{
			max_Uy=*(U+ndof*i+1);
			node_Uy_max = i;
		}
		if( min_Uy > *(U+ndof*i+1))
		{
			min_Uy=*(U+ndof*i+1);
			node_Uy_min = i;
		}
		if( max_Uz < *(U+ndof*i+2))
		{
			max_Uz=*(U+ndof*i+2);
			node_Uz_max = i;
		}
		if( min_Uz > *(U+ndof*i+2))
		{
			min_Uz=*(U+ndof*i+2);
			node_Uz_min = i;
		}
	}

/* Because Mesa has problems with Meshes that have dimensions larger than 1000
   or smaller than .1, I am rescaling everything so that things are on the order
   of 10.0.
*/

	absolute_max_coord = fabs(init_left);
	if(absolute_max_coord < fabs(init_right)) absolute_max_coord = fabs(init_right);
	if(absolute_max_coord < fabs(init_bottom)) absolute_max_coord = fabs(init_bottom);
	if(absolute_max_coord < fabs(init_top)) absolute_max_coord = fabs(init_top);

	coord_rescale = 1.0;
	if( absolute_max_coord > 10.0 )     coord_rescale = 10.0;
	if( absolute_max_coord > 100.0 )    coord_rescale = 100.0;
	if( absolute_max_coord > 1000.0 )   coord_rescale = 1000.0;
	if( absolute_max_coord > 10000.0 )  coord_rescale = 10000.0;
	if( absolute_max_coord > 100000.0 ) coord_rescale = 100000.0;

	if( absolute_max_coord < 1.0 )     coord_rescale = 0.1;
	if( absolute_max_coord < 0.1 )     coord_rescale = 0.01;
	if( absolute_max_coord < 0.01 )    coord_rescale = 0.001;
	if( absolute_max_coord < 0.001 )   coord_rescale = 0.0001;
	if( absolute_max_coord < 0.0001 )  coord_rescale = 0.00001;
	if( absolute_max_coord < 0.00001 ) coord_rescale = 0.000001;

/* Rescale coordinates and displacements */

	
	if( coord_rescale > 1.01 || coord_rescale < .99 )
	{
		for( i = 0; i < numnp; ++i )
		{
			*(coord+nsd*i) /= coord_rescale;
			*(coord+nsd*i+1) /= coord_rescale;
			*(coord+nsd*i+2) /= coord_rescale;

			*(U+ndof*i) /= coord_rescale;
			*(U+ndof*i+1) /= coord_rescale;
			*(U+ndof*i+2) /= coord_rescale;
		}

		init_left /= coord_rescale;
		init_right /= coord_rescale;
		init_bottom /= coord_rescale;
		init_top /= coord_rescale;
		init_near /= coord_rescale;
		true_far /= coord_rescale;

		min_Ux /= coord_rescale;
		max_Ux /= coord_rescale;
		min_Uy /= coord_rescale;
		max_Uy /= coord_rescale;
		min_Uz /= coord_rescale;
		max_Uz /= coord_rescale;
	}

/* Search for largest absolute value U */

	absolute_max_U = fabs(min_Ux);
	if( absolute_max_U < fabs(max_Ux))
		absolute_max_U = fabs(max_Ux);
	if( absolute_max_U < fabs(min_Uy))
		absolute_max_U = fabs(min_Uy);
	if( absolute_max_U < fabs(max_Uy))
		absolute_max_U = fabs(max_Uy);
	if( absolute_max_U < fabs(min_Uz))
		absolute_max_U = fabs(min_Uz);
	if( absolute_max_U < fabs(max_Uz))
		absolute_max_U = fabs(max_Uz);

	if( init_far > true_far)
		init_far=true_far;

	for( i = 0; i < numnp; ++i )
	{

/* Find extreme strains */

		if( max_strain.xx < strain_node[i].xx )
		{
			max_strain.xx = strain_node[i].xx;
			node_strain_max.xx = i;
		}
		if( min_strain.xx > strain_node[i].xx )
		{
			min_strain.xx = strain_node[i].xx;
			node_strain_min.xx = i;
		}
		if( max_strain.yy < strain_node[i].yy )
		{
			max_strain.yy = strain_node[i].yy;
			node_strain_max.yy = i;
		}
		if( min_strain.yy > strain_node[i].yy )
		{
			min_strain.yy = strain_node[i].yy;
			node_strain_min.yy = i;
		}
		if( max_strain.xy < strain_node[i].xy )
		{
			max_strain.xy = strain_node[i].xy;
			node_strain_max.xy = i;
		}
		if( min_strain.xy > strain_node[i].xy )
		{
			min_strain.xy = strain_node[i].xy;
			node_strain_min.xy = i;
		}
		if( max_strain.I < strain_node[i].I )
		{
			max_strain.I = strain_node[i].I;
			node_strain_max.I = i;
		}
		if( min_strain.I > strain_node[i].I )
		{
			min_strain.I = strain_node[i].I;
			node_strain_min.I = i;
		}
		if( max_strain.II < strain_node[i].II )
		{
			max_strain.II = strain_node[i].II;
			node_strain_max.II = i;
		}
		if( min_strain.II > strain_node[i].II )
		{
			min_strain.II = strain_node[i].II;
			node_strain_min.II = i;
		}
/* Find extreme stresses */

		if( max_stress.xx < stress_node[i].xx )
		{
			max_stress.xx = stress_node[i].xx;
			node_stress_max.xx = i;
		}
		if( min_stress.xx > stress_node[i].xx )
		{
			min_stress.xx = stress_node[i].xx;
			node_stress_min.xx = i;
		}
		if( max_stress.yy < stress_node[i].yy )
		{
			max_stress.yy = stress_node[i].yy;
			node_stress_max.yy = i;
		}
		if( min_stress.yy > stress_node[i].yy )
		{
			min_stress.yy = stress_node[i].yy;
			node_stress_min.yy = i;
		}
		if( max_stress.xy < stress_node[i].xy )
		{
			max_stress.xy = stress_node[i].xy;
			node_stress_max.xy = i;
		}
		if( min_stress.xy > stress_node[i].xy )
		{
			min_stress.xy = stress_node[i].xy;
			node_stress_min.xy = i;
		}
		if( max_stress.I < stress_node[i].I )
		{
			max_stress.I = stress_node[i].I;
			node_stress_max.I = i;
		}
		if( min_stress.I > stress_node[i].I )
		{
			min_stress.I = stress_node[i].I;
			node_stress_min.I = i;
		}
		if( max_stress.II < stress_node[i].II )
		{
			max_stress.II = stress_node[i].II;
			node_stress_max.II = i;
		}
		if( min_stress.II > stress_node[i].II )
		{
			min_stress.II = stress_node[i].II;
			node_stress_min.II = i;
		}
	}

/* Set the axes parameters */

	AxisMax_x = 1.2*init_right + 1.0;
	AxisMax_y = 1.2*init_top + 1.0;
	AxisMax_z = 1.2*init_near + 1.0;
	AxisMin_x = 1.2*init_left - 1.0;
	AxisMin_y = 1.2*init_bottom - 1.0;
	AxisMin_z = 1.2*true_far - 1.0;

	if( AxisMax_x < 0.0 )
		AxisMax_x = -.1*AxisMin_x;
	if( AxisMax_y < 0.0 )
		AxisMax_y = -.1*AxisMin_y;
	if( AxisMax_z < 0.0 )
		AxisMax_z = -.1*AxisMin_z;
	if( AxisMin_x > 0.0 )
		AxisMin_x = -.1*AxisMax_x;
	if( AxisMin_y > 0.0 )
		AxisMin_y = -.1*AxisMax_y;
	if( AxisMin_z > 0.0 )
		AxisMin_z = -.1*AxisMax_z;

	IAxisMin_x = (double)(int)AxisMin_x; 
	IAxisMin_y = (double)(int)AxisMin_y;
	IAxisMin_z = (double)(int)AxisMin_z;

	AxisLength_x = (int)(AxisMax_x - AxisMin_x);
	AxisLength_y = (int)(AxisMax_y - AxisMin_y);
	AxisLength_z = (int)(AxisMax_z - AxisMin_z);
	AxisLength_max = AxisLength_x;
	if( AxisLength_max < AxisLength_y )
		AxisLength_max = AxisLength_y;
	if( AxisLength_max < AxisLength_z )
		AxisLength_max = AxisLength_z;

	AxisPoint_step = .1;
	if( AxisLength_max > 1.0 )
		AxisPoint_step = 1.0;
	if( AxisLength_max > 10.0 )
		AxisPoint_step = 10.0;
	if( AxisLength_max > 100.0 )
		AxisPoint_step = 100.0;
	if( AxisLength_max > 1000.0 )
		AxisPoint_step = 1000.0;
	if( AxisLength_max > 10000.0 )
		AxisPoint_step = 10000.0;
	AxisLength_max *= .05;
	/*printf(" AxisPoint_step %10.5e\n",AxisPoint_step);*/

/* Determine amplification step size */

	amplify_step0 = .5*AxisLength_max/(absolute_max_U+SMALL);
	/*printf(" amplify_step0 %10.5e\n",amplify_step0);*/

/* Calculate orthographic viewport parameters */

	right = init_right + (init_right - init_left) / 10.0;
	left = init_left - (init_right - init_left) / 10.0;
	top = init_top + (init_top - init_bottom) / 10.0;
	bottom = init_bottom - (init_top - init_bottom) / 10.0;
	near = init_near + (init_near - init_far) / 10.0;
	far = init_far - (init_near - init_far) / 10.0;

	dim_max = right - left;
	if( dim_max < top - bottom )
		dim_max = top - bottom;

	ortho_right0 = left + dim_max + 1.0;
	ortho_left0 = left - 1.0;
	ortho_top0 = bottom + dim_max + 1.0;
	ortho_bottom0 = bottom - 1.0;

	ortho_right = ortho_right0;
	ortho_left = ortho_left0;
	ortho_top = ortho_top0;
	ortho_bottom = ortho_bottom0;

/* Set the Viewer parameters */

	step_sizex = (right - left) / 20.0;
	step_sizey = (top - bottom) / 20.0;
	step_sizez = (near - far) / 5.0;

	left_right0 = -(left + right) / 2.0;
	up_down0 = - (top + bottom ) / 2.0;
	/*in_out  = (far + near ) / 2.0 - 5.0;*/
	in_out0 = far - 20.0*AxisLength_max;
	left_right = left_right0;
	up_down = up_down0;
	in_out = in_out0;

/* Set the Cross Section Plane parameters */

	cross_sec_left_right0 = AxisMax_x;
	cross_sec_up_down0 = AxisMax_y;
	cross_sec_in_out0 = AxisMax_z;

	cross_sec_left_right = cross_sec_left_right0;
	cross_sec_up_down = cross_sec_up_down0;
	cross_sec_in_out = cross_sec_in_out0;

	mesh_width = mesh_width0;
	mesh_height = mesh_height0;

/* Print the above data in the file "trview.dat" */

	fprintf( view_data, "                            node\n");
	fprintf( view_data, "                          min  max       min            max\n");
	fprintf( view_data,"displacement Ux        %5d %5d   %14.6e %14.6e\n", node_Ux_min,
		node_Ux_max, min_Ux*coord_rescale, max_Ux*coord_rescale);
	fprintf( view_data,"displacement Uy        %5d %5d   %14.6e %14.6e\n", node_Uy_min,
		node_Uy_max, min_Uy*coord_rescale, max_Uy*coord_rescale);
	fprintf( view_data,"displacement Uz        %5d %5d   %14.6e %14.6e\n", node_Uz_min,
		node_Uz_max, min_Uz*coord_rescale, max_Uz*coord_rescale);
	fprintf( view_data,"\n");
	fprintf( view_data, "                            node\n");
	fprintf( view_data, "                        min       max         min           max\n");
	fprintf( view_data,"stress xx            %5d     %5d   %14.6e %14.6e\n", node_stress_min.xx,
		node_stress_max.xx, min_stress.xx, max_stress.xx);
	fprintf( view_data,"stress yy            %5d     %5d   %14.6e %14.6e\n", node_stress_min.yy,
		node_stress_max.yy, min_stress.yy, max_stress.yy);
	fprintf( view_data,"stress xy            %5d     %5d   %14.6e %14.6e\n", node_stress_min.xy,
		node_stress_max.xy, min_stress.xy, max_stress.xy);
	fprintf( view_data,"stress I             %5d     %5d   %14.6e %14.6e\n", node_stress_min.I,
		node_stress_max.I, min_stress.I, max_stress.I);
	fprintf( view_data,"stress II            %5d     %5d   %14.6e %14.6e\n", node_stress_min.II,
		node_stress_max.II, min_stress.II, max_stress.II);
	fprintf( view_data,"\n");
	fprintf( view_data,"strain xx            %5d     %5d   %14.6e %14.6e\n", node_strain_min.xx,
		node_strain_max.xx, min_strain.xx, max_strain.xx);
	fprintf( view_data,"strain yy            %5d     %5d   %14.6e %14.6e\n", node_strain_min.yy,
		node_strain_max.yy, min_strain.yy, max_strain.yy);
	fprintf( view_data,"strain xy            %5d     %5d   %14.6e %14.6e\n", node_strain_min.xy,
		node_strain_max.xy, min_strain.xy, max_strain.xy);
	fprintf( view_data,"strain I             %5d     %5d   %14.6e %14.6e\n", node_strain_min.I,
		node_strain_max.I, min_strain.I, max_strain.I);
	fprintf( view_data,"strain II            %5d     %5d   %14.6e %14.6e\n", node_strain_min.II,
		node_strain_max.II, min_strain.II, max_strain.II);
	fprintf( view_data,"\n");
	fprintf( view_data,"Orthographic viewport parameters(right, left, top, bottom, near, far)\n ");
	fprintf( view_data,"%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n", ortho_right, ortho_left,
		ortho_top, ortho_bottom, near, 1000.0);
	fprintf( view_data,"Perspective viewport parameters( mesh width and height)\n ");
	fprintf( view_data,"%6d %6d\n", mesh_width, mesh_height);
	fprintf( view_data,"Step sizes in x, y, z\n ");
	fprintf( view_data,"%14.6e %14.6e %14.6e\n",step_sizex, step_sizey, step_sizez);
	fprintf( view_data,"Amplification size\n ");
	fprintf( view_data,"%14.6e\n",amplify_step0);

	fclose( view_data );

	return 1;    /* ANSI C requires main to return int. */
}
