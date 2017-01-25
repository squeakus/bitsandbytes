/*
    This program rereads in the parameters for the FEM GUI
    for 3 dimensional elements with displacements
    Ux, Uy, Uz and stresses xx, yy, zz, xy, zx, yz, I, II, III.

    It will also handle thermal load data.

                        Last Update 1/23/06

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
#if TETRA1
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra/testruct.h"
#include "../tetra/tetra_gr/testrcgr.h"
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
extern double left, right, top, bottom, near, far, fscale, coord_rescale;
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

extern double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
extern SDIM del_stress, del_strain, max_stress, min_stress,
	max_strain, min_strain;
extern double max_Ux, min_Ux, del_Ux, max_Uy, min_Uy, del_Uy,
	max_Uz, min_Uz, del_Uz, absolute_max_U;
#if BRICK2
extern double max_T, min_T, del_T, max_Q, min_Q, del_Q;
#endif


void ReGetparameter( void)
{
	int i, j, check;
	int node_Ux_max, node_Ux_min, node_Uy_max, node_Uy_min,
		node_Uz_max, node_Uz_min; 
	int node_T_max, node_T_min, node_Q_max, node_Q_min;
 	ISDIM node_stress_max, node_stress_min, node_strain_max, node_strain_min;
	char char_dum[20], char_dum2[5], buf[ BUFSIZ ];
	double fdum;
	FILE *view_data;

/*   view_data contains all the parameters and extreme values 
*/
#if BRICK1
	view_data = fopen( "brview.dat","r" );
#endif
#if BRICK2
	view_data = fopen( "br2view.dat","r" );
#endif
#if TETRA1
	view_data = fopen( "teview.dat","r" );
#endif
#if WEDGE1
	view_data = fopen( "weview.dat","r" );
#endif

/* Read data from the file "brview.dat" */

	fgets( buf, BUFSIZ, view_data );
	fgets( buf, BUFSIZ, view_data );
	fscanf( view_data,"%20s %5s      %d %d   %lf %lf\n", char_dum, char_dum2, &node_Ux_min,
		&node_Ux_max, &min_Ux, &max_Ux);
	fscanf( view_data,"%20s %5s      %d %d   %lf %lf\n", char_dum, char_dum2, &node_Uy_min,
		&node_Uy_max, &min_Uy, &max_Uy);
	fscanf( view_data,"%20s %5s      %d %d   %lf %lf\n", char_dum, char_dum2, &node_Uz_min,
		&node_Uz_max, &min_Uz, &max_Uz);

/* Rescale the displacement data */

	min_Ux /= coord_rescale;
	max_Ux /= coord_rescale;
	min_Uy /= coord_rescale;
	max_Uy /= coord_rescale;
	min_Uz /= coord_rescale;
	max_Uz /= coord_rescale;

	fscanf( view_data,"\n");

#if BRICK2
	fscanf( view_data,"%20s          %d %d   %lf %lf\n", char_dum, &node_T_min,
		&node_T_max, &min_T, &max_T);
	fscanf( view_data,"%20s %5s      %d %d   %lf %lf\n", char_dum, char_dum2, &node_Q_min,
		&node_Q_max, &min_Q, &max_Q);
	fscanf( view_data,"\n");
#endif

	fgets( buf, BUFSIZ, view_data );
	fgets( buf, BUFSIZ, view_data );
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.xx, &node_stress_max.xx,
		&min_stress.xx, &max_stress.xx);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.yy, &node_stress_max.yy,
		&min_stress.yy, &max_stress.yy);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.zz, &node_stress_max.zz,
		&min_stress.zz, &max_stress.zz);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.xy, &node_stress_max.xy,
		&min_stress.xy, &max_stress.xy);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.zx, &node_stress_max.zx,
		&min_stress.zx, &max_stress.zx);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.yz, &node_stress_max.yz,
		&min_stress.yz, &max_stress.yz);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.I, &node_stress_max.I,
		&min_stress.I, &max_stress.I);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.II, &node_stress_max.II,
		&min_stress.II, &max_stress.II);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_stress_min.III, &node_stress_max.III,
		&min_stress.III, &max_stress.III);
	fscanf( view_data,"\n");
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.xx, &node_strain_max.xx,
		&min_strain.xx, &max_strain.xx);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.yy, &node_strain_max.yy,
		&min_strain.yy, &max_strain.yy);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.zz, &node_strain_max.zz,
		&min_strain.zz, &max_strain.zz);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.xy, &node_strain_max.xy,
		&min_strain.xy, &max_strain.xy);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.zx, &node_strain_max.zx,
		&min_strain.zx, &max_strain.zx);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.yz, &node_strain_max.yz,
		&min_strain.yz, &max_strain.yz);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.I, &node_strain_max.I,
		&min_strain.I, &max_strain.I);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.II, &node_strain_max.II,
		&min_strain.II, &max_strain.II);
	fscanf( view_data,"%20s %5s    %d %d  %lf %lf\n", char_dum, char_dum2,
		&node_strain_min.III, &node_strain_max.III,
		&min_strain.III, &max_strain.III);
	fscanf( view_data,"\n");
	fgets( buf, BUFSIZ, view_data );
	fscanf( view_data,"%lf %lf %lf %lf %lf %lf\n", &ortho_right, &ortho_left,
		&ortho_top, &ortho_bottom, &near, &fdum);
	fgets( buf, BUFSIZ, view_data );
	fscanf( view_data,"%d %d\n", &mesh_width, &mesh_height);
	fgets( buf, BUFSIZ, view_data );
	fscanf( view_data,"%lf %lf %lf\n", &step_sizex, &step_sizey, &step_sizez);
	fgets( buf, BUFSIZ, view_data );
	fscanf( view_data,"%lf\n", &amplify_step0);

	fclose( view_data );

	printf( "                            node\n");
	printf( "                          min  max       min            max\n");
	printf("displacement Ux        %5d %5d   %14.6e %14.6e\n", node_Ux_min,
		node_Ux_max, min_Ux*coord_rescale, max_Ux*coord_rescale);
	printf("displacement Uy        %5d %5d   %14.6e %14.6e\n", node_Uy_min,
		node_Uy_max, min_Uy*coord_rescale, max_Uy*coord_rescale);
	printf("displacement Uz        %5d %5d   %14.6e %14.6e\n", node_Uz_min,
		node_Uz_max, min_Uz*coord_rescale, max_Uz*coord_rescale);
	printf("\n");

#if BRICK2
	printf("Temperature            %5d %5d   %14.6e %14.6e\n", node_T_min,
		node_T_max, min_T, max_T);
	printf("heat Q                 %5d %5d   %14.6e %14.6e\n", node_Q_min,
		node_Q_max, min_Q, max_Q);
	printf("\n");
#endif

	printf( "                        el. gauss pt.\n");
	printf( "                        min       max         min           max\n");
	printf("stress xx            %5d     %5d  %14.6e %14.6e\n", node_stress_min.xx,
		node_stress_max.xx, min_stress.xx, max_stress.xx);
	printf("stress yy            %5d     %5d  %14.6e %14.6e\n", node_stress_min.yy,
		node_stress_max.yy, min_stress.yy, max_stress.yy);
	printf("stress zz            %5d     %5d  %14.6e %14.6e\n", node_stress_min.zz,
		node_stress_max.zz, min_stress.zz, max_stress.zz);
	printf("stress xy            %5d     %5d  %14.6e %14.6e\n", node_stress_min.xy,
		node_stress_max.xy, min_stress.xy, max_stress.xy);
	printf("stress zx            %5d     %5d  %14.6e %14.6e\n", node_stress_min.zx,
		node_stress_max.zx, min_stress.zx, max_stress.zx);
	printf("stress yz            %5d     %5d  %14.6e %14.6e\n", node_stress_min.yz,
		node_stress_max.yz, min_stress.yz, max_stress.yz);
	printf("stress I             %5d     %5d  %14.6e %14.6e\n", node_stress_min.I,
		node_stress_max.I, min_stress.I, max_stress.I);
	printf("stress II            %5d     %5d  %14.6e %14.6e\n", node_stress_min.II,
		node_stress_max.II, min_stress.II, max_stress.II);
	printf("stress III           %5d     %5d  %14.6e %14.6e\n", node_stress_min.III,
		node_stress_max.III, min_stress.III, max_stress.III);
	printf("\n");
	printf("strain xx            %5d     %5d  %14.6e %14.6e\n", node_strain_min.xx,
		node_strain_max.xx, min_strain.xx, max_strain.xx);
	printf("strain yy            %5d     %5d  %14.6e %14.6e\n", node_strain_min.yy,
		node_strain_max.yy, min_strain.yy, max_strain.yy);
	printf("strain zz            %5d     %5d  %14.6e %14.6e\n", node_strain_min.zz,
		node_strain_max.zz, min_strain.zz, max_strain.zz);
	printf("strain xy            %5d     %5d  %14.6e %14.6e\n", node_strain_min.xy,
		node_strain_max.xy, min_strain.xy, max_strain.xy);
	printf("strain zx            %5d     %5d  %14.6e %14.6e\n", node_strain_min.zx,
		node_strain_max.zx, min_strain.zx, max_strain.zx);
	printf("strain yz            %5d     %5d  %14.6e %14.6e\n", node_strain_min.yz,
		node_strain_max.yz, min_strain.yz, max_strain.yz);
	printf("strain I             %5d     %5d  %14.6e %14.6e\n", node_strain_min.I,
		node_strain_max.I, min_strain.I, max_strain.I);
	printf("strain II            %5d     %5d  %14.6e %14.6e\n", node_strain_min.II,
		node_strain_max.II, min_strain.II, max_strain.II);
	printf("strain III           %5d     %5d  %14.6e %14.6e\n", node_strain_min.III,
		node_strain_max.III, min_strain.III, max_strain.III);
	printf("\n");
	printf("Orthographic viewport parameters(right, left, top, bottom, near, far)\n ");
	printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n", ortho_right, ortho_left,
		ortho_top, ortho_bottom, near, 1000.0);
	printf("Perspective viewport parameters( mesh width and height)\n ");
	printf("%6d %6d\n", mesh_width, mesh_height);
	printf("Step sizes in x, y, z\n ");
	printf("%14.6e %14.6e %14.6e\n",step_sizex, step_sizey, step_sizez);
	printf("Amplification size\n ");
	printf("%14.6e\n",amplify_step0);
}

