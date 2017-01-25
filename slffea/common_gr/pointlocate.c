/*
    This program uses xAngle, yAngle, and zAngle
    to rotate a point about the origin.

 		Last Update 4/26/01

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int matX( double *, double *, double *, int, int, int);

int quaternion( double , double *, double *);

int PointLocate(double *point_vec, double xAngle, double yAngle, double zAngle )
{
	double rotate[9], point_vec2[3], point_vec3[3], point_vec4[3];
	double x_axis[3], y_axis[3], z_axis[3];
	double x2_axis[3], y2_axis[3], z2_axis[3];
	double x3_axis[3], y3_axis[3], z3_axis[3];
	int check, i, j, k;

	x_axis[0] = 1.0;
	x_axis[1] = 0.0;
	x_axis[2] = 0.0;

	y_axis[0] = 0.0;
	y_axis[1] = 1.0;
	y_axis[2] = 0.0;

	z_axis[0] = 0.0;
	z_axis[1] = 0.0;
	z_axis[2] = 1.0;

/* Determine rotation matrix rotate which is calculated from a
   rotation of xAngle about the x_axis
*/
	check = quaternion( xAngle, x_axis, rotate);
	if(!check) printf( " Problems with quaternion\n");

/* Find location of new y and z axes after being rotated though by xAngle about
   the x axis */

	check = matX( y2_axis, rotate, y_axis, 3, 1, 3); 
	if(!check) printf( " Problems with matX\n");
	check = matX( z2_axis, rotate, z_axis, 3, 1, 3); 
	if(!check) printf( " Problems with matX\n");

/* Find location of point after being rotated though by xAngle about
   the x axis */

	check = matX( point_vec2, rotate, point_vec, 3, 1, 3); 
	if(!check) printf( " Problems with matX\n");

/* Determine rotation matrix rotate which is calculated from a
   rotation of yAngle about the new y2_axis.
*/
	check = quaternion( yAngle, y2_axis, rotate);
	if(!check) printf( " Problems with quaternion\n");

/* Find location of new z2_axis after being rotated though by yAngle about
   the new y2_axis */

	check = matX( z3_axis, rotate, z2_axis, 3, 1, 3); 
	if(!check) printf( " Problems with matX\n");

/* Find location of point after being rotated though by yAngle about
   the new y2_axis */

	check = matX( point_vec3, rotate, point_vec2, 3, 1, 3); 
	if(!check) printf( " Problems with matX\n");

/* Determine rotation matrix rotate which is calculated from a
   rotation of zAngle about the new z3_axis.
*/
	check = quaternion( zAngle, z3_axis, rotate);
	if(!check) printf( " Problems with quaternion\n");

/* Find location of point after being rotated though by zAngle about
   the new z3_axis */

	check = matX( point_vec4, rotate, point_vec3, 3, 1, 3); 
	if(!check) printf( " Problems with matX\n");

	point_vec[0] = point_vec4[0];
	point_vec[1] = point_vec4[1];
	point_vec[2] = point_vec4[2];

	return 1;
}

