/*
    meshspring.c
    This program generates the mesh for a spring based on wedge elements.

     Updated 12/9/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define pi              3.141592654

int mesh_spring(void)
{
	FILE *o1, *o2;
	int i, j, k, dum, dum2;
	int numnp, numel, numel_cross_sec, numel_spring_loop, numel_loop,
		el_cross_sec, numnp_cross_sec, numnp_loop, num_loop, loop_up_flag;
	double angle, S, C, Ro, loop_R, loop_R2, RT, X, Y, Z;
	double loop_space, dY_per_loop, dY_per_el, dloop_space, dloop_space_per_el;
	double arc_angle, c_arc_angle, cos_arc_ang, sin_arc_ang,
		c_spring_angle, dspring_angle, darc_angle;
	o1 = fopen( "spring", "w" );
	o2 = fopen( "spring.in", "r" );
	if(o2 == NULL ) {
		printf("Can't find file spring.in\n");
		exit(1);
	}

	char name[30], buf[ BUFSIZ ];

	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&numel_cross_sec);  /* elements in the spring's cross section */
	printf( "elements in the spring's cross section: %d\n",numel_cross_sec);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%lf\n",&Ro);                /* spring Radius */
	printf( "spring outer Radius: %lf\n",Ro);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%lf\n",&loop_R);            /* spring loop Radius */
	printf( "spring loop Radius: %lf\n",loop_R);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&num_loop);          /* number of loops */
	printf( "number of loops: %d\n", num_loop);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&numel_spring_loop);  /* elements in one spring loop */
	printf( "elements in the spring loop: %d\n",numel_spring_loop);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%lf\n",&loop_space);        /* space between loops */
	printf( "space between loops: %lf\n", loop_space);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&loop_up_flag);       /* tells whether last loop points up */
	printf( "last loop points up?: ", loop_up_flag);
	if(loop_up_flag) printf( "yes\n");
	else printf( "no\n");

	numnp_cross_sec = numel_cross_sec + 1;
	numnp_loop = numnp_cross_sec*numel_spring_loop;
	numel_loop = numel_cross_sec*numel_spring_loop;

	if( !loop_up_flag )
	{
	    numnp = numnp_loop*num_loop + numnp_cross_sec;
	    numel = numel_cross_sec*numel_spring_loop*num_loop;
	}
	else
	{
	    dum2 = numel_spring_loop%3;
	    dum2 = numel_spring_loop - dum2;
	    dum2 /= 3;
	    numnp = numnp_loop*num_loop + numnp_cross_sec*dum2;
	    numel = numel_cross_sec*(numel_spring_loop*num_loop + dum2 - 1);
	}

	fprintf(o1, "   numel numnp nmat nmode   (This is for a wedge spring)\n");
	fprintf(o1, "%d %d %d %d\n", numel, numnp, 1, 0);
	fprintf(o1, "matl no., E modulus, Poisson Ratio, density\n");
	fprintf(o1, "%d %16.4f %16.4f %16.4f\n",0, 1000000.0, 0.27, 2.77e3);
	fprintf(o1, "el no.,connectivity, matl no.\n");
	dum = 0;
	for( k = 0; k < num_loop; ++k )
	{
	    for( i = 0; i < numel_spring_loop; ++i )
	    {
		for( j = 0; j < numel_cross_sec-1; ++j )
		{
		    fprintf(o1, "%4d   ",dum);
		    ++dum;
		    fprintf(o1, "%4d %4d %4d %4d %4d %4d %4d\n",
			numnp_cross_sec*i + numnp_loop*k,
			j+2 + numnp_cross_sec*i + numnp_loop*k,
			j+1 + numnp_cross_sec*i + numnp_loop*k,
			numnp_cross_sec*(i+1) + numnp_loop*k,
			j+2 + numnp_cross_sec*(i+1) + numnp_loop*k,
			j+1 + numnp_cross_sec*(i+1) + numnp_loop*k,
			0);
		}
		fprintf(o1, "%4d   ",dum);
		++dum;
		j = numel_cross_sec-1;
		fprintf(o1, "%4d %4d %4d %4d %4d %4d %4d\n",
		    numnp_cross_sec*i + numnp_loop*k,
		    1 + numnp_cross_sec*i + numnp_loop*k,
		    j+1 + numnp_cross_sec*i + numnp_loop*k,
		    numnp_cross_sec*(i+1) + numnp_loop*k,
		    1 + numnp_cross_sec*(i+1) + numnp_loop*k,
		    j+1 + numnp_cross_sec*(i+1) + numnp_loop*k,
		    0);
	    }
	}

	if( loop_up_flag )
	{
	    for( i = 0; i < numel_spring_loop/3 - 1; ++i )
	    {
		for( j = 0; j < numel_cross_sec-1; ++j )
		{
		    fprintf(o1, "%4d   ",dum);
		    ++dum;
		    fprintf(o1, "%4d %4d %4d %4d %4d %4d %4d\n",
			numnp_cross_sec*i + numnp_loop*k,
			j+2 + numnp_cross_sec*i + numnp_loop*k,
			j+1 + numnp_cross_sec*i + numnp_loop*k,
			numnp_cross_sec*(i+1) + numnp_loop*k,
			j+2 + numnp_cross_sec*(i+1) + numnp_loop*k,
			j+1 + numnp_cross_sec*(i+1) + numnp_loop*k,
			0);
		}
		fprintf(o1, "%4d   ",dum);
		++dum;
		j = numel_cross_sec-1;
		fprintf(o1, "%4d %4d %4d %4d %4d %4d %4d\n",
		    numnp_cross_sec*i + numnp_loop*k,
		    1 + numnp_cross_sec*i + numnp_loop*k,
		    j+1 + numnp_cross_sec*i + numnp_loop*k,
		    numnp_cross_sec*(i+1) + numnp_loop*k,
		    1 + numnp_cross_sec*(i+1) + numnp_loop*k,
		    j+1 + numnp_cross_sec*(i+1) + numnp_loop*k,
		    0);
	    }
	}

	fprintf(o1, "node no. coordinates\n");

	c_arc_angle=0.0;
	dum = 0;
	dspring_angle = 2*pi/(double)numel_cross_sec;
	darc_angle = 2.0*pi/(double)numel_spring_loop;
	dloop_space_per_el = (loop_space + 2*Ro)/(double)(numel_spring_loop);
	dloop_space = (loop_space + 2*Ro);

/* Everything inside the "k" loop forms the main base of the spring. */

	dY_per_loop = 0.0;
	for( k = 0; k < num_loop; ++k )
	{
	    dY_per_el = 0.0;
	    c_arc_angle = 0.0;
	    for( i = 0; i < numel_spring_loop; ++i )
	    {
		cos_arc_ang = cos(c_arc_angle);
		sin_arc_ang = sin(c_arc_angle);
		c_spring_angle=0.0;

		X= loop_R*sin_arc_ang;
		Y= dY_per_el + dY_per_loop;
		Z= loop_R*cos_arc_ang;
		fprintf(o1, "%4d ",dum);
		++dum;
		fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

		for( j = 0; j < numel_cross_sec; ++j )
		{
		    X = (Ro*cos(c_spring_angle) + loop_R)*sin_arc_ang;
		    Y = Ro*sin(c_spring_angle) + dY_per_el + dY_per_loop;
		    Z = (Ro*cos(c_spring_angle) + loop_R)*cos_arc_ang;
		    fprintf(o1, "%4d ",dum);
		    ++dum;
		    fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

		    c_spring_angle += dspring_angle;
		}
		dY_per_el += dloop_space_per_el;
		c_arc_angle += darc_angle;
	    }
	    dY_per_loop += dloop_space;
	}

#if 0

/* This section is one additional cross section which falls on the plane of y=0, z=0.

   This code being commented out was used to generate "spring1" in "~/data/we/" as
   well as "tespring" in "~/data/te/".  These meshes are very big and take a long
   time to run so I don't want to regenerate the slightly different meshes with my
   modification and re-run everything.
*/

	cos_arc_ang = cos(c_arc_angle);
	sin_arc_ang = sin(c_arc_angle);
	c_spring_angle=0.0;

	X= loop_R*sin_arc_ang;
	Y= dY_per_loop + dloop_space_per_el;
	Z= loop_R*cos_arc_ang;
	fprintf(o1, "%4d ",dum);
	++dum;
	fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

	for( j = 0; j < numel_cross_sec; ++j )
	{
	    X = Ro*cos(c_spring_angle)*sin_arc_ang + loop_R*sin_arc_ang;
	    Y = Ro*sin(c_spring_angle) + dY_per_el + dY_per_loop - (dloop_space_per_el + dloop_space);
	    Z = Ro*cos(c_spring_angle)*cos_arc_ang + loop_R*cos_arc_ang;
	    fprintf(o1, "%4d ",dum);
	    ++dum;
	    fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

	    c_spring_angle += dspring_angle;
	}

#endif

	if( !loop_up_flag )
	{

/* This section semi-replaces the above code. */

	    dY_per_el = 0.0;
	    c_arc_angle = 0.0;
	    cos_arc_ang = cos(c_arc_angle);
	    sin_arc_ang = sin(c_arc_angle);
	    c_spring_angle=0.0;

	    X= loop_R*sin_arc_ang;
	    Y= dY_per_el + dY_per_loop;
	    Z= loop_R*cos_arc_ang;
	    fprintf(o1, "%4d ",dum);
	    ++dum;
	    fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

	    for( j = 0; j < numel_cross_sec; ++j )
	    {
		X = Ro*cos(c_spring_angle)*sin_arc_ang + loop_R*sin_arc_ang;
		Y = Ro*sin(c_spring_angle) + dY_per_el + dY_per_loop;
		Z = Ro*cos(c_spring_angle)*cos_arc_ang + loop_R*cos_arc_ang;
		fprintf(o1, "%4d ",dum);
		++dum;
		fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

		c_spring_angle += dspring_angle;
	    }

	}

/* This section will curve the last loop of the spring upward so that the last face is
   parallel to the y axis.
*/
	if( loop_up_flag )
	{
	    dum2 = numel_spring_loop%3;
	    dum2 = numel_spring_loop - dum2;
	    dum2 /= 3;

	    dY_per_el = 0.0;
	    c_arc_angle = 0.0;
	    loop_R2 = loop_R/2.0;
	    for( i = 0; i < dum2; ++i )
	    {
		cos_arc_ang = cos(c_arc_angle);
		sin_arc_ang = sin(c_arc_angle);
		c_spring_angle=0.0;

		X= loop_R2*sin_arc_ang;
		Y= (1.0 - cos_arc_ang)*loop_R2 + dY_per_loop;
		Z= loop_R;
		fprintf(o1, "%4d ",dum);
		++dum;
		fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

		for( j = 0; j < numel_cross_sec; ++j )
		{
		    X = (-Ro*sin(c_spring_angle) + loop_R2)*sin_arc_ang;
		    Y = loop_R2 - (loop_R2 - Ro*sin(c_spring_angle))*cos_arc_ang + dY_per_loop;
		    Z = Ro*cos(c_spring_angle) + loop_R;
		    fprintf(o1, "%4d ",dum);
		    ++dum;
		    fprintf(o1, "%9.5f %9.5f %9.5f \n", X, Y, Z);

		    c_spring_angle += dspring_angle;
		}
		dY_per_el += dloop_space_per_el;
		c_arc_angle += darc_angle;
	    }
	    dY_per_loop += dloop_space;
	}


	fprintf( o1, "prescribed displacement x: node  disp value\n");
	for( j = 0; j < numnp_cross_sec; ++j )
	{
		fprintf(o1, "%4d %9.5f \n", j, 0.0);
	}
	fprintf( o1, "%4d\n ",-10);
	fprintf( o1, "prescribed displacement y: node  disp value\n");
	for( j = 0; j < numnp_cross_sec; ++j )
	{
		fprintf(o1, "%4d %9.5f \n", j, 0.0);
	}
	fprintf( o1, "%4d\n ",-10);
	fprintf( o1, "prescribed displacement z: node  disp value\n");
	for( j = 0; j < numnp_cross_sec; ++j )
	{
		fprintf(o1, "%4d %9.5f \n", j, 0.0);
	}
	fprintf( o1, "%4d\n ",-10);
	fprintf( o1, "node with point load and load vector in x,y,z\n");
	if( !loop_up_flag )
	{
	    for( j = numnp-numnp_cross_sec; j < numnp; ++j )
	    {
		fprintf(o1, "%4d %9.5f  %9.5f  %9.5f\n",
			j, 20.0, 0.0, 0.0);
	    }
	}
	else
	{
	    for( j = numnp-numnp_cross_sec; j < numnp; ++j )
	    {
		fprintf(o1, "%4d %9.5f  %9.5f  %9.5f\n",
			j, 0.0, 20.0, 0.0);
	    }
	}
	fprintf( o1, "%4d\n ",-10);
	fprintf( o1, "node no. with stress and stress vector xx,yy,xy,zx,yz\n");
	fprintf( o1, "%4d ",-10);

	return 1;
}

main(int argc, char** argv)
{
	int check;
	check=mesh_spring();
} 
