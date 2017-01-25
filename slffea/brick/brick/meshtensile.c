/*
     This program generates the mesh for a tensile test.

		Updated 1/15/07

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define nsd             3
#define pi              3.141592654
#define sq2             1.414213462
#define SMALL           1.e-20

int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, nmode;
	int i, i2, j, k, dum, dum2, dum3, dum3a, dum4, dum5, dum6, dum7, dum8, dum9, dum10;
	double fdum, fdum1, fdum2, fdum3, fdum4, ux[9], uy[9];
	double angle, height_length, square_length, cylinder_length, ray_angle,
		height_el_length, square_el_length, cylinder_el_length,
		ratio;
	double *coord;
	int height_el_num, half_height_el_num, square_el_num, cylinder_el_num, angle_div,
		angleh_div, nodes_per_layer;
	char name[30];
	char buf[ BUFSIZ ];
	FILE *o1;
	char text;

	o1 = fopen( "ttest","w" );
	printf( "\n What is the length of the inner square?\n ");
	scanf( "%lf",&square_length);
	printf( "\n How many elements on the inner square?\n ");
	scanf( "%d",&square_el_num);
	printf( "\n What is the length of the cylinder height?\n ");
	scanf( "%lf",&height_length);
	printf( "\n How many elements on the cylinder height?\n ");
	scanf( "%d",&height_el_num);
	printf( "\n What is the length of the outer cylinder?\n ");
	scanf( "%lf",&cylinder_length);
	printf( "\n How many elements on the outer cylinder?\n ");
	scanf( "%d",&cylinder_el_num);
	printf( "\n What is the ratio?(ratio > 1)\n ");
	scanf( "%lf",&ratio);

	ray_angle = 45.0/((double)square_el_num);
	angle_div = (int)(90.0/ray_angle);
	angleh_div = (int)(((double)angle_div)/2.0);
	/*nodes_per_layer = (angle_div+1)*(cylinder_el_num+square_el_num+1);*/
	nodes_per_layer = (square_el_num+1)*(square_el_num+1) +
	    (angle_div + 1)*(cylinder_el_num) +
	    2*(square_el_num)*(square_el_num+1) +
	    2*(angle_div)*(cylinder_el_num) +
	    (square_el_num)*(square_el_num) +
	    (angle_div - 1)*(cylinder_el_num);

	nmat = 1; 

	nmode = 0;
	numel = (square_el_num*square_el_num + (angle_div)*(cylinder_el_num))*height_el_num;
	numel *= 4;
	numnp = nodes_per_layer*(height_el_num+2);
	coord=(double *)calloc(nsd*numnp,sizeof(double));

	half_height_el_num = (height_el_num - height_el_num%2)/2;
	cylinder_el_length = cylinder_length/((double)cylinder_el_num);
	square_el_length = square_length/((double)square_el_num);
	height_el_length = height_length/((double)height_el_num);
	printf( "    %4d %4d %4d %4d \n ", numel,numnp,nmat,nodes_per_layer);
	printf( "    %10.6f %10.6f %10.6f\n ", square_el_length, height_el_length, cylinder_el_length );

	fprintf( o1, "   numel numnp nmat nmode  This is the Saint Venant mesh \n ");
	fprintf( o1, "    %4d %4d %4d %4d\n ", numel,numnp,nmat,nmode);

	fprintf( o1, "matl no., E modulus, Poisson Ratio, density \n");
	for( i = 0; i < nmat; ++i )
	{
	   fprintf( o1, " %3d",0);
	   fprintf( o1, " %14.6e %14.6e %14.6e\n ", 2.410000e+11, 0.2, 2.77e3);
	}

	ray_angle *= pi/180.0;
	dum = 0;
	dum2 = 0;
	dum3 = (square_el_num+cylinder_el_num+1);
	dum3a = (square_el_num+cylinder_el_num);
	dum4 = 0;
	dum6 = 0;
	dum8 = (square_el_num+1)*(square_el_num+1) +
	    (angle_div + 1)*(cylinder_el_num) +
	    2*(square_el_num)*(square_el_num+1) +
	    2*(angle_div)*(cylinder_el_num);
	dum9 = 0;

	fprintf( o1, "el no., connectivity, matl no. \n");
	for( k = 0; k < height_el_num; ++k )
	{
	    dum5 = 0;

/* This is the 1st quarter */

	    for( i = 0; i < square_el_num; ++i )
	    {
		for( j = 0; j < square_el_num + cylinder_el_num; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    i*dum3 + j + dum2 - dum9,
			    i*dum3 + j + 1 + dum2 - dum9,
			    (i+1)*dum3 + j + 1 + dum2 - dum9,
			    (i+1)*dum3 + j + dum2 - dum9,
			    i*dum3 + j + nodes_per_layer + dum2,
			    i*dum3 + j + 1 + nodes_per_layer + dum2,
			    (i+1)*dum3 + j + 1 + nodes_per_layer + dum2,
			    (i+1)*dum3 + j + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	    }
	    dum4 = square_el_num*dum3;
	    for( j = 0; j < square_el_num - 1; ++j )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		    dum4 + j + dum2 - dum9,
		    dum4 + j + 1 + dum2 - dum9,
		    dum4 + j + dum3 + 1 + dum2 - dum9,
		    dum4 + j + dum3 + dum2 - dum9,
		    dum4 + j + nodes_per_layer + dum2,
		    dum4 + j + 1 + nodes_per_layer + dum2,
		    dum4 + j + dum3 + 1 + nodes_per_layer + dum2,
		    dum4 + j + dum3 + nodes_per_layer + dum2,
		    0);
		fprintf( o1, "\n");
		++dum;
	    }
	    dum4 = (1+square_el_num)*dum3;

	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		for( j = 0; j < square_el_num - 1; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    dum4 + i*square_el_num + j + dum2 - dum9,
			    dum4 + i*square_el_num + j + 1 + dum2 - dum9,
			    dum4 + (i+1)*square_el_num + j + 1 + dum2 - dum9,
			    dum4 + (i+1)*square_el_num + j + dum2 - dum9,
			    dum4 + i*square_el_num + j + nodes_per_layer + dum2,
			    dum4 + i*square_el_num + j + 1 + nodes_per_layer + dum2,
			    dum4 + (i+1)*square_el_num + j + 1 + nodes_per_layer + dum2,
			    dum4 + (i+1)*square_el_num + j + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	    }
	    dum4 = square_el_num*dum3;
	    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		dum4 + square_el_num - 1 + dum2 - dum9,
		dum4 + square_el_num + dum2 - dum9,
		dum4 + square_el_num + 1 + dum2 - dum9,
		dum4 + dum3 - 1 + square_el_num + dum2 - dum9,
		dum4 + square_el_num - 1 + nodes_per_layer + dum2,
		dum4 + square_el_num + nodes_per_layer + dum2,
		dum4 + square_el_num + 1 + nodes_per_layer + dum2,
		dum4 + dum3 - 1 + square_el_num + nodes_per_layer + dum2,
		0);
	    fprintf( o1, "\n");
	    ++dum;
	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			(i+1)*square_el_num - 1 + dum3 + dum4 + dum2 - dum9,
			dum4 + square_el_num + i + 1 + dum2 - dum9,
			dum4 + square_el_num + i + 2 + dum2 - dum9,
			(i+1)*square_el_num - 1 + dum3 + dum4 + square_el_num + dum2 - dum9,
			(i+1)*square_el_num - 1 + dum3 + dum4 + nodes_per_layer + dum2,
			dum4 + square_el_num + i + 1 + nodes_per_layer + dum2,
			dum4 + square_el_num + i + 2 + nodes_per_layer + dum2,
			(i+1)*square_el_num - 1 + dum3 + dum4 + square_el_num + nodes_per_layer + dum2,
			0);
		fprintf( o1, "\n");
		++dum;
	    }

/* This is the 2nd quarter */

	    dum5 += (square_el_num+1)*(square_el_num+1) + (angle_div+1)*(cylinder_el_num);

	    for( j = 0; j < square_el_num + 1; ++j )
	    {
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    j*(square_el_num + cylinder_el_num + 1) + dum2 - dum9,
			    (j + 1)*(square_el_num + cylinder_el_num + 1) + dum2 - dum9,
			    j + 1 + dum5 + dum2 - dum9,
			    j + dum5 + dum2 - dum9,
			    j*(square_el_num + cylinder_el_num + 1) + nodes_per_layer + dum2,
			    (j + 1)*(square_el_num + cylinder_el_num + 1) + nodes_per_layer + dum2,
			    j + 1 + dum5 + nodes_per_layer + dum2,
			    j + dum5 + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
	    }

	    dum6 = (square_el_num + 1)*(square_el_num + cylinder_el_num + 1);
	    for( j = 1; j < cylinder_el_num; ++j )
	    {
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (j - 1)*(square_el_num) + dum6 + dum2 - dum9,
			    j*square_el_num + dum6 + dum2 - dum9,
			    j + square_el_num + 1 + dum5 + dum2 - dum9,
			    j + square_el_num + dum5 + dum2 - dum9,
			    (j - 1)*(square_el_num) + dum6 + nodes_per_layer + dum2,
			    j*square_el_num + dum6 + nodes_per_layer + dum2,
			    j + square_el_num + 1 + dum5 + nodes_per_layer + dum2,
			    j + square_el_num + dum5 + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
	    }

	    for( i = 1; i < square_el_num; ++i )
	    {
		for( j = 0; j < square_el_num + cylinder_el_num; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (i-1)*dum3 + j + dum5 + dum2 - dum9,
			    (i-1)*dum3 + j + 1 + dum5 + dum2 - dum9,
			    i*dum3 + j + 1 + dum5 + dum2 - dum9,
			    i*dum3 + j + dum5 + dum2 - dum9,
			    (i-1)*dum3 + j + nodes_per_layer + dum5 + dum2,
			    (i-1)*dum3 + j + 1 + nodes_per_layer + dum5 + dum2,
			    i*dum3 + j + 1 + nodes_per_layer + dum5 + dum2,
			    i*dum3 + j + nodes_per_layer + dum5 + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	    }
	    dum4 = square_el_num*dum3;
	    for( j = 0; j < square_el_num - 1; ++j )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		    dum3*(square_el_num - 1) + j + dum5 + dum2 - dum9,
		    dum3*(square_el_num - 1) + j + 1 + dum5 + dum2 - dum9,
		    dum3*square_el_num + j + 1 + dum5 + dum2 - dum9,
		    dum3*square_el_num + j + dum5 + dum2 - dum9,
		    dum3*(square_el_num - 1) + j + nodes_per_layer + dum5 + dum2,
		    dum3*(square_el_num - 1) + j + 1 + nodes_per_layer + dum5 + dum2,
		    dum3*square_el_num + j + 1 + nodes_per_layer + dum5 + dum2,
		    dum3*square_el_num + j + nodes_per_layer + dum5 + dum2,
		    0);
		fprintf( o1, "\n");
		++dum;
	    }
	    dum4 = (1+square_el_num)*dum3;
	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		for( j = 0; j < square_el_num - 1; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    dum3*square_el_num + i*square_el_num + j + dum5 + dum2 - dum9,
			    dum3*square_el_num + i*square_el_num + j + 1 + dum5 + dum2 - dum9,
			    dum3*square_el_num + (i+1)*square_el_num + j + 1 + dum5 + dum2 - dum9,
			    dum3*square_el_num + (i+1)*square_el_num + j + dum5 + dum2 - dum9,
			    dum3*square_el_num + i*square_el_num + j + nodes_per_layer + dum5 + dum2,
			    dum3*square_el_num + i*square_el_num + j + 1 + nodes_per_layer + dum5 + dum2,
			    dum3*square_el_num + (i+1)*square_el_num + j + 1 + nodes_per_layer + dum5 + dum2,
			    dum3*square_el_num + (i+1)*square_el_num + j + nodes_per_layer + dum5 + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	    }
	    dum4 = square_el_num*dum3;
	    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		dum3*(square_el_num-1) + square_el_num - 1 + dum5 + dum2 - dum9,
		dum3*(square_el_num-1) + square_el_num + dum5 + dum2 - dum9,
		dum3*(square_el_num-1) + square_el_num + 1 + dum5 + dum2 - dum9,
		dum3*square_el_num - 1 + square_el_num + dum5 + dum2 - dum9,
		dum3*(square_el_num-1) + square_el_num - 1 + nodes_per_layer + dum5 + dum2,
		dum3*(square_el_num-1) + square_el_num + nodes_per_layer + dum5 + dum2,
		dum3*(square_el_num-1) + square_el_num + 1 + nodes_per_layer + dum5 + dum2,
		dum3*square_el_num - 1 + square_el_num + nodes_per_layer + dum5 + dum2,
		0);
	    fprintf( o1, "\n");
	    ++dum;
	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			(i+1+dum3)*square_el_num - 1 + dum5 + dum2 - dum9,
			dum3*(square_el_num-1) + square_el_num + i + 1 + dum5 + dum2 - dum9,
			dum3*(square_el_num-1) + square_el_num + i + 2 + dum5 + dum2 - dum9,
			(i+1+dum3)*square_el_num - 1 + square_el_num + dum5 + dum2 - dum9,
			(i+1+dum3)*square_el_num - 1 + nodes_per_layer + dum5 + dum2,
			dum3*(square_el_num-1) + square_el_num + i + 1 + nodes_per_layer + dum5 + dum2,
			dum3*(square_el_num-1) + square_el_num + i + 2 + nodes_per_layer + dum5 + dum2,
			(i+1+dum3)*square_el_num - 1 + square_el_num + nodes_per_layer + dum5 + dum2,
			0);
		fprintf( o1, "\n");
		++dum;
	    }

/* This is the 3rd quarter */

	    /*dum7 = dum5;*/
	    dum7 = (square_el_num)*(square_el_num+1) + (angle_div)*(cylinder_el_num);
	    dum5 += (square_el_num)*(square_el_num+1) + (angle_div)*(cylinder_el_num);

	    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		dum2 - dum9,
		(square_el_num + cylinder_el_num + 1) + dum7 + dum2 - dum9,
		1 + dum5 + dum2 - dum9,
		dum5 + dum2 - dum9,
		nodes_per_layer + dum2,
		(square_el_num + cylinder_el_num + 1) + dum7 + nodes_per_layer + dum2,
		1 + dum5 + nodes_per_layer + dum2,
		dum5 + nodes_per_layer + dum2,
		0);
	    fprintf( o1, "\n");
	    ++dum;

	    for( j = 1; j < square_el_num + 1; ++j )
	    {
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    j*(square_el_num + cylinder_el_num + 1) + dum7 + dum2 - dum9,
			    (j + 1)*(square_el_num + cylinder_el_num + 1) + dum7 + dum2 - dum9,
			    j + 1 + dum5 + dum2 - dum9,
			    j + dum5 + dum2 - dum9,
			    j*(square_el_num + cylinder_el_num + 1) + dum7 + nodes_per_layer + dum2,
			    (j + 1)*(square_el_num + cylinder_el_num + 1) + dum7 + nodes_per_layer + dum2,
			    j + 1 + dum5 + nodes_per_layer + dum2,
			    j + dum5 + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
	    }

	    dum6 = (square_el_num + 1)*(square_el_num + cylinder_el_num + 1);
	    for( j = 1; j < cylinder_el_num; ++j )
	    {
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (j - 1)*(square_el_num) + dum6 + dum7 + dum2 - dum9,
			    j*square_el_num + dum6 + dum7 + dum2 - dum9,
			    j + square_el_num + 1 + dum5 + dum2 - dum9,
			    j + square_el_num + dum5 + dum2 - dum9,
			    (j - 1)*(square_el_num) + dum6 + dum7 + nodes_per_layer + dum2,
			    j*square_el_num + dum6 + dum7 + nodes_per_layer + dum2,
			    j + square_el_num + 1 + dum5 + nodes_per_layer + dum2,
			    j + square_el_num + dum5 + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
	    }

	    for( i = 1; i < square_el_num; ++i )
	    {
		for( j = 0; j < square_el_num + cylinder_el_num; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (i-1)*dum3 + j + dum5 + dum2 - dum9,
			    (i-1)*dum3 + j + 1 + dum5 + dum2 - dum9,
			    i*dum3 + j + 1 + dum5 + dum2 - dum9,
			    i*dum3 + j + dum5 + dum2 - dum9,
			    (i-1)*dum3 + j + nodes_per_layer + dum5 + dum2,
			    (i-1)*dum3 + j + 1 + nodes_per_layer + dum5 + dum2,
			    i*dum3 + j + 1 + nodes_per_layer + dum5 + dum2,
			    i*dum3 + j + nodes_per_layer + dum5 + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	    }
	    dum4 = square_el_num*dum3;
	    for( j = 0; j < square_el_num - 1; ++j )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		    dum3*(square_el_num-1) + j + dum5 + dum2 - dum9,
		    dum3*(square_el_num-1) + j + 1 + dum5 + dum2 - dum9,
		    dum3*square_el_num + j + 1 + dum5 + dum2 - dum9,
		    dum3*square_el_num + j + dum5 + dum2 - dum9,
		    dum3*(square_el_num-1) + j + nodes_per_layer + dum5 + dum2,
		    dum3*(square_el_num-1) + j + 1 + nodes_per_layer + dum5 + dum2,
		    dum3*square_el_num + j + 1 + nodes_per_layer + dum5 + dum2,
		    dum3*square_el_num + j + nodes_per_layer + dum5 + dum2,
		    0);
		fprintf( o1, "\n");
		++dum;
	    }
	    dum4 = (1+square_el_num)*dum3;
	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		for( j = 0; j < square_el_num - 1; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (dum3+i)*square_el_num + j + dum5 + dum2 - dum9,
			    (dum3+i)*square_el_num + j + 1 + dum5 + dum2 - dum9,
			    (dum3+i+1)*square_el_num + j + 1 + dum5 + dum2 - dum9,
			    (dum3+i+1)*square_el_num + j + dum5 + dum2 - dum9,
			    (dum3+i)*square_el_num + j + nodes_per_layer + dum5 + dum2,
			    (dum3+i)*square_el_num + j + 1 + nodes_per_layer + dum5 + dum2,
			    (dum3+i+1)*square_el_num + j + 1 + nodes_per_layer + dum5 + dum2,
			    (dum3+i+1)*square_el_num + j + nodes_per_layer + dum5 + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	    }
	    dum4 = square_el_num*dum3;
	    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		dum3*(square_el_num-1) + square_el_num - 1 + dum5 + dum2 - dum9,
		dum3*(square_el_num-1) + square_el_num + dum5 + dum2 - dum9,
		dum3*(square_el_num-1) + square_el_num + 1 + dum5 + dum2 - dum9,
		dum3*square_el_num - 1 + square_el_num + dum5 + dum2 - dum9,
		dum3*(square_el_num-1) + square_el_num - 1 + nodes_per_layer + dum5 + dum2,
		dum3*(square_el_num-1) + square_el_num + nodes_per_layer + dum5 + dum2,
		dum3*(square_el_num-1) + square_el_num + 1 + nodes_per_layer + dum5 + dum2,
		dum3*square_el_num - 1 + square_el_num + nodes_per_layer + dum5 + dum2,
		0);
	    fprintf( o1, "\n");
	    ++dum;
	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			(i+1+dum3)*square_el_num - 1 + dum5 + dum2 - dum9,
			dum3*(square_el_num-1) + square_el_num + i + 1 + dum5 + dum2 - dum9,
			dum3*(square_el_num-1) + square_el_num + i + 2 + dum5 + dum2 - dum9,
			(i+1+dum3)*square_el_num - 1 + square_el_num + dum5 + dum2 - dum9,
			(i+1+dum3)*square_el_num - 1 + nodes_per_layer + dum5 + dum2,
			dum3*(square_el_num-1) + square_el_num + i + 1 + nodes_per_layer + dum5 + dum2,
			dum3*(square_el_num-1) + square_el_num + i + 2 + nodes_per_layer + dum5 + dum2,
			(i+1+dum3)*square_el_num - 1 + square_el_num + nodes_per_layer + dum5 + dum2,
			0);
		fprintf( o1, "\n");
		++dum;
	    }


/* This is the 4th quarter */

	    /*dum7 = dum5;
	    dum5 += (square_el_num)*(square_el_num+1) + (angle_div)*(cylinder_el_num);*/
	    dum7 = 2*((square_el_num)*(square_el_num+1) + (angle_div)*(cylinder_el_num));
	    dum5 += (square_el_num)*(square_el_num+1) + (angle_div)*(cylinder_el_num);

	    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		dum2 - dum9,
		dum3 + dum7 + dum2 - dum9,
		dum5 + dum2 - dum9,
		1 + dum2 - dum9,
		nodes_per_layer + dum2,
		dum3 + dum7 + nodes_per_layer + dum2,
		dum5 + nodes_per_layer + dum2,
		1 + nodes_per_layer + dum2,
	    0);
	    fprintf( o1, "\n");
	    ++dum;

	    for( j = 1; j < square_el_num + 1; ++j )
	    {
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    j*(square_el_num + cylinder_el_num + 1) + dum7 + dum2 - dum9,
			    (j + 1)*(square_el_num + cylinder_el_num + 1) + dum7 + dum2 - dum9,
			    j + dum5 + dum2 - dum9,
			    j - 1 + dum5 + dum2 - dum9,
			    j*(square_el_num + cylinder_el_num + 1) + dum7 + nodes_per_layer + dum2,
			    (j + 1)*(square_el_num + cylinder_el_num + 1) + dum7 + nodes_per_layer + dum2,
			    j + dum5 + nodes_per_layer + dum2,
			    j - 1 + dum5 + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
	    }

	    dum6 = (square_el_num + 1)*(square_el_num + cylinder_el_num + 1);
	    for( j = 1; j < cylinder_el_num; ++j )
	    {
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (j - 1)*(square_el_num) + dum6 + dum7 + dum2 - dum9,
			    j*square_el_num + dum6 + dum7 + dum2 - dum9,
			    j + square_el_num + dum5 + dum2 - dum9,
			    j - 1 + square_el_num + dum5 + dum2 - dum9,
			    (j - 1)*(square_el_num) + dum6 + dum7 + nodes_per_layer + dum2,
			    j*square_el_num + dum6 + dum7 + nodes_per_layer + dum2,
			    j + square_el_num + dum5 + nodes_per_layer + dum2,
			    j - 1 + square_el_num + dum5 + nodes_per_layer + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
	    }

	    for( i = 1; i < square_el_num; ++i )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		    i + dum2 - dum9,
		    (i-1)*dum3a + dum5 + dum2 - dum9,
		    (i)*dum3a + dum5 + dum2 - dum9,
		    i + 1 + dum2 - dum9,
		    i + nodes_per_layer + dum2,
		    (i-1)*dum3a + dum5 + nodes_per_layer + dum2,
		    (i)*dum3a + dum5 + nodes_per_layer + dum2,
		    i + 1 + nodes_per_layer + dum2,
		0);
		fprintf( o1, "\n");
		++dum;

		for( j = 0; j < square_el_num + cylinder_el_num - 1; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (i-1)*dum3a + j + dum5 + dum2 - dum9,
			    (i-1)*dum3a + j + 1 + dum5 + dum2 - dum9,
			    i*dum3a + 1 + j + dum5 + dum2 - dum9,
			    i*dum3a + j + dum5 + dum2 - dum9,
			    (i-1)*dum3a + j + nodes_per_layer + dum5 + dum2,
			    (i-1)*dum3a + j + 1 + nodes_per_layer + dum5 + dum2,
			    i*dum3a + 1 + j + nodes_per_layer + dum5 + dum2,
			    i*dum3a + j + nodes_per_layer + dum5 + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	    }

	    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		    square_el_num + dum2 - dum9,
		    dum3a*(square_el_num-1) + dum5 + dum2 - dum9,
		    dum3a*square_el_num + dum5 + dum2 - dum9,
		    square_el_num + 1 + dum2 - dum9,
		    square_el_num + nodes_per_layer + dum2,
		    dum3a*(square_el_num-1) + dum5 + nodes_per_layer + dum2,
		    dum3a*square_el_num + dum5 + nodes_per_layer + dum2,
		    square_el_num + 1 + nodes_per_layer + dum2,
	    0);
	    fprintf( o1, "\n");
	    ++dum;

	    dum4 = square_el_num*dum3a;
	    for( j = 1; j < square_el_num - 1; ++j )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		    dum3a*(square_el_num-1) - 1 + j + dum5 + dum2 - dum9,
		    dum3a*(square_el_num-1) + j + dum5 + dum2 - dum9,
		    dum3a*square_el_num + j + dum5 + dum2 - dum9,
		    dum3a*square_el_num -1 + j + dum5 + dum2 - dum9,
		    dum3a*(square_el_num-1) -1 + j + nodes_per_layer + dum5 + dum2,
		    dum3a*(square_el_num-1) + j + nodes_per_layer + dum5 + dum2,
		    dum3a*square_el_num + j + nodes_per_layer + dum5 + dum2,
		    dum3a*square_el_num - 1 + j + nodes_per_layer + dum5 + dum2,
		    0);
		fprintf( o1, "\n");
		++dum;
	    }
	    dum4 = (1+square_el_num)*dum3a;
	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		dum10 = 0;
		if(i == 0) dum10 = dum9;
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		    square_el_num + i + 1 + dum2 - dum9,
		    i*(square_el_num-1) + dum3a*(square_el_num) + dum5 + dum2 - dum10,
		    (i+1)*(square_el_num-1) + dum3a*(square_el_num) + dum5 + dum2,
		    square_el_num + i + 2 + dum2 - dum9,
		    square_el_num + i + 1 + nodes_per_layer + dum2,
		    i*(square_el_num-1) + dum3a*(square_el_num) + dum5 + nodes_per_layer + dum2,
		    (i+1)*(square_el_num-1) + dum3a*(square_el_num) + dum5 + nodes_per_layer + dum2,
		    square_el_num + i + 2 + nodes_per_layer + dum2,
		0);
		fprintf( o1, "\n");
		++dum;

		for( j = 0; j < square_el_num - 3; ++j )
		{
		    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (dum3a+i)*square_el_num - i + j + dum5 + dum2 - dum10,
			    (dum3a+i)*square_el_num - i + j + 1 + dum5 + dum2 - dum10,
			    (dum3a+i+1)*square_el_num - i + j + dum5 + dum2,
			    (dum3a+i+1)*square_el_num - i + j - 1 + dum5 + dum2,
			    (dum3a+i)*square_el_num - i + j + nodes_per_layer + dum5 + dum2,
			    (dum3a+i)*square_el_num - i + j + 1 + nodes_per_layer + dum5 + dum2,
			    (dum3a+i+1)*square_el_num - i + j + nodes_per_layer + dum5 + dum2,
			    (dum3a+i+1)*square_el_num - i + j - 1 + nodes_per_layer + dum5 + dum2,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}

		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			    (dum3a+i)*square_el_num - i + square_el_num - 3 + dum5 + dum2 - dum10,
			    (dum3a+i)*square_el_num - i + square_el_num - 3 + 1 + dum5 + dum2 - dum9,
			    (dum3a+i+1)*square_el_num - i + square_el_num - 3 + dum5 + dum2 - dum9,
			    (dum3a+i+1)*square_el_num - i + square_el_num - 3 - 1 + dum5 + dum2,
			    (dum3a+i)*square_el_num - i + square_el_num - 3 + nodes_per_layer + dum5 + dum2,
			    (dum3a+i)*square_el_num - i + square_el_num - 3 + 1 + nodes_per_layer + dum5 + dum2,
			    (dum3a+i+1)*square_el_num - i + square_el_num - 3 + nodes_per_layer + dum5 + dum2,
			    (dum3a+i+1)*square_el_num - i + square_el_num - 3 - 1 + nodes_per_layer + dum5 + dum2,
			    0);
		fprintf( o1, "\n");
		++dum;

	    }
	    dum4 = square_el_num*dum3a;
	    fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
		dum3a*(square_el_num-1) + square_el_num - 2 + dum5 + dum2 - dum9,
		dum3a*(square_el_num-1) + square_el_num - 1 + dum5 + dum2 - dum9,
		dum3a*(square_el_num-1) + square_el_num + dum5 + dum2 - dum9,
		dum3a*square_el_num - 2 + square_el_num + dum5 + dum2 - dum9,
		dum3a*(square_el_num-1) + square_el_num - 2 + nodes_per_layer + dum5 + dum2,
		dum3a*(square_el_num-1) + square_el_num - 1 + nodes_per_layer + dum5 + dum2,
		dum3a*(square_el_num-1) + square_el_num + nodes_per_layer + dum5 + dum2,
		dum3a*square_el_num - 2 + square_el_num + nodes_per_layer + dum5 + dum2,
		0);
	    fprintf( o1, "\n");
	    ++dum;
	    for( i = 0; i < cylinder_el_num-1; ++i )
	    {
		fprintf( o1, "%6d %9d %9d %9d %9d %9d %9d %9d %9d %9d",dum,
			- i + (i+1)*square_el_num - 2 + dum4 + dum5 + dum2 - dum9,
			dum3a*(square_el_num-1) + square_el_num + i + dum5 + dum2 - dum9,
			dum3a*(square_el_num-1) + square_el_num + i + 1 + dum5 + dum2 - dum9,
			- i + (i+1)*square_el_num - 3 + dum4 + square_el_num + dum5 + dum2 - dum9,
			- i + (i+1)*square_el_num - 2 + dum4 + nodes_per_layer + dum5 + dum2,
			dum3a*(square_el_num-1) + square_el_num + i + nodes_per_layer + dum5 + dum2,
			dum3a*(square_el_num-1) + square_el_num + i + 1 + nodes_per_layer + dum5 + dum2,
			- i + (i+1)*square_el_num - 3 + dum4 + square_el_num + nodes_per_layer + dum5 + dum2,
			0);
		fprintf( o1, "\n");
		++dum;
	    }

	    dum2 += nodes_per_layer;
	    dum9 = 0;
	    if( k == half_height_el_num-1 )
	    {
		dum2 += nodes_per_layer;
		dum9 = nodes_per_layer;
	    }
	}

/* These are the nodes for the 1st quarter */

	dum = 0;
	fdum3 = 0.0;
	fdum4 = square_length + square_el_length;
	fprintf( o1, "node no., coordinates \n");
	for( k = 0; k < height_el_num+2; ++k )
	{
	    for( i = 0; i < square_el_num+1; ++i )
	    {
		for( j = 0; j < square_el_num+1; ++j )
		{
			fdum1 = square_el_length*((double)j);
			fdum2 = square_el_length*((double)i);
			fprintf( o1, "%5d  ",dum);
			fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
			fprintf( o1, "\n");
			*(coord + nsd*dum) = fdum1;
			*(coord + nsd*dum + 1) = fdum2;
			*(coord + nsd*dum + 2) = fdum3;
			++dum;
		
		}
		angle = ray_angle*((double)i);
		for( j = 0; j < cylinder_el_num; ++j )
		{
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	    }
	    for( j = 0; j < cylinder_el_num; ++j )
	    {
		for( i = 0; i < square_el_num; ++i )
		{
		   angle = pi/2.0 - ray_angle*((double)i);
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	    }

/* Add the extra nodes of the crack */


#if 0
	    if( k == half_height_el_num )
	    {
		for( j = 0; j < cylinder_el_num; ++j )
		{
		    for( i = 0; i < square_el_num; ++i )
		    {
			angle = pi/2.0 - ray_angle*((double)i);
			fdum1 =
			    (ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
			fdum2 =
			    (ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
			fprintf( o1, "%5d  ",dum);
			fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
			fprintf( o1, "\n");
			*(coord + nsd*dum) = fdum1;
			*(coord + nsd*dum + 1) = fdum2;
			*(coord + nsd*dum + 2) = fdum3;
			++dum;
		    }
		}
	    }
#endif

/* These are the nodes for the 2nd quarter */

	    for( i = 1; i < square_el_num+1; ++i )
	    {
		for( j = 0; j < square_el_num + 1; ++j )
		{
		   fdum1 = -square_el_length*((double)i);
		   fdum2 = square_el_length*((double)j);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
		angle = ray_angle*((double)i) + pi/2.0;
		for( j = 0; j < cylinder_el_num; ++j )
		{
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	      }
	      for( j = 0; j < cylinder_el_num; ++j )
	      {
		for( i = 0; i < square_el_num; ++i )
		{
		   angle = pi/2.0 - ray_angle*((double)i) + pi/2.0;
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	    }

/* These are the nodes for the 3rd quarter */

	    for( i = 1; i < square_el_num+1; ++i )
	    {
		for( j = 0; j < square_el_num + 1; ++j )
		{
		   fdum1 = -square_el_length*((double)j);
		   fdum2 = -square_el_length*((double)i);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
		angle = ray_angle*((double)i) + pi;
		for( j = 0; j < cylinder_el_num; ++j )
		{
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	      }
	      for( j = 0; j < cylinder_el_num; ++j )
	      {
		for( i = 0; i < square_el_num; ++i )
		{
		   angle = pi/2.0 - ray_angle*((double)i) + pi;
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	    }

/* These are the nodes for the 4th quarter */

	    for( i = 1; i < square_el_num+1; ++i )
	    {
		for( j = 1; j < square_el_num + 1; ++j )
		{
		   fdum1 = square_el_length*((double)i);
		   fdum2 = -square_el_length*((double)j);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
		angle = ray_angle*((double)i) + 3.0*pi/2.0;
		for( j = 0; j < cylinder_el_num; ++j )
		{
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	    }
	    for( j = 0; j < cylinder_el_num; ++j )
	    {
		for( i = 1; i < square_el_num; ++i )
		{
		   angle = pi/2.0 - ray_angle*((double)i) + 3.0*pi/2.0;
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
		   fprintf( o1, "%5d  ",dum);
		   fprintf( o1, "%14.6e  %14.6e  %14.6e",fdum1,fdum2,fdum3);
		   fprintf( o1, "\n");
		   *(coord + nsd*dum) = fdum1;
		   *(coord + nsd*dum + 1) = fdum2;
		   *(coord + nsd*dum + 2) = fdum3;
		   ++dum;
		}
	    }

	    fdum3 += height_el_length;
	    if( k == half_height_el_num ) fdum3 -= height_el_length;
	    
	}

        dum= 0;
        dum2= nodes_per_layer*height_el_num;
        fprintf( o1, "prescribed displacement x: node  disp value\n");
        for( k = dum2; k < numnp; ++k )
	{
                fprintf( o1, "%4d %12.6e\n",k,0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement y: node  disp value\n");
        for( k = dum2; k < numnp; ++k )
	{
                fprintf( o1, "%4d %12.6e\n",k,0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement z: node  disp value\n");
        for( k = dum2; k < numnp; ++k )
	{
                fprintf( o1, "%4d %12.6e\n",k,0.0);
        }
        fprintf( o1, "%4d\n ",-10);

	dum2 = nodes_per_layer;
	dum3 = (square_el_num+cylinder_el_num+1);
	dum3a = (square_el_num+cylinder_el_num);
	fdum4 = 2.0*square_length*square_length;
	fdum4 = sqrt(fdum4);

        fprintf( o1, "node with point load and load vector in x,y\n");
	for( i = 0; i < nodes_per_layer; ++i )
	{
		dum = i;
		fdum = *(coord + nsd*i)*(*(coord + nsd*i)) + *(coord + nsd*i + 1)*(*(coord + nsd*i + 1));
		fdum = sqrt(fdum);
		if(fdum > fdum4)
		{
			fdum1 = *(coord + nsd*i)*2.415600e+07/(fdum + SMALL);
			fdum2 = *(coord + nsd*i + 1)*2.415600e+07/(fdum + SMALL);
			/*fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum, -fdum2, fdum1, -1.0e9);*/
			fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum, 0.0, 0.0, -1.0e9);
		}
	}

#if 0
/* The code below is excessively complex.  So I am using the code in meshtorq.c which
   assigns loads.
*/
/* These are the forces for the 1st quarter */

        for( i = 0; i < square_el_num + 1; ++i )
        {
            for( j = 0; j < cylinder_el_num; ++j )
            {
		dum = square_el_num + 1 + i*dum3 + j;
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,0.0,50000.0,0.0);
	    }
        }
	dum5 = (square_el_num+1)*(square_el_num+1) + (angle_div + 1)*(cylinder_el_num);
        for( i = dum+1; i < dum5; ++i )
        {
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",i,0.0,50000.0,0.0);
	}

/* These are the forces for the 2nd quarter */

        for( i = 0; i < square_el_num; ++i )
        {
            for( j = 0; j < cylinder_el_num; ++j )
            {
		dum = square_el_num + 1 + i*dum3 + j + dum5;
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,0.0,50000.0,0.0);
	    }
        }
	dum5 += (square_el_num)*(square_el_num+1) + (angle_div)*(cylinder_el_num);
        for( i = dum+1; i < dum5; ++i )
        {
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",i,0.0,50000.0,0.0);
	}

/* These are the forces for the 3rd quarter */

        for( i = 0; i < square_el_num; ++i )
        {
            for( j = 0; j < cylinder_el_num; ++j )
            {
		dum = square_el_num + 1 + i*dum3 + j + dum5;
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,0.0,50000.0,0.0);
	    }
        }

	dum5 += (square_el_num)*(square_el_num+1) + (angle_div)*(cylinder_el_num);
        for( i = dum+1; i < dum5; ++i )
        {
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",i,0.0,50000.0,0.0);
	}

/* These are the forces for the 4th quarter */

        for( i = 0; i < square_el_num; ++i )
        {
            for( j = 0; j < cylinder_el_num; ++j )
            {
		dum = square_el_num + i*dum3a + j + dum5;
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,0.0,50000.0,0.0);
	    }
        }
	dum5 += (square_el_num)*(square_el_num) + (angle_div - 1)*(cylinder_el_num);
        for( i = dum+1; i < dum5; ++i )
        {
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",i,0.0,50000.0,0.0);
	}
#endif

        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "node no. with stress and stress vector in xx,yy,xy,zx,yz\n");
        fprintf( o1, "%4d ",-10);

        return 1;
}

