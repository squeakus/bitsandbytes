/*
    This program generates the mesh for the qdinc quadrilateral
    problem extruded for a brick.

		Updated 6/29/01

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

#define nsd             3
#define pi              3.141592654

int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, nmode;
        int i,j,k,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, fdum4, ux[9], uy[9];
	double angle, height_length, side_length, ray_angle, height_el_length, side_el_length;
	int height_el_num, side_el_num, angle_div, angleh_div, nodes_per_layer;
        char name[30];
	char buf[ BUFSIZ ];
        FILE *o1;
	char text;

        o1 = fopen( "wedge","w" );
        printf( "\n What is the length of one side?\n ");
	scanf( "%lf",&side_length);
        printf( "\n How many elements on one side?\n ");
	scanf( "%d",&side_el_num);
        printf( "\n What is the length of the cylinder height?\n ");
	scanf( "%lf",&height_length);
        printf( "\n How many elements on the cylinder height?\n ");
	scanf( "%d",&height_el_num);
        printf( "\n What angle(in degrees) between rays?\n ");
	scanf( "%lf",&ray_angle);
	angle_div = (int)(90.0/ray_angle);
	angleh_div = (int)(((double)angle_div)/2.0);
	nodes_per_layer = (angle_div+1)*(side_el_num+1);

	nmat = 1; 

	nmode = 0;
	numel = side_el_num*(height_el_num-1)*angle_div;
	numnp = (side_el_num+1)*(height_el_num)*(1+angle_div);
	side_el_length = side_length/((double)side_el_num+1);
	height_el_length = height_length/((double)height_el_num);

        fprintf( o1, "   numel numnp nmat nmode  This is the wedge mesh \n ");
        fprintf( o1, "    %4d %4d %4d %4d \n ", numel,numnp,nmat,nmode);

        fprintf( o1, "matl no., E modulus, Poisson Ratio, density \n");
        for( i = 0; i < nmat; ++i )
        {
           fprintf( o1, "%3d ",0);
           fprintf( o1, " %9.4f %9.4f %9.4f\n ",10000000.0, 0.3, 2.77e3);
        }

	ray_angle *= pi/180.0;
	dum = 0;
	dum2 = 0;
        fprintf( o1, "el no., connectivity, matl no. \n");
        for( k = 0; k < height_el_num-1; ++k )
	{
            for( i = 0; i < angle_div; ++i )
            {
        	for( j = 0; j < side_el_num; ++j )
        	{
           		fprintf( o1, "%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d",dum,
				i*(side_el_num+1) + j + dum2,
				i*(side_el_num+1)+j+1 + dum2, 
				(i+1)*(side_el_num+1)+j+1 + dum2,
				(i+1)*(side_el_num+1)+j + dum2,
           			i*(side_el_num+1) + j + nodes_per_layer + dum2,
				i*(side_el_num+1)+j+1 + nodes_per_layer + dum2,
				(i+1)*(side_el_num+1)+j+1 + nodes_per_layer + dum2,
				(i+1)*(side_el_num+1)+j + nodes_per_layer + dum2,
				0);
           		fprintf( o1, "\n");
			++dum;
         	}
            }
	    dum2 += nodes_per_layer;
        }

	dum = 0;
	fdum3 = 0.0;
        fprintf( o1, "node no., coordinates \n");
        for( k = 0; k < height_el_num; ++k )
        {
            for( i = 0; i < angleh_div; ++i )
            {
		angle = ray_angle*((double)i);
        	for( j = 0; j < side_el_num; ++j )
        	{
			fdum1 = side_el_length*((double)j+1)*cos(angle);
			fdum2 = side_el_length*((double)j+1)*sin(angle);
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f %9.4f %9.4f ",fdum1,fdum2,fdum3);
           		fprintf( o1, "\n");
			++dum;
         	}
		fdum1 = side_length;
		fdum2 = side_length*tan(angle);
           	fprintf( o1, "%d ",dum);
           	fprintf( o1, "%9.4f %9.4f %9.4f ",fdum1,fdum2,fdum3);
           	fprintf( o1, "\n");
		++dum;
            }
            for( j = 0; j < side_el_num; ++j )
            {
		angle = ray_angle*((double)angleh_div);
		fdum1 = side_el_length*((double)j+1)*cos(angle);
		fdum2 = side_el_length*((double)j+1)*sin(angle);
           	fprintf( o1, "%d ",dum);
           	fprintf( o1, "%9.4f %9.4f %9.4f ",fdum1,fdum2,fdum3);
           	fprintf( o1, "\n");
		++dum;
            }
            fprintf( o1, "%d ",dum);
            fprintf( o1, "%9.4f %9.4f %9.4f ",side_length,side_length,fdum3);
            fprintf( o1, "\n");
	    ++dum;
            for( i = angleh_div+1; i < angle_div+1; ++i )
            {
		angle = ray_angle*((double)i);
        	for( j = 0; j < side_el_num; ++j )
        	{
			fdum1 = side_el_length*((double)j+1)*cos(angle);
			fdum2 = side_el_length*((double)j+1)*sin(angle);
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f %9.4f %9.4f ",fdum1,fdum2,fdum3);
           		fprintf( o1, "\n");
			++dum;
         	}
		fdum1 = side_length/tan(angle);
		fdum2 = side_length;
           	fprintf( o1, "%d ",dum);
           	fprintf( o1, "%9.4f %9.4f %9.4f ",fdum1,fdum2,fdum3);
           	fprintf( o1, "\n");
		++dum;
            }
	    fdum3 += height_el_length;
        }

        dum= 0;
        dum2= 0;
        fprintf( o1, "prescribed displacement x: node  disp value\n");
        for( k = 0; k < height_el_num; ++k )
	{
            for( i = 0; i < side_el_num+1; ++i )
            {
		dum = (side_el_num+1)*(angle_div)+i+dum2;
                fprintf( o1, "%4d %14.6e\n",dum,0.0);
            }
	    dum2 += nodes_per_layer;
        }
        fprintf( o1, "%4d\n ",-10);
        dum2= 0;
        fprintf( o1, "prescribed displacement y: node  disp value\n");
        for( k = 0; k < height_el_num; ++k )
	{
            for( i = 0; i < side_el_num+1; ++i )
            {
                fprintf( o1, "%4d %14.6e\n",i+dum2,0.0);
            }
	    dum2 += nodes_per_layer;
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement z: node  disp value\n");
        for( i = 0; i < 1; ++i )
        {
                fprintf( o1, "%4d %14.6e\n",i,0.0);
        }
        fprintf( o1, "%4d\n ",-10);

	dum2 = 0;
        fprintf( o1, "node with point load and load vector in x,y\n");
        for( k = 0; k < height_el_num; ++k )
	{
            for( i = 0; i < angleh_div+1; ++i )
            {
		dum = side_el_num+i*(side_el_num+1)+dum2;
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,50000.0,0.0,0.0);
            }
	    dum2 += nodes_per_layer;
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "node no. with stress and stress vector xx,yy,xy,zx,yz\n");
        fprintf( o1, "%4d ",-10);

        return 1;
}

