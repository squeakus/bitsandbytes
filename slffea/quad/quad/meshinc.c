/*
    This program generates the mesh for the plane
    stress inclusion problem for a quad.

		Updated 8/26/00

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

#define pi              3.141592654

int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, plane_stress_flag;
        int i,j,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, ux[9], uy[9];
	double angle, side_length, ray_angle, el_length;
	int side_el_num, angle_div, angleh_div;
        char name[20];
	char buf[ BUFSIZ ];
        FILE *o1;
	char text;

        o1 = fopen( "qdinc","w" );
        printf( "\n What is the length of one side?\n ");
	scanf( "%lf",&side_length);
        printf( "\n How many elements on one side?\n ");
	scanf( "%d",&side_el_num);
        printf( "\n What angle(in degrees) between rays?\n ");
	scanf( "%lf",&ray_angle);
	angle_div = (int)(90.0/ray_angle);
	angleh_div = (int)(((double)angle_div)/2.0);

	nmat = 1; 

	plane_stress_flag = 1;
	numel = side_el_num*angle_div;
	numnp = (side_el_num+1)*(1+angle_div);
	el_length = side_length/((double)side_el_num+1);

        fprintf( o1, "   numel numnp nmat nmode plane_stress_flag  This is the inclusion problem \n ");
        fprintf( o1, "    %4d %4d %4d %4d %4d \n ", numel,numnp,nmat,0,plane_stress_flag);

        fprintf( o1, "matl no., E modulus, Poisson Ratio, density \n");
        for( i = 0; i < nmat; ++i )
        {
           fprintf( o1, "%3d ",0);
           fprintf( o1, " %9.4f %9.4f     %9.4f \n ",10000000.0, 0.3, 2.77e3);
        }

	ray_angle *= pi/180.0;
	dum = 0;
        fprintf( o1, "el no., connectivity, matl no. \n");
        for( i = 0; i < angle_div; ++i )
        {
        	for( j = 0; j < side_el_num; ++j )
        	{
           		fprintf( o1, "%4d %4d %4d %4d %4d %4d",dum,i*(side_el_num+1) + j,
				i*(side_el_num+1)+j+1, (i+1)*(side_el_num+1)+j+1,
				(i+1)*(side_el_num+1)+j,0);
           		fprintf( o1, "\n");
			++dum;
         	}
        }

	dum = 0;
	fdum3 = 0.0;
        fprintf( o1, "node no., coordinates \n");
        for( i = 0; i < angleh_div; ++i )
        {
		angle = ray_angle*fdum3;
        	for( j = 0; j < side_el_num; ++j )
        	{
			fdum1 = el_length*((double)j+1)*cos(angle);
			fdum2 = el_length*((double)j+1)*sin(angle);
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f %9.4f ",fdum1,fdum2);
           		fprintf( o1, "\n");
			++dum;
         	}
		fdum1 = side_length;
		fdum2 = side_length*tan(angle);
		fdum3 += 1.0;
           	fprintf( o1, "%d ",dum);
           	fprintf( o1, "%9.4f %9.4f ",fdum1,fdum2);
           	fprintf( o1, "\n");
		++dum;
        }
        for( j = 0; j < side_el_num; ++j )
        {
		angle = ray_angle*((double)angleh_div);
		fdum1 = el_length*((double)j+1)*cos(angle);
		fdum2 = el_length*((double)j+1)*sin(angle);
           	fprintf( o1, "%d ",dum);
           	fprintf( o1, "%9.4f %9.4f ",fdum1,fdum2);
           	fprintf( o1, "\n");
		++dum;
        }
        fprintf( o1, "%d ",dum);
        fprintf( o1, "%9.4f %9.4f ",side_length,side_length);
        fprintf( o1, "\n");
	++dum;
	fdum3 = (double)(angleh_div+1);
        for( i = angleh_div+1; i < angle_div+1; ++i )
        {
		angle = ray_angle*fdum3;
        	for( j = 0; j < side_el_num; ++j )
        	{
			fdum1 = el_length*((double)j+1)*cos(angle);
			fdum2 = el_length*((double)j+1)*sin(angle);
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f %9.4f ",fdum1,fdum2);
           		fprintf( o1, "\n");
			++dum;
         	}
		fdum1 = side_length/tan(angle);
		fdum2 = side_length;
		fdum3 += 1.0;
           	fprintf( o1, "%d ",dum);
           	fprintf( o1, "%9.4f %9.4f ",fdum1,fdum2);
           	fprintf( o1, "\n");
		++dum;
        }

        dum= 0;
        fprintf( o1, "prescribed displacement x: node  disp value\n");
        for( i = 0; i < side_el_num+1; ++i )
        {
		dum = (side_el_num+1)*(angle_div)+i;
                fprintf( o1, "%4d %14.6e\n",dum,0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement y: node  disp value\n");
        for( i = 0; i < side_el_num+1; ++i )
        {
                fprintf( o1, "%4d %14.6e\n",i,0.0);
        }
        fprintf( o1, "%4d\n ",-10);

        fprintf( o1, "node with point load and load vector in x,y\n");
        for( i = 0; i < angleh_div+1; ++i )
        {
		dum = side_el_num+i*(side_el_num+1);
                fprintf( o1, "%4d %14.6e %14.6e\n",dum,100000.0,0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "element and gauss pt. with stress and stress vector in xx,yy,xy\n");
        fprintf( o1, "%4d ",-10);

        return 1;
}

