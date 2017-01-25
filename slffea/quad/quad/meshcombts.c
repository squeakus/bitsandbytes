/*
    This program generates the mesh for a honeycomb
    using quad elements.

		Updated 12/26/08

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

int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, plane_stress_flag;
        int i,j,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, fdum4, ux[9], uy[9], ang60, ang30;
	int horiz_el_num, vert_el_num;
	double horiz_length, el_length, rad_length, A, B;
	int side_el_num, angle_div, angleh_div;
        char name[20];
	char buf[ BUFSIZ ];
        FILE *o1;
	char text;

        o1 = fopen( "comb3d","w" );
        printf( "\n What is the length of horizontal side?\n ");
	scanf( "%lf",&horiz_length);
        printf( "\n How many combs on the horizontal side?\n ");
	scanf( "%d",&horiz_el_num);
        printf( "\n How many combs on the vertical side?\n ");
	scanf( "%d",&vert_el_num);

	nmat = 1; 

	plane_stress_flag = 1;
	numel = vert_el_num*(4*horiz_el_num + 2*horiz_el_num - 1) + horiz_el_num-1;
	numnp = (2*vert_el_num + 1)*4*horiz_el_num;
	el_length = horiz_length/((double)horiz_el_num+1);
	ang60 = 60.0*pi/180;
	ang30 = 30.0*pi/180;
	rad_length = el_length/(2.0*sin(ang30));

        printf( "   %9.6e %9.6e \n ", el_length, rad_length);

        fprintf( o1, "   numel numnp nmat nmode plane_stress_flag  This is the inclusion problem \n ");
        fprintf( o1, "    %4d %4d %4d %4d \n ", numel,numnp,nmat,0);

        fprintf( o1, "matl no., E modulus, Poisson Ratio, density \n");
        for( i = 0; i < nmat; ++i )
        {
           fprintf( o1, "%3d ",0);
           fprintf( o1, " %9.4f %9.4f     %9.4f \n ",10000000.0, 0.3, 2.77e3);
        }

	dum = 0;
        fprintf( o1, "el no., connectivity, matl no. \n");
        for( i = 0; i < vert_el_num; ++i )
        {
        	for( j = 0; j < 2*horiz_el_num; ++j )
        	{
           		fprintf( o1, "%4d %4d %4d %4d",dum,
				i*8*horiz_el_num + j,
				i*8*horiz_el_num + j + 4*horiz_el_num,
				0);
           		fprintf( o1, "\n");
			++dum;
         	}
        	for( j = 0; j < horiz_el_num-1; ++j )
        	{
           		fprintf( o1, "%4d %4d %4d %4d",dum,
				i*8*horiz_el_num + 2*j + 1,
				i*8*horiz_el_num + 2*j + 2,
				0);
           		fprintf( o1, "\n");
			++dum;
         	}
        	for( j = 0; j < horiz_el_num; ++j )
        	{
           		fprintf( o1, "%4d %4d %4d %4d",dum,
				i*8*horiz_el_num + 2*j + 4*horiz_el_num,
				i*8*horiz_el_num + 2*j + 1 + 4*horiz_el_num,
				0);
           		fprintf( o1, "\n");
			++dum;
         	}
        	for( j = 0; j < 2*horiz_el_num; ++j )
        	{
           		fprintf( o1, "%4d %4d %4d %4d",dum,
				i*8*horiz_el_num + j + 4*horiz_el_num,
				i*8*horiz_el_num + j + 8*horiz_el_num,
				0);
           		fprintf( o1, "\n");
			++dum;
         	}
        }

        i = vert_el_num;
        for( j = 0; j < horiz_el_num-1; ++j )
        {
        	fprintf( o1, "%4d %4d %4d %4d",dum,
			i*8*horiz_el_num + 2*j + 1,
			i*8*horiz_el_num + 2*j + 2,
			0);
           	fprintf( o1, "\n");
		++dum;
         }

	dum = 0;
	fdum3 = 0.0;
	fdum4 = 0.0;
	A = 2.0*rad_length;
	B = el_length;
        fprintf( o1, "node no., coordinates \n");
        for( i = 0; i < 2*vert_el_num + 1; ++i )
        {
		fdum1 = 0.0;
        	for( j = 0; j < horiz_el_num; ++j )
        	{
			fdum2 = fdum3;
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f   %9.4f  %9.4f ",fdum1 + fdum4, fdum2, 0.0);
           		fprintf( o1, "\n");
			++dum;
			fdum1 += A;
			fdum2 = fdum3;
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f   %9.4f  %9.4f ",fdum1 + fdum4, fdum2, 0.0);
           		fprintf( o1, "\n");
			++dum;
			fdum1 += B;
         	}
		fdum1 = 0.0;
        	for( j = 0; j < horiz_el_num; ++j )
        	{
			fdum2 = fdum3;
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f   %9.4f  %9.4f ",fdum1 + fdum4, fdum2, 1.0);
           		fprintf( o1, "\n");
			++dum;
			fdum1 += A;
			fdum2 = fdum3;
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f   %9.4f  %9.4f ",fdum1 + fdum4, fdum2, 1.0);
           		fprintf( o1, "\n");
			++dum;
			fdum1 += B;
         	}
		if(fdum4 < 1.0e-5)
		{
			fdum4 = (2.0*rad_length - el_length)/2.0;
			A = el_length;
			B = 2.0*rad_length;
		}
		else
		{
			fdum4 = 0.0;
			A = 2.0*rad_length;
			B = el_length;
		}
		fdum3 += el_length*cos(ang30);
        }

        dum= 0;
        fprintf( o1, "prescribed displacement x: node  disp value\n");
        for( i = 0; i < 2*vert_el_num + 1; ++i )
        {
           	dum = i*4*horiz_el_num;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
           	dum = i*4*horiz_el_num + 2*horiz_el_num;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
           	dum = i*4*horiz_el_num + 2*horiz_el_num - 1;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
           	dum = i*4*horiz_el_num + 2*horiz_el_num + 2*horiz_el_num - 1;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
        }
        for( i = 0; i < 2*horiz_el_num; ++i )
        {
           	dum = i;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement y: node  disp value\n");
        for( i = 0; i < 2*vert_el_num + 1; ++i )
        {
           	dum = i*4*horiz_el_num;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
           	dum = i*4*horiz_el_num + 2*horiz_el_num;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
           	dum = i*4*horiz_el_num + 2*horiz_el_num - 1;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
           	dum = i*4*horiz_el_num + 2*horiz_el_num + 2*horiz_el_num - 1;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
        }
        for( i = 0; i < 2*horiz_el_num; ++i )
        {
           	dum = i;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement z: node  disp value\n");
        for( i = 0; i < numnp; ++i )
        {
           	dum = i;
           	fprintf( o1, "%4d %12.6e\n",dum, 0.0);
        }
        fprintf( o1, "%4d\n ",-10);

        fprintf( o1, "node with point load and load vector in x, y, z\n");
        for( i = 0; i < 2*horiz_el_num; ++i )
        {
           	dum = 2*vert_el_num*4*horiz_el_num + i;
                fprintf( o1, "%4d %14.6e %14.6e %14.6e\n", dum, 0.0, 100000.0, 0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "element and gauss pt. with stress and stress vector in xx,yy,xy\n");
        fprintf( o1, "%4d ",-10);

        return 1;
}

