/*
    This program generates a rectangular mesh for the
    electromagnetic waveguide problem for quad elements.

		Updated 3/4/03

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

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
	int dof, nmat, numel, numed, numnp, plane_stress_flag;
        int i,j,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, ux[9], uy[9];
	double side_length, side_width, el_length, el_width;
	int length_el_num, width_el_num, el_type;
        char name[20];
	char buf[ BUFSIZ ];
        FILE *o1;
	char text;

        printf( "\n What is the value of the length?\n ");
	scanf( "%lf",&side_length);
        printf( "\n How many elements on the length?\n ");
	scanf( "%d",&length_el_num);
        printf( "\n What is the length of the width?\n ");
	scanf( "%lf",&side_width);
        printf( "\n How many elements on the width?\n ");
	scanf( "%d",&width_el_num);
        printf("What type of element is this for?\n");
        printf("1) quad \n");
        printf("2) quad3 \n");
        scanf( "%d",&el_type);

	if(el_type < 2 )
	{
        	o1 = fopen( "rect","w" );
	}

	if(el_type > 1 )
	{
        	o1 = fopen( "rect3","w" );
	}

	nmat = 1; 
	plane_stress_flag = 1;
	numel = length_el_num*width_el_num;
	numed = length_el_num*(width_el_num+1) + (length_el_num+1)*width_el_num;
	numnp = (length_el_num+1)*(width_el_num+1);
	el_length = side_length/((double)length_el_num);
	el_width = side_width/((double)width_el_num);

	if(el_type < 2 )
	{
            fprintf( o1, "   numel numnp nmat nmode plane_stress_flag");
            fprintf( o1, "  This is the rectangular waveguide problem \n ");
            fprintf( o1, "    %4d %4d %4d %4d %4d \n ",
		numel,numnp,nmat,0,plane_stress_flag);
	}

	if(el_type > 1 )
	{
            fprintf( o1, "   numed numel numnp nmat nmode plane_stress_flag");
            fprintf( o1, "  This is the rectangular waveguide problem \n ");
            fprintf( o1, "  %4d %4d %4d %4d %4d %4d \n ",
		numed,numel,numnp,nmat,2,plane_stress_flag);
	}

	if(el_type < 2 )
	{
            fprintf( o1, "matl no., E modulus, Poisson Ratio, density \n");
            for( i = 0; i < nmat; ++i )
            {
        	fprintf( o1, "%3d ",0);
        	fprintf( o1, " %9.4f %9.4f     %9.4f \n ",10000000.0, 0.3, 2.77e3);
            }
	}


	if(el_type > 1 )
	{
            fprintf( o1, "matl no., permeability, permittivity, operating fq\n");
            for( i = 0; i < nmat; ++i )
            {
        	fprintf( o1, "%3d ",0);
        	fprintf( o1, " %9.4f %9.4f %9.4f\n ",.25, 0.3, 0.3);
            }
	}

	if(el_type > 1 )
	{
	    dum = 0;
            fprintf( o1, "edge no., node connectivity, matl no.\n");
            for( i = 0; i < width_el_num; ++i )
            {
        	for( j = 0; j < length_el_num; ++j )
        	{
           	    fprintf( o1, "%4d    %4d %4d",dum,i*(length_el_num+1) + j,
			i*(length_el_num+1)+j+1);
           	    fprintf( o1, "\n");
		    ++dum;
         	}
        	for( j = 0; j < length_el_num + 1; ++j )
        	{
           	    fprintf( o1, "%4d    %4d %4d",dum,i*(length_el_num+1) + j,
			(i+1)*(length_el_num+1)+j);
           	    fprintf( o1, "\n");
		    ++dum;
         	}
            }
	    i = width_el_num;
            for( j = 0; j < length_el_num; ++j )
            {
           	fprintf( o1, "%4d    %4d %4d",dum,i*(length_el_num+1) + j,
		    i*(length_el_num+1)+j+1);
           	fprintf( o1, "\n");
		++dum;
            }
        }

	dum = 0;
	if(el_type < 2)
		fprintf( o1, "el no., connectivity, matl no. \n");
	if(el_type > 1)
		fprintf( o1, "el no., node connectivity, edge connectivity, matl no.\n");
	for( i = 0; i < width_el_num; ++i )
	{
		for( j = 0; j < length_el_num; ++j )
		{
			fprintf( o1, "%4d ",dum);
			fprintf( o1, " %4d %4d %4d %4d",i*(length_el_num+1) + j,
				i*(length_el_num+1)+j+1, (i+1)*(length_el_num+1)+j+1,
				(i+1)*(length_el_num+1)+j);
		        if(el_type > 1)
			{
			    fprintf( o1, "    %4d %4d %4d %4d",i*(2*length_el_num+1) + j,
			    (i+1)*(2*length_el_num+1)+j, i*(2*length_el_num+1)+length_el_num+j+1,
				i*(2*length_el_num+1)+length_el_num+j);
			}
			fprintf( o1, "%4d ",0);
			fprintf( o1, "\n");
			++dum;
		}
	}

	dum = 0;
        fprintf( o1, "node no., coordinates \n");
	for( i = 0; i < width_el_num + 1; ++i )
	{
		for( j = 0; j < length_el_num + 1; ++j )
		{
			fdum1 = el_length*((double)j);
			fdum2 = el_width*((double)i);
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f %9.4f ",fdum1,fdum2);
           		fprintf( o1, "\n");
			++dum;
         	}
        }

	if(el_type < 2 )
	{
            dum= 0;
            fprintf( o1, "prescribed displacement x: node  disp value\n");
            for( i = 0; i < width_el_num+1; ++i )
            {
		dum = (length_el_num+1)*i;
                fprintf( o1, "%4d %14.6e\n",dum,0.0);
            }
            fprintf( o1, "%4d\n ",-10);
            fprintf( o1, "prescribed displacement y: node  disp value\n");
            for( i = 0; i < length_el_num+1; ++i )
            {
                fprintf( o1, "%4d %14.6e\n",i,0.0);
            }
            fprintf( o1, "%4d\n ",-10);

            fprintf( o1, "node with point load and load vector in x,y\n");
	    dum = length_el_num;
            fprintf( o1, "%4d %14.6e %14.6e\n",dum,50000.0,0.0);
            for( i = 1; i < width_el_num; ++i )
            {
		dum = (length_el_num+1)*i + length_el_num;
                fprintf( o1, "%4d %14.6e %14.6e\n",dum,100000.0,0.0);
            }
	    i = width_el_num;
	    dum = (length_el_num+1)*i + length_el_num;
            fprintf( o1, "%4d %14.6e %14.6e\n",dum,50000.0,0.0);
            fprintf( o1, "%4d\n ",-10);
            fprintf( o1, "element and gauss pt. with stress and stress vector in xx,yy,xy\n");
            fprintf( o1, "%4d ",-10);
	}

	if(el_type > 1 )
	{
            dum= 0;
            fprintf( o1, "prescribed edge: edge value\n");
            for( i = 0; i < length_el_num; ++i )
            {
                fprintf( o1, "%4d %14.6e\n",i,0.0);
            }
            for( i = 0; i < width_el_num; ++i )
            {
		dum = length_el_num + (2*length_el_num + 1)*i;
                fprintf( o1, "%4d %14.6e\n",dum,0.0);
            }
            for( i = 0; i < width_el_num; ++i )
            {
		dum = length_el_num + (2*length_el_num + 1)*i + length_el_num;
                fprintf( o1, "%4d %14.6e\n",dum,0.0);
            }
            for( i = numed - length_el_num; i < numed; ++i )
            {
                fprintf( o1, "%4d %14.6e\n",i,0.0);
            }
            fprintf( o1, "%4d\n ",-10);

            fprintf( o1, "node with point charge\n");
            for( i = 0; i < length_el_num+1; ++i )
            {
		dum = (length_el_num+1)*i + length_el_num;
                fprintf( o1, "%4d %14.6e %14.6e\n",dum,100000.0,0.0);
            }
            fprintf( o1, "%4d\n ",-10);
            fprintf( o1, "element and gauss pt. with stress and stress vector in xx,yy,xy\n");
            fprintf( o1, "%4d ",-10);
	}

        return 1;
}

