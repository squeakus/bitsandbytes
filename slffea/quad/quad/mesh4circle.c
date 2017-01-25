/*
    This program generates the mesh for a quater cylindrical heat mesh.

		Updated 4/3/03

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
#define sq2             1.414213462
#define SMALL           1.e-20

int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, nmode;
        int i,j,k,dum,dum2,dum3,dum4;
	double fdum, fdum1, fdum2, fdum3, fdum4, ux[9], uy[9];
	double angle, height_length, square_length, cylinder_length, ray_angle,
		height_el_length, square_el_length, cylinder_el_length,
		ratio;
	double *coord, moment;
	int square_el_num, cylinder_el_num, angle_div,
		angleh_div, nodes_per_layer;
        char name[30];
	char buf[ BUFSIZ ];
        FILE *o1;
	char text;

        o1 = fopen( "torq","w" );
        printf( "\n What is the length of the inner square?\n ");
	scanf( "%lf",&square_length);
        printf( "\n How many elements on the inner square?\n ");
	scanf( "%d",&square_el_num);
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
            (angle_div + 1)*(cylinder_el_num);

	nmat = 1; 

	nmode = 0;
	numel = (square_el_num*square_el_num + (angle_div)*(cylinder_el_num));
	numnp = 1*nodes_per_layer;
	coord=(double *)calloc(3*numnp,sizeof(double));

	cylinder_el_length = cylinder_length/((double)cylinder_el_num);
	square_el_length = square_length/((double)square_el_num);
        printf( "    %4d %4d %4d %4d \n ", numel,numnp,nmat,nodes_per_layer);
        printf( "    %10.6f %10.6f %10.6f\n ", square_el_length, cylinder_el_length );

        fprintf( o1, "  numel numnp nmat nmode  plane_stress_flag   This is the heat mesh \n ");
        fprintf( o1, "    %4d %4d %4d %4d %4d \n ", numel,numnp,nmat,nmode,1);

        fprintf( o1, "matl no., E modulus, Poisson Ratio, density \n");
        for( i = 0; i < nmat; ++i )
        {
           fprintf( o1, "%3d ",0);
           fprintf( o1, " %9.4f %9.4f %9.4f\n ",241.0e9, 0.2, 2.77e3);
        }

	ray_angle *= pi/180.0;
	dum = 0;
	dum3 = (square_el_num+cylinder_el_num+1);
        fprintf( o1, "el no., connectivity, matl no. \n");
	for( i = 0; i < square_el_num; ++i )
	{
		for( j = 0; j < square_el_num + cylinder_el_num; ++j )
		{
		    fprintf( o1, "%4d %4d %4d %4d %4d %4d",dum,
			    i*dum3 + j,
			    i*dum3 + j + 1,
			    (i+1)*dum3 + j + 1,
			    (i+1)*dum3 + j,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	}
	dum4 = square_el_num*dum3;
	for( j = 0; j < square_el_num - 1; ++j )
	{
	        fprintf( o1, "%4d %4d %4d %4d %4d %4d",dum,
	    	    dum4 + j,
	    	    dum4 + j + 1,
	    	    dum4 + j + dum3 + 1,
	    	    dum4 + j + dum3,
	    	    0);
	        fprintf( o1, "\n");
	        ++dum;
	}
	dum4 = (1+square_el_num)*dum3;
	for( i = 0; i < cylinder_el_num-1; ++i )
	{
		for( j = 0; j < square_el_num - 1; ++j )
		{
		    fprintf( o1, "%4d %4d %4d %4d %4d %4d",dum,
			    dum4 + i*square_el_num + j,
			    dum4 + i*square_el_num + j + 1,
			    dum4 + (i+1)*square_el_num + j + 1,
			    dum4 + (i+1)*square_el_num + j,
			    0);
		    fprintf( o1, "\n");
		    ++dum;
		}
	}
	dum4 = square_el_num*dum3;
	fprintf( o1, "%4d %4d %4d %4d %4d %4d",dum,
	    	dum4 + square_el_num - 1,
	    	dum4 + square_el_num,
	    	dum4 + square_el_num + 1,
	    	dum4 + dum3 - 1 + square_el_num,
	    	0);
	fprintf( o1, "\n");
	++dum;
	for( i = 0; i < cylinder_el_num-1; ++i )
	{
		fprintf( o1, "%4d %4d %4d %4d %4d %4d",dum,
			(i+1)*square_el_num - 1 + dum3 + dum4,
			dum4 + square_el_num + i + 1,
			dum4 + square_el_num + i + 2,
			(i+1)*square_el_num - 1 + dum3 + dum4 + square_el_num,
			0);
		fprintf( o1, "\n");
		++dum;
	}

	dum = 0;
	fdum3 = 0.0;
	fdum4 = square_length + square_el_length;
        fprintf( o1, "node no., coordinates \n");
        for( i = 0; i < square_el_num+1; ++i )
        {
        	for( j = 0; j < square_el_num+1; ++j )
        	{
			fdum1 = square_el_length*((double)j);
			fdum2 = square_el_length*((double)i);
           		fprintf( o1, "%d ",dum);
           		fprintf( o1, "%9.4f %9.4f",fdum1,fdum2);
           		fprintf( o1, "\n");
			*(coord + 2*dum) = fdum1;
			*(coord + 2*dum + 1) = fdum2;
			++dum;
		
	        }
		angle = ray_angle*((double)i);
        	for( j = 0; j < cylinder_el_num; ++j )
        	{
		   fdum1 =
			(ratio*fdum4+cylinder_el_length*((double)j))*cos(angle);
		   fdum2 =
			(ratio*fdum4+cylinder_el_length*((double)j))*sin(angle);
           	   fprintf( o1, "%d ",dum);
           	   fprintf( o1, "%9.4f %9.4f",fdum1,fdum2);
           	   fprintf( o1, "\n");
		   *(coord + 2*dum) = fdum1;
		   *(coord + 2*dum + 1) = fdum2;
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
           	   fprintf( o1, "%d ",dum);
           	   fprintf( o1, "%9.4f %9.4f",fdum1,fdum2);
           	   fprintf( o1, "\n");
		   *(coord + 2*dum) = fdum1;
		   *(coord + 2*dum + 1) = fdum2;
		   ++dum;
         	}
	}

        dum= 0;
        dum2= 0;
	dum3 = (square_el_num+1)*(square_el_num+cylinder_el_num+1);

        fprintf( o1, "prescribed displacement x: node  disp value\n");
        for( i = 0; i < nodes_per_layer; ++i )
        {
                fprintf( o1, "%4d %14.6e\n",i,0.0);
        }
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement y: node  disp value\n");
        for( i = 0; i < nodes_per_layer; ++i )
        {
                fprintf( o1, "%4d %14.6e\n",i,0.0);
        }
        fprintf( o1, "%4d\n ",-10);

	dum2 = nodes_per_layer;
	fdum4 = 2.0*square_length*square_length;
        fdum4 = sqrt(fdum4);
	moment = 0.0;
        fprintf( o1, "node with point load and load vector in x,y\n");
        printf("\n");
        for( i = 0; i < nodes_per_layer; ++i )
        {
		dum = i;
		fdum = *(coord + 2*i)*(*(coord + 2*i)) + *(coord + 2*i + 1)*(*(coord + 2*i + 1));
		fdum = sqrt(fdum);
		if(fdum > fdum4)
		{
			fdum1 = *(coord + 2*i)*314.37e5/(fdum + SMALL);
			fdum2 = *(coord + 2*i + 1)*314.37e5/(fdum + SMALL);
			moment += 314.37e5*fdum;
                	fprintf( o1, "%4d %14.6e %14.6e\n",dum, -fdum2, fdum1);
                	printf(  "%4d %14.6e %14.6e\n",dum, -fdum2, fdum1);
		}
        }
        fprintf( o1, "%4d\n ",-10);

        fprintf( o1, "node no. with stress and stress vector xx,yy,xy,zx,yz\n");
        fprintf( o1, "%4d ",-10);

        return 1;
}

