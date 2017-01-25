/*
    meshring.c
    This program generates the mesh for a ring based on brick elements.

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

int writer(void)
{
        FILE *o1;
        int j,j2,n,dum;
	double angle,S,C,R,T,X,Y,Z;
        o1 = fopen( "ring","w" );

    	printf("How many elements do you want ? \n");
    	scanf( "%d",&n);
    	printf("What is the Radius ? \n");
    	scanf( "%lf",&R);
    	printf("What is the Thickness ? \n");
    	scanf( "%lf",&T);

	angle=0.0;
        fprintf(o1, "   numel numnp nmat nmode   (This is for a brick ring)\n");
        fprintf(o1, "%d %d %d %d\n",n,4*n+4,1,0);
        fprintf(o1, "matl no., E modulus, Poisson Ratio, density\n");
        fprintf(o1, "%d %16.4f %16.4f %16.4f\n",0, 1000000.0, 0.27, 2.77e3);
        fprintf(o1, "el no.,connectivity, matl no.\n");
	dum = 0;
        for( j = 0; j < n; ++j )
        {
        	fprintf(o1, "%4d   ",dum);
		++dum;
        	fprintf(o1, "%4d %4d %4d %4d %4d %4d %4d %4d %4d\n",
		   	4*j,4*j+1,4*j+2,4*j+3,4*j+4,4*j+5,4*j+6,4*j+7,0);
	}
        fprintf(o1, "node no. coordinates\n");
	dum = 0;
        for( j = 0; j < n+1; ++j )
        {
		X=R*cos(angle);
		Y=R*sin(angle);
        	fprintf(o1, "%4d ",dum);
		++dum;
                fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,0.0);
        	fprintf(o1, "%4d ",dum);
		++dum;
                fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,T);
		X=(R+1)*cos(angle);
		Y=(R+1)*sin(angle);
        	fprintf(o1, "%4d ",dum);
		++dum;
                fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,T);
        	fprintf(o1, "%4d ",dum);
		++dum;
                fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,0.0);

	   	/*angle+=.314159;*/
	   	angle+=6.283185307/(double)n;

        }
        fprintf( o1, "prescribed displacement x: node  disp value\n");
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement y: node  disp value\n");
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "prescribed displacement z: node  disp value\n");
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "node with point load and load vector in x,y,z\n");
        fprintf( o1, "%4d\n ",-10);
        fprintf( o1, "node no. with stress and stress vector xx,yy,xy,zx,yz\n");
        fprintf( o1, "%4d ",-10);

        return 1;
}

main(int argc, char** argv)
{
	int check;
	check=writer();
} 
