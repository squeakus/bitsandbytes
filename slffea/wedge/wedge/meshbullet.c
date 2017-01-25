/*
    meshbullet.c
    This program generates the mesh for a bullet based on
    wedge elements.
  
  	Updated 12/1/01

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

#define PI                  3.14159265358979323846

int writer(void)
{
        FILE *o1, *o2;
	char buf[ BUFSIZ ];
        int j, j2, levels, blevels, dum, el_type, numel, numnp, nmat, div;
	double angle, angle2, dt, dt2, S, C, R, T, X, Y, Z, length, blength,
		sections, fdum;

	o1 = fopen( "bullet.in","r" );

        if(o1 == NULL ) {
                printf("Can't find file bullet.in\n");
                exit(1);
        }

	o2 = fopen( "bullet","w" );

    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&blength);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&blevels);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&R);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&div);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&length);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&levels);

	numel = (div)*(blevels + levels);
	numnp = (div + 1)*(blevels + levels + 1);
	nmat = 1;

        fprintf(o2, "   numel numnp nmat nmode  (This is for a bullet)\n");
        fprintf(o2, "%d %d %d %d\n", numel, numnp, nmat, 0);
        fprintf(o2, "matl no., E modulus, Poisson Ratio, density\n");
        fprintf(o2, "%d %16.4f %16.4f %16.4f\n",0, 100000000.0, 0.27, 2.77e3);
        fprintf(o2, "el no.,connectivity, matl no.\n");

	dum = 0;
        for( j = 0; j < blevels + levels; ++j )
	{
        	for( j2 = 0; j2 < div - 1; ++j2 )
        	{
        	     fprintf(o2, "%4d  ", dum);
        	     fprintf(o2, "%4d %4d %4d %4d %4d %4d", 
			(div+1)*j,
			(div+1)*j + j2 + 1,
			(div+1)*j + j2 + 2,
			(div+1)*(j + 1),
			(div+1)*(j + 1) + j2 + 1,
			(div+1)*(j + 1) + j2 + 2);
		     fprintf(o2, " %4d\n",0);
		     ++dum;
		}
        	fprintf(o2, "%4d  ", dum);
        	fprintf(o2, "%4d %4d %4d %4d %4d %4d", 
		     (div+1)*j,
		     (div+1)*j + j2 + 1,
		     (div+1)*j + 1,
		     (div+1)*(j + 1),
		     (div+1)*(j + 1) + j2 + 1,
		     (div+1)*(j + 1) + 1);
		fprintf(o2, " %4d\n",0);
		++dum;
	}

	X=0.0;
	Y=0.0;
	Z=0.0;
	dum = 0;
        fprintf(o2, "node no. coordinates\n");
        fprintf(o2, "%4d  ", dum);
        fprintf(o2, "%9.5f %9.5f %9.5f \n",0.0,0.0,0.0);
	dt=2.0*PI/((double)div);
	dt2=blength/((double)blevels);
	Y += dt2/2.0;
	++dum;
        for( j2 = 0; j2 < div; ++j2 )
        {
		fdum = R*sqrt(Y/blength);
	        Z=fdum*cos(angle2);
               	X=fdum*sin(angle2);
        	fprintf(o2, "%4d  ", dum);
		++dum;
               	fprintf(o2, "%9.5f %9.5f %9.5f \n",X,Y,Z);
		angle2+=dt;
	}
	Y += dt2/2.0;
        for( j = 0; j < blevels; ++j )
        {
		angle2=0.0;
		X = 0.0;
		Z = 0.0;
        	fprintf(o2, "%4d  ", dum);
                fprintf(o2, "%9.5f %9.5f %9.5f \n",X,Y,Z);
		++dum;
        	for( j2 = 0; j2 < div; ++j2 )
        	{
			fdum = R*sqrt(Y/blength);
	        	Z=fdum*cos(angle2);
                	X=fdum*sin(angle2);
        		fprintf(o2, "%4d  ", dum);
			++dum;
                	fprintf(o2, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			angle2+=dt;
		}
		Y += dt2;
	}

	sections = ((double)length)/((double)levels);

	dt2=length/((double)levels);
	Y += dt2;
        for( j = 0; j < levels; ++j )
        {
		angle2=0.0;
		X = 0.0;
		Z = 0.0;
        	fprintf(o2, "%4d  ", dum);
                fprintf(o2, "%9.5f %9.5f %9.5f \n",X,Y,Z);
		++dum;
        	for( j2 = 0; j2 < div; ++j2 )
        	{
	        	Z=R*cos(angle2);
                	X=R*sin(angle2);
        		fprintf(o2, "%4d  ", dum);
			++dum;
                	fprintf(o2, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			angle2+=dt;
		}
		Y += dt2;
	}

        fprintf( o2, "prescribed displacement x: node  disp value\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "prescribed displacement y: node  disp value\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "prescribed displacement z: node  disp value\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "node with point load and load vector in x,y,z\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "node no. with stress and stress vector xx,yy,xy,zx,yz\n");
        fprintf( o2, "%4d ",-10);


        return 1;
}

main(int argc, char** argv)
{
	int check;
	check=writer();
} 
