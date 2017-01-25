/*
    meshball.c
    This program generates the mesh for a ball based on brick elements.
  
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
        int j, j2, n, dum, el_type, numel, numnp, nmat;
	double angle, angle2, dt, dt2, S, C, R, T, X, Y, Z;

    	printf("How many levels you want(2-10) ? \n");
    	scanf( "%d",&n);
    	printf("What is the Radius ? \n");
    	scanf( "%lf",&R);
    	printf("What is the Thickness ? \n");
    	scanf( "%lf",&T);
	printf("What type of element is this for?\n");
	printf("1) brick \n");
	printf("2) shell \n");
	scanf( "%d",&el_type);

	if(el_type < 2 ) o1 = fopen( "ball","w" );
        else o1 = fopen( "ball.sh","w" );

	if(el_type > 1 )
	{
	   numel = 4*n*(n-1);
	   numnp = 4*n*n;
	   nmat = 1;

	   fprintf( o1, "   numel numnp nmat nmode integ_flag This is a shell cylinder\n");
	   fprintf(o1, "   %4d %4d %4d %4d %4d\n",numel,numnp,1,0,1);
	   fprintf( o1, " matl no., E mod., Poiss. Ratio, density, shear fac. \n");
	   fprintf(o1, " %4d %9.4f %9.4f %9.4f %9.4f\n", 0, 4.32e8, 0.0, 2.77e3, 0.8333333);
	   fprintf( o1, "el no., connectivity, matl no.\n");

	   dum = 0;
           for( j = 0; j < n-1; ++j )
	   {
        	for( j2 = 0; j2 < 4*n-1; ++j2 )
        	{
        	     fprintf(o1, "%4d  ", dum);
        	     fprintf(o1, "%4d %4d %4d %4d",
			4*n*j+j2, 4*n*j+j2+1, 4*n*j+j2+4*n+1, 4*n*j+j2+4*n);
		     fprintf(o1, "%4d\n",0);
		     ++dum;
		}
		j2=4*n-1;
        	fprintf(o1, "%4d  ", dum);
        	fprintf(o1, "%4d %4d %4d %4d", 
			4*n*j+j2, 4*n*j, 4*n*j+4*n, 4*n*j+j2+4*n);
		fprintf(o1, "%4d\n",0);
		++dum;


	   }
	}

	if(el_type < 2 )
	{
	   numel = 4*n*(n-1);
	   numnp = 8*n*n;
	   nmat = 1;

           fprintf(o1, "   numel numnp nmat nmode  (This is for a brick ball)\n");
           fprintf(o1, "%d %d %d %d\n", numel, numnp, nmat, 0);
           fprintf(o1, "matl no., E modulus, Poisson Ratio, density\n");
           fprintf(o1, "%d %16.4f %16.4f %16.4f\n",0, 100000000.0, 0.27, 2.77e3);
           fprintf(o1, "el no.,connectivity, matl no.\n");
	   dum = 0;
           for( j = 0; j < n-1; ++j )
	   {
        	for( j2 = 0; j2 < 4*n-1; ++j2 )
        	{
        	     fprintf(o1, "%4d  ", dum);
        	     fprintf(o1, "%4d %4d %4d %4d %4d %4d %4d %4d", 
			8*n*j+j2, 8*n*j+j2+4*n, 8*n*j+j2+4*n+1, 8*n*j+j2+1, 8*n*j+j2+8*n, 
			8*n*j+j2+12*n, 8*n*j+j2+12*n+1, 8*n*j+j2+8*n+1);
		     fprintf(o1, " %4d\n",0);
		     ++dum;
		}
		j2=4*n-1;
        	fprintf(o1, "%4d  ", dum);
        	fprintf(o1, "%4d %4d %4d %4d %4d %4d %4d %4d", 
			8*n*j+j2, 8*n*j+j2+4*n, 8*n*j+4*n, 8*n*j, 8*n*j+j2+8*n, 
			8*n*j+j2+12*n, 8*n*j+12*n, 8*n*j+8*n);
		fprintf(o1, " %4d\n",0);
		++dum;
	   }
	}

	if(el_type > 1 )
	{
           fprintf(o1, "node no. coordinates\n");
	   dum = 0;
	   dt=90.0*3.14159/180.0/(double)n;
	   dt2=180.0*3.14159/180.0/(double)(n+1);
	   angle=-(double)(n)*dt2;
           for( j = 0; j < n; ++j )
           {
		S=sin(angle);
		Y=R*cos(angle);
		angle2=0.0;
        	for( j2 = 0; j2 < 4*n; ++j2 )
        	{
	                Z=R*cos(angle2)*S;
                	X=R*sin(angle2)*S;
        		fprintf(o1, "%4d  ", dum);
			++dum;
                	fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			angle2+=dt;
		}
		angle+=dt2;
	   }
           fprintf(o1, "The corresponding top nodes are:\n");
	   dum = 0;
	   angle=-(double)(n)*dt2;
           for( j = 0; j < n; ++j )
           {
		S=sin(angle);
		Y=(R+T)*cos(angle);
		angle2=0.0;
        	for( j2 = 0; j2 < 4*n; ++j2 )
        	{
	                Z=(R+T)*cos(angle2)*S;
                	X=(R+T)*sin(angle2)*S;
        		fprintf(o1, "%4d  ", dum);
			++dum;
                	fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			angle2+=dt;
		}
		angle+=dt2;
	   }
	}

	if(el_type < 2 )
	{
           fprintf(o1, "node no. coordinates\n");
	   dum = 0;
	   dt=90.0*3.14159/180.0/(double)n;
	   dt2=180.0*3.14159/180.0/(double)(n+1);
	   angle=-(double)(n)*dt2;
           for( j = 0; j < n; ++j )
           {
		S=sin(angle);
		Y=R*cos(angle);
		angle2=0.0;
        	for( j2 = 0; j2 < 4*n; ++j2 )
        	{
	                Z=R*cos(angle2)*S;
                	X=R*sin(angle2)*S;
        		fprintf(o1, "%4d  ", dum);
			++dum;
                	fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			angle2+=dt;
		}
		angle2=0.0;
		Y=(R+T)*cos(angle);
        	for( j2 = 0; j2 < 4*n; ++j2 )
        	{
	                Z=(R+T)*cos(angle2)*S;
                	X=(R+T)*sin(angle2)*S;
        		fprintf(o1, "%4d  ", dum);
			++dum;
                	fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			angle2+=dt;
		}
		angle+=dt2;
	   }
	}

	if(el_type > 1 )
	{
	   fprintf( o1, "prescribed displacement x: node  disp value\n");
	   fprintf( o1, "%4d\n",-10);
	   fprintf( o1, "prescribed displacement y: node  disp value\n");
	   fprintf( o1, "%4d\n",-10);
	   fprintf( o1, "prescribed displacement z: node  disp value\n");
	   fprintf( o1, "%4d\n",-10);
	   fprintf( o1, "prescribed angle phi x: node angle value\n");
	   fprintf( o1, "%4d\n",-10);
	   fprintf( o1, "prescribed angle phi y: node angle value\n");
	   fprintf( o1, "%4d\n",-10);
	   fprintf( o1, "node with point load x, y, z and 2 moments phi x, phi y\n");
	   fprintf( o1, "%4d\n",-10);
	   fprintf( o1, "node no. with stress and stress vector in lamina xx,yy,xy,zx,yz\n");
	   fprintf( o1, "%4d\n",-10);
	}

	if(el_type < 2 )
	{
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
	}


        return 1;
}

main(int argc, char** argv)
{
	int check;
	check=writer();
} 
