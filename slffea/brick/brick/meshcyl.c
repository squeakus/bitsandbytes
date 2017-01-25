/*
    This program generates the mesh for a cylinder based on
    either brick or shell elements.
  
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

#define pi   		3.141592654	

int writer(void)
{
        FILE *o1;
        int i,j,nh,nv,numnp,numel,el_type,dum;
	double S,C,R,T,X,Y,Z,length,length_inc,angle,angle_T,angle_inc,angle_start;

    	printf("How many elements do you want in horizontal?\n");
    	scanf( "%d",&nh);
    	printf("How many elements do you want in vertical?\n");
    	scanf( "%d",&nv);
    	printf("What is the Radius?\n");
    	scanf( "%lf",&R);
    	printf("What is the Angle?\n");
    	scanf( "%lf",&angle_T);
    	printf("What is the Length?\n");
    	scanf( "%lf",&length);
    	printf("What is the Thickness?\n");
    	scanf( "%lf",&T);
    	printf("What type of element is this for?\n");
	printf("1) brick \n");
	printf("2) shell \n");
    	scanf( "%d",&el_type);

	numnp=(nv+1)*(nh+1);
	numel=(nv)*(nh);
	angle_inc=angle_T*pi/180.0/(double)nv;
	angle_start=(90.0-angle_T)*pi/180.0;
	angle=angle_start;
	length_inc=length/(double)nv;
	if(el_type < 2 ) o1 = fopen( "cylinder.br","w" );
	else o1 = fopen( "cylinder.sh","w" );
	if(el_type > 1 )
	{
		fprintf( o1, "   numel numnp nmat nmode integ_flag This is a shell cylinder\n");
        	fprintf(o1, "   %4d %4d %4d %4d %4d\n", numel, numnp, 1, 0, 1);
		fprintf( o1, " matl no., E mod., Poiss. Ratio, density, shear fac. \n");
        	fprintf(o1, " %4d %9.4f %9.4f %9.4f %9.4f\n", 0, 4.32e8, 0.0, 2.77e3, 0.8333333);
		fprintf( o1, "el no., connectivity, matl no.\n");
        	for( i = 0; i < nh; ++i )
        	{
        	    for( j = 0; j < nv; ++j )
        	    {
        		fprintf(o1, "%4d ",i*nv+j);
        		fprintf(o1, "%4d %4d %4d %4d",
			   	(nv+1)*i+j,(nv+1)*(i+1)+j,
				(nv+1)*(i+1)+1+j,(nv+1)*i+j+1);
        		fprintf(o1, "%4d\n",0);
		    }
		}
	}
	if(el_type < 2 )
	{
        	fprintf(o1, "   numel numnp nmat nmode  (This is for a brick cylinder\n");
        	fprintf(o1, "%d %d %d %d\n",numel,2*numnp,1,0);
        	fprintf(o1, "matl no., E modulus, Poisson Ratio, density\n");
        	fprintf(o1, "%d %16.4f %16.4f %16.4f\n",0, 4.32e8, 0.0, 2.77e3);
        	fprintf(o1, "el no.,connectivity, matl no.\n");
		dum = 0;
        	for( i = 0; i < nh; ++i )
        	{
        	    for( j = 0; j < nv; ++j )
        	    {
        		fprintf(o1, "%4d ",dum);
        		fprintf(o1, "%4d %4d %4d %4d ",
			   	(nv+1)*i+j,(nv+1)*(i+1)+j,
				(nv+1)*(i+1)+1+j,(nv+1)*i+j+1);
        		fprintf(o1, "%4d %4d %4d %4d %4d\n",
			   	(nv+1)*i+j+numnp,(nv+1)*(i+1)+j+numnp,
				(nv+1)*(i+1)+1+j+numnp,(nv+1)*i+j+1+numnp,0);
			++dum;
		    }
		}
	}
	dum = 0;
	Y=0.0;
        fprintf(o1, "node no. coordinates\n");
        for( i = 0; i < nh+1; ++i )
        {
            for( j = 0; j < nv+1; ++j )
            {
		X=R*cos(angle);
		Z=R*sin(angle);
        	fprintf(o1, "%4d ",dum);
                fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
	   	angle+=angle_inc;
		++dum;
            }
	    Y+=length_inc;
	    angle=angle_start;
        }
	if(el_type > 1 )
	{
                fprintf(o1, "The corresponding top nodes are:\n");
		dum = 0;
	}
	Y=0.0;
	angle=angle_start;
        for( i = 0; i < nh+1; ++i )
        {
            for( j = 0; j < nv+1; ++j )
            {
		X=(R+T)*cos(angle);
		Z=(R+T)*sin(angle);
        	fprintf(o1, "%4d ",dum);
                fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
	   	angle+=angle_inc;
		++dum;
            }
	    Y+=length_inc;
	    angle=angle_start;
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
           fprintf( o1, "node no. with stress and stress vector in xx,yy,zz,xy,zx,yz\n");
           fprintf( o1, "%4d ",-10);
	}
        return 1;
}

main(int argc, char** argv)
{
	int check;
	check=writer();
} 
