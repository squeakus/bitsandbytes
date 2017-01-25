/*
    This library function reads in data from a finite element
    data set to prepare it for the patch test analysis on a
    plate element .  First, it reads in the data, then
    it creates the prescribed displacements so that the main
    finite element program can do the analysis.  It also splits
    the patch test into 2, one for the z coordintate and one for
    phi x,y angles.

		Updated 5/19/99

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "plconst.h"
#include "plstruct.h"


int main(int argc, char** argv)
{
	int dof, nmat, nmode, numel, numnp;
        int i,j,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, fdum4, fdum5, uz[9], uphix[9], uphiy[9];
        char name[20];
	char buf[ BUFSIZ ];
        FILE *o1, *o2, *o3, *o4;
	char text;

        printf( "What is the name of the file containing the \n");
        printf( "plate structural data? \n");
        scanf( "%20s",name);

        o1 = fopen( name,"r" );
        if(o1 == NULL ) {
                printf("error on open\n");
                exit(1);
        }
        o2 = fopen( "patch","w" );
        o3 = fopen( "patch.z","w" );
        o4 = fopen( "patch.phi","w" );

        fprintf( o2, "      numel numnp nmat nmode   This is the patch test \n ");
        fprintf( o3, "      numel numnp nmat nmode   This is the patch test \n ");
        fprintf( o4, "      numel numnp nmat nmode   This is the patch test \n ");
        fgets( buf, BUFSIZ, o1 );
        fscanf( o1, "%d %d %d %d\n ",&numel,&numnp,&nmat,&nmode);
        fprintf( o2, "    %4d %4d %4d %4d\n ", numel,numnp,nmat,nmode);
        fprintf( o3, "    %4d %4d %4d %4d\n ", numel,numnp,nmat,nmode);
        fprintf( o4, "    %4d %4d %4d %4d\n ", numel,numnp,nmat,nmode);
        fgets( buf, BUFSIZ, o1 );

        fprintf( o2, " matl no., E mod., Poiss. Ratio, density, thick, shear fac.\n");
        fprintf( o3, " matl no., E mod., Poiss. Ratio, density, thick, shear fac.\n");
        fprintf( o4, " matl no., E mod., Poiss. Ratio, density, thick, shear fac.\n");
        for( i = 0; i < nmat; ++i )
        {
           fscanf( o1, "%d\n ",&dum);
           fprintf( o2, "%3d ",dum);
           fprintf( o3, "%3d ",dum);
           fprintf( o4, "%3d ",dum);
           fscanf( o1, " %lf %lf %lf %lf %lf\n",&fdum1, &fdum2, &fdum3, &fdum4, &fdum5);
           fprintf( o2, " %9.4f %9.4f %9.4f %9.4f %9.4f\n ",fdum1, fdum2, fdum3, fdum4, fdum5);
           fprintf( o3, " %9.4f %9.4f %9.4f %9.4f %9.4f\n ",fdum1, fdum2, fdum3, fdum4, fdum5);
           fprintf( o4, " %9.4f %9.4f %9.4f %9.4f %9.4f\n ",fdum1, fdum2, fdum3, fdum4, fdum5);
        }
        fgets( buf, BUFSIZ, o1 );

        fprintf( o2, "el no., connectivity, matl no. \n");
        fprintf( o3, "el no., connectivity, matl no. \n");
        fprintf( o4, "el no., connectivity, matl no. \n");
        for( i = 0; i < numel; ++i )
        {
           fscanf( o1,"%d ",&dum);
           fprintf( o2, "%4d ",dum);
           fprintf( o3, "%4d ",dum);
           fprintf( o4, "%4d ",dum);
           for( j = 0; j < npel; ++j )
           {
                fscanf( o1, "%d",&dum3);
                fprintf( o2, "%4d ",dum3);
                fprintf( o3, "%4d ",dum3);
                fprintf( o4, "%4d ",dum3);
           }
           fscanf( o1,"%d\n",&dum4);
           fprintf( o2, " %3d\n",dum4);
           fprintf( o3, " %3d\n",dum4);
           fprintf( o4, " %3d\n",dum4);
        }
        fgets( buf, BUFSIZ, o1 );

        fprintf( o2, "node no., coordinates \n");
        fprintf( o3, "node no., coordinates \n");
        fprintf( o4, "node no., coordinates \n");
        for( i = 0; i < numnp; ++i )
        {
           fscanf( o1,"%d ",&dum);
           fprintf( o2, "%d ",dum);
           fprintf( o3, "%d ",dum);
           fprintf( o4, "%d ",dum);
           fscanf( o1, "%lf %lf",&fdum1,&fdum2);
           fprintf( o2, "%9.4f %9.4f ",fdum1,fdum2);
           fprintf( o3, "%9.4f %9.4f ",fdum1,fdum2);
           fprintf( o4, "%9.4f %9.4f ",fdum1,fdum2);
	   *(uz+i) = 6.000e-04*fdum1 - 4.000e-04*fdum2;
	   *(uphix+i) = 1.000e-04*fdum1 + 3.000e-04*fdum2;
	   *(uphiy+i) = 2.000e-04*fdum1 + 4.000e-04*fdum2;
           fscanf( o1,"\n");
           fprintf( o2, "\n");
           fprintf( o3, "\n");
           fprintf( o4, "\n");
        }
        fgets( buf, BUFSIZ, o1 );

        dum= 0;
        fprintf( o2, "prescribed displacement z: node  disp value\n");
        fprintf( o3, "prescribed displacement z: node  disp value\n");
        fprintf( o4, "prescribed displacement z: node  disp value\n");
        for( i = 0; i < 4; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uz+i));
                fprintf( o3, "%4d %14.6e\n",i,*(uz+i));
        }
        for( i = 5; i < numnp; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uz+i));
                fprintf( o3, "%4d %14.6e\n",i,*(uz+i));
        }
        fprintf( o4, "%4d %14.6e\n",0,0.0);
        fprintf( o2, "%4d\n ",-10);
        fprintf( o3, "%4d\n ",-10);
        fprintf( o4, "%4d\n ",-10);
        fprintf( o2, "prescribed angle phi x: node angle value\n");
        fprintf( o3, "prescribed angle phi x: node angle value\n");
        fprintf( o4, "prescribed angle phi x: node angle value\n");
        for( i = 0; i < 4; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uphix+i));
                fprintf( o4, "%4d %14.6e\n",i,*(uphix+i));
        }
        for( i = 5; i < numnp; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uphix+i));
                fprintf( o4, "%4d %14.6e\n",i,*(uphix+i));
        }
        fprintf( o4, "%4d %14.6e\n",0,0.0);
        fprintf( o2, "%4d\n ",-10);
        fprintf( o3, "%4d\n ",-10);
        fprintf( o4, "%4d\n ",-10);
        fprintf( o2, "prescribed angle phi y: node angle value\n");
        fprintf( o3, "prescribed angle phi y: node angle value\n");
        fprintf( o4, "prescribed angle phi y: node angle value\n");
        for( i = 0; i < 4; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uphiy+i));
                fprintf( o4, "%4d %14.6e\n",i,*(uphiy+i));
        }
        for( i = 5; i < numnp; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uphiy+i));
                fprintf( o4, "%4d %14.6e\n",i,*(uphiy+i));
        }
        fprintf( o4, "%4d %14.6e\n",0,0.0);
        fprintf( o2, "%4d\n ",-10);
        fprintf( o3, "%4d\n ",-10);
        fprintf( o4, "%4d\n ",-10);

        fgets( buf, BUFSIZ, o1 );
        fgets( buf, BUFSIZ, o1 );
        fprintf( o2, "node with point load z and 2 moments phi x, phi y\n");
        fprintf( o3, "node with point load z and 2 moments phi x, phi y\n");
        fprintf( o4, "node with point load z and 2 moments phi x, phi y\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o3, "%4d\n ",-10);
        fprintf( o4, "%4d\n ",-10);
        fprintf( o2, "element no. and gauss pt. no. with moment ");
        fprintf( o3, "element no. and gauss pt. no. with moment ");
        fprintf( o4, "element no. and gauss pt. no. with moment ");
        fprintf( o2, "xx,yy,xy and stress vector in zx,yz\n");
        fprintf( o3, "xx,yy,xy and stress vector in zx,yz\n");
        fprintf( o4, "xx,yy,xy and stress vector in zx,yz\n");
        fprintf( o2, "%4d ",-10);
        fprintf( o3, "%4d ",-10);
        fprintf( o4, "%4d ",-10);

        fprintf( o2, "\n\n\n Note: Remove the prescribed angles and disp. for node 4.\n");
        fprintf( o2, "%4d %14.6e\n",4,*(uz+4));
        fprintf( o2, "%4d %14.6e\n",4,*(uphix+4));
        fprintf( o2, "%4d %14.6e\n",4,*(uphiy+4));
        fprintf( o3, "\n\n\n Note: Remove the prescribed angles and disp. for node 4.\n");
        fprintf( o3, "%4d %14.6e\n",4,*(uz+4));
        fprintf( o4, "\n\n\n Note: Remove the prescribed angles and disp. for node 4.\n");
        fprintf( o4, "%4d %14.6e\n",4,*(uphix+4));
        fprintf( o4, "%4d %14.6e\n",4,*(uphiy+4));
        fprintf( o2, "\n You also need to divide this file into 2 patch test");
        fprintf( o2, "\n The first you can call patch.z and remove all the");
        fprintf( o2, "\n prescribed angles phi.  The second, call patch.phi");
        fprintf( o2, "\n and remove the prescribed z.");
        fprintf( o2, "\n");

        return 1;
}

