/*
    This library function reads in data from a finite element
    data set to prepare it for the patch test analysis on a
    quadrilateral element .  First, it reads in the data, then
    it creates the prescribed displacements so that the main
    finite element program can do the analysis. 

		Updated 5/24/00

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "qdconst.h"
#include "qdstruct.h"


int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, nmode, plane_stress_flag;
        int i,j,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, ux[9], uy[9];
        char name[20];
	char buf[ BUFSIZ ];
        FILE *o1, *o2;
	char text;

        printf( "What is the name of the file containing the \n");
        printf( "quad structural data? \n");
        scanf( "%20s",name);

        o1 = fopen( name,"r" );
        if(o1 == NULL ) {
                printf("error on open\n");
                exit(1);
        }
        o2 = fopen( "patch","w" );

        fprintf( o2, "   numel numnp nmat plane_stress_flag  This is the patch test \n ");
        fgets( buf, BUFSIZ, o1 );
        fscanf( o1, "%d %d %d %d %d\n ", &numel, &numnp, &nmat, &nmode, &plane_stress_flag);
        fprintf( o2, "    %4d %4d %4d %4d %4d\n ", numel, numnp, nmat, nmode, plane_stress_flag);
        fgets( buf, BUFSIZ, o1 );

        fprintf( o2, "matl no., E modulus, Poisson Ratio, density \n");
        for( i = 0; i < nmat; ++i )
        {
           fscanf( o1, "%d\n ",&dum);
           fprintf( o2, "%3d ",dum);
           fscanf( o1, " %lf %lf %lf\n",&fdum1, &fdum2, &fdum3);
           fprintf( o2, " %9.4f %9.4f %9.4f\n ",fdum1, fdum2, fdum3);
        }
        fgets( buf, BUFSIZ, o1 );

        fprintf( o2, "el no., connectivity, matl no. \n");
        for( i = 0; i < numel; ++i )
        {
           fscanf( o1,"%d ",&dum);
           fprintf( o2, "%4d ",dum);
           for( j = 0; j < npel; ++j )
           {
                fscanf( o1, "%d",&dum3);
                fprintf( o2, "%4d ",dum3);
           }
           fscanf( o1,"%d\n",&dum4);
           fprintf( o2, " %3d\n",dum4);
        }
        fgets( buf, BUFSIZ, o1 );

        fprintf( o2, "node no., coordinates \n");
        for( i = 0; i < numnp; ++i )
        {
           fscanf( o1,"%d ",&dum);
           fprintf( o2, "%d ",dum);
           fscanf( o1, "%lf %lf",&fdum1,&fdum2);
           fprintf( o2, "%9.4f %9.4f ",fdum1,fdum2);
	   *(ux+i) = 1.000e-04*fdum1 + 3.000e-04*fdum2;
	   *(uy+i) = 2.000e-04*fdum1 + 4.000e-04*fdum2;
           fscanf( o1,"\n");
           fprintf( o2, "\n");
        }
        fgets( buf, BUFSIZ, o1 );

        dum= 0;
        fprintf( o2, "prescribed displacement x: node  disp value\n");
        for( i = 0; i < 4; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(ux+i));
        }
        for( i = 5; i < numnp; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(ux+i));
        }
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "prescribed displacement y: node  disp value\n");
        for( i = 0; i < 4; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uy+i));
        }
        for( i = 5; i < numnp; ++i )
        {
                fprintf( o2, "%4d %14.6e\n",i,*(uy+i));
        }
        fprintf( o2, "%4d\n ",-10);

        fgets( buf, BUFSIZ, o1 );
        fgets( buf, BUFSIZ, o1 );
        fprintf( o2, "node with point load and load vector in x,y\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "element and gauss pt. with stress and stress vector in xx,yy,xy\n");
        fprintf( o2, "%4d ",-10);

        fprintf( o2, "\n\n\n Note: Remove the prescribed displacements for node 4.\n");
        fprintf( o2, "%4d %14.6e\n",4,*(ux+4));
        fprintf( o2, "%4d %14.6e\n",4,*(uy+4));

        return 1;
}

