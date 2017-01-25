/*
    This library function reads in data from a finite element
    data set to prepare it for the patch test analysis on a
    shell element .  First, it reads in the data of a 4 element
    mesh which lies in the xy plane and then uses the equation
    of a plane:

	a*x + b*y + c*z = d

    to move the mesh out of plane.  The new coordinates will
    be printed to the screen.  You should copy it into "shpatch.in".

		Updated 5/12/99

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "shconst.h"
#include "shstruct.h"


int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, integ_flag;
        int i,j,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, fdum4, a, b, c, d, e;
        char name[20];
	char buf[ BUFSIZ ];
        FILE *o1;
	char text;

	a = 2.0;
	b = -3.0;
	c = 4.0;
	d = -2.1;

        printf( "What is the name of the file containing the \n");
        printf( "shell structural data? \n");
        scanf( "%20s",name);

        o1 = fopen( name,"r" );
        if(o1 == NULL ) {
                printf("error on open\n");
                exit(1);
        }

        printf( "      numel numnp nmat    This is the patch test \n ");
        fgets( buf, BUFSIZ, o1 );
        fscanf( o1, "%d %d %d %d\n ",&numel,&numnp,&nmat,&integ_flag);
        printf( "    %4d %4d %4d %4d\n ", numel,numnp,nmat,integ_flag);
        fgets( buf, BUFSIZ, o1 );

        printf( " matl no., E mod., Poiss. Ratio, shear fac. \n");
        for( i = 0; i < nmat; ++i )
        {
           fscanf( o1, "%d\n ",&dum);
           printf( "%3d ",dum);
           fscanf( o1, " %lf %lf %lf %lf\n",&fdum1, &fdum2, &fdum3);
           printf( " %9.4f %9.4f %9.4f\n ",fdum1, fdum2, fdum3);
        }
        fgets( buf, BUFSIZ, o1 );

        printf( "el no., connectivity, matl no. \n");
        for( i = 0; i < numel; ++i )
        {
           fscanf( o1,"%d ",&dum);
           printf( "%4d ",dum);
           for( j = 0; j < npell4; ++j )
           {
                fscanf( o1, "%d",&dum3);
                printf( "%4d ",dum3);
           }
           fscanf( o1,"%d\n",&dum4);
           printf( " %3d\n",dum4);
        }
        fgets( buf, BUFSIZ, o1 );

        printf( "node no., coordinates \n");
        for( i = 0; i < numnp; ++i )
        {
           fscanf( o1,"%d ",&dum);
           printf( "%d ",dum);
           fscanf( o1, "%lf %lf %lf",&fdum1,&fdum2,&fdum3);

/* Calculate the z coordinates */

	   fdum3 = (d - a*fdum1 -b*fdum2)/c;
           printf( "%9.4f %9.4f %9.4f",fdum1,fdum2,fdum3);
           fscanf( o1,"\n");
           printf( "\n");
        }
        fgets( buf, BUFSIZ, o1 );
        printf( "The corresponding top nodes are:\n");
        for( i = 0; i < numnp; ++i )
        {
           fscanf( o1,"%d ",&dum);
           printf( "%d ",dum);
           fscanf( o1, "%lf %lf %lf",&fdum1,&fdum2,&fdum3);

/* Calculate the z coordinates */
	   fdum3 = (d - a*fdum1 -b*fdum2)/c + .1;

           printf( "%9.4f %9.4f %9.4f",fdum1,fdum2,fdum3);
           fscanf( o1,"\n");
           printf( "\n");
        }

        return 1;
}

