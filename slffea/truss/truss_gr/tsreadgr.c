/*
    This library function reads in additional strain
    data for the graphics program for truss elements.

		Updated 8/14/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "../truss/tsconst.h"
#include "../truss/tsstruct.h"

extern int dof, nmat, numel, numnp;

int tsreader_gr( FILE *o1, SDIM *strain)
{
	int i,j,dum;
	char buf[ BUFSIZ ];
	char text;

	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );

	printf("strain for ele: ");
	fscanf( o1,"%d",&dum);
	printf( "(%6d)",dum);
	while( dum > -1 )
	{
	   fscanf( o1,"%lf ",&strain[dum].xx);
	   printf(" %12.5e",strain[dum].xx);
	   fscanf( o1,"\n");
	   printf( "\n");
	   printf("strain for ele: ");
	   fscanf( o1,"%d",&dum);
	   printf( "(%6d)",dum);
	}
	printf( "\n\n");

	return 1;
}

