/*
    This library function reads in the connectivity data
    for the surface elements only from the file *.con.
    This smaller connectivity set will speed up the graphics.

		Updated 8/12/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if BRICK1
#include "../brick/brick/brconst.h"
#include "../brick/brick/brstruct.h"
#endif
#if BRICK2
#include "../brick/brick/brconst.h"
#include "../brick/brick2/br2struct.h"
#endif
#if TETRA1
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra/testruct.h"
#endif
#if WEDGE1
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge/westruct.h"
#endif


extern int numel;

int ConnectSurfreader( int *connecter, int *el_matl, char *name)
{
	int i,j,dum,connect_read_flag, name_length;
	char *ccheck;
	char buf[ BUFSIZ ];
	char text, connect_dat[30];
	FILE *o4;

/*  o4 contains the connectivity for the surface elements
*/

/* Search for the file (name).con which contains the surface
   conectivity data
*/

	name_length = strlen(name);
	if( name_length > 25) name_length = 25;

/* Open surface connectivity data file */

	memset(connect_dat,0,30*sizeof(char));

	ccheck = strncpy(connect_dat, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(connect_dat+name_length, ".con", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

	o4 = fopen( connect_dat,"r" );

	connect_read_flag = 1;
	if(o4 == NULL ) {
		printf("There is no surface connectivity file %30s\n",
			connect_dat);
		connect_read_flag = 0;
	}

	if( connect_read_flag )
	{
	    fgets( buf, BUFSIZ, o4 );
	    printf( "\n");

	    fscanf( o4, "%d\n",&numel);
	    printf( "%d\n",numel);

	    fgets( buf, BUFSIZ, o4 );
	    printf( "\n");

	    for( i = 0; i < numel; ++i )
	    {
		fscanf( o4,"%d ",&dum);
		printf( "connectivity for element (%6d) ",dum);
		for( j = 0; j < npel; ++j )
		{
			fscanf( o4, "%d",(connecter+npel*dum+j));
			printf( "%6d ",*(connecter+npel*dum+j));
		}
		fscanf( o4,"%d\n",(el_matl+dum));
		printf( " with matl %3d\n",*(el_matl+dum));
	    }
	}
	printf( "\n");

	return 1;
}

