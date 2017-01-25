/* This progam converts brick data to wedge
   data by modifying the connectivity.

    Updated 6/25/01

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main()
{
	FILE *o1, *o2;
	int check, i, j, k, dum, dum1, dum2, dum3, connect_el[8];
	unsigned short int dum_vec[3];
	double fdum, fdum2, rndnum;
	char name[30], chardum[30], one_char;
	char *ccheck, buf[ BUFSIZ ];
	int numel, numnp, npel;

	printf( "\nWhat is the name of the file\n");
	scanf( "%s", name);
	o2 = fopen( name,"r" );

	if(o2 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	fgets( buf, BUFSIZ, o2 );
	npel = 8;

	fscanf(o2, "%d %d %d %d", &numel, &numnp, &dum, &dum2);
	printf( "%d %d %d %d", numel, numnp, dum, dum2);
	for( i = 0; i < 4; ++i)
	{
		fgets( buf, BUFSIZ, o2 );
	}

	dum3 = 0;
	printf( "\n");
	for( i = 0; i < numel; ++i)
	{

/* connectivity */


	    fscanf( o2, "\n");
            fscanf(o2, "%d", &dum2);
	    for( j = 0; j < npel; ++j)
	    {
        	fscanf(o2, "%d", (connect_el + j));
	    }
            fscanf(o2, "%d", &dum2);

            printf( "%4d", dum3);
            printf( "  %8d", *(connect_el + 0));
            printf( "  %8d", *(connect_el + 1));
            printf( "  %8d", *(connect_el + 2));
            printf( "  %8d", *(connect_el + 4));
            printf( "  %8d", *(connect_el + 5));
            printf( "  %8d", *(connect_el + 6));
            printf( "  %8d\n", 0);
	    ++dum3;
            printf( "%4d", dum3);
            printf( "  %8d", *(connect_el + 0));
            printf( "  %8d", *(connect_el + 2));
            printf( "  %8d", *(connect_el + 3));
            printf( "  %8d", *(connect_el + 4));
            printf( "  %8d", *(connect_el + 6));
            printf( "  %8d", *(connect_el + 7));
            printf( "  %8d\n", 0);
	    ++dum3;

	}

	return 1;
}	

