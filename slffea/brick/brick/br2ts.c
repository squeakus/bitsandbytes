/* This progam converts brick data to truss
   data by modifying the connectivity.

    Updated 12/12/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 2008  San Le

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
	int numel, numnp, nmat, npel;

	printf( "\nWhat is the name of the file\n");
	scanf( "%s", name);
	o2 = fopen( name,"r" );

	if(o2 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	fgets( buf, BUFSIZ, o2 );
	npel = 8;

	fscanf(o2, "%d %d %d %d\n", &numel, &numnp, &nmat, &dum2);
	printf( "%d %d %d %d\n", numel, numnp, nmat, dum2);
	fgets( buf, BUFSIZ, o2 );
	for( i = 0; i < nmat; ++i)
	{
		fgets( buf, BUFSIZ, o2 );
	}
	fgets( buf, BUFSIZ, o2 );

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

            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 1), *(connect_el + 2), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 2), *(connect_el + 3), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 3), *(connect_el), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 3), *(connect_el + 1), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 2), *(connect_el), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 2), *(connect_el + 6), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 1), *(connect_el + 6), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el + 3), *(connect_el + 6), 0);
	    ++dum3;
            printf( "%8d  %8d  %8d  %8d\n", dum3, *(connect_el), *(connect_el + 6), 0);
	    ++dum3;

	}

/* The code below is specific for the ball mesh. */

/*
	for( i = 480; i < 511; ++i)
	{
            printf( "%8d  %8d  %8d  %8d\n", dum3, i, i+1, 0);
	    ++dum3;
	}
        printf( "%8d  %8d  %8d  %8d\n", dum3, 511, 480, 0);
	++dum3;
*/

	return 1;
}	

