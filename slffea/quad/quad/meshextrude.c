/* This progam extrudes either a triangle or quadrilateral mesh in the
   z direction to create either wedge or brick elements respectively.

    Updated 3/4/03

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003  San Le

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
	int check, i, j, k, dum, dum1, dum2, dum3;
	int *connect, nlayers, numel, numnp, nmat, npel, el_type;
	unsigned short int dum_vec[3];
	double fdum, fdum1, fdum2, fdum3;
	double *coord, width;
	char name[30], chardum[30], one_char;
	char *ccheck, buf[ BUFSIZ ];

	memset(chardum,0,30*sizeof(char));
	memset(name,0,20*sizeof(char));

	printf( "\nWhat type of element is being extruded?\n");
	printf("1) triangle \n");
	printf("2) quadrilateral \n");
	scanf( "%d", &el_type);

	nmat = 1;
	npel = 3;
	if(el_type == 2) npel = 4;

	printf( "\nWhat is the name of the file?\n");
	scanf( "%s", name);
	o2 = fopen( name,"r" );
	if(o2 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}
	printf( "\nHow many layers do you want?\n");
	scanf( "%d", &nlayers);
	printf( "\nWhat is the distance bewteen layers?\n");
	scanf( "%lf", &width);

	fgets( buf, BUFSIZ, o2 );
	if(el_type == 2) printf( "   numel numnp nmat nmode This is for an 8 node brick\n");
	else printf( "   numel numnp nmat nmode This is for a 6 node wedge\n");

	fscanf(o2, "%d %d %d %d %d\n", &numel, &numnp, &nmat, &dum, &dum2);
	printf( "%d %d %d %d\n",
		numel*nlayers, numnp*(nlayers+1), nmat, dum);

	fgets( buf, BUFSIZ, o2 );
	printf( "matl no., E modulus, Poisson Ratio, and density\n");
        fscanf( o2, "%d %lf %lf %lf\n", &dum, &fdum, &fdum1, &fdum2);
        printf("%4d %14.6e %14.6e %14.6e\n", dum, fdum, fdum1, fdum2);

	connect=(int *)calloc(numel*nlayers*npel,sizeof(int));
	coord=(double *)calloc((numnp+1)*nlayers,sizeof(double));

	fgets( buf, BUFSIZ, o2 );
	printf( "el no., connectivity, matl no.\n");
	for( i = 0; i < numel; ++i)
	{
/* connectivity */

            fscanf(o2, "%d", &dum2);
            printf( "  %4d", dum2);

	    for( j = 0; j < npel; ++j)
	    {
        	fscanf(o2, "%d", (connect + npel*i + j));
        	printf( "  %8d", *(connect + npel*i + j));
	    }
	    for( j = 0; j < npel; ++j)
	    {
        	printf( "  %8d", *(connect + npel*i + j) + numnp);
	    }
            fscanf(o2, "%d\n", &dum);
            printf( "  %8d\n", dum);
	}

	dum2 = numel;
	for( k = 0; k < nlayers - 1; ++k)
	{
	    for( i = 0; i < numel; ++i)
	    {
/* connectivity for the additional layers*/

            	printf( "  %4d", dum2);
		for( j = 0; j < npel; ++j)
		{
        	    printf( "  %8d", *(connect + npel*i + j) + numnp*(k+1));
		}
		for( j = 0; j < npel; ++j)
		{
        	    printf( "  %8d", *(connect + npel*i + j) + numnp*(k+2));
		}
        	printf( "  %8d\n", 0);
		++dum2;
	    }
	}

	fgets( buf, BUFSIZ, o2 );
	printf( "node no., coordinates\n" );

	for( i = 0; i < numnp; ++i)
	{
        	fscanf( o2, "%d", &dum2);
        	printf( " %4d", dum2);
		for( j = 0; j < 2; ++j)
		{
        	   fscanf( o2, "%lf", (coord + 2*i + j));
        	   printf( "  %14.6e", *(coord + 2*i + j));
		}
        	printf( "  %14.6e", 0.0);
        	printf( "\n");
	}

	dum2 = numnp;
	for( k = 0; k < nlayers; ++k)
	{
	  for( i = 0; i < numnp; ++i)
	  {
        	printf( " %4d", dum2);
		for( j = 0; j < 2; ++j)
		{
        	   printf( "  %14.6e", *(coord + 2*i + j));
		}
        	printf( "  %14.6e", width*((double)(k+1)));
        	printf( "\n");
		++dum2;
	   }
	}

	printf( "prescribed displacement x: node  disp value\n");
	printf( "%4d\n ",-10);
	printf( "prescribed displacement y: node  disp value\n");
	printf( "%4d\n ",-10);
	printf( "prescribed displacement z: node  disp value\n");
	printf( "%4d\n ",-10);
	printf( "node with point load and load vector in x,y,z\n");
	printf( "%4d\n ",-10);
	printf( "node no. with stress and stress vector xx,yy,xy,zx,yz\n");
	printf( "%4d ",-10);

	return 1;
}	

