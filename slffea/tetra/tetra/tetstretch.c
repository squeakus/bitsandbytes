/* This progam takes the square mesh made up of 24 tetrahedron elements
   and 15 nodes and expands it. 

    Updated 8/29/01

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
	int check, i, j, k, dum, dum1, dum2, dum3;
	int *connect, nlayers, nlayers2, remainder;
	int numel, numnp, numnp2, nmat, npel, nsd, matl_num;
	unsigned short int dum_vec[3];
	double fdum, fdum1, fdum2, fdum3;
	double *coord, width;
	char name[30], chardum[30], one_char;
	char *ccheck, buf[ BUFSIZ ];

	memset(chardum,0,30*sizeof(char));
	memset(name,0,20*sizeof(char));

	npel = 4;
	nmat = 1;
	nsd = 3;
	width = 1.0;

	ccheck = strncpy(name, "te24", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

	o2 = fopen( name,"r" );
	if(o2 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}
	printf( "\nHow many additional layers do you want?\n");
	scanf( "%d", &nlayers);

	fgets( buf, BUFSIZ, o2 );
	printf( "   numel numnp nmat nmode This is for a 4 node tetrahedron\n");

	fscanf(o2, "%d %d %d %d %d\n", &numel, &numnp, &nmat, &dum, &dum2);
	numnp2 = numnp + (numnp - 5)*nlayers;
	printf( "%d %d %d %d\n",
		numel*(nlayers+1), numnp2, nmat, dum);

	fgets( buf, BUFSIZ, o2 );
	printf( "matl no., E modulus, Poisson Ratio, and density\n");
        fscanf( o2, "%d %lf %lf %lf\n", &dum, &fdum, &fdum1, &fdum2);
        printf("%4d %14.6e %14.6e %14.6e\n", dum, fdum, fdum1, fdum2);

	connect=(int *)calloc(numel*npel,sizeof(int));
	coord=(double *)calloc(numnp2,sizeof(double));

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
        	fscanf(o2, "%d\n", &matl_num);
        	printf( "  %8d\n", matl_num);
	}

	dum2 = numel;
	for( k = 0; k < nlayers; ++k)
	{
	    for( i = 0; i < numel; ++i)
	    {
/* connectivity of additional layers */

		printf( "  %4d", dum2);

		for( j = 0; j < npel; ++j)
		{
        		printf( "  %8d", *(connect + npel*i + j) + 10*(k+1));
		}
		printf( "  %8d\n", matl_num);
		++dum2;
	    }
	}

	fgets( buf, BUFSIZ, o2 );
	printf( "node no., coordinates\n" );

	for( i = 0; i < numnp; ++i)
	{
        	fscanf( o2, "%d", &dum2);
        	printf( " %4d", dum2);
		for( j = 0; j < nsd; ++j)
		{
        	   fscanf( o2, "%lf", (coord + nsd*i + j));
        	   printf( "  %14.6e", *(coord + nsd*i + j));
		}
        	printf( "\n");
	}

	dum2 = numnp;
	for( k = 0; k < nlayers; ++k)
	{
	  for( i = 5; i < numnp; ++i)
	  {
        	printf( " %4d", dum2);
        	printf( "  %14.6e", *(coord + nsd*i));
        	printf( "  %14.6e", *(coord + nsd*i + 1) + width*((double)(k+1)));
        	printf( "  %14.6e", *(coord + nsd*i + 2));
        	printf( "\n");
		++dum2;
	   }
	}

	printf( "prescribed displacement x: node  disp value\n");
	printf( "%4d    %14.6e\n", numnp2 - 5, 0.00);
	printf( "%4d    %14.6e\n", numnp2 - 4, 0.00);
	printf( "%4d\n ",-10);
	printf( "prescribed displacement y: node  disp value\n");
	printf( "%4d    %14.6e\n", numnp2 - 5, 0.00);
	printf( "%4d    %14.6e\n", numnp2 - 4, 0.00);
	printf( "%4d    %14.6e\n", numnp2 - 3, 0.00);
	printf( "%4d    %14.6e\n", numnp2 - 2, 0.00);
	printf( "%4d    %14.6e\n", numnp2 - 1, 0.00);
	printf( "%4d\n ",-10);
	printf( "prescribed displacement z: node  disp value\n");
	printf( "%4d    %14.6e\n", numnp2 - 5, 0.00);
	printf( "%4d    %14.6e\n", numnp2 - 2, 0.00);
	printf( "%4d\n ",-10);
	printf( "node with point load and load vector in x,y,z\n");
	printf( "%4d    %14.6e   %14.6e   %14.6e\n", 0, 0.00, -80000.0000, 0.00);
	printf( "%4d    %14.6e   %14.6e   %14.6e\n", 1, 0.00, -80000.0000, 0.00);
	printf( "%4d    %14.6e   %14.6e   %14.6e\n", 2, 0.00, -80000.0000, 0.00);
	printf( "%4d    %14.6e   %14.6e   %14.6e\n", 3, 0.00, -80000.0000, 0.00);
	printf( "%4d    %14.6e   %14.6e   %14.6e\n", 4, 0.00, -80000.0000, 0.00);
	printf( "%4d\n ",-10);
	printf( "node no. with stress and stress vector xx,yy,xy,zx,yz\n");
	printf( "%4d ",-10);

	return 1;
}	

