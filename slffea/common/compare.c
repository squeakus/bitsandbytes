/*
    This program compares 2 slffea 1.3 output files to see if they match.
    It does this by comparing the displacement data.
  
             Updated 5/9/05

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define  SMALL         1.00000e-20

main(int argc, char** argv)
{
	FILE *o1, *o2;
	int i, j, j2, n, dum, dum2, el_type, numel, numnp, nmat,
		numel2, numnp2, nmat2;
	char name[30], name2[30];
	char buf[ BUFSIZ ];
	double fdum, fdum2, fdum3;


	printf("What is the name of the first file?\n");
	scanf( "%30s",name);
	printf("What is the name of the second file?\n");
	scanf( "%30s",name2);

#if 0
	printf("What type of element is this for?\n");
	printf("1) brick \n");
	printf("2) shell \n");
	scanf( "%d",&el_type);
#endif


/*   o1 contains the data of the first file.  */

	o1 = fopen( name,"r" );

	if(o1 == NULL ) {
		printf("error on open\n");
		exit(1);
	}
	fgets( buf, BUFSIZ, o1 );
	fscanf( o1, "%d %d %d",&numel,&numnp,&nmat);
	printf( "%d %d %d\n ",numel,numnp,nmat);
	fgets( buf, BUFSIZ, o1 );

/*   o2 contains all the data of the second file.  */

	o2 = fopen( name2,"r" );

	if(o2 == NULL ) {
		printf("error on open\n");
		exit(1);
	}
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d %d %d",&numel2,&numnp2,&nmat2);
	printf( "%d %d %d\n",numel2,numnp2,nmat2);
	fgets( buf, BUFSIZ, o2 );

	n = 1 + nmat + 1 + numel + 1 + numnp + 3 + numel;
	printf( "%d\n",n);

	for( i = 0; i < n; ++i )
	{
		fgets( buf, BUFSIZ, o1 );
		fgets( buf, BUFSIZ, o2 );
	}

	for( i = 0; i < numnp; ++i )
	{
		fscanf( o1, "%d %lf\n",&dum, &fdum);
		fscanf( o2, "%d %lf\n",&dum2, &fdum2);
		fdum3 = fdum2/(SMALL + fdum);
		printf( "%4d   %16.8e %16.8e %16.8e\n",dum, fdum, fdum2, fdum3);
	}
	printf( "\n");

	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o2 );
	fgets( buf, BUFSIZ, o2 );

	for( i = 0; i < numnp; ++i )
	{
		fscanf( o1, "%d %lf\n",&dum, &fdum);
		fscanf( o2, "%d %lf\n",&dum2, &fdum2);
		fdum3 = fdum2/(SMALL + fdum);
		printf( "%4d   %16.8e %16.8e %16.8e\n",dum, fdum, fdum2, fdum3);
	}
	printf( "\n");

	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o2 );
	fgets( buf, BUFSIZ, o2 );

	for( i = 0; i < numnp; ++i )
	{
		fscanf( o1, "%d %lf\n",&dum, &fdum);
		fscanf( o2, "%d %lf\n",&dum2, &fdum2);
		fdum3 = fdum2/(SMALL + fdum);
		printf( "%4d   %16.8e %16.8e %16.8e\n",dum, fdum, fdum2, fdum3);
	}
	printf( "\n");

	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o2 );
	fgets( buf, BUFSIZ, o2 );

	for( i = 0; i < numnp; ++i )
	{
		fscanf( o1, "%d %lf\n",&dum, &fdum);
		fscanf( o2, "%d %lf\n",&dum2, &fdum2);
		fdum3 = fdum2/(SMALL + fdum);
		printf( "%4d   %16.8e %16.8e %16.8e\n",dum, fdum, fdum2, fdum3);
	}
	printf( "\n");

	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o2 );
	fgets( buf, BUFSIZ, o2 );

	for( i = 0; i < numnp; ++i )
	{
		fscanf( o1, "%d %lf\n",&dum, &fdum);
		fscanf( o2, "%d %lf\n",&dum2, &fdum2);
		fdum3 = fdum2/(SMALL + fdum);
		printf( "%4d   %16.8e %16.8e %16.8e\n",dum, fdum, fdum2, fdum3);
	}
	printf( "\n");

	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o2 );
	fgets( buf, BUFSIZ, o2 );

	for( i = 0; i < numnp; ++i )
	{
		fscanf( o1, "%d %lf\n",&dum, &fdum);
		fscanf( o2, "%d %lf\n",&dum2, &fdum2);
		fdum3 = fdum2/(SMALL + fdum);
		printf( "%4d   %16.8e %16.8e %16.8e\n",dum, fdum, fdum2, fdum3);
	}
	printf( "\n");

	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o2 );
	fgets( buf, BUFSIZ, o2 );

	return 1;
}

