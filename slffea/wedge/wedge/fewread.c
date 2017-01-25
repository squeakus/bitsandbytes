/* This progam reads in the data from the sample
   data used in the program Femwater.f and converts
   it to the SLFFEA data format.

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001  San Le

    Updated 3/21/01

 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int readwrite( FILE *o1, FILE *o2, char *name)
{
	int numel, numnp, nmat;
	int dum, dum2, dum3, dum4, dum5, dum6, dum7, dum8, i, j, k;
	double fdum, fdum2, fdum3, fdum4;
	char char_dum[20], char_dum2[5], buf[ BUFSIZ ];

	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );
	fgets( buf, BUFSIZ, o1 );

	numel = 1318;
        numnp = 860;
        nmat = 3;

	fprintf( o2, "   numel numnp nmat nmode This is for a 6 node wedge\n");
        fprintf( o2, "   %4d %4d %4d %4d\n",numel,numnp,nmat,0);
	fprintf( o2, "matl no., E modulus, Poisson Ratio, and density \n");
	fprintf( o2, " %4d %9.4f %9.4f %9.4f\n", 0, 4.32e8, 0.0, 2.77e3);
	fprintf( o2, " %4d %9.4f %9.4f %9.4f\n", 1, 4.32e8, 0.0, 2.77e3);
	fprintf( o2, " %4d %9.4f %9.4f %9.4f\n", 2, 4.32e8, 0.0, 2.77e3);
	fprintf( o2, "el no., connectivity, matl no. \n");

	for( i = 0; i < 1318; ++i)
	{
	   fscanf( o1,"%4s  %d %d %d %d %d %d %d %d\n", char_dum, &dum,
                &dum2, &dum3, &dum4, &dum5, &dum6, &dum7, &dum8);
	   fprintf( o2,"%4d %4d %4d %4d %4d %4d %4d %4d\n", dum-1,
                dum2-1, dum3-1, dum4-1, dum5-1, dum6-1, dum7-1, dum8-1);
	}
	fprintf( o2, "node no., coordinates \n");
	for( i = 0; i < 860; ++i)
	{
	   fscanf( o1,"%4s  %d %lf %lf %lf\n", char_dum, &dum, 
                &fdum, &fdum2, &fdum3);
	   fprintf( o2," %4d   %14.8e %14.8e %14.8e \n", dum-1,
                fdum, fdum2, fdum3);
	}

        fprintf( o2, "prescribed displacement x: node  disp value\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "prescribed displacement y: node  disp value\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "prescribed displacement z: node  disp value\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "node with point load and load vector in x,y,z\n");
        fprintf( o2, "%4d\n ",-10);
        fprintf( o2, "node no. with stress and stress vector in lamina xx,yy,xy,zx,yz\n");
        fprintf( o2, "%4d ",-10);

	return 1;
}

int main()
{
	FILE *o1, *o2;
	int dum, check;
	char name[20];
	char *ccheck;

	memset(name,0,20*sizeof(char));

        printf("What is the name of the file containing the \n");
        printf("Femwater data? \n");
        scanf( "%30s",name);

	o1 = fopen( name,"r" );
	o2 = fopen( "fewsamp","w");

        if(o1 == NULL ) {
                printf("Can't find file %30s\n",name);
                exit(1);
        }

	check = readwrite( o1, o2, name);
	if(!check) printf( " Problems with readwrite \n");

	return 1;
}	

