/*
    This library function reads in additional strain and
    curvature data for the graphics program for beam elements.

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
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"

extern int dof, nmat, numel, numnp;
extern int stress_xyzx_flag;

int bmreader_gr( FILE *o1, CURVATURE *curve, STRAIN *strain)
{
	int i, j, dum, dum2, file_loc;
	double fdum1, fdum2, fdum3, fdum4, fdum5, fdum6;
	char one_char;
	char buf[ BUFSIZ ];
	char text;

	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );

/* Store this location in the input file. We start reading again from this point
   after determing whether this is a new or old beam file.
*/
	file_loc = ftell(o1);

	fscanf( o1,"%d",&dum);
	if( dum > -1 )
	{

/* Check if this is a new beam file where there is strain xy and strain zx data
and read the data if it exists.  If not, stress_xyzx_flag = 0 and we are reading
an old beam data file.
*/

	    fscanf( o1,"%d",&dum2);

	    fscanf( o1,"%lf %lf %lf %lf", &fdum1, &fdum2, &fdum3, &fdum4);
	    while(( one_char = (unsigned char) fgetc(o1)) != '\n')
	    {
		stress_xyzx_flag = 0;
		fdum5 = 0.0;
		fdum6 = 0.0;
		if(one_char != ' ' )
		{
		    ungetc( one_char, o1);
		    fscanf( o1,"%lf %lf", &fdum5, &fdum6);
		    stress_xyzx_flag = 1;
		    break;
		}
	    }
	}

	fseek(o1, file_loc, 0);

	printf("strain for ele: ");
	fscanf( o1,"%d",&dum);
	printf( "(%6d)",dum);
	if(stress_xyzx_flag)
	{
	   while( dum > -1 )
	   {
		fscanf( o1,"%d",&dum2);
		printf( " node (%1d)",dum2);
		fscanf( o1,"%lf ",&strain[dum].pt[dum2].xx);
		fscanf( o1,"%lf ",&strain[dum].pt[dum2].xy);
		fscanf( o1,"%lf ",&strain[dum].pt[dum2].zx);
		fscanf( o1,"%lf ",&curve[dum].pt[dum2].xx);
		fscanf( o1,"%lf ",&curve[dum].pt[dum2].yy);
		fscanf( o1,"%lf ",&curve[dum].pt[dum2].zz);
		printf(" %12.5e",strain[dum].pt[dum2].xx);
		printf(" %12.5e",strain[dum].pt[dum2].xy);
		printf(" %12.5e",strain[dum].pt[dum2].zx);
		printf(" %12.5e",curve[dum].pt[dum2].xx);
		printf(" %12.5e",curve[dum].pt[dum2].yy);
		printf(" %12.5e",curve[dum].pt[dum2].zz);
		fscanf( o1,"\n");
		printf( "\n");
		printf("strain for ele: ");
		fscanf( o1,"%d",&dum);
		printf( "(%6d)",dum);
	   }
	}
	else
	{
	   while( dum > -1 )
	   {
		fscanf( o1,"%d",&dum2);
		printf( " node (%1d)",dum2);
		fscanf( o1,"%lf ",&strain[dum].pt[dum2].xx);
		fscanf( o1,"%lf ",&curve[dum].pt[dum2].xx);
		fscanf( o1,"%lf ",&curve[dum].pt[dum2].yy);
		fscanf( o1,"%lf ",&curve[dum].pt[dum2].zz);
		printf(" %12.5e",strain[dum].pt[dum2].xx);
		printf(" %12.5e",strain[dum].pt[dum2].xy);
		printf(" %12.5e",strain[dum].pt[dum2].zx);
		printf(" %12.5e",curve[dum].pt[dum2].xx);
		printf(" %12.5e",curve[dum].pt[dum2].yy);
		printf(" %12.5e",curve[dum].pt[dum2].zz);
		fscanf( o1,"\n");
		printf( "\n");
		printf("strain for ele: ");
		fscanf( o1,"%d",&dum);
		printf( "(%6d)",dum);
	   }
	}
	printf( "\n\n");

	return 1;
}

