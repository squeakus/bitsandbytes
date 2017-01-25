/*
     This library function assembles the B matrix in [Btrans][D][B]
     for plate elements.

     It is based on the subroutine QDCB from the book
     "The Finite Element Method" by Thomas Hughes, page 780.

     I have also used the notes from "AMES 232 B Course Notes" by
     David Benson, although I have switched the angles of rotation
     to make it consistant with my shell as well as the right hand
     rule.  

     SLFFEA source file
     Version:  1.5
     Copyright (C) 1999, 2000, 2001, 2002  San Le

     The source code contained in this file is released under the
     terms of the GNU Library General Public License.
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "plconst.h"

/* FOR THE BENDING */

int plateB4pt(double *shg, double *B)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX FOR 2X2 INTEGRATION

		Updated 9/5/06 
*/
	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel4; ++i )
	{
		i2      =ndof3*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

	     /* *(B+i2m2)           = 0.0; 
		*(B+i2m1)           = 0.0; */
		*(B+i2)             = -*(shg+i);
	     /* *(B+neqel12*1+i2m2) = 0.0; */
		*(B+neqel12*1+i2m1) = *(shg+npel4*1+i); 
	     /* *(B+neqel12*1+i2)   = 0.0; */
	     /* *(B+neqel12*2+i2m2) = 0.0; */
		*(B+neqel12*2+i2m1) = *(shg+i);
		*(B+neqel12*2+i2)   = -*(shg+npel4*1+i);
	}
	return 1;
}

int plateB1pt(double *shg, double *B)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX FOR 1 POINT INTEGRATION

		Updated 9/5/06 
*/
	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel4; ++i )
	{
		i2      =ndof3*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B+i2m2)           = *(shg+i);
	     /* *(B+i2m1)           = 0.0; */
		*(B+i2)             = *(shg+npel4*2+i);
		*(B+neqel12*1+i2m2) = *(shg+npel4*1+i);
		*(B+neqel12*1+i2m1) = -*(shg+npel4*2+i); 
	     /* *(B+neqel12*1+i2)   = 0.0; */
	}
	return 1;
}

int plateB4pt_node(double *shg, double *B)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX FOR 2X2 INTEGRATION.  

       THIS DIFFERS FROM "plateB4pt" IN THAT THE CALCULATION
       IS DONE FOR THE ENTIRE "B" MATRIX RATHER THAN JUST 
       THE BENDING.  THIS IS USED FOR WHEN THE STRESSES AND
       STRAINS ARE CALCULATED AT THE NODES RATHER THAN GAUSS
       POINTS.
       
		Updated 9/5/06 
*/
	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel4; ++i )
	{
		i2      =ndof3*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B+i2)             = -*(shg+i);
		*(B+neqel12*1+i2m1) = *(shg+npel4*1+i); 
		*(B+neqel12*2+i2m1) = *(shg+i);
		*(B+neqel12*2+i2)   = -*(shg+npel4*1+i);
		*(B+neqel12*3+i2m2) = *(shg+i);
		*(B+neqel12*3+i2)   = *(shg+npel4*2+i);
		*(B+neqel12*4+i2m2) = *(shg+npel4*1+i);
		*(B+neqel12*4+i2m1) = -*(shg+npel4*2+i); 
	}
	return 1;
}


int plateB_mass(double *shg, double *B_mass)
{
/*
     This library function assembles the B_mass matrix in
     [B_mass trans][B_mass] for plate elements.

     The specifications for the plate element mass matrix
     was taken from the AMES 232B Winter 1997 notes by
     David Benson, page 79.

		Updated 9/5/06
*/

	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel4; ++i )
	{
		i2      =ndof3*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B_mass+i2)             = *(shg+i);
		*(B_mass+neqel12*1+i2m1) = -*(shg+i);
		*(B_mass+neqel12*2+i2m2) = *(shg+i);
	}
	return 1;
}

/* FOR THE MEMBRANE */

int plateBtr(double *shgtr, double *B)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX FOR 1 POINT INTEGRATION

		Updated 8/30/06
*/
	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel3; ++i )
	{
		i2      =ndof3*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B+i2)            = -*(shgtr+i);
		*(B+neqel9*1+i2m1) = *(shgtr+npel3*1+i); 
		*(B+neqel9*2+i2m1) = *(shgtr+i);
		*(B+neqel9*2+i2)   = -*(shgtr+npel3*1+i);
		*(B+neqel9*3+i2m2) = *(shgtr+i);
		*(B+neqel9*3+i2)   = *(shgtr+npel3*2+i);
		*(B+neqel9*4+i2m2) = *(shgtr+npel3*1+i);
		*(B+neqel9*4+i2m1) = -*(shgtr+npel3*2+i); 
	}
	return 1;
}


int plateBtr_mass(double *shgtr, double *B_mass)
{
/*
     This library function assembles the B_mass matrix in
     [B_mass trans][B_mass] for plate elements.

     The specifications for the plate element mass matrix
     was taken from the AMES 232B Winter 1997 notes by
     David Benson, page 79.

		Updated 10/28/06
*/

	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel3; ++i )
	{
		i2      =ndof3*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B_mass+i2)            = *(shgtr+i);
		*(B_mass+neqel9*1+i2m1) = -*(shgtr+i);
		*(B_mass+neqel9*2+i2m2) = *(shgtr+i);
	}
	return 1;
}
