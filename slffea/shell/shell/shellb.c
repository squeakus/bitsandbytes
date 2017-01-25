/*
     This library function assembles the B matrix in [Btrans][D][B]
     for shell elements.

     It is based on the subroutine QDCB from the book
     "The Finite Element Method" by Thomas Hughes, page 780.

     SLFFEA source file
     Version:  1.5
     Copyright (C) 1999-2008  San Le

     The source code contained in this file is released under the
     terms of the GNU Library General Public License.
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "shconst.h"

extern int flag_quad_element;

int dotX(double *, double *, double *, int );

int shellBNpt(double *shg, double *shg_z, double *znode, double *B, 
	double *rotate_l, double *rotate_f)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX FOR 2X2 INTEGRATION IN LAMINA

		Updated 10/5/08
*/
	int i,i2,i2m1,i2m2,i2m3,i2m4,check;
	double vec_dum[nsd],w1[ndof-3],w2[ndof-3],w3[ndof-3];
	double dNdxl1,dNdxl2,dNdxl3,dNzdxl1,dNzdxl2,dNzdxl3;
	int npell_, neqel_;

	npell_ = npell4;
	neqel_ = neqel20;
	if(!flag_quad_element)
	{
	    npell_ = npell3;
	    neqel_ = neqel15;
	}

	for( i = 0; i < npell_; ++i )
	{
		i2      =ndof*i+4;
		i2m1    =i2-1;
		i2m2    =i2-2;
		i2m3    =i2-3;
		i2m4    =i2-4;

/* calculate derivatives in local x,y,z */

		*(vec_dum)   = *(shg+i);
		*(vec_dum+1) = *(shg+npell_*1+i);
		*(vec_dum+2) = *(shg+npell_*2+i);
		check = dotX(&dNdxl1,rotate_l,vec_dum,nsd);
		check = dotX(&dNdxl2,(rotate_l+nsd*1),vec_dum,nsd);
		check = dotX(&dNdxl3,(rotate_l+nsd*2),vec_dum,nsd);
		*(vec_dum)   = *(znode+i)*(*(shg+i))+
			*(shg_z+i)*(*(shg+npell_*3+i));
		*(vec_dum+1) = *(znode+i)*(*(shg+npell_*1+i))+
			*(shg_z+npell_*1+i)*(*(shg+npell_*3+i));
		*(vec_dum+2) = *(znode+i)*(*(shg+npell_*2+i))+
			*(shg_z+npell_*2+i)*(*(shg+npell_*3+i));
		check = dotX(&dNzdxl1,(rotate_l),vec_dum,nsd);
		check = dotX(&dNzdxl2,(rotate_l+nsd*1),vec_dum,nsd);
		check = dotX(&dNzdxl3,(rotate_l+nsd*2),vec_dum,nsd);

/* calculate w1,w2,w3(alpha) */

		check = dotX((w1),(rotate_l),(rotate_f+nsdsq*i+nsd*1),nsd);
		*(w1)*= -1;
		check = dotX((w1+1),(rotate_l),(rotate_f+nsdsq*i),nsd);
		check = dotX((w2),(rotate_l+nsd*1),(rotate_f+nsdsq*i+nsd*1),nsd);
		*(w2)*= -1;
		check = dotX((w2+1),(rotate_l+nsd*1),(rotate_f+nsdsq*i),nsd);
		check = dotX((w3),(rotate_l+nsd*2),(rotate_f+nsdsq*i+nsd*1),nsd);
		*(w3)*= -1;
		check = dotX((w3+1),(rotate_l+nsd*2),(rotate_f+nsdsq*i),nsd);

		*(B+i2m4) = *(rotate_l)*dNdxl1;
		*(B+i2m3) = *(rotate_l+1)*dNdxl1;
		*(B+i2m2) = *(rotate_l+2)*dNdxl1;
		*(B+i2m1) = *(w1)*dNzdxl1;
		*(B+i2)   = *(w1+1)*dNzdxl1;

		*(B+neqel_*1+i2m4) = *(rotate_l+nsd*1)*dNdxl2;
		*(B+neqel_*1+i2m3) = *(rotate_l+nsd*1+1)*dNdxl2;
		*(B+neqel_*1+i2m2) = *(rotate_l+nsd*1+2)*dNdxl2;
		*(B+neqel_*1+i2m1) = *(w2)*dNzdxl2;
		*(B+neqel_*1+i2)   = *(w2+1)*dNzdxl2; 

		*(B+neqel_*2+i2m4) = *(rotate_l)*dNdxl2+
			*(rotate_l+nsd*1)*dNdxl1;
		*(B+neqel_*2+i2m3) = *(rotate_l+1)*dNdxl2+
			*(rotate_l+nsd*1+1)*dNdxl1;
		*(B+neqel_*2+i2m2) = *(rotate_l+2)*dNdxl2+
			*(rotate_l+nsd*1+2)*dNdxl1;
		*(B+neqel_*2+i2m1) = *(w1)*dNzdxl2+*(w2)*dNzdxl1;
		*(B+neqel_*2+i2)   = *(w1+1)*dNzdxl2+*(w2+1)*dNzdxl1; 

		*(B+neqel_*3+i2m4) = *(rotate_l+nsd*2)*dNdxl1+
			*(rotate_l)*dNdxl3;
		*(B+neqel_*3+i2m3) = *(rotate_l+nsd*2+1)*dNdxl1+
			*(rotate_l+1)*dNdxl3;
		*(B+neqel_*3+i2m2) = *(rotate_l+nsd*2+2)*dNdxl1+
			*(rotate_l+2)*dNdxl3;
		*(B+neqel_*3+i2m1) = *(w3)*dNzdxl1+*(w1)*dNzdxl3;
		*(B+neqel_*3+i2)   = *(w3+1)*dNzdxl1+*(w1+1)*dNzdxl3; 

		*(B+neqel_*4+i2m4) = *(rotate_l+nsd*1)*dNdxl3+
			*(rotate_l+nsd*2)*dNdxl2;
		*(B+neqel_*4+i2m3) = *(rotate_l+nsd*1+1)*dNdxl3+
			*(rotate_l+nsd*2+1)*dNdxl2;
		*(B+neqel_*4+i2m2) = *(rotate_l+nsd*1+2)*dNdxl3+
			*(rotate_l+nsd*2+2)*dNdxl2;
		*(B+neqel_*4+i2m1) = *(w2)*dNzdxl3+*(w3)*dNzdxl2;
		*(B+neqel_*4+i2)   = *(w2+1)*dNzdxl3+*(w3+1)*dNzdxl2;
	}
	return 1;
}

int shellB1ptM(double *shg, double *B, double *rotate_l, double *rotate_f)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX FOR 1X1 POINT INTEGRATION OVER THE
       LAMINA FOR THE MEMBRANE SHEAR TERM

		Updated 10/5/08
*/
	int i,i2,i2m1,i2m2,i2m3,i2m4,check;
	double vec_dum[nsd],dNdxl1,dNdxl2,dNdxl3;
	int npell_, neqel_;

	npell_ = npell4;
	neqel_ = neqel20;
	if(!flag_quad_element)
	{
	    npell_ = npell3;
	    neqel_ = neqel15;
	}

	for( i = 0; i < npell_; ++i )
	{
		i2      =ndof*i+4;
		i2m1    =i2-1;
		i2m2    =i2-2;
		i2m3    =i2-3;
		i2m4    =i2-4;

/* calculate derivatives in local x,y,z */

		*(vec_dum)   = *(shg+i);
		*(vec_dum+1) = *(shg+npell_*1+i);
		*(vec_dum+2) = *(shg+npell_*2+i);
		check = dotX(&dNdxl1,(rotate_l),vec_dum,nsd);
		check = dotX(&dNdxl2,(rotate_l+nsd*1),vec_dum,nsd);
		check = dotX(&dNdxl3,(rotate_l+nsd*2),vec_dum,nsd);

		*(B+i2m4) = *(rotate_l)*dNdxl2+
			*(rotate_l+nsd*1)*dNdxl1;
		*(B+i2m3) = *(rotate_l+1)*dNdxl2+
			*(rotate_l+nsd*1+1)*dNdxl1;
		*(B+i2m2) = *(rotate_l+2)*dNdxl2+
			*(rotate_l+nsd*1+2)*dNdxl1;
	}
	return 1;
}

int shellB1ptT(double *shg, double *shg_z, double *znode, double *B, double *rotate_l, 
	double *rotate_f)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX FOR 1X1 POINT INTEGRATION OVER THE
       LAMINA FOR THE MEMBRANE SHEAR TERM

		Updated 10/5/08
*/
	int i,i2,i2m1,i2m2,i2m3,i2m4,check;
	double vec_dum[nsd],w1[ndof-3],w2[ndof-3],w3[ndof-3];
	double dNdxl1,dNdxl2,dNdxl3,dNzdxl1,dNzdxl2,dNzdxl3;
	int npell_, neqel_;

	npell_ = npell4;
	neqel_ = neqel20;
	if(!flag_quad_element)
	{
	    npell_ = npell3;
	    neqel_ = neqel15;
	}

	for( i = 0; i < npell_; ++i )
	{
		i2      =ndof*i+4;
		i2m1    =i2-1;
		i2m2    =i2-2;
		i2m3    =i2-3;
		i2m4    =i2-4;

/* calculate derivatives in local x,y,z */

		*(vec_dum)   = *(shg+i);
		*(vec_dum+1) = *(shg+npell_*1+i);
		*(vec_dum+2) = *(shg+npell_*2+i);
		check = dotX(&dNdxl1,rotate_l,vec_dum,nsd);
		check = dotX(&dNdxl2,(rotate_l+nsd*1),vec_dum,nsd);
		check = dotX(&dNdxl3,(rotate_l+nsd*2),vec_dum,nsd);
		*(vec_dum)   = *(znode+i)*(*(shg+i))+
			*(shg_z+i)*(*(shg+npell_*3+i));
		*(vec_dum+1) = *(znode+i)*(*(shg+npell_*1+i))+
			*(shg_z+npell_*1+i)*(*(shg+npell_*3+i));
		*(vec_dum+2) = *(znode+i)*(*(shg+npell_*2+i))+
			*(shg_z+npell_*2+i)*(*(shg+npell_*3+i));
		check = dotX(&dNzdxl1,rotate_l,vec_dum,nsd);
		check = dotX(&dNzdxl2,(rotate_l+nsd*1),vec_dum,nsd);
		check = dotX(&dNzdxl3,(rotate_l+nsd*2),vec_dum,nsd);

/* calculate w1,w2,w3(alpha) */

		check = dotX((w1),(rotate_l),(rotate_f+nsdsq*i+nsd*1),nsd);
		*(w1)*= -1;
		check = dotX((w1+1),(rotate_l),(rotate_f+nsdsq*i),nsd);
		check = dotX((w2),(rotate_l+nsd*1),(rotate_f+nsdsq*i+nsd*1),nsd);
		*(w2)*= -1;
		check = dotX((w2+1),(rotate_l+nsd*1),(rotate_f+nsdsq*i),nsd);
		check = dotX((w3),(rotate_l+nsd*2),(rotate_f+nsdsq*i+nsd*1),nsd);
		*(w3)*= -1;
		check = dotX((w3+1),(rotate_l+nsd*2),(rotate_f+nsdsq*i),nsd);

		*(B+neqel_*0+i2m4) = *(rotate_l+nsd*2)*dNdxl1+
			*(rotate_l)*dNdxl3;
		*(B+neqel_*0+i2m3) = *(rotate_l+nsd*2+1)*dNdxl1+
			*(rotate_l+1)*dNdxl3;
		*(B+neqel_*0+i2m2) = *(rotate_l+nsd*2+2)*dNdxl1+
			*(rotate_l+2)*dNdxl3;
		*(B+neqel_*0+i2m1) = *(w3)*dNzdxl1+*(w1)*dNzdxl3;
		*(B+neqel_*0+i2)   = *(w3+1)*dNzdxl1+*(w1+1)*dNzdxl3;

		*(B+neqel_*1+i2m4) = *(rotate_l+nsd*1)*dNdxl3+
			*(rotate_l+nsd*2)*dNdxl2;
		*(B+neqel_*1+i2m3) = *(rotate_l+nsd*1+1)*dNdxl3+
			*(rotate_l+nsd*2+1)*dNdxl2;
		*(B+neqel_*1+i2m2) = *(rotate_l+nsd*1+2)*dNdxl3+
			*(rotate_l+nsd*2+2)*dNdxl2;
		*(B+neqel_*1+i2m1) = *(w2)*dNzdxl3+*(w3)*dNzdxl2;
		*(B+neqel_*1+i2)   = *(w2+1)*dNzdxl3+*(w3+1)*dNzdxl2;

	}
	return 1;
}


int shellB_mass(double *shg, double *znode, double *B, double *rotate_f)
{
/*
     This library function assembles the B_mass matrix in
     [B_mass trans][B_mass] for shell elements.

		Updated 10/5/08
*/
	int i,i2,i2m1,i2m2,i2m3,i2m4;
	double N,Nz;
	int npell_, neqel_;

	npell_ = npell4;
	neqel_ = neqel20;
	if(!flag_quad_element)
	{
	    npell_ = npell3;
	    neqel_ = neqel15;
	}

	for( i = 0; i < npell_; ++i )
	{
		i2      =ndof*i+4;
		i2m1    =i2-1;
		i2m2    =i2-2;
		i2m3    =i2-3;
		i2m4    =i2-4;

		N = *(shg+i);
		Nz = *(shg+i)*(*(znode+i));

		*(B+i2m4) = N;
		*(B+i2m3) = 0.0;
		*(B+i2m2) = 0.0;
		*(B+i2m1) = - *(rotate_f+nsdsq*i+nsd*1)*Nz;
		*(B+i2)   = *(rotate_f+nsdsq*i)*Nz;

		*(B+neqel_*1+i2m4) = 0.0;
		*(B+neqel_*1+i2m3) = N;
		*(B+neqel_*1+i2m2) = 0.0;
		*(B+neqel_*1+i2m1) = - *(rotate_f+nsdsq*i+nsd*1+1)*Nz;
		*(B+neqel_*1+i2)   = *(rotate_f+nsdsq*i+1)*Nz; 

		*(B+neqel_*2+i2m4) = 0.0;
		*(B+neqel_*2+i2m3) = 0.0;
		*(B+neqel_*2+i2m2) = N;
		*(B+neqel_*2+i2m1) = - *(rotate_f+nsdsq*i+nsd*1+2)*Nz;
		*(B+neqel_*2+i2)   = *(rotate_f+nsdsq*i+2)*Nz; 
	}
	return 1;
}

