/*
    This utility function assembles the Conductivity matrix for a finite 
    element program which does analysis on a thermal brick.

    The equations for the heat and temperature can be seen in
    equations 6.1-1 to 6.2-7 of the ANSYS manual:

      Kohneke, Peter, *ANSYS User's Manual for Revision 5.0,
         Vol IV Theory, Swanson Analysis Systems Inc., 1992.

    pages 6-1 to 6-6.  They show how the thermal loads and
    orthortrophy are included.

		Updated 9/26/01

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../brick/brconst.h"
#include "br2struct.h"

extern int temp_analysis_flag, dof, Tdof, numel, numel_film, numnp, sof;
extern int TLU_decomp_flag, Tnumel_K, TBnumel_K, Tnumel_P;
extern double shg[sosh], shl[sosh], shl_film[sosh_film], shl_node[sosh],
	shl_node2[sosh_node2], w[num_int], *Vol0;

int brcubic( double *);

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int brshface( double *, int , double *, double *);

int dotX(double *, double *, double *, int);

int brickB_T2( double *,double *);

int brickB_T( double *,double *);

int brshg( double *, int, double *, double *, double *);

int brstress_shg( double *, int, double *, double *, double * );
	       
int brCassemble(double *A, int *connect, int *connect_film, double *coord, int *el_matl,
	int *el_matl_film, double *heat_el, double *heat_node, int *Tid, int *Tidiag,
	int *Tlm, int *TBlm, MATL *matl, double *Q, double *T, double *TB, double *TK_diag)
{
	int i, i1, i2, j, k, Tdof_el[Tneqel], TBdof_el[TBneqel],
		sdof_el[npel*nsd];
	int check, counter, node, dum;
	int matl_num, matl_num_film;
	XYZF thrml_cond;
	double film_const;
	double fdum;
	double B_T[TsoB], B_TB[TBsoB], B_heat[Tneqel], DB[TsoB];
	double K_temp[Tneqlsq], K_el[Tneqlsq];
	double heat_node_el[Tneqel], Q_el[Tneqel], Q_el_film[TBneqel],
		T_el[Tneqel], TB_el[TBneqel];
	double coord_el_trans[npel*nsd];
	double det[num_int], dArea[num_int_film], wXdet, wXdArea;

	for( k = 0; k < numel; ++k )
	{

		matl_num = *(el_matl+k);

		thrml_cond.x = matl[matl_num].thrml_cond.x;
		thrml_cond.y = matl[matl_num].thrml_cond.y;
		thrml_cond.z = matl[matl_num].thrml_cond.z;

     /*printf("thrml K x, K y, K z %f %f %f \n",
		thrml_cond.x, thrml_cond.y, thrml_cond.z );*/

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(Tdof_el+Tndof*j) = Tndof*node;
		}


/* Assembly of the shg matrix for each integration point */

		check=brshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with brshg \n");

/* The loop over j below calculates the 8 points of the gaussian integration 
   for several quantities */

		memset(K_el,0,Tneqlsq*sof);
		memset(Q_el,0,Tneqel*sof);
		memset(T_el,0,Tneqel*sof);

		for( j = 0; j < num_int; ++j )
		{
		    memset(B_T,0,TsoB*sof);
		    memset(DB,0,TsoB*sof);
		    memset(K_temp,0,Tneqlsq*sof);

/* Assembly of the B matrix */

		    check = brickB_T((shg+npel*(nsd+1)*j),B_T);
		    if(!check) printf( "Problems with brickB_T \n");

		    for( i1 = 0; i1 < Tneqel; ++i1 )
		    {
			*(DB+i1) = *(B_T+i1)*thrml_cond.x;
			*(DB+Tneqel*1+i1) = *(B_T+Tneqel*1+i1)*thrml_cond.y;
			*(DB+Tneqel*2+i1) = *(B_T+Tneqel*2+i1)*thrml_cond.z;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_temp, B_T, DB, Tneqel, Tneqel, Tdim);
		    if(!check) printf( "error \n");

/* Compute the element diffusion conductivity matrix.  Look at the [Ktb] matrix
   on page 6-6 of the ANSYS manual.
*/
		    for( i2 = 0; i2 < Tneqlsq; ++i2 )
		    {
			  *(K_el+i2) += *(K_temp+i2)*wXdet;
		    }
		}

/* To debug the program */

		for( j = 0; j < Tneqel; ++j )
		{
			*(T_el + j) = *(T + *(Tdof_el+j));
			*(heat_node_el + j) = *(heat_node + *(Tdof_el+j));
		}

		check = matX(Q_el, K_el, T_el, Tneqel, 1, Tneqel);
		if(!check) printf( "Problems with matX \n");

		if(temp_analysis_flag == 1)
		{

/* Compute the equivalant heat based on prescribed temperature */

		  for( j = 0; j < Tneqel; ++j )
		  {
			*(Q + *(Tdof_el+j)) -= *(Q_el + j);
		  }

		  memset(Q_el,0,Tneqel*sof);

		  for( j = 0; j < num_int; ++j )
		  {
			memset(B_heat,0,Tneqel*sof);

/* Assembly of the B_heat matrix */

			check = brickB_T2((shg+npel*(nsd+1)*j + npel*3),B_heat);
			if(!check) printf( "Problems with brickB_T2 \n");

/* Calculate the value of heat at the integration point */

			wXdet = *(w+j)*(*(det+j));

			check=dotX( &fdum, heat_node_el, B_heat, Tneqel);
			if(!check) printf( "Problems with dotX \n");

/* Compute the heat based on nodes and elements with heat generation.  Look at
   the third term on the right hand side of equation 6.2-7 on page 6-6
   of the ANSYS manual.  Note that the ANSYS manual doesn't break down
   the heat as I have done.
*/

			for( i2 = 0; i2 < Tneqel; ++i2 )
			{
			  *(Q_el+i2) +=
			     (fdum + *(heat_el+k))*(*(B_heat+i2))*wXdet;
			}
		  }

/* Compute the equivalant nodal heat based on prescribed temperature */

		  for( j = 0; j < Tneqel; ++j )
		  {
			*(Q + *(Tdof_el+j)) += *(Q_el + j);
		  }

/* Assembly of either the global skylined conductivity matrix or Tnumel_K of the
   element conductivity matrices if the Conjugate Gradient method is used */

		  if(TLU_decomp_flag)
		  {
			check = globalKassemble(A, Tidiag, K_el, (Tlm + k*Tneqel),
			    Tneqel);
			if(!check) printf( "Problems with globalKassemble \n");
		  }
		  else
		  {
			check = globalConjKassemble(A, Tdof_el, k, TK_diag, K_el,
				Tneqel, Tneqlsq, Tnumel_K);
			if(!check) printf( "Problems with globalConjKassemble \n");
		  }
		}
		else
		{
/* Calculate the element reaction heat */

			for( j = 0; j < Tneqel; ++j )
			{
				*(Q + *(Tdof_el+j)) += *(Q_el + j);
			}

		}
	}

/* Add convection to the equvivalant nodal heat Q and add convection conductivity
   to the global conductivity */

	if(temp_analysis_flag == 1)
	{
	    for( k = 0; k < numel_film; ++k )
	    {
		matl_num_film = *(el_matl_film+k);
		film_const = matl[matl_num_film].film;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel_film; ++j )
		{
			node = *(connect_film+npel_film*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel_film*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel_film*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(TBdof_el+Tndof*j) = Tndof*node;
		}

		memset(TB_el,0,TBneqel*sof);

		for( j = 0; j < TBneqel; ++j )
		{
			*(TB_el + j) = *(TB + *(TBdof_el+j));
		}

		check = brshface( dArea, k, shl_film, coord_el_trans);
		if(!check) printf( "Problems with brshface \n");

		memset(K_el,0,TBneqlsq*sof);
		memset(Q_el_film,0,TBneqel*sof);

		for( j = 0; j < num_int_film; ++j )
		{
		   memset(B_TB,0,TBneqel*sof);
		   memset(DB,0,TBneqel*sof);
		   memset(K_temp,0,TBneqlsq*sof);

		   check = brickB_T2((shl_film+npel_film*nsd*j+npel_film*2),B_TB);
		   if(!check) printf( "Problems with brickB_T2 \n");

		   *(DB) = *(B_TB)*film_const;
		   *(DB+1) = *(B_TB+1)*film_const;
		   *(DB+2) = *(B_TB+2)*film_const;
		   *(DB+3) = *(B_TB+3)*film_const;

		   wXdArea = *(w+j)*(*(dArea+j));

		   check=matXT(K_temp, B_TB, DB, TBneqel, TBneqel, 1);
		   if(!check) printf( "error \n");

/* Compute the element convection surface conductivity matrix.  Look at the [Ktc] matrix
   on page 6-6 of the ANSYS manual.
*/
		   for( i2 = 0; i2 < TBneqlsq; ++i2 )
		   {
		       *(K_el+i2) += *(K_temp+i2)*wXdArea;
		   }

		   check=dotX( &fdum, TB_el, B_TB, TBneqel);
		   if(!check) printf( "Problems with dotX \n");

/* Compute the heat based on convection.  Look at the second term
   on the right hand side of equation 6.2-7 on page 6-6 of the ANSYS manual.
*/

		   for( i2 = 0; i2 < TBneqel; ++i2 )
		   {
		       *(Q_el_film+i2) +=
			   fdum*film_const*(*(B_TB+i2))*wXdArea;
		   }
		}

		for( j = 0; j < TBneqel; ++j )
		{
			*(Q + *(TBdof_el+j)) += *(Q_el_film + j);
			/*printf("Q %3d %14.5f\n",*(TBdof_el+j),
				*(Q + *(TBdof_el+j)));*/
		}

/* Assembly of either the global skylined conductivity matrix or TBnumel_K of the
   element convection matrices if the Conjugate Gradient method is used */

		if(TLU_decomp_flag)
		{
			check = globalKassemble(A, Tidiag, K_el, (TBlm + k*TBneqel),
			    TBneqel);
			if(!check) printf( "Problems with globalKassemble \n");
		}
		else
		{

/* If the Conjugate Gradient method is used, add Tnumel_K*Tneqlsq to A */

			dum = Tnumel_K*Tneqlsq;

			check = globalConjKassemble((A+dum), TBdof_el, k, TK_diag, K_el,
				TBneqel, TBneqlsq, TBnumel_K);
			if(!check) printf( "Problems with globalConjKassemble \n");
		}
	    }
	}

	if(temp_analysis_flag == 1)
	{

/* Contract the global heat matrix using the id array only if LU decomposition
   is used. */

	  if(TLU_decomp_flag)
	  {
	     counter = 0;
	     for( i = 0; i < Tdof ; ++i )
	     {
		if( *(Tid + i ) > -1 )
		{
			*(Q + counter ) = *(Q + i );
			/*printf("Q %5d %16.8e\n", counter, *(Q + counter));*/
			++counter;
		}
	     }
	  }
	}

	return 1;
}

