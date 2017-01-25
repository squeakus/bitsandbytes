/*
    This utility function calculates the Volume of each
    shell element using shape functions and gaussian
    integration.

                Updated 10/5/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License. 
*/

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "shconst.h"
#include "shstruct.h"

extern int numel, numnp, sof, flag_quad_element;
extern SH shg, shl, shl_tri;
extern ROTATE rotate;
extern double w[num_int8], w_tri[num_int6];

int shshg( double *, int , SH , SH , XL , double *, double *, double *,
	double *, ROTATE );

int shVolume( int *connect, double *coord, double *Vol)
{
        int i, i1, i2, i3, i4, i5, j, k;
	int check, node, npell_, num_intb_;
	int matl_num;
	double fdum1, fdum2, fdum3, fdum4;
        double coord_el_trans[npel8*nsd], zm1[npell4], zp1[npell4],
		znode[npell4*num_ints], dzdt_node[npell4];
        double vec_dum[nsd];
        double det[num_int8+num_ints];
	XL xl;

	num_intb_ = num_intb4;
	npell_ = npell4;
	if(!flag_quad_element)
	{
		npell_ = npell3;
		num_intb_ = num_intb3;
	}

        for( k = 0; k < numel; ++k )
        {

/* Create the coord transpose vector and other variables for one element */

                for( j = 0; j < npell_; ++j )
                {
			node = *(connect+npell_*k+j);

/* Create the coord -/+*/

                        *(coord_el_trans+j) = *(coord+nsd*node);
                        *(coord_el_trans+npel8*1+j) = *(coord+nsd*node+1);
                        *(coord_el_trans+npel8*2+j) = *(coord+nsd*node+2);

                        *(coord_el_trans+npell_+j) = *(coord+nsd*(node+numnp));
                        *(coord_el_trans+npel8*1+npell_+j) = *(coord+nsd*(node+numnp)+1);
                        *(coord_el_trans+npel8*2+npell_+j) = *(coord+nsd*(node+numnp)+2);

/* Create the coord_bar and coord_hat vector for one element */

                        xl.bar[j]=.5*( *(coord_el_trans+j)*(1.0-zeta)+
				*(coord_el_trans+npell_+j)*(1.0+zeta));
                        xl.bar[npell_*1+j]=.5*( *(coord_el_trans+npel8*1+j)*(1.0-zeta)+
				*(coord_el_trans+npel8*1+npell_+j)*(1.0+zeta));
                        xl.bar[npell_*2+j]=.5*( *(coord_el_trans+npel8*2+j)*(1.0-zeta)+
				*(coord_el_trans+npel8*2+npell_+j)*(1.0+zeta));

                        xl.hat[j]=*(coord_el_trans+npell_+j)-*(coord_el_trans+j);
                        xl.hat[npell_*1+j]=*(coord_el_trans+npel8*1+npell_+j)-
				*(coord_el_trans+npel8*1+j);
                        xl.hat[npell_*2+j]=*(coord_el_trans+npel8*2+npell_+j)-
				*(coord_el_trans+npel8*2+j);

			fdum1=fabs(xl.hat[j]);
			fdum2=fabs(xl.hat[npell_*1+j]);
			fdum3=fabs(xl.hat[npell_*2+j]);
			fdum4=sqrt(fdum1*fdum1+fdum2*fdum2+fdum3*fdum3);
                        xl.hat[j] /= fdum4;
                        xl.hat[npell_*1+j] /= fdum4;
                        xl.hat[npell_*2+j] /= fdum4;
			*(zp1+j)=.5*(1.0-zeta)*fdum4;
			*(zm1+j)=-.5*(1.0+zeta)*fdum4;
/*
   calculating rotation matrix for the fiber q[i,j] matrix 
*/
        	       	memset(vec_dum,0,nsd*sof);
			i2=1;
			if( fdum1 > fdum3)
			{
				fdum3=fdum1;
				i2=2;
			}
			if( fdum2 > fdum3) i2=3;
			*(vec_dum+(i2-1))=1.0;
                	rotate.f_shear[nsdsq*j+2*nsd]=xl.hat[j];
                	rotate.f_shear[nsdsq*j+2*nsd+1]=xl.hat[npell_*1+j];
                	rotate.f_shear[nsdsq*j+2*nsd+2]=xl.hat[npell_*2+j];
                	check = normcrossX((rotate.f_shear+nsdsq*j+2*nsd),
			    vec_dum,(rotate.f_shear+nsdsq*j+1*nsd));
                	if(!check) printf( "Problems with normcrossX \n");
                	check = normcrossX((rotate.f_shear+nsdsq*j+1*nsd),
			    (rotate.f_shear+nsdsq*j+2*nsd),(rotate.f_shear+nsdsq*j));
                	if(!check) printf( "Problems with normcrossX \n");
                }

/* Assembly of the shg matrix for each integration point */

		if(!flag_quad_element)
		{
/* Triangle elements */
		    check=shshg( det, k, shl_tri, shg, xl, zp1, zm1, znode,
			dzdt_node, rotate);
		    if(!check) printf( "Problems with shshg \n");

/* Calculate the Volume from determinant of the Jacobian */

		    for( i4 = 0; i4 < num_ints; ++i4 )
		    {
		       for( j = 0; j < num_intb_; ++j )
		       {
			    *(Vol + k) +=
				0.5*(*(w_tri+num_intb_*i4+j))*(*(det+num_intb_*i4+j));

		       }
		    }
		}
		else
		{
		    check=shshg( det, k, shl, shg, xl, zp1, zm1, znode,
			dzdt_node, rotate);
		    if(!check) printf( "Problems with shshg \n");

/* Calculate the Volume from determinant of the Jacobian */

		    for( i4 = 0; i4 < num_ints; ++i4 )
		    {
		       for( j = 0; j < num_intb_; ++j )
		       {
			    *(Vol + k) += *(w+num_intb_*i4+j)*(*(det+num_intb_*i4+j));

		       }
		    }
		}

	}

	return 1;
}

