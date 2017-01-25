/*
     SLFFEA source file
     Version:  1.5
     Copyright (C) 1999-2008  San Le

     The source code contained in this file is released under the
     terms of the GNU Library General Public License.
*/

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "shconst.h"
#include "shstruct.h"

extern int flag_quad_element;

int normal(double *);

int normcrossX(double *, double *, double *);

int dotX(double *,double *, double *, int );

int shshl_triangle( double g, SH shl_tri, double *w_tri)
{
/* 
     This subroutine calculates the local shape function derivatives for
     a shell element at the gauss points for 3/6 node elements.

     It is based on the subroutine QDCSHL from the book
     "The Finite Element Method" by Thomas Hughes, page 784.

     The positions of the integration points are also very similar to the
     shape functions for wedges.  Look on page 171-172 and "weshape.c".
     Also, look at "trshape.c" which may be more informative since it
     shows the interpolation of the triangle by itself.

     Also, look on page 13-3 to 13-6 in the "ANSYS User's Manual".  It
     discusses integration points for triangles.

     It should be noted that I have permutated the node sequences
     given by Hughes so that in terms of the triangle given in
     Figure 3.I.5 on page 167, node 3 in Hughes is node 1 in SLFFEA,
     node 2 is node 3, and node 1 goes to node 2.  This is because
     I want to be consistant with the tetrahedron.  You can read
     more about this change in teshl for the tetrahedron.

     Also, look on page 13-3 to 13-6 in the "ANSYS User's Manual".  It
     discusses integration points for triangles.

 ....  CALCULATE INTEGRATION-RULE WEIGHTS, SHAPE FUNCTIONS AND
       LOCAL DERIVATIVES FOR A EIGHT-NODE HEXAHEDRAL ELEMENT
       r, s = LOCAL ELEMENT COORDINATES along triangle faces
       xi = LOCAL ELEMENT COORDINATES along length of body
       *(shl_tri+npel*(nsd+1)*k+i) = LOCAL ("r") DERIVATIVE OF SHAPE FUNCTION
       *(shl_tri+npel*(nsd+1)*k+npel*1+i) = LOCAL ("s") DERIVATIVE OF SHAPE FUNCTION
       *(shl_tri+npel*(nsd+1)*k+npel*2+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl_tri+npel*(nsd+1)*k+npel*3+i) = LOCAL SHAPE FUNCTION
       *(w_tri+k)    = INTEGRATION-RULE WEIGHT
       i           = LOCAL NODE NUMBER
       k           = INTEGRATION POINT
       num_intb6   = NUMBER OF INTEGRATION POINTS FOR LAMINA, 4
       num_ints    = NUMBER OF INTEGRATION POINTS FOR FIBER, 2

                        Updated 10/5/08
*/
	double ra[]={ pt1667, pt6667, pt1667, pt3333};
	double sa[]={ pt1667, pt1667, pt6667, pt3333};
	double ta[]={-0.50, 0.50 };
	double temps,tempr,tempt,r,s,t,fdum;
	int i,j,k;

	if( g > 1.99 )
	{

/* Set ra, sa, ta for shl_node calculation */

		*(ra) = 0.0;        *(sa) = 0.0;
		*(ra + 1) = 1.0;    *(sa + 1) = 0.0;
		*(ra + 2) = 0.0;    *(sa + 2) = 1.0;
		*(ra + 3) = pt3333; *(sa + 3) = pt3333;
	}

	for( k = 0; k < num_intb3; ++k )
	{
/* 
   calculating the weights and local dN/ds,dr matrix for 3
   integration for shl_tri
*/
		*(w_tri+k)=pt3333;
		*(w_tri+num_intb3+k)=pt3333;

		r = *(ra+k);
		s = *(sa+k);
		fdum = (1.0 - r - s);

/* dN/dr */
		shl_tri.bend[npell3*(nsdl+1)*k]   = -1.0;                /* node 0 */
		shl_tri.bend[npell3*(nsdl+1)*k+1] =  1.0;                /* node 1 */
		shl_tri.bend[npell3*(nsdl+1)*k+2] =  0.0;                /* node 2 */

/* dN/ds */
		shl_tri.bend[npell3*(nsdl+1)*k+npell3*1]   = -1.0;       /* node 0 */
		shl_tri.bend[npell3*(nsdl+1)*k+npell3*1+1] =  0.0;       /* node 1 */
		shl_tri.bend[npell3*(nsdl+1)*k+npell3*1+2] =  1.0;       /* node 2 */

/* N */
		shl_tri.bend[npell3*(nsdl+1)*k+npell3*2]   = fdum;   /* node 0 */
		shl_tri.bend[npell3*(nsdl+1)*k+npell3*2+1] = r;      /* node 1 */
		shl_tri.bend[npell3*(nsdl+1)*k+npell3*2+2] = s;      /* node 2 */

		/*printf("\n");*/
	}
/* 
   calculating the weights and local dN/ds,dr matrix for 1X1 
   point integration for shl_tri in lamina
*/
	r = *(ra+num_intb3);
	s = *(sa+num_intb3);
	fdum = (1.0 - r - s);

/* dN/dr */
	shl_tri.shear[0] = -1.0;                /* node 0 */
	shl_tri.shear[1] =  1.0;                /* node 1 */
	shl_tri.shear[2] =  0.0;                /* node 2 */

/* dN/ds */
	shl_tri.shear[npell3*1]   = -1.0;       /* node 0 */
	shl_tri.shear[npell3*1+1] =  0.0;       /* node 1 */
	shl_tri.shear[npell3*1+2] =  1.0;       /* node 2 */

/* N */
	shl_tri.shear[npell3*2]   = fdum;   /* node 0 */
	shl_tri.shear[npell3*2+1] = r;      /* node 1 */
	shl_tri.shear[npell3*2+2] = s;      /* node 2 */


/* 
   calculating the local N and dN/dt matrix for 2X2
   integration for shl_tri in fiber 
*/

   t=.25*g;

   shl_tri.bend_z[0]= *(ta);                 shl_tri.bend_z[1]= *(ta+1);
   shl_tri.bend_z[2]=pt5+t;                  shl_tri.bend_z[3]=pt5-t;
   shl_tri.bend_z[npelf*num_ints]= *(ta);    shl_tri.bend_z[npelf*num_ints+1]= *(ta+1);
   shl_tri.bend_z[npelf*num_ints+2]=pt5-t;   shl_tri.bend_z[npelf*num_ints+3]=pt5+t;

/* 
   calculating the local N and dN/dt matrix for 1X1 point
   integration for shl_tri in fiber 
*/
   t=.25*g;
   shl_tri.shear_z[0]= *(ta);                shl_tri.shear_z[1]= *(ta+1);
   shl_tri.shear_z[npelf*(num_ints-1)]=pt5;  shl_tri.shear_z[npelf*(num_ints-1)+1]=pt5;

	return 1;
}


int shshl( double g, SH shl, double *w)
{
/* 
     This subroutine calculates the local shape function derivatives for
     a shell element at the gauss points for a 4/8 node element.

     It is based on the subroutine QDCSHL from the book
     "The Finite Element Method" by Thomas Hughes, page 784.

                        Updated 5/5/99
*/
	double ra[]={-0.50, 0.50, 0.50,-0.50, 0.00};
	double sa[]={-0.50,-0.50, 0.50, 0.50, 0.00};
	double ta[]={-0.50, 0.50 };
	double temps,tempr,tempt,r,s,t;
	int i,j,k;

	for( k = 0; k < num_intb4; ++k )
	{
/* 
   calculating the weights and local dN/ds,dr matrix for 2X2
   integration for shl
*/
		*(w+k)=1.0;
		*(w+num_intb4+k)=1.0;

		r=g*(*(ra+k));
		s=g*(*(sa+k));
		for( i = 0; i < npell4; ++i )
		{
		   tempr = pt5 + *(ra+i)*r;
		   temps = pt5 + *(sa+i)*s;
		   shl.bend[npell4*(nsdl+1)*k+i]=*(ra+i)*temps;
		   shl.bend[npell4*(nsdl+1)*k+npell4*1+i]=tempr*(*(sa+i));
		   shl.bend[npell4*(nsdl+1)*k+npell4*2+i]=tempr*temps;
		}
		/*printf("\n");*/
	}
/* 
   calculating the weights and local dN/ds,dr matrix for 1X1 
   point integration for shl in lamina
*/
	r=g*(*(ra+num_intb4));
	s=g*(*(sa+num_intb4));
	/* Actually, 4.0 but I'm adding 4 times in shKassemble */
	for( i = 0; i < npell4; ++i )
	{
	   tempr = pt5 + *(ra+i)*r;
	   temps = pt5 + *(sa+i)*s;
	   shl.shear[i]=*(ra+i)*temps;
	   shl.shear[npell4*1+i]=tempr*(*(sa+i));
	   shl.shear[npell4*2+i]=tempr*temps;
	}

/* 
   calculating the local N and dN/dt matrix for 2X2 point
   integration for shl in fiber 
*/

   t=.25*g;

   shl.bend_z[0]= *(ta);                 shl.bend_z[1]= *(ta+1);
   shl.bend_z[2]=pt5+t;                  shl.bend_z[3]=pt5-t;
   shl.bend_z[npelf*num_ints]= *(ta);    shl.bend_z[npelf*num_ints+1]= *(ta+1);
   shl.bend_z[npelf*num_ints+2]=pt5-t;   shl.bend_z[npelf*num_ints+3]=pt5+t;

/* 
   calculating the local N and dN/dt matrix for 1X1 point
   integration for shl in fiber 
*/
   t=.25*g;
   shl.shear_z[0]= *(ta);                shl.shear_z[1]= *(ta+1);
   shl.shear_z[npelf*(num_ints-1)]=pt5;  shl.shear_z[npelf*(num_ints-1)+1]=pt5;

	return 1;
}

int shshg( double *det, int el, SH shl, SH shg, XL xl, double *zp1, double *zm1, 
	double *znode, double *dzdt_node, ROTATE rotate)
{
/*
     This subroutine calculates the global shape function derivatives for
     a shell element at the gauss points.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR A FOUR-NODE QUADRALATERAL ELEMENT

       xl.bar[j+npell_*i] = GLOBAL COORDINATES ON LAMINA 
       xl.hat[j+npell_*i] = GLOBAL COORDINATES ON FIBER
       *(det+num_intb_*k2+k)  = JACOBIAN DETERMINANT
       *(det+num_int_+k2)  = JACOBIAN DETERMINANT FOR 1X1 GAUSS

       FOR 2X2 or 3 PT. GAUSS
       shl.bend[npell_*(nsdl+1)*k+i] = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       shl.bend[npell_*(nsdl+1)*k+npell_*1+i] = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       shl.bend[npell_*(nsdl+1)*k+npell_*2+i] = LOCAL SHAPE FUNCTION(in XI and ETA)
       shg.bend[npell_*(nsd+1)*k+i] = X-DERIVATIVE OF SHAPE FUNCTION
       shg.bend[npell_*(nsd+1)*k+npell_*1+i] = Y-DERIVATIVE OF SHAPE FUNCTION
       shg.bend[npell_*(nsd+1)*k+npell_*2+i] = Z-DERIVATIVE OF SHAPE FUNCTION
       shg.bend[npell_*(nsd+1)*k+npell_*3+i] = shl(npell_*(nsd)*k+npell_*2+i)
       shg.bend_z[npell_*(nsd)*k+i] = X-DERIVATIVE OF SHAPE FUNCTION
       shg.bend_z[npell_*(nsd)*k+npell_*1+i] = Y-DERIVATIVE OF SHAPE FUNCTION
       shg.bend_z[npell_*(nsd)*k+npell_*2+i] = Z-DERIVATIVE OF SHAPE FUNCTION

       FOR 1X1 PT. GAUSS IN LAMINA
       shl.shear[i] = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION 
       shl.shear[npell_*1+i] = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       shl.shear[npell_*2+i] = LOCAL SHAPE FUNCTION
       shg.shear[i] = X-DERIVATIVE OF SHAPE FUNCTION
       shg.shear[npell_*1+i] = Y-DERIVATIVE OF SHAPE FUNCTION
       shg.shear[npell_*2+i] = Z-DERIVATIVE OF SHAPE FUNCTION
       shg.shear[npell_*3+i] = shl(npell_*(nsd)*num_intb_+npell_*2+i)
       shg.shear_z[i] = X-DERIVATIVE OF SHAPE FUNCTION
       shg.shear_z[npell_*1+i] = Y-DERIVATIVE OF SHAPE FUNCTION
       shg.shear_z[npell_*2+i] = Z-DERIVATIVE OF SHAPE FUNCTION

       FOR 2X2 PT. GAUSS
       shl.bend_z[i] = LOCAL ("ZETA") DERIVATIVE OF SHAPE FUNCTION
       shl.bend_z[npelf*num_ints+npelf*k2+i] = LOCAL SHAPE FUNCTION(in ZETA)
       FOR 1X1 AND 2X2 PT. GAUSS  IN LAMINA
       shl.shear_z[i] = LOCAL ("ZETA") DERIVATIVE OF SHAPE FUNCTION
       shl.shear_z[npelf+i] = LOCAL SHAPE FUNCTION(in ZETA) 

       *(xs+2*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER FOR LAMINA 
          k2   = INTEGRATION-POINT NUMBER FOR FIBER
       num_intb_   = NUMBER OF INTEGRATION POINTS FOR LAMINA, 4
       num_ints    = NUMBER OF INTEGRATION POINTS FOR FIBER, 2

                        Updated 10/3/08
*/
	double xs[soxshat],temp[nsdsq],col1[nsdl],col2[nsdl],temp1,temp2;
	double shlshl2_vec[npell4],xfib2[2];
	double vecl_xi[nsd],vecl_eta[nsd],vecl_alp[nsd],vecl_beta[nsd],
		vecl_1[nsd],vecl_2[nsd],vecl_3[nsd];
	XLXS xlxs;
	int check,i,j,k,k2,npell_,num_int_,num_intb_;

	npell_ = npell4;
	num_intb_ = num_intb4;
	num_int_ = num_int8;
	if(!flag_quad_element)
	{
	    npell_ = npell3;
	    num_intb_ = num_intb3;
	    num_int_ = num_int6;
	}

/* 
   calculating the dN/dx,dy,dz dza/dx,dy,dz matrices for 2X2 or 3 point
   integration
*/
	for( k2 = 0; k2 < num_ints; ++k2 )
	{
	   for( i = 0; i < npell_; ++i )
	   {
/* 
   calculating the values of the za and dzdt matrix for 2X2 point
   integration
*/
	       *(znode+npell_*k2+i)=
		    shl.bend_z[npelf*(nsdf+1)*k2+npelf*1+1]*(*(zp1+i))+
		    shl.bend_z[npelf*(nsdf+1)*k2+npelf*1]*(*(zm1+i));
	       *(dzdt_node+i)=
		    shl.bend_z[npelf*(nsdf+1)*k2+1]*(*(zp1+i))+
		    shl.bend_z[npelf*(nsdf+1)*k2]*(*(zm1+i));
	   }
	   for( k = 0; k < num_intb_; ++k )
	   {
/* 
   calculating dx,dy,dz/ds,dr matrix for 2X2 or 3 point
   integration
*/
		for( j = 0; j < nsdl; ++j ) /* j loops over s,r */
		{
			for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
			{
			    check=dotX((xlxs.bar+nsdl*i+j),(shl.bend+npell_*(nsdl+1)*k+npell_*j),
				(xl.bar+npell_*i),npell_);
			}
		}
		for( j = 0; j < nsdl; ++j ) /* j loops over s,r */
		{
		   for( i = 0; i < npell_; ++i )
		   {
			*(shlshl2_vec+i)=
			   shl.bend[npell_*(nsdl+1)*k+npell_*j+i]*(*(znode+npell_*k2+i));
		   }
		   for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
		   {
		       check=dotX((xlxs.hat+nsd*i+j),shlshl2_vec,
				(xl.hat+npell_*i), npell_);
		       *(xs+nsd*i+j)=xlxs.bar[nsdl*i+j]+xlxs.hat[nsd*i+j];
		   }
		}
/* 
   calculating dx,dy,dz/dt matrix for 2X2 or 3 point
   integration
*/
		for( i = 0; i < npell_; ++i )
		{
		   *(shlshl2_vec+i)=
			shl.bend[npell_*(nsdl+1)*k+npell_*2+i]*(*(dzdt_node+i));
		}
		for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
		{
		   check= dotX((xlxs.hat+nsd*i+2),shlshl2_vec,
			(xl.hat+npell_*i), npell_);
		   *(xs+nsd*i+2)=xlxs.hat[nsd*i+2];
		}
/* 
   calculating rotation matrix for lamina q[i,j] matrix for 2X2 or 3 point
   integration
*/
		*(vecl_xi)=*(xs);               *(vecl_eta)=*(xs+1);
		*(vecl_xi+1)=*(xs+nsd*1);       *(vecl_eta+1)=*(xs+nsd*1+1);
		*(vecl_xi+2)=*(xs+nsd*2);       *(vecl_eta+2)=*(xs+nsd*2+1);

		check = normal(vecl_xi);
		if(!check) printf( " Problems with vecl_xi element %d\n",el);
		check = normal(vecl_eta);
		if(!check) printf( " Problems with vecl_eta element %d\n",el);
		check = normcrossX(vecl_xi,vecl_eta,vecl_3);
		if(!check) printf( " Problems with normcrossX \n");

		*(vecl_alp)=*(vecl_xi)+*(vecl_eta);
		*(vecl_alp+1)=*(vecl_xi+1)+*(vecl_eta+1);
		*(vecl_alp+2)=*(vecl_xi+2)+*(vecl_eta+2);
		check = normal(vecl_alp);
		if(!check) printf( " Problems with vecl_alp element %d\n",el);
		check = normcrossX(vecl_3,vecl_alp,vecl_beta);
		if(!check) printf( " Problems with normcrossX \n");

		*(vecl_1)=sqpt5*(*(vecl_alp)-*(vecl_beta));
		*(vecl_1+1)=sqpt5*(*(vecl_alp+1)-*(vecl_beta+1));
		*(vecl_1+2)=sqpt5*(*(vecl_alp+2)-*(vecl_beta+2));
		*(vecl_2)=sqpt5*(*(vecl_alp)+*(vecl_beta));
		*(vecl_2+1)=sqpt5*(*(vecl_alp+1)+*(vecl_beta+1));
		*(vecl_2+2)=sqpt5*(*(vecl_alp+2)+*(vecl_beta+2));

		for( i = 0; i < nsd; ++i ) 
		{
		    rotate.l_bend[nsdsq*num_intb_*k2+nsdsq*k+i]=*(vecl_1+i);
		    rotate.l_bend[nsdsq*num_intb_*k2+nsdsq*k+nsd*1+i]=*(vecl_2+i);
		    rotate.l_bend[nsdsq*num_intb_*k2+nsdsq*k+nsd*2+i]=*(vecl_3+i);
		}

		*(temp)=*(xs+4)*(*(xs+8))-*(xs+7)*(*(xs+5));
		*(temp+3)=*(xs+6)*(*(xs+5))-*(xs+3)*(*(xs+8));
		*(temp+6)=*(xs+3)*(*(xs+7))-*(xs+6)*(*(xs+4));

		*(det+num_intb_*k2+k)=
			*(xs)*(*(temp))+*(xs+1)*(*(temp+3))+*(xs+2)*(*(temp+6));
		/*printf(" %d %f\n", k, *(det+num_intb_*k2+k));*/
 
		if(*(det+num_intb_*k2+k) <= 0.0 )
		{
		    printf("the element (%d) is inverted; det:%f; integ pt.:%d\n",
			el,*(det+num_intb_*k2+k),k);
		    return 0;
		}
 
/* The inverse of the jacobian, dc/dx, is calculated below */
 
		*(temp+1)=*(xs+7)*(*(xs+2))-*(xs+1)*(*(xs+8));
		*(temp+4)=*(xs)*(*(xs+8))-*(xs+6)*(*(xs+2));
		*(temp+7)=*(xs+6)*(*(xs+1))-*(xs)*(*(xs+7));
		*(temp+2)=*(xs+1)*(*(xs+5))-*(xs+4)*(*(xs+2));
		*(temp+5)=*(xs+3)*(*(xs+2))-*(xs)*(*(xs+5));
		*(temp+8)=*(xs)*(*(xs+4))-*(xs+3)*(*(xs+1));
     
		for( j = 0; j < nsd; ++j )
		{
		    for( i = 0; i < nsd; ++i )
		    {
		       *(xs+nsd*i+j)=*(temp+nsd*i+j)/(*(det+num_intb_*k2+k));
		    }
		}
		for( i = 0; i < npell_; ++i )
		{
		   *(col1)=shl.bend[npell_*(nsd)*k+i];
		   *(col1+1)=shl.bend[npell_*(nsd)*k+npell_*1+i];
		   *(col2)=*(xs);
		   *(col2+1)=*(xs+nsd*1);
		   check= dotX((shg.bend+npell_*(nsd+1)*num_intb_*k2+npell_*(nsd+1)*k+i),
			(col2),(col1), nsdl);
		   *(col2)=*(xs+1);
		   *(col2+1)=*(xs+nsd*1+1);
		   check= dotX((shg.bend+npell_*(nsd+1)*num_intb_*k2+npell_*(nsd+1)*k+npell_*1+i),
			(col2),(col1), nsdl);
		   *(col2)=*(xs+2);
		   *(col2+1)=*(xs+nsd*1+2);
		   check=dotX((shg.bend+npell_*(nsd+1)*num_intb_*k2+npell_*(nsd+1)*k+npell_*2+i),
			(col2),(col1), nsdl);
		   shg.bend[npell_*(nsd+1)*num_intb_*k2+npell_*(nsd+1)*k+npell_*3+i]=
		      shl.bend[npell_*(nsd)*k+npell_*2+i];

		   shg.bend_z[npell_*(nsd)*num_intb_*k2+npell_*(nsd)*k+i] =  
		      *(dzdt_node+i)*(*(xs+nsd*2));
		   shg.bend_z[npell_*(nsd)*num_intb_*k2+npell_*(nsd)*k+npell_*1+i] = 
		      *(dzdt_node+i)*(*(xs+nsd*2+1));
		   shg.bend_z[npell_*(nsd)*num_intb_*k2+npell_*(nsd)*k+npell_*2+i] = 
		      *(dzdt_node+i)*(*(xs+nsd*2+2));

		}
	   }
	}

/* 
   calculating the dN/dx,dy,dz and dza/dx,dy,dz matrices for 1X1 
   integration in lamina 
*/
/* 
   calculating dx,dy,dz/ds,dr matrix for 1X1 
   integration in lamina
*/
	for( j = 0; j < nsdl; ++j ) /* j loops over s,r */
	{
		for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
		{
		    check=dotX((xlxs.bar+nsdl*i+j),(shl.shear+npell_*j),
			(xl.bar+npell_*i), npell_);
		}
	}
	for( k2 = 0; k2 < num_ints; ++k2 )
	{
		for( j = 0; j < nsdl; ++j ) /* j loops over s,r */
		{
		   for( i = 0; i < npell_; ++i )
		   {
			*(shlshl2_vec+i)=
				shl.shear[npell_*j+i]*(*(znode+npell_*k2+i));
		   }

		   for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
		   {
		       check=dotX((xlxs.hat+nsd*i+j),shlshl2_vec,
				(xl.hat+npell_*i), npell_);
		       *(xs+nsd*i+j)=xlxs.bar[nsdl*i+j]+xlxs.hat[nsd*i+j];
		   }
		}
/* 
   calculating dx,dy,dz/dt matrix for 1X1 
   integration in lamina
*/
		for( i = 0; i < npell_; ++i )
		{
		   *(shlshl2_vec+i)=
			shl.shear[npell_*2+i]*(*(dzdt_node+i));
		}
		for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
		{
		   check= dotX((xlxs.hat+nsd*i+2),shlshl2_vec,
			(xl.hat+npell_*i), npell_);
		   *(xs+nsd*i+2)=xlxs.hat[nsd*i+2];
		}

/* 
   calculating rotation matrix for lamina q[i,j] matrix for 1X1 
   integration in lamina
*/
		*(vecl_xi)=*(xs);               *(vecl_eta)=*(xs+1);
		*(vecl_xi+1)=*(xs+nsd*1);       *(vecl_eta+1)=*(xs+nsd*1+1);
		*(vecl_xi+2)=*(xs+nsd*2);       *(vecl_eta+2)=*(xs+nsd*2+1);

		check = normal(vecl_xi);
		if(!check) printf( " Problems with vecl_xi element %d\n",el);
		check = normal(vecl_eta);
		if(!check) printf( " Problems with vecl_eta element %d\n",el);
		check = normcrossX(vecl_xi,vecl_eta,vecl_3);
		if(!check) printf( " Problems with normcrossX \n");

		*(vecl_alp)=*(vecl_xi)+*(vecl_eta);
		*(vecl_alp+1)=*(vecl_xi+1)+*(vecl_eta+1);
		*(vecl_alp+2)=*(vecl_xi+2)+*(vecl_eta+2);
		check = normal(vecl_alp);
		if(!check) printf( " Problems with vecl_alp element %d\n",el);
		check = normcrossX(vecl_3,vecl_alp,vecl_beta);
		if(!check) printf( " Problems with normcrossX \n");

		*(vecl_1)=sqpt5*(*(vecl_alp)-*(vecl_beta));
		*(vecl_1+1)=sqpt5*(*(vecl_alp+1)-*(vecl_beta+1));
		*(vecl_1+2)=sqpt5*(*(vecl_alp+2)-*(vecl_beta+2));
		*(vecl_2)=sqpt5*(*(vecl_alp)+*(vecl_beta));
		*(vecl_2+1)=sqpt5*(*(vecl_alp+1)+*(vecl_beta+1));
		*(vecl_2+2)=sqpt5*(*(vecl_alp+2)+*(vecl_beta+2));

		for( i = 0; i < nsd; ++i ) 
		{
		    rotate.l_shear[nsdsq*k2+i]=*(vecl_1+i);
		    rotate.l_shear[nsdsq*k2+nsd*1+i]=*(vecl_2+i);
		    rotate.l_shear[nsdsq*k2+nsd*2+i]=*(vecl_3+i);
		}

		*(temp)=*(xs+4)*(*(xs+8))-*(xs+7)*(*(xs+5));
		*(temp+3)=*(xs+6)*(*(xs+5))-*(xs+3)*(*(xs+8));
		*(temp+6)=*(xs+3)*(*(xs+7))-*(xs+6)*(*(xs+4));
 
		*(det+num_int_+k2)=
			*(xs)*(*(temp))+*(xs+1)*(*(temp+3))+*(xs+2)*(*(temp+6));
		/*printf("%d %f\n", k2, *(det+num_int_+k2));*/
 
		if(*(det+num_int_+k2) <= 0.0 )
		{
		    printf("the element (%d) is inverted; det:%f; fiber integ pt.:%d\n",
			el,*(det+num_int_+k2),k2);
		    return 0;
		}
/* The inverse of the jacobian, dc/dx, is calculated below */
 
		*(temp+1)=*(xs+7)*(*(xs+2))-*(xs+1)*(*(xs+8));
		*(temp+4)=*(xs)*(*(xs+8))-*(xs+6)*(*(xs+2));
		*(temp+7)=*(xs+6)*(*(xs+1))-*(xs)*(*(xs+7));
		*(temp+2)=*(xs+1)*(*(xs+5))-*(xs+4)*(*(xs+2));
		*(temp+5)=*(xs+3)*(*(xs+2))-*(xs)*(*(xs+5));
		*(temp+8)=*(xs)*(*(xs+4))-*(xs+3)*(*(xs+1));
     
		for( j = 0; j < nsd; ++j )
		{
		    for( i = 0; i < nsd; ++i )
		    {
		       *(xs+nsd*i+j)=
			   *(temp+nsd*i+j)/(*(det+num_int_+k2));
		    }
		}
		for( i = 0; i < npell_; ++i )
		{
		    *(col1)=shl.shear[i];
		    *(col1+1)=shl.shear[npell_*1+i];
		    *(col2)=*(xs);
		    *(col2+1)=*(xs+nsd*1);
		    check=dotX((shg.shear+npell_*(nsd+1)*k2+i),
			(col2),(col1),nsdl);
		    *(col2)=*(xs+1);
		    *(col2+1)=*(xs+nsd*1+1);
		    check=dotX((shg.shear+npell_*(nsd+1)*k2+npell_*1+i),
			(col2),(col1),nsdl);
		    *(col2)=*(xs+2);
		    *(col2+1)=*(xs+nsd*1+2);
		    check=dotX((shg.shear+npell_*(nsd+1)*k2+npell_*2+i),
			(col2),(col1),nsdl);
		    shg.shear[npell_*(nsd+1)*k2+npell_*3+i]=
			shl.shear[npell_*2+i];

		    shg.shear_z[npell_*(nsd)*k2+i] =  
			*(dzdt_node+i)*(*(xs+nsd*2));
		    shg.shear_z[npell_*(nsd)*k2+npell_*1+i] = 
			*(dzdt_node+i)*(*(xs+nsd*2+1));
		    shg.shear_z[npell_*(nsd)*k2+npell_*2+i] = 
			*(dzdt_node+i)*(*(xs+nsd*2+2));
		}
 
	}
	return 1; 
}


int shshg_mass( double *det, int el, SH shl, XL xl, double *zp1, double *zm1, 
	double *znode, double *dzdt_node)
{
/*
     This subroutine calculates the determinant for
     a shell element at the gauss points for the calculation of
     the global mass matrix.  Unlike shshg, we do not have to
     calculate the shape function derivatives with respect to
     the global coordinates x, y, z which are only needed for
     strains and stresses.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.
     This subroutine calculates the global shape function derivatives for
     a shell element at the gauss points.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR A FOUR-NODE QUADRALATERAL ELEMENT

       xl.bar[j+npell_*i] = GLOBAL COORDINATES ON LAMINA 
       xl.hat[j+npell_*i] = GLOBAL COORDINATES ON FIBER
       *(det+num_intb_*k2+k)  = JACOBIAN DETERMINANT

       FOR 2X2 or 3 PT. GAUSS
       shl.bend[npell_*(nsdl+1)*k+i] = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       shl.bend[npell_*(nsdl+1)*k+npell_*1+i] = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       shl.bend[npell_*(nsdl+1)*k+npell_*2+i] = LOCAL SHAPE FUNCTION(in XI and ETA)

       FOR 2X2 PT. GAUSS
       shl.bend_z[i] = LOCAL ("ZETA") DERIVATIVE OF SHAPE FUNCTION
       shl.bend_z[npelf*num_ints+npelf*k2+i] = LOCAL SHAPE FUNCTION(in ZETA)

       *(xs+2*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER FOR LAMINA 
          k2   = INTEGRATION-POINT NUMBER FOR FIBER
       num_intb_   = NUMBER OF INTEGRATION POINTS FOR LAMINA, 4
       num_ints    = NUMBER OF INTEGRATION POINTS FOR FIBER, 2

                        Updated 10/3/08
*/
	double xs[soxshat],temp[nsdsq],col1[nsdl],col2[nsdl],temp1,temp2;
	double shlshl2_vec[npell4],xfib2[2];
	XLXS xlxs;
	int check,i,j,k,k2,npell_,num_intb_;

	npell_ = npell4;
	num_intb_ = num_intb4;
	if(!flag_quad_element)
	{
	    npell_ = npell3;
	    num_intb_ = num_intb3;
	}
/* 
   calculating the dN/dx,dy,dz dza/dx,dy,dz matrices for 2X2 or 3 point
   integration
*/
	for( k2 = 0; k2 < num_ints; ++k2 )
	{
	   for( i = 0; i < npell_; ++i )
	   {
/* 
   calculating the values of the za and dzdt matrix for 2X2 point
   integration
*/
	       *(znode+npell_*k2+i)=
		    shl.bend_z[npelf*(nsdf+1)*k2+npelf*1+1]*(*(zp1+i))+
		    shl.bend_z[npelf*(nsdf+1)*k2+npelf*1]*(*(zm1+i));
	       *(dzdt_node+i)=
		    shl.bend_z[npelf*(nsdf+1)*k2+1]*(*(zp1+i))+
		    shl.bend_z[npelf*(nsdf+1)*k2]*(*(zm1+i));
	   }
	   for( k = 0; k < num_intb_; ++k )
	   {
/* 
   calculating dx,dy,dz/ds,dr matrix for 2X2 or 3 point
   integration
*/
		for( j = 0; j < nsdl; ++j ) /* j loops over s,r */
		{
			for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
			{
			    check=dotX((xlxs.bar+nsdl*i+j),(shl.bend+npell_*(nsdl+1)*k+npell_*j),
				(xl.bar+npell_*i),npell_);
			}
		}
		for( j = 0; j < nsdl; ++j ) /* j loops over s,r */
		{
		   for( i = 0; i < npell_; ++i )
		   {
			*(shlshl2_vec+i)=
			   shl.bend[npell_*(nsdl+1)*k+npell_*j+i]*(*(znode+npell_*k2+i));
		   }
		   for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
		   {
		       check=dotX((xlxs.hat+nsd*i+j),shlshl2_vec,
				(xl.hat+npell_*i), npell_);
		       *(xs+nsd*i+j)=xlxs.bar[nsdl*i+j]+xlxs.hat[nsd*i+j];
		   }
		}
/* 
   calculating dx,dy,dz/dt matrix for 2X2 or 3 point
   integration
*/
		for( i = 0; i < npell_; ++i )
		{
		   *(shlshl2_vec+i)=
			shl.bend[npell_*(nsdl+1)*k+npell_*2+i]*(*(dzdt_node+i));
		}
		for( i = 0; i < nsd; ++i ) /* i loops over x,y,z */
		{
		   check= dotX((xlxs.hat+nsd*i+2),shlshl2_vec,
			(xl.hat+npell_*i), npell_);
		   *(xs+nsd*i+2)=xlxs.hat[nsd*i+2];
		}

		*(temp)=*(xs+4)*(*(xs+8))-*(xs+7)*(*(xs+5));
		*(temp+3)=*(xs+6)*(*(xs+5))-*(xs+3)*(*(xs+8));
		*(temp+6)=*(xs+3)*(*(xs+7))-*(xs+6)*(*(xs+4));

		*(det+num_intb_*k2+k)=
			*(xs)*(*(temp))+*(xs+1)*(*(temp+3))+*(xs+2)*(*(temp+6));
		/*printf(" %d %f\n", k, *(det+num_intb_*k2+k));*/
 
		if(*(det+num_intb_*k2+k) <= 0.0 )
		{
		    printf("the element (%d) is inverted; det:%f; integ pt.:%d\n",
			el,*(det+num_intb_*k2+k),k);
		    return 0;
		}
 
	   }
	}

	return 1; 
}

