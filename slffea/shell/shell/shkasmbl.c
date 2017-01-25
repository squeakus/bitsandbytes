/*
    This utility function assembles the stiffness matrix for a finite 
    element program which does analysis on a shell element.  

		Updated 10/10/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "shconst.h"
#include "shstruct.h"

extern int analysis_flag, dof, sdof, integ_flag, doubly_curved_flag,
	numel, numnp, neqn, sof, flag_quad_element;
extern int gauss_stress_flag;
extern int LU_decomp_flag, numel_K, numel_P;
extern SH shg, shg_node, shl, shl_node, shl_tri, shl_tri_node;
extern ROTATE rotate, rotate_node;
extern double w[num_int8], w_tri[num_int6];

int cubic( double *);

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matX( double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int shellB1ptT(double *, double *,double *, double *, double *, double *);

int shellB1ptM(double *, double *, double *, double *);

int shellBNpt(double *, double *,double *, double *, double *, double *);

int shshg( double *, int , SH , SH , XL , double *, double *, double *,
	double *, ROTATE );

int shKassemble(double *A, int *connect, double *coord, int *el_matl, double *fiber_vec,
	double *force, int *id, int *idiag, double *K_diag, int *lm, double *lamina_ref,
	double *fiber_xyz, MATL *matl, double *node_counter, STRAIN *strain, SDIM *strain_node,
	STRESS *stress, SDIM *stress_node, double *U)
{
	int i, i1, i2, i3, i4, i5, j, k, dof_el[neqel20], sdof_el[npel8*nsd];
	int check, counter, node;
	int matl_num;
	double Emod, Pois, G1, G2, G3, shearK, thickness, const1, const2,
		fdum1, fdum2, fdum3, fdum4;
	double D11,D12,D21,D22;
	double B[soB100], DB[soB100];
	double K_temp[neqlsq400], K_el[neqlsq400];
	double force_el[neqel20], U_el[neqel20];
	double coord_el_trans[npel8*nsd], zm1[npell4], zp1[npell4],
		znode[npell4*num_ints], dzdt_node[npell4];
	double stress_el[sdim], strain_el[sdim], xxaddyy, xxsubyy, xysq, invariant[nsd],
		yzsq, zxsq, xxyy;
	double det[num_int8+num_ints], wXdet;
	XL xl;
	int npell_, neqel_, num_intb_, neqlsq_, soB_, npel_;
	SH *shl_, *shl_node_;

	npell_ = npell4;
	neqel_ = neqel20;
	num_intb_ = num_intb4;
	neqlsq_ = neqlsq400;
	soB_ = soB100;
	npel_ = npel8;
	shl_ = &shl;
	shl_node_ = &shl_node;

	if(!flag_quad_element)
	{
		npell_ = npell3;
		neqel_ = neqel15;
		num_intb_ = num_intb3;
		neqlsq_ = neqlsq225;
		soB_ = soB75;
		npel_ = npel6;
		shl_ = &shl_tri;
		shl_node_ = &shl_tri_node;
	}


	for( k = 0; k < numel; ++k )
	{

		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		shearK = matl[matl_num].shear;
		thickness = matl[matl_num].thick;

/* The constants below are for plane stress */

		const1 = Emod/(1.0-Pois*Pois);
		const2 = .5*(1.0-Pois);

		/*printf("Emod, Pois %f %f \n", Emod, Pois);*/

		D11 = const1;
		D12 = Pois*const1;
		D21 = Pois*const1;
		D22 = const1;

		G1 = const1*const2;
		G2 = const1*const2*shearK;
		G3 = const1*const2*shearK;

/* Create the coord transpose vector and other variables for one element */

		for( j = 0; j < npell_; ++j )
		{
			node = *(connect+npell_*k+j);

			*(sdof_el+nsd*j)=nsd*node;
			*(sdof_el+nsd*j+1)=nsd*node+1;
			*(sdof_el+nsd*j+2)=nsd*node+2;

			*(sdof_el+nsd*npell_+nsd*j)=nsd*(node+numnp);
			*(sdof_el+nsd*npell_+nsd*j+1)=nsd*(node+numnp)+1;
			*(sdof_el+nsd*npell_+nsd*j+2)=nsd*(node+numnp)+2;

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
			*(dof_el+ndof*j+3) = ndof*node+3;
			*(dof_el+ndof*j+4) = ndof*node+4;

/* Count the number of times a particular node is part of an element */

			if(analysis_flag == 1)
			{
			    *(node_counter + node) += 1.0;
			    *(node_counter + node + numnp) += 1.0;
#if 0
/* This code was for when the singly curved 4 node shells were of constant thickness.
   Currently, an averaging of the thicknesses over each node is being done in
   "shTopCoordinates" to calculate the corresponding top nodes for the entire mesh.
*/
			    if(!doubly_curved_flag)
			    {
				*(coord+*(sdof_el+nsd*j)+sdof) =
				    *(coord+*(sdof_el+nsd*j)) +
				    *(fiber_vec+*(sdof_el+nsd*j))*thickness;
				*(coord+*(sdof_el+nsd*j+1)+sdof) =
				    *(coord+*(sdof_el+nsd*j+1)) +
				    *(fiber_vec+*(sdof_el+nsd*j+1))*thickness;
				*(coord+*(sdof_el+nsd*j+2)+sdof) =
				    *(coord+*(sdof_el+nsd*j+2)) +
				    *(fiber_vec+*(sdof_el+nsd*j+2))*thickness;
			    }
#endif
			}

/* Create the coord -/+*/

			*(coord_el_trans+j) =
				*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel_*1+j) =
				*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel_*2+j) =
				*(coord+*(sdof_el+nsd*j+2));

			*(coord_el_trans+npell_+j) =
				*(coord+*(sdof_el+nsd*npell_+nsd*j));
			*(coord_el_trans+npel_*1+npell_+j) =
				*(coord+*(sdof_el+nsd*npell_+nsd*j+1));
			*(coord_el_trans+npel_*2+npell_+j) =
				*(coord+*(sdof_el+nsd*npell_+nsd*j+2));

/* Create the coord_bar and coord_hat vector for one element */

			xl.bar[j] = *(lamina_ref + nsd*node);
			xl.bar[npell_*1+j] = *(lamina_ref + nsd*node + 1);
			xl.bar[npell_*2+j] = *(lamina_ref + nsd*node + 2);

			fdum1=*(coord_el_trans+npell_+j)-*(coord_el_trans+j);
			fdum2=*(coord_el_trans+npel_*1+npell_+j)-*(coord_el_trans+npel_*1+j);
			fdum3=*(coord_el_trans+npel_*2+npell_+j)-*(coord_el_trans+npel_*2+j);
			fdum4=sqrt(fdum1*fdum1+fdum2*fdum2+fdum3*fdum3);

			*(zp1+j)=.5*(1.0-zeta)*fdum4;
			*(zm1+j)=-.5*(1.0+zeta)*fdum4;

			xl.hat[j] = *(fiber_xyz + nsdsq*node + 2*nsd);
			xl.hat[npell_*1+j] = *(fiber_xyz + nsdsq*node + 2*nsd + 1);
			xl.hat[npell_*2+j] = *(fiber_xyz + nsdsq*node + 2*nsd + 2);


/* Create the rotation matrix */

			for( i1 = 0; i1 < nsd; ++i1 )
			{
			    rotate.f_shear[nsdsq*j + i1] =
				*(fiber_xyz + nsdsq*node + i1);
			    rotate.f_shear[nsdsq*j + 1*nsd + i1] =
				*(fiber_xyz + nsdsq*node + 1*nsd + i1);
			    rotate.f_shear[nsdsq*j + 2*nsd + i1] =
				*(fiber_xyz + nsdsq*node + 2*nsd + i1);
			}
		}

		memcpy(rotate_node.f_shear,rotate.f_shear,sorfs36*sizeof(double));

/* Assembly of the shg matrix for each integration point */

		check=shshg( det, k, *shl_, shg, xl, zp1, zm1, znode,
			dzdt_node, rotate);
		if(!check) printf( "Problems with shshg \n");

		memset(U_el,0,neqel_*sof);
		memset(K_el,0,neqlsq_*sof);
		memset(force_el,0,neqel_*sof);

/* The loop over i4 below calculates the 2 fiber points of the gaussian integration */

		for( i4 = 0; i4 < num_ints; ++i4 )
		{

/* The loop over j below calculates the 2X2 or 3 points of the gaussian integration
   over the lamina for several quantities */

		   for( j = 0; j < num_intb_; ++j )
		   {
			memset(B,0,soB_*sof);
			memset(DB,0,soB_*sof);
			memset(K_temp,0,neqlsq_*sof);

/* Assembly of the B matrix */

			check =
			   shellBNpt((shg.bend+npell_*(nsd+1)*num_intb_*i4+npell_*(nsd+1)*j),
			   (shg.bend_z+npell_*(nsd)*num_intb_*i4+npell_*(nsd)*j),
			   (znode+npell_*i4),B,(rotate.l_bend+nsdsq*num_intb_*i4+nsdsq*j),
			   rotate.f_shear);
			if(!check) printf( "Problems with shellBNpt \n");

			if( integ_flag == 0 || integ_flag == 2 ) 
			{
/* Calculate the membrane shear terms in B using 1X1 point gaussian integration in lamina */

			    check = shellB1ptM((shg.shear+npell_*(nsd+1)*i4),
				(B+(sdim-2)*neqel_),
				(rotate.l_shear+nsdsq*i4), rotate.f_shear);
			    if(!check) printf( "Problems with shellB1pt \n");
			}

			if( integ_flag == 1 || integ_flag == 2 ) 
			{
/* Calculate the transverse shear terms in B using 1X1 point gaussian integration in lamina */

			    check = shellB1ptT((shg.shear+npell_*(nsd+1)*i4),
				(shg.bend_z+npell_*(nsd)*num_intb_*i4+npell_*(nsd)*j),
				(znode+npell_*i4),(B+(sdim-2)*neqel_),
				(rotate.l_shear+nsdsq*i4), rotate.f_shear);
			    if(!check) printf( "Problems with shellB1pt \n");
			}

			for( i1 = 0; i1 < neqel_; ++i1 )
			{
				*(DB+i1)=*(B+i1)*D11 +
					*(B+neqel_*1+i1)*D12;
				*(DB+neqel_*1+i1)=*(B+i1)*D21 +
					*(B+neqel_*1+i1)*D22;
				*(DB+neqel_*2+i1)=*(B+neqel_*2+i1)*G1;
				*(DB+neqel_*3+i1)=*(B+neqel_*3+i1)*G2;
				*(DB+neqel_*4+i1)=*(B+neqel_*4+i1)*G3;
			}
#if 0
			fprintf(oB,"\n\n\n");
			fprintf(oS,"\n\n\n");
			for( i3 = 0; i3 < 3; ++i3 )
			{
			    fprintf(oB,"\n ");
			    fprintf(oS,"\n ");
			    for( i2 = 0; i2 < neqel_; ++i2 )
			    {
				fprintf(oB,"%14.5e ",*(B+neqel_*i3+i2));
				fprintf(oS,"%14.5e ",*(DB+neqel_*i3+i2));
			    }
			}
#endif

			wXdet = *(w+num_intb_*i4+j)*(*(det+num_intb_*i4+j));
			if(!flag_quad_element)
				wXdet = 0.5*(*(w_tri+num_intb_*i4+j))*(*(det+num_intb_*i4+j));

			check=matXT(K_temp, B, DB, neqel_, neqel_, sdim);
			if(!check) printf( "Problems with matXT \n");

			for( i2 = 0; i2 < neqlsq_; ++i2 )
			{
			   *(K_el+i2) += *(K_temp+i2)*wXdet;
			}
		   }
		}

		for( j = 0; j < neqel_; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(force_el, K_el, U_el, neqel_, 1, neqel_);
		if(!check) printf( "Problems with matX \n");

		if(analysis_flag == 1)
		{

/* Compute the equivalant nodal forces based on prescribed displacements */

			for( j = 0; j < neqel_; ++j )
			{
				*(force + *(dof_el+j)) -= *(force_el + j);
			}

/* Assembly of either the global skylined stiffness matrix or numel_K of the
   element stiffness matrices if the Conjugate Gradient method is used */

			if(LU_decomp_flag)
			{
			    check = globalKassemble(A, idiag, K_el, (lm + k*neqel_),
				neqel_);
			    if(!check) printf( "Problems with globalKassemble \n");
			}
			else
			{
			    check = globalConjKassemble(A, dof_el, k, K_diag, K_el,
				neqel_, neqlsq_, numel_K);
			    if(!check) printf( "Problems with globalConjKassemble \n");
			}
		}
		else
		{
/* Calculate the element reaction forces */

			for( j = 0; j < neqel_; ++j )
			{
				*(force + *(dof_el+j)) += *(force_el + j);
			}

/* Calculate the element stresses */

/* Unlike the brick or quad, there is no assembly of the of the shstress_shg matrix
   for each nodal point.  The complexity of this element, with all the geometric
   transformation terms, makes such calculations prohibitive and of questionable
   value. 
*/

			if(gauss_stress_flag)
			{
/* Calculate shg at integration point */
			    check=shshg( det, k, *shl_, shg, xl, zp1, zm1, znode,
				dzdt_node, rotate);
			    if(!check) printf( "Problems with shshg \n");
			}
			else
			{
/* Calculate shg at nodal point */
			    check=shshg( det, k, *shl_node_, shg_node, xl, zp1, zm1, znode,
				dzdt_node, rotate_node);
			    if(!check) printf( "Problems with shshg \n");
			}


/* The loop over i4 below calculates the 2 fiber points of the gaussian integration */

			for( i4 = 0; i4 < num_ints; ++i4 )
			{

/* The loop over j below calculates the 2X2 or 3 points of the gaussian integration
   over the lamina for several quantities */

			    for( j = 0; j < num_intb_; ++j )
			    {
				memset(B,0,soB_*sof);
				memset(stress_el,0,sdim*sof);
				memset(strain_el,0,sdim*sof);

				node = *(connect+npell_*k+j) + i4*numnp;

/* Assembly of the B matrix */

				if(gauss_stress_flag)
				{
				   check =
					shellBNpt(
					(shg.bend+npell_*(nsd+1)*num_intb_*i4+npell_*(nsd+1)*j),
					(shg.bend_z+npell_*(nsd)*num_intb_*i4+npell_*(nsd)*j),
					(znode+npell_*i4),B,
					(rotate.l_bend+nsdsq*num_intb_*i4+nsdsq*j),
					rotate.f_shear);
				   if(!check) printf( "Problems with shellBNpt \n");

				   if( integ_flag == 0 || integ_flag == 2 )
				   {
/* Calculate the membrane shear terms in B using 1X1 point gaussian integration in lamina */

					check = shellB1ptM((shg.shear+npell_*(nsd+1)*i4),
					    (B+(sdim-2)*neqel_),
					    (rotate.l_shear+nsdsq*i4), rotate.f_shear);
					if(!check) printf( "Problems with shellB1pt \n");
				   }

				   if( integ_flag == 1 || integ_flag == 2 )
				   {
/* Calculate the transverse shear terms in B using 1X1 point gaussian integration in lamina */

					check = shellB1ptT((shg.shear+npell_*(nsd+1)*i4),
					    (shg.bend_z+npell_*(nsd)*num_intb_*i4+npell_*(nsd)*j),
					    (znode+npell_*i4),(B+(sdim-2)*neqel_),
					    (rotate.l_shear+nsdsq*i4), rotate.f_shear);
					if(!check) printf( "Problems with shellB1pt \n");
				   }
				}
				else
				{
/* Calculate B matrix at nodal point */
				   check =
					shellBNpt(
					(shg_node.bend+npell_*(nsd+1)*num_intb_*i4+npell_*(nsd+1)*j),
					(shg_node.bend_z+npell_*(nsd)*num_intb_*i4+npell_*(nsd)*j),
					(znode+npell_*i4),B,
					(rotate_node.l_bend+nsdsq*num_intb_*i4+nsdsq*j),
					rotate_node.f_shear);
					if(!check) printf( "Problems with shellBNpt \n");
				}

/* Calculation of the local strain matrix */

				check=matX(strain_el, B, U_el, sdim, 1, neqel_ );
				if(!check) printf( "Problems with matX \n");
#if 0
				for( i1 = 0; i1 < sdim; ++i1 )
				{
				       printf("%12.8f ",*(stress_el+i1));
				       /*printf("%12.2f ",*(stress_el+i1));
				       printf("%12.8f ",*(B+i1));*/
				}
				printf("\n");
#endif

/* Update of the global strain matrix */

				i5 = num_intb_*i4+j;

				strain[k].pt[i5].xx = *(strain_el);
				strain[k].pt[i5].yy = *(strain_el+1);
				strain[k].pt[i5].xy = *(strain_el+2);
				strain[k].pt[i5].zx = *(strain_el+3);
				strain[k].pt[i5].yz = *(strain_el+4);

/* Calculate the principal straines */

				memset(invariant,0,nsd*sof);
				xysq = strain[k].pt[i5].xy*strain[k].pt[i5].xy;
				zxsq = strain[k].pt[i5].zx*strain[k].pt[i5].zx;
				yzsq = strain[k].pt[i5].yz*strain[k].pt[i5].yz;
				xxyy = strain[k].pt[i5].xx*strain[k].pt[i5].yy;

				*(invariant) = - strain[k].pt[i5].xx -
				     strain[k].pt[i5].yy;
				*(invariant+1) = xxyy - yzsq - zxsq - xysq;
				*(invariant+2) = -
				     2*strain[k].pt[i5].yz*strain[k].pt[i5].zx*strain[k].pt[i5].xy +
				     yzsq*strain[k].pt[i5].xx + zxsq*strain[k].pt[i5].yy;
				
				check = cubic(invariant);

				strain[k].pt[i5].I = *(invariant);
				strain[k].pt[i5].II = *(invariant+1);
				strain[k].pt[i5].III = *(invariant+2);

/* Add all the strains for a particular node from all the elements which share that node */

				strain_node[node].xx += strain[k].pt[i5].xx;
				strain_node[node].yy += strain[k].pt[i5].yy;
				strain_node[node].xy += strain[k].pt[i5].xy;
				strain_node[node].zx += strain[k].pt[i5].zx;
				strain_node[node].yz += strain[k].pt[i5].yz;
				strain_node[node].I += strain[k].pt[i5].I;
				strain_node[node].II += strain[k].pt[i5].II;
				strain_node[node].III += strain[k].pt[i5].III;

/* Calculation of the local stress matrix */

				*(stress_el)= strain[k].pt[i5].xx*D11 +
					strain[k].pt[i5].yy*D12;
				*(stress_el+1)= strain[k].pt[i5].xx*D21 +
					strain[k].pt[i5].yy*D22;
				*(stress_el+2)= strain[k].pt[i5].xy*G1;
				    *(stress_el+3)= strain[k].pt[i5].zx*G2;
				    *(stress_el+4)= strain[k].pt[i5].yz*G3;

/* Update of the global stress matrix */

				stress[k].pt[i5].xx += *(stress_el);
				stress[k].pt[i5].yy += *(stress_el+1);
				stress[k].pt[i5].xy += *(stress_el+2);
				stress[k].pt[i5].zx += *(stress_el+3);
				stress[k].pt[i5].yz += *(stress_el+4);

/* Calculate the principal stresses */

				memset(invariant,0,nsd*sof);
				xysq = stress[k].pt[i5].xy*stress[k].pt[i5].xy;
				zxsq = stress[k].pt[i5].zx*stress[k].pt[i5].zx;
				yzsq = stress[k].pt[i5].yz*stress[k].pt[i5].yz;
				xxyy = stress[k].pt[i5].xx*stress[k].pt[i5].yy;

				*(invariant) = - stress[k].pt[i5].xx -
				     stress[k].pt[i5].yy;
				*(invariant+1) = xxyy - yzsq - zxsq - xysq;
				*(invariant+2) = -
				     2*stress[k].pt[i5].yz*stress[k].pt[i5].zx*stress[k].pt[i5].xy +
				     yzsq*stress[k].pt[i5].xx + zxsq*stress[k].pt[i5].yy;

				check = cubic(invariant);

				stress[k].pt[i5].I = *(invariant);
				stress[k].pt[i5].II = *(invariant+1);
				stress[k].pt[i5].III = *(invariant+2);

/* Add all the stresses for a particular node from all the elements which share that node */

				stress_node[node].xx += stress[k].pt[i5].xx;
				stress_node[node].yy += stress[k].pt[i5].yy;
				stress_node[node].xy += stress[k].pt[i5].xy;
				stress_node[node].zx += stress[k].pt[i5].zx;
				stress_node[node].yz += stress[k].pt[i5].yz;
				stress_node[node].I += stress[k].pt[i5].I;
				stress_node[node].II += stress[k].pt[i5].II;
				stress_node[node].III += stress[k].pt[i5].III;

/*
				printf("%14.6e ",stress[k].pt[i5].xx);
				printf("%14.6e ",stress[k].pt[i5].yy);
				printf("%14.6e ",stress[k].pt[i5].xy);
				printf( "\n");
*/
			    }
			    /*printf( "\n");*/
			}
		}
	}

	if(analysis_flag == 1)
	{

/* Contract the global force matrix using the id array only if LU decomposition
   is used. */

	  if(LU_decomp_flag)
	  {
	     counter = 0;
	     for( i = 0; i < dof ; ++i )
	     {
		if( *(id + i ) > -1 )
		{
			*(force + counter ) = *(force + i );
			++counter;
		}
	     }
	  }
	}
	if(analysis_flag == 2)
	{

/* Average all the stresses and strains at the nodes */

	  for( i = 0; i < 2*numnp ; ++i )
	  {
		   strain_node[i].xx /= *(node_counter + i);
		   strain_node[i].yy /= *(node_counter + i);
		   strain_node[i].xy /= *(node_counter + i);
		   strain_node[i].zx /= *(node_counter + i);
		   strain_node[i].yz /= *(node_counter + i);
		   strain_node[i].I /= *(node_counter + i);
		   strain_node[i].II /= *(node_counter + i);
		   strain_node[i].III /= *(node_counter + i);

		   stress_node[i].xx /= *(node_counter + i);
		   stress_node[i].yy /= *(node_counter + i);
		   stress_node[i].xy /= *(node_counter + i);
		   stress_node[i].zx /= *(node_counter + i);
		   stress_node[i].yz /= *(node_counter + i);
		   stress_node[i].I /= *(node_counter + i);
		   stress_node[i].II /= *(node_counter + i);
		   stress_node[i].III /= *(node_counter + i);
	  }
	}

	return 1;
}

