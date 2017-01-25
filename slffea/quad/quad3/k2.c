/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on a quad.  It is for
    eletromagnetism.

		Updated 12/7/00

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qd3const.h"
#include "qd3struct.h"

extern int analysis_flag, dof, EMdof, neqn, EMneqn, numed, numel, numnp,
	plane_stress_flag, sof;
extern int B_matrix_store, Bzz_matrix_store, gauss_stress_flag;
extern int LU_decomp_flag, Bzz_LU_decomp_flag, numel_EM, numel_P;
extern double dcdx[nsdsq*num_int], shg[sosh], shg_node[sosh], shl[sosh],
	shl_node[sosh], shl_node2[sosh_node2], w[num_int], *Area0;

int globalConjKassemble(double *, int *, int , double *,
        double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matX( double *,double *,double *, int ,int ,int );

int dyadicX( double *,double *,double *, int);

int matXT( double *, double *, double *, int, int, int);

int quad3B( double *, double *, double *, double *, double *, double *, double *, double *);

int qd3shg( double *, double *, int, double *, double *, double *, double *);

int qd3Kassemble(double *Att, double *Btt, double *Btz, double *Bzt, double *Bzz,
	int *connect, int *edge_connect, int *el_edge_connect, double *coord, int *el_matl,
	double *force, int *id, int *idiag, double *Att_diag, double *Bzz_diag, int *lm,
	MATL *matl, double *node_counter, double *edge_counter, STRAIN *strain,
	SDIM *strain_node, STRESS *stress, SDIM *stress_node, double *EMedge,
	double *Emnode, double *Arean)
{
        int i, i1, i2, j, k, dof_el[neqel], EMdof_el[EMneqel], sdof_el[npel*nsd];
	int check, counter, node, node1, node2, edge;
	int matl_num;
	double permit, permea, op_fq, G;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[npel], gradB[EMsoB], Bedge[EMsoB], curlBedge[EMsoB], BXB[EMneqlsq];
	double Att_temp[EMneqlsq], Btt_temp[EMneqlsq], Bzt_temp[EMneqlsq],
		Btz_temp[EMneqlsq], Bzz_temp[EMneqlsq];
	double Att_el[EMneqlsq], Btt_el[EMneqlsq], Bzt_el[EMneqlsq],
		Btz_el[EMneqlsq], Bzz_el[EMneqlsq];
	double K_temp[EMneqlsq], K_el[EMneqlsq];
	double force_el[neqel], EMedge_el[EMneqel];
        double coord_el_trans[npel*nsd], length[epel];
	double stress_el[sdim], strain_el[sdim], xxaddyy, xxsubyy, xysq;
	double det[num_int];
	double fdum, fdum2;

        for( k = 0; k < numel; ++k )
        {

                matl_num = *(el_matl+k);
                permit = matl[matl_num].eta;
		permea = matl[matl_num].nu;
		op_fq = matl[matl_num].ko*matl[matl_num].ko;

        	/*printf("lamda, mu, permit, permea  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

		D11 = lamda+2.0*mu;
		D12 = lamda;
		D21 = lamda;
		D22 = lamda+2.0*mu;

/* Create the coord_el transpose vector for one element */

                for( j = 0; j < npel; ++j )
                {
			node = *(connect+npel*k+j);

                	*(sdof_el+nsd*j) = nsd*node;
                	*(sdof_el+nsd*j+1) = nsd*node+1;

                        *(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
                        *(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));

                	*(dof_el+ndof*j) = ndof*node;
                	*(dof_el+ndof*j+1) = ndof*node+1;

/* Count the number of times a particular node is part of an element */

			if(analysis_flag == 1)
				*(node_counter + node) += 1.0;
		}

/* Create edge_el for one element */

                for( j = 0; j < epel; ++j )
                {
			edge = *(el_edge_connect+epel*k+j);
			node1 = *(edge_connect+edge*nped);
			node2 = *(edge_connect+edge*nped + 1);

                	*(EMdof_el+edof*j) = edof*edge;

/* Calculate the edge lengths */

			fdum = *(coord+node2*nsd) - *(coord+node1*nsd);
			fdum2 = *(coord+node2*nsd+1) - *(coord+node1*nsd+1);
			*(length + j) = sqrt(fdum*fdum + fdum2*fdum2);

/* Count the number of times a particular edge is part of an element */

			if(analysis_flag == 1)
				*(edge_counter + edge) += 1.0;
		}

/* Assembly of the dcdx and shg matrix for each integration point */

		check=qd3shg(dcdx, det, k, shl, shg, coord_el_trans, (Arean+k));
		if(!check) printf( "Problems with qd3shg \n");

/* The loop over j below calculates the 4 points of the gaussian integration 
   for several quantities */

		printf("\n hhhhhhh %4d \n", EMneqlsq);

		memset(EMedge_el,0,EMneqel*sof);
                memset(K_el,0,EMneqlsq*sof);
                memset(Att_el,0,EMneqlsq*sof);
                memset(Btt_el,0,EMneqlsq*sof);
                memset(Btz_el,0,EMneqlsq*sof);
                memset(Bzt_el,0,EMneqlsq*sof);
                memset(Bzz_el,0,EMneqlsq*sof);
		memset(force_el,0,neqel*sof);

                for( j = 0; j < num_int; ++j )
                {

                    memset(B,0,npel*sof);
                    memset(gradB,0,EMsoB*sof);
                    memset(Bedge,0,EMsoB*sof);
                    memset(curlBedge,0,EMsoB*sof);
                    memset(Att_temp,0,EMneqlsq*sof);
                    memset(Btt_temp,0,EMneqlsq*sof);
                    memset(Btz_temp,0,EMneqlsq*sof);
                    memset(Bzt_temp,0,EMneqlsq*sof);
                    memset(Bzz_temp,0,EMneqlsq*sof);
                    memset(BXB,0,EMneqlsq*sof);
                    memset(K_temp,0,EMneqlsq*sof);

/* Assembly of the component matrices */

		    check = quad3B((dcdx+nsdsq*j), (shl+npel*(nsd+1)*j),
			(shg+npel*(nsd+1)*j),B, gradB, Bedge, curlBedge, length);
                    if(!check) printf( "Problems with quad3B \n");

                    check=matXT(Att_temp, curlBedge, curlBedge, EMneqel, EMneqel, nsd);
                    if(!check) printf( "Problems with matXT  \n");

                    check=matXT(Btt_temp, Bedge, Bedge, EMneqel, EMneqel, nsd);
                    if(!check) printf( "Problems with matXT  \n");

		    fdum = *(w+j)*(*(det+j));

                    for( i2 = 0; i2 < EMneqlsq; ++i2 )
                    {
                          *(Att_temp+i2) = *(Att_temp+i2)/permea -
				*(Btt_temp+i2)*op_fq*permit;
			  *(Att_el+i2) += *(Att_temp+i2)*fdum;
                    }

		    if(Bzz_LU_decomp_flag || Bzz_matrix_store)
		    {
			check=matXT(Bzz_temp, gradB, gradB, EMneqel, EMneqel, nsd);
			if(!check) printf( "Problems with matXT  \n");

			check=dyadicX(BXB, B, B, EMneqel);
			if(!check) printf( "Problems with dyadicX \n");

			for( i2 = 0; i2 < EMneqlsq; ++i2 )
			{
			   *(Bzz_temp+i2) = *(Bzz_temp+i2)/permea -
				*(BXB+i2)*op_fq*permit;
			   *(Bzz_el+i2) += *(Bzz_temp+i2)*fdum;
			}

			for( i2 = 0; i2 < EMneqel; ++i2 )
			{
			   for( i1 = 0; i1 < EMneqel; ++i1 )
			   {
				printf("%12.8f ",*(Bzz_temp + EMneqel*i2 + i1));
			   }
			   printf("\n ");
			}
			printf("\n ");
		    }

		    if(B_matrix_store)
		    {
			check=matXT(Bzt_temp, Bedge, gradB, EMneqel, EMneqel, nsd);
			if(!check) printf( "Problems with matXT  \n");

			check=matXT(Btz_temp, gradB, Bedge, EMneqel, EMneqel, nsd);
			if(!check) printf( "Problems with matXT  \n");

			check=dyadicX(BXB, B, B, EMneqel);
			if(!check) printf( "Problems with dyadicX \n");

			for( i2 = 0; i2 < EMneqlsq; ++i2 )
			{
			   *(Btt_temp + i2) /= permea;
			   *(Btz_temp + i2) /= permea;
			   *(Bzt_temp + i2) /= permea;

			   *(Btt_el+i2) += *(Btt_temp+i2)*fdum;
			   *(Btz_el+i2) += *(Btz_temp+i2)*fdum;
			   *(Bzt_el+i2) += *(Bzt_temp+i2)*fdum;
			}
                    }
                }

		for( j = 0; j < EMneqel; ++j )
                {
			*(EMedge_el + j) = *(EMedge + *(EMdof_el+j));
		}

                check = matX(force_el, K_el, EMedge_el, EMneqel, 1, EMneqel);
                if(!check) printf( "Problems with matX \n");

		if(analysis_flag == 1)
		{

/* Compute the equivalant nodal forces based on prescribed displacements */

			for( j = 0; j < neqel; ++j )
                	{
				*(force + *(dof_el+j)) -= *(force_el + j);
			}

/* Assembly of either the global skylined Att matrix or numel_EM of the
   element Att matrices if the Conjugate Gradient method is used */

			if(LU_decomp_flag)
			{
			    check = globalKassemble(Att, idiag, Att_el, (lm + k*EMneqel),
				EMneqel);
			    if(!check) printf( "Problems with globalKassemble \n");
			}
			else
			{
			    check = globalConjKassemble(Att, EMdof_el, k, Att_diag, Att_el,
				EMneqel, EMneqlsq, numel_EM);
			    if(!check) printf( "Problems with globalConjKassemble \n");
			}

			if(Bzz_LU_decomp_flag)
			{
			    check = globalKassemble(Bzz, idiag, Bzz_el, (lm + k*EMneqel),
				EMneqel);
			    if(!check) printf( "Problems with globalKassemble \n");
			}
			else if(Bzz_matrix_store)
			{
			    check = globalConjKassemble(Bzz, EMdof_el, k, Bzz_diag, Bzz_el,
				EMneqel, EMneqlsq, numel_EM);
			    if(!check) printf( "Problems with globalConjKassemble \n");
			}

/* If there is enough memory, store all the element Btt, Btz, Bzt, Bzz matrices */

			if(B_matrix_store)
			{
			   for( j = 0; j < EMneqlsq; ++j )
			   {
				*(Btt + EMneqlsq*k + j) = *(Btt_el + j);
				*(Btz + EMneqlsq*k + j) = *(Btz_el + j);
				*(Bzt + EMneqlsq*k + j) = *(Bzt_el + j);
			   }

			}

		}
		else
		{
/* Calculate the element reaction forces */

			for( j = 0; j < neqel; ++j )
                        {
				*(force + *(dof_el+j)) += *(force_el + j);
			}

/* Calculate the element stresses */

/* Assembly of the qdstress_shg matrix for each nodal point */

/*
   qdstress_shg calculates stress at the nodes.  It is more efficient than qd3shg
   because it removes all the zero multiplications from the calculation of shg. 
   You can use either function when calculating shg at the nodes.  Because stress
   is calculated at the Gauss points, qd3shg is used.

			check=qdstress_shg(det, k, shl_node2, shg, coord_el_trans);
			check=qd3shg(det, k, shl_node, shg, coord_el_trans, (Arean+k));
*/

			if(gauss_stress_flag)
			{
/* Calculate shg at integration point */
			    check=qd3shg(dcdx, det, k, shl, shg, coord_el_trans, (Arean+k));
			    if(!check) printf( "Problems with qd3shg \n");
			}
			else
			{
/* Calculate shg at nodal point */
			    check=qd3shg(dcdx, det, k, shl, shg, coord_el_trans, (Arean+k));
			    if(!check) printf( "Problems with qd3shg \n");
			}

			for( j = 0; j < num_int; ++j )
                	{

                    	   memset(B,0,EMsoB*sof);
                    	   memset(stress_el,0,sdim*sof);
                    	   memset(strain_el,0,sdim*sof);

                           node = *(connect+npel*k+j);

/* Assembly of the B matrix */

			   if(gauss_stress_flag)
			   {
/* Calculate B matrix at integration point */
				check = quad3B((dcdx+nsdsq*j), (shl+npel*(nsd+1)*j),
				    (shg+npel*(nsd+1)*j),B, gradB, Bedge, curlBedge,
				    length);
				if(!check) printf( "Problems with quad3B \n");
			   }
			   else
			   {
/* Calculate B matrix at nodal point */
				check = quad3B((dcdx+nsdsq*j), (shl_node+npel*(nsd+1)*j),
				    (shg_node+npel*(nsd+1)*j),B, gradB, Bedge, curlBedge,
				    length);
				if(!check) printf( "Problems with quad3B \n");
			   }
		
/* Calculation of the local strain matrix */

#if 0
                	   check=matX(strain_el, B, EMedge_el, sdim, 1, EMneqel );
                    	   if(!check) printf( "Problems with matX \n");

                    	   for( i1 = 0; i1 < sdim; ++i1 )
                    	   {
                          	     printf("%12.8f ",*(stress_el+i1));
                          	     /*printf("%12.2f ",*(stress_el+i1));
                          	     printf("%12.8f ",*(B+i1));*/
                    	   }
                    	   printf("\n");
#endif

/* Update of the global strain matrix */

		       	   strain[k].pt[j].xx = *(strain_el);
		     	   strain[k].pt[j].yy = *(strain_el+1);
		    	   strain[k].pt[j].xy = *(strain_el+2);

/* Calculate the principal straines */

			   xxaddyy = .5*(strain[k].pt[j].xx + strain[k].pt[j].yy);
			   xxsubyy = .5*(strain[k].pt[j].xx - strain[k].pt[j].yy);
			   xysq = strain[k].pt[j].xy*strain[k].pt[j].xy;

		    	   strain[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
		    	   strain[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);
           		   /*printf("%14.6e %14.6e %14.6e\n",xxaddyy,xxsubyy,xysq);*/

/* Add all the strains for a particular node from all the elements which share that node */

			   strain_node[node].xx += strain[k].pt[j].xx;
			   strain_node[node].yy += strain[k].pt[j].yy;
			   strain_node[node].xy += strain[k].pt[j].xy;
			   strain_node[node].I += strain[k].pt[j].I;
			   strain_node[node].II += strain[k].pt[j].II;

/* Calculation of the local stress matrix */

			   *(stress_el) = strain[k].pt[j].xx*D11+
				strain[k].pt[j].yy*D12;
			   *(stress_el+1) = strain[k].pt[j].xx*D21+
				strain[k].pt[j].yy*D22;
			   *(stress_el+2) = strain[k].pt[j].xy*G;

/* Update of the global stress matrix */

		       	   stress[k].pt[j].xx += *(stress_el);
		     	   stress[k].pt[j].yy += *(stress_el+1);
		    	   stress[k].pt[j].xy += *(stress_el+2);

/* Calculate the principal stresses */

			   xxaddyy = .5*(stress[k].pt[j].xx + stress[k].pt[j].yy);
			   xxsubyy = .5*(stress[k].pt[j].xx - stress[k].pt[j].yy);
			   xysq = stress[k].pt[j].xy*stress[k].pt[j].xy;

		    	   stress[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
		    	   stress[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);

/* Add all the stresses for a particular node from all the elements which share that node */

			   stress_node[node].xx += stress[k].pt[j].xx;
			   stress_node[node].yy += stress[k].pt[j].yy;
			   stress_node[node].xy += stress[k].pt[j].xy;
			   stress_node[node].I += stress[k].pt[j].I;
			   stress_node[node].II += stress[k].pt[j].II;

/*
           		   printf("%14.6e ",stress[k].pt[j].xx);
           		   printf("%14.6e ",stress[k].pt[j].yy);
           		   printf("%14.6e ",stress[k].pt[j].xy);
           		   printf( "\n");
*/
			}
           		/*printf( "\n");*/
		}
	}

#if 0
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

	  for( i = 0; i < numnp ; ++i )
	  {
		   strain_node[i].xx /= *(node_counter + i);
		   strain_node[i].yy /= *(node_counter + i);
		   strain_node[i].xy /= *(node_counter + i);
		   strain_node[i].I /= *(node_counter + i);
		   strain_node[i].II /= *(node_counter + i);

		   stress_node[i].xx /= *(node_counter + i);
		   stress_node[i].yy /= *(node_counter + i);
		   stress_node[i].xy /= *(node_counter + i);
		   stress_node[i].I /= *(node_counter + i);
		   stress_node[i].II /= *(node_counter + i);
	  }
	}
#endif

	return 1;
}

