/*
    This utility function calculates the vectors in the fiber direction
    for each node in the case of a singly curved shell.  It does it by
    taking the cross product of the 2 adjacent sides which share the node.
    Because there are multiple elements which may share the node, averaging
    is done at the end based on the number of elements which share the
    node.

		Updated 3/25/05

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005  San Le 

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

extern int numel, numnp, sof;

int normal(double *);

int normcrossX(double *, double *, double *);

int matXT(double *, double *, double *, int, int, int);

int shfiber(int *connect, double *coord, double *fiber_vec, double *node_counter)
{
        int i, i1, i2, j, k, dof_el[neqel20], sdof_el[npel8*nsd];
	int check, counter, node;
	int matl_num;
	double fdum1, fdum2, fdum3, fdum4;
        double coord_el_trans[npel8*nsd], zm1[npell4], zp1[npell4],
		znode[npell4*num_ints], dzdt_node[npell4];
        double vec_dum1[nsd], vec_dum2[nsd], vec_dum3[nsd];

        for( k = 0; k < numel; ++k )
        {

/* Create the coord transpose vector and other variables for one element */

                for( j = 0; j < npell4; ++j )
                {
			node = *(connect+npell4*k+j);

			*(sdof_el+nsd*j)=nsd*node;
			*(sdof_el+nsd*j+1)=nsd*node+1;
			*(sdof_el+nsd*j+2)=nsd*node+2;

/* Count the number of times a particular node is part of an element */

			*(node_counter+node) += 1.0;

/* Create the coord -/+*/

                        *(coord_el_trans+j) =
				*(coord+*(sdof_el+nsd*j));
                        *(coord_el_trans+npell4*1+j) =
				*(coord+*(sdof_el+nsd*j+1));
                        *(coord_el_trans+npell4*2+j) =
				*(coord+*(sdof_el+nsd*j+2));

                printf("cccc %14.5e %14.5e %14.5e\n", *(coord_el_trans+j),
			*(coord_el_trans+npell4*1+j),
			*(coord_el_trans+npell4*2+j));

		}

/* Calculate the fiber vectors */

/* for node 0 */
		*(vec_dum1) = *(coord_el_trans+1) - *(coord_el_trans);
		*(vec_dum1+1) = *(coord_el_trans+npell4+1) - *(coord_el_trans+npell4);
		*(vec_dum1+2) = *(coord_el_trans+2*npell4+1) - *(coord_el_trans+2*npell4);

		*(vec_dum2) = *(coord_el_trans+3) - *(coord_el_trans);
		*(vec_dum2+1) = *(coord_el_trans+npell4+3) - *(coord_el_trans+npell4);
		*(vec_dum2+2) = *(coord_el_trans+2*npell4+3) - *(coord_el_trans+2*npell4);

                printf("vvvv %14.5e %14.5e %14.5e\n", *(vec_dum1), *(vec_dum1+1), *(vec_dum1+2));
                printf("vvvv %14.5e %14.5e %14.5e\n", *(vec_dum2), *(vec_dum2+1), *(vec_dum2+2));
		check = normcrossX(vec_dum1, vec_dum2, vec_dum3);
		if(!check) printf( "Problems with normcrossX \n");
		*(fiber_vec+*(sdof_el)) += *(vec_dum3);
		*(fiber_vec+*(sdof_el+1)) += *(vec_dum3+1);
		*(fiber_vec+*(sdof_el+2)) += *(vec_dum3+2);
                printf("1111 %14.5e %14.5e %14.5e\n", *(fiber_vec+*(sdof_el)),
			*(fiber_vec+*(sdof_el+1)),
			*(fiber_vec+*(sdof_el+2)));
/* for node 1 */
		*(vec_dum1) = *(coord_el_trans+2) - *(coord_el_trans+1);
		*(vec_dum1+1) = *(coord_el_trans+npell4+2) - *(coord_el_trans+npell4+1);
		*(vec_dum1+2) = *(coord_el_trans+2*npell4+2) - *(coord_el_trans+2*npell4+1);

		*(vec_dum2) = *(coord_el_trans) - *(coord_el_trans+1);
		*(vec_dum2+1) = *(coord_el_trans+npell4) - *(coord_el_trans+npell4+1);
		*(vec_dum2+2) = *(coord_el_trans+2*npell4) - *(coord_el_trans+2*npell4+1);

		check = normcrossX(vec_dum1, vec_dum2, vec_dum3);
		if(!check) printf( "Problems with normcrossX \n");
		*(fiber_vec+*(sdof_el+nsd*1)) += *(vec_dum3);
		*(fiber_vec+*(sdof_el+nsd*1+1)) += *(vec_dum3+1);
		*(fiber_vec+*(sdof_el+nsd*1+2)) += *(vec_dum3+2);
                printf("2222 %14.5e %14.5e %14.5e\n", *(fiber_vec+*(sdof_el+nsd*1)),
			*(fiber_vec+*(sdof_el+nsd*1+1)),
			*(fiber_vec+*(sdof_el+nsd*1+2)));
/* for node 2 */
		*(vec_dum1) = *(coord_el_trans+3) - *(coord_el_trans+2);
		*(vec_dum1+1) = *(coord_el_trans+npell4+3) - *(coord_el_trans+npell4+2);
		*(vec_dum1+2) = *(coord_el_trans+2*npell4+3) - *(coord_el_trans+2*npell4+2);

		*(vec_dum2) = *(coord_el_trans+1) - *(coord_el_trans+2);
		*(vec_dum2+1) = *(coord_el_trans+npell4+1) - *(coord_el_trans+npell4+2);
		*(vec_dum2+2) = *(coord_el_trans+2*npell4+1) - *(coord_el_trans+2*npell4+2);

		check = normcrossX(vec_dum1, vec_dum2, vec_dum3);
		if(!check) printf( "Problems with normcrossX \n");
		*(fiber_vec+*(sdof_el+nsd*2)) += *(vec_dum3);
		*(fiber_vec+*(sdof_el+nsd*2+1)) += *(vec_dum3+1);
		*(fiber_vec+*(sdof_el+nsd*2+2)) += *(vec_dum3+2);
                printf("3333 %14.5e %14.5e %14.5e\n", *(fiber_vec+*(sdof_el+nsd*2)),
			*(fiber_vec+*(sdof_el+nsd*2+1)),
			*(fiber_vec+*(sdof_el+nsd*2+2)));
/* for node 3 */
		*(vec_dum1) = *(coord_el_trans) - *(coord_el_trans+3);
		*(vec_dum1+1) = *(coord_el_trans+npell4) - *(coord_el_trans+npell4+3);
		*(vec_dum1+2) = *(coord_el_trans+2*npell4) - *(coord_el_trans+2*npell4+3);

		*(vec_dum2) = *(coord_el_trans+2) - *(coord_el_trans+3);
		*(vec_dum2+1) = *(coord_el_trans+npell4+2) - *(coord_el_trans+npell4+3);
		*(vec_dum2+2) = *(coord_el_trans+2*npell4+2) - *(coord_el_trans+2*npell4+3);

		check = normcrossX(vec_dum1, vec_dum2, vec_dum3);
		if(!check) printf( "Problems with normcrossX \n");
		*(fiber_vec+*(sdof_el+nsd*3)) += *(vec_dum3);
		*(fiber_vec+*(sdof_el+nsd*3+1)) += *(vec_dum3+1);
		*(fiber_vec+*(sdof_el+nsd*3+2)) += *(vec_dum3+2);

                printf("4444 %14.5e %14.5e %14.5e\n", *(fiber_vec+*(sdof_el+nsd*3)),
			*(fiber_vec+*(sdof_el+nsd*3+1)),
			*(fiber_vec+*(sdof_el+nsd*3+2)));
#if 0
                	fprintf(oB,"\n\n\n");
                	fprintf(oS,"\n\n\n");
                	for( i3 = 0; i3 < 3; ++i3 )
                	{
                   	    fprintf(oB,"\n ");
                   	    fprintf(oS,"\n ");
                   	    for( i2 = 0; i2 < neqel20; ++i2 )
                   	    {
                            	fprintf(oB,"%14.5e ",*(B+neqel20*i3+i2));
                            	fprintf(oS,"%14.5e ",*(DB+neqel20*i3+i2));
                   	    }
                	}
#endif


	}

/* Average all the components of the fiber vectors at the nodes */

	for( i = 0; i < numnp ; ++i )
	{
		*(fiber_vec+nsd*i) /= *(node_counter+i);
		*(fiber_vec+nsd*i+1) /= *(node_counter+i);
		*(fiber_vec+nsd*i+2) /= *(node_counter+i);
		check = normal((fiber_vec+nsd*i));
		if(!check) printf( "Problems with normal \n");
	}

	memset(node_counter,0,2*numnp*sof);

	return 1;
}

