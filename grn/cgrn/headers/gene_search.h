#ifndef REG_SITES_H
#define REG_SITES_H

#include "typedefs.h"

#define GENE_SIZE 8
#define STARTING_INDEX -(GENE_SIZE - 2)

#define PROMO_MASK  0x000000FF
#define P_PROMOTER  0x00000000
#define TF_PROMOTER 0x000000FF

Gene *next_tf_gene(Chromosome *c);

Gene *next_p_gene(Chromosome *c);

void reset_tf_gene_search();

void reset_p_gene_search();

int gene_index(Chromosome *c, Gene *g);

#endif
