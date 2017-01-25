#ifndef GENE_EXPRESSION_H
#define GENE_EXPRESSION_H

#include "typedefs.h"

#define COUNT_MASK 0x0001

Protein *express(Gene *g);

//double production(Gene **g, Protein **p, int index, int total_proteins);
double production(Gene *g, Protein *p, Protein **proteins, int total_proteins);

double enhancer_signal(Gene *g, Protein **p, int total_proteins);

double inhibitor_signal(Gene *g, Protein **p, int total_proteins);

double regulatory_signal(RegulatorySite r, Protein **p, int total_proteins);

int count_complementary_bits(RegulatorySite r, Protein *p);

#endif
