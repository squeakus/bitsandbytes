#ifndef GRN_H
#define GRN_H

#include "typedefs.h"

/**
 *
 */
GRN *init_grn(Chromosome *chromo, Protein *input_proteins, int number_of_inputs);

/**
 *
 */
void run(GRN *grn, int time_steps);

#endif
