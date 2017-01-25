#include "gene_search.h"
#include "typedefs.h"

#include <stdio.h>
#include <stdlib.h>

/** Internal index of last found TF promoter */
static int _current_tf_promoter_index = STARTING_INDEX;

/** Internal index of last found production promoter */
static int _current_p_promoter_index = STARTING_INDEX;

/** Internal store of current chromosome being searched */
static Chromosome *_current_chromosome = NULL;

/** Used to store the location of found genes, to stop overlap */
static _Bool *_found_genes = NULL;

/**
 * Checks if the codon at this index of the Chromosome is a promoter.
 *
 * @param c The Chromosome
 * @param index Codon index to check if that codon is a promoter
 * @return True if the codon is a promoter, otherwise false
 */
_Bool is_a_promoter(Chromosome *c, int index, int mask) {
  /* if (index == 21) { */
  /*   printf("%d %d %d %d %d\n", c->codons[index], PROMO_MASK, (c->codons[index] & PROMO_MASK), mask, (c->codons[index] & PROMO_MASK) ^ mask); */
  /*   printf("%d %d", _found_genes[21-2], _found_genes[21+5]); */
  /* } */
  return !((c->codons[index] & PROMO_MASK) ^ mask);
  //Alternative: 
  //return (c->codons[index] << 24) == PROMOTER; 
  //where PROMOTER is already leftshifted.
  //Alternate is probably better.
}

/**
 * Find the next promotor (for the given mask) along the chromosome.
 * 
 * @param c Chromosome to be searched
 * @param prev_index Index along the chromosome of the previous promoter
 * @param mask The promoter mask to match with
 * @return The index of the next promoter in the chromosome, otherwise -1
 */
int next_promoter(Chromosome *c, int prev_index, int mask) {
  /* If search is beginning from the start/restarting, do some initialisation */
  if (_current_chromosome != c) {
    if (_found_genes != NULL)
      free(_found_genes);

    _found_genes = calloc(c->length, sizeof(_Bool));
    _current_chromosome = c;
  }
  if (prev_index < 0) {
    prev_index = STARTING_INDEX;
  }

  /* Increment the size of a gene from the last promotor found */
  prev_index += GENE_SIZE;

  /* Search for a gene, checking that we're within bounds, and aren't overlapping */
  while (++prev_index < c->length && (!is_a_promoter(c, prev_index, mask) ||
                                      (_found_genes[prev_index-2] || 
                                       _found_genes[prev_index+5])));

  /* Return -1 if we are out of bounds */
  if (prev_index > c->length - 6)
    prev_index = - 1;

  /* Otherwise fill out the found array so the next gene won't overlap */
  else {
    int i;
    for (i = prev_index-2; i < prev_index+6; i++)
      _found_genes[i] = 1;
  }

  return prev_index;
}

/**
 * Returns a pointer to the next Gene found in the Chromosome,
 * beginning search from the end of the last previously found Gene, or
 * if none have yet been found, from the beginning of the
 * Chromosome. If no Genes are found, NULL is returned.
 *
 * @param c  Chromosome to search
 * @return Pointer to the next Gene, or NULL if no Genes are found
 */
Gene *next_tf_gene(Chromosome *c) {
  _current_tf_promoter_index = next_promoter(c, _current_tf_promoter_index, TF_PROMOTER);
  if (_current_tf_promoter_index == -1)
    return NULL;

/* printf("Found TF Gene: %d\n", _current_tf_promoter_index); */

  return (Gene *)(c->codons+_current_tf_promoter_index - 2);
}

Gene *next_p_gene(Chromosome *c) {
  _current_p_promoter_index = next_promoter(c, _current_p_promoter_index, P_PROMOTER);
  if (_current_p_promoter_index == -1)
    return NULL;

  /* printf("Found P Gene: %d\n", _current_p_promoter_index); */

  return (Gene *)(c->codons+_current_p_promoter_index - 2);
}

/**
 * Resets the search to the beginning of the Chromosome
 */
void reset_tf_gene_search() {
  _current_tf_promoter_index = STARTING_INDEX;
}

/**
 * Resets the search to the beginning of the Chromosome
 */
void reset_p_gene_search() {
  _current_p_promoter_index = STARTING_INDEX;
}

/**
 * Find the index of the Gene in the Chromosome's codon array.
 *
 * @param c Chromosome containing the Gene
 * @param g The Gene whose index in the Chromosome is being sought
 * @return The index of the Gene in the Chromosome's codon array
 */
int gene_index(Chromosome *c, Gene *g) {
  return (unsigned int *)g - c->codons;
}

