#include <stdlib.h>
#include <stdio.h>

#include "gene_expression.h"
#include "gene_search.h"
#include "grn.h"
#include "util.h"
/**
 *
 */
GRN *init_grn(Chromosome *chromo, Protein *input_proteins, int number_of_inputs) {
  int max_genes = 1 + chromo->length/GENE_SIZE;
  int i;

  GRN *grn = malloc(sizeof(GRN));
  grn->genes = (Gene **)malloc(max_genes*sizeof(Gene *));

  grn->chromosome = chromo;
  grn->n_input_proteins = number_of_inputs;

  /* Fint all TF Genes */
  for (i = 0; i < max_genes && (grn->genes[i] = next_tf_gene(chromo)); i++);
  grn->n_tf_genes = i;
  grn->n_tf_proteins = grn->n_tf_genes + number_of_inputs;

  /* Find all P Genes */
  grn->p_genes = grn->genes + grn->n_tf_genes;
  for (; i < max_genes && (grn->genes[i] = next_p_gene(chromo)); i++);
  grn->n_p_genes = i - grn->n_tf_genes;

  /* Build the protein pool; Express each TF gene; Add the input proteins. Finally express the P genes */
  grn->proteins = (Protein **)malloc(grn->n_tf_proteins*sizeof(Protein *));
  grn->p_proteins = (Protein **)malloc(grn->n_p_genes*sizeof(Protein *));

  for (i = 0; i < grn->n_tf_genes; i++)
    grn->proteins[i] = express(grn->genes[i]);
  for (i = 0; i < number_of_inputs; i++)
    grn->proteins[grn->n_tf_genes + i] = input_proteins + i;
  for (i = 0; i < grn->n_p_genes; i++)
    grn->p_proteins[i] = express(grn->p_genes[i]);

  /* Calculate the input protein's initial level of concentration */
  grn->input_concentration = 0.0;
  for (i = 0; i < number_of_inputs; i++)
    grn->input_concentration += (input_proteins+i)->concentration;

  /* Set initial TF concentrations */
  for (i = 0; i < grn->n_tf_genes; i++)
    grn->proteins[i]->concentration = (1.0 - grn->input_concentration)/(double)grn->n_tf_genes;

  /* Set initial P concentrations */
  for (i = 0; i < grn->n_p_genes; i++)
    grn->p_proteins[i]->concentration = 1.0/(double)grn->n_p_genes;

  return grn;
}

/**
 *
 */
void run(GRN *grn, int time_steps) {

  //Feedback loop
  int t,i;

  for (i = 0; i < grn->n_tf_genes; i++)
    printf("p%d ", i);
  for (i = 0; i < grn->n_input_proteins; i++)
    printf("i%d ", i);
  for (i = 0; i < grn->n_p_genes; i++)
    printf("o%d ", i);
  printf("\n");

  for (i = 0; i < grn->n_tf_proteins; i++)
    printf("%f ", grn->proteins[i]->concentration);
  for (i = 0; i < grn->n_p_genes; i++)
    printf("%f ", grn->p_proteins[i]->concentration);

  for (t = 0; t < time_steps; t++) {
    printf("\n");
    //Dynamics test
    /*if (t % 2000 == 0)
      for (i = 0; i < grn->n_input_proteins; i++)
        grn->proteins[grn->n_tf_genes+i]->concentration = 0.1 - grn->proteins[grn->n_tf_genes+i]->concentration;
    */

    /* Update the protein concentrations */
    int total_genes = grn->n_tf_genes + grn->n_p_genes;
    double prod_rates[total_genes];
    for (i = 0; i < grn->n_tf_genes; i++){
      printf("LOOP %d \n",i);
      prod_rates[i] = production(grn->genes[i], grn->proteins[i], grn->proteins, grn->n_tf_proteins);
    }
    for (i = 0; i < grn->n_p_genes; i++)
      prod_rates[grn->n_tf_genes + i] = production(grn->p_genes[i], grn->p_proteins[i], grn->proteins, grn->n_tf_proteins);

    for (i = 0; i < grn->n_tf_genes; i++) {
      grn->proteins[i]->concentration += prod_rates[i];
      if (grn->proteins[i]->concentration < ZERO)
        grn->proteins[i]->concentration = 0.0;
    }
    for (i = 0; i < grn->n_p_genes; i++) {
      grn->p_proteins[i]->concentration += prod_rates[grn->n_tf_genes + i];
      if (grn->p_proteins[i]->concentration < ZERO)
        grn->p_proteins[i]->concentration = 0.0;
    }

    /* Normalise TF Levels */
    double total = 0;
    for (i = 0; i < grn->n_tf_genes; i++)
      total += grn->proteins[i]->concentration;
    if (total > ZERO)
      for (i = 0; i < grn->n_tf_genes; i++) {
        grn->proteins[i]->concentration *= 1.0 - grn->input_concentration;
        grn->proteins[i]->concentration /= total;
      }

    /* Normalise P Levels */
    total = 0;
    for (i = 0; i < grn->n_p_genes; i++)
      total += grn->p_proteins[i]->concentration;
    if (total > ZERO)
      for (i = 0; i < grn->n_p_genes; i++)
        grn->p_proteins[i]->concentration /= total;
    
    /* Print data */
    /* for (i = 0; i < grn->n_tf_proteins; i++) */
    /*   printf("%f ", grn->proteins[i]->concentration); */
    /* for (i = 0; i < grn->n_p_genes; i++) */
    /*   printf("%f ", grn->p_proteins[i]->concentration); */
  }
}
