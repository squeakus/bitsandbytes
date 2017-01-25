// MAKE SURE THE EQNS IN GENE_EXPRESSION MATCH THE BANZHAF PAPER

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "gene_search.h"
#include "gene_expression.h"
#include "grn.h"
#include "typedefs.h"

#define TIME_STEPS 10000
#define N_GENES 10

int main(int argc, char **argv) {

  int seed = 1;
  if (argc > 1)
    seed = atoi(argv[1]);

  Chromosome *chromo = malloc(sizeof(Chromosome));
  chromo->codons = malloc(N_GENES*sizeof(int));

  int i;
  srand(seed);
  for (i = 0; i < N_GENES; i++)
    chromo->codons[i] = rand();
  chromo->length = N_GENES;  


  //for (i = 0; i < N_GENES; i++)
  //  printf("%d ", chromo->codons[i]);
  //printf("\n");
  //return 0;

  Protein *p = malloc(sizeof(Protein)*4);
  p->value = 0x0000;
  p->concentration = 0.025;
  (p+1)->value = 0x0000FFFF;
  (p+1)->concentration = 0.05;
  (p+2)->value = 0xFFFF0000;
  (p+2)->concentration = 0.1;
  (p+3)->value = 0xFFFFFFFF;
  (p+3)->concentration = 0.075;
  int curtime = time(NULL);
  
  for (i = 0; i < 100; i++){
    GRN *grn = init_grn(chromo, p, 4);
    run(grn, TIME_STEPS);
  }
  
  int endtime = time(NULL);
  int total = endtime - curtime;
  printf("the time is %d", total);  
  return 0;
}
