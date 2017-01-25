#include <assert.h>
#include <math.h>
//#include <popcntintrin.h>
#include <stdlib.h>
#include <stdio.h>

#include "gene_expression.h"
#include "util.h"

#define PRINT_GENE_EXPRESSION
#undef PRINT_GENE_EXPRESSION

/** */
static double scaling_factor = 1.0;

/** */
static double delta = 1.0;


/**
 *
 *
 * @param
 * @return 
 */
double phi(double x) {
  return 0.0;
}

/**
 *
 *
 * @param
 * @return 
 */
Protein *express(Gene *g) {
  Protein *p = (Protein *)malloc(sizeof(Protein));
  int bits[32] = {0};
  int i,j;
  for (i = 0; i < 5; i++) {
    int codon = g->codons[i];
    for (j = 0; j < 32; j++) {
      bits[j] += (codon >> j) & 1; 
    }
  }

  p->value = 0;
  for (j = 31; j > -1; j--) {
    p->value <<= 1;
    p->value += bits[j] > 2 ? 1 : 0;
  }

#ifdef PRINT_GENE_EXPRESSION
  char ge[33];
  char gi[33];
  char gp[33];
  char gc0[33];
  char gc1[33];
  char gc2[33];
  char gc3[33];
  char gc4[33];
  char pv[33];
  itoa(g->e, ge, 2);
  itoa(g->i, gi, 2);
  itoa(g->p, gp, 2);
  itoa(g->codons[0], gc0, 2);
  itoa(g->codons[1], gc1, 2);
  itoa(g->codons[2], gc2, 2);
  itoa(g->codons[3], gc3, 2);
  itoa(g->codons[4], gc4, 2);
  itoa(p->value, pv, 2);
  printf("Gene: %s %s %s\n%s\n%s\n%s\n%s\n%s\n\n%s\n\n", ge,gi,gp,gc0,gc1,gc2,gc3,gc4,pv);
#endif

  return p;
}

/**
 *
 *
 * @param
 * @param
 * @return 
 */
double production(Gene *g, Protein *p, Protein **proteins, int total_proteins) {
  return (delta * (enhancer_signal(g, proteins, total_proteins) - inhibitor_signal(g, proteins, total_proteins)) * p->concentration - phi(1.0));
}

/**
 *
 *
 * @param
 * @param
 * @param
 * @return 
 */
double enhancer_signal(Gene *g, Protein **p, int total_proteins) {
  //printf("Getting enhancer signal: %d\n", g->e);
  return regulatory_signal(g->e, p, total_proteins);
}

/**
 *
 *
 * @param
 * @param
 * @param
 * @return 
 */
double inhibitor_signal(Gene *g, Protein **p, int total_proteins) {
  //printf("Getting inhibitor signal: %d\n", g->i);
  return regulatory_signal(g->i, p, total_proteins);
}

/**
 *
 *
 * @param
 * @param
 * @param
 * @return 
 */
double regulatory_signal(RegulatorySite r, Protein **p, int total_proteins) {
  double signal = 0.0;
  unsigned int i;
  
  int max_cbits = -1;
  int cbits[total_proteins];
  for (i = 0; i < total_proteins; i++) {
    cbits[i] = count_complementary_bits(r, p[i]);
    if(cbits[i] > max_cbits)
      max_cbits = cbits[i];
  }

#ifdef PRINT_GENE_EXPRESSION
  printf("max_cbits: %d\n", max_cbits);
#endif

  for (i = 0; i < total_proteins; i++) {
    signal += p[i]->concentration * exp(scaling_factor * (cbits[i] - max_cbits));

#ifdef PRINT_GENE_EXPRESSION
    char buffer[33], pbuffer[33];
    itoa(r, buffer, 2); itoa(p[i]->value, pbuffer, 2);
    printf("%s\t%f\t%f\n%s\t%d\n\n", pbuffer, p[i]->concentration, p[i]->concentration * exp(scaling_factor * (cbits[i] - max_cbits)), buffer, cbits[i]);
#endif
  }

#ifdef PRINT_GENE_EXPRESSION
  printf("final signal: %f\n\n", signal/(double)total_proteins);
#endif
  return signal/(double)total_proteins;
}

/**
 *
 *
 * @param
 * @param
 * @return 
 */
int count_complementary_bits(RegulatorySite r, Protein *p) {
  unsigned int counter = 0;
  unsigned int v = r ^ p->value;
  while (v) {
    counter++;
    v &= v - 1;
  }
  return counter;
  
  //return _mm_popcnt_u32(r ^ p->value);
}
