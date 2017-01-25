#ifndef TYPEDEFS_H
#define TYPEDEFS_H

typedef struct chromosome {
  unsigned int *codons;
  unsigned int length;
} Chromosome;

typedef unsigned int RegulatorySite;

typedef RegulatorySite Enhancer;

typedef RegulatorySite Inhibitor;

typedef RegulatorySite Promotor;

typedef struct gene {
  Enhancer e;
  Inhibitor i;
  Promotor p;
  unsigned int codons[5];
} Gene;

typedef struct protein {
  unsigned int value;
  double concentration;
} Protein;

typedef struct grn {
  Chromosome *chromosome;
  Gene **genes;
  Gene **p_genes;
  Protein **proteins;
  Protein **p_proteins;
  int n_tf_genes, n_input_proteins, n_tf_proteins, n_p_genes;
  double input_concentration;
} GRN;

#endif
