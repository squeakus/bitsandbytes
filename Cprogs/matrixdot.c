#include <stdio.h>

void  regulate(int conc_count, float concs[conc_count],
	       int rows, size_t cols, float weights[rows][cols],
	       int tf_count, int tf_genes[tf_count],
	       int p_count, int p_genes[p_count]){
  float updates[rows];
  int signal;
  int debug = 0;

  /* printf("CONC\n"); */
  /* for(int i = 0; i < conc_count; i++){ */
  /*     printf("%f ", concs[i]); */
  /* } */

  /* printf("\nWEIGHTS\n"); */
  /* for(int i = 0; i < rows; i++){ */
  /*   for (int j = 0; j < cols; j++){ */
  /*     printf("%f ", weights[i][j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  /* Get dot product of concs and weights for each gene */
  /* and create an update array */
  for(int i = 0; i < cols; i++){
    signal = 0;
    for (int j = 0; j < rows; j++){
      signal += (concs[j] * weights[j][i]);
    }
    updates[i] = signal;
  }

  if(debug) printf("updates\n");
  for(int i = 0; i < conc_count; i++){
    if(debug) printf("%f ", updates[i]);
  }

  if (debug) printf("\nupdated\n");
  for(int i = 0; i < conc_count; i++){
    concs[i] += updates[i];
    if(debug) printf("%f ", concs[i]);
  }

  float tf_total = 0;
  for(int i =0 ; i < tf_count; i++){
    tf_total += concs[tf_genes[i]];
  }

  for(int i =0 ; i < tf_count; i++){
    concs[tf_genes[i]] = concs[tf_genes[i]] / tf_total;
  } 
  if (debug) printf("\ntf total %f\n", tf_total);

  float p_total = 0;
  for(int i =0 ; i < p_count; i++){
    p_total += concs[p_genes[i]];
  }
  for(int i =0 ; i < p_count; i++){
    concs[p_genes[i]] = concs[p_genes[i]] / p_total;
  } 

  if(debug) printf("p total %f\n", p_total);
}


int main(){
  float concs[6] = { 1, 2, 3, 4, 5, 6};
  int conc_count = sizeof(concs)/sizeof(int);
  int p_genes[2] = {1,4};
  int p_count = sizeof(p_genes)/sizeof(int);
  int tf_genes[3] = {0,2,3};
  int tf_count = sizeof(tf_genes)/sizeof(int);

  float weights[6][5] = {{ 1, 1, 1, 1, 1},
			 { 0, 0, 0, 0, 0},
			 { 1, 2, 1, 1, 1},
			 { 3, 3, 3, 3, 3},
			 { 0, 0, 0, 0, 0},
			 { 1, 1, 1, 1, 1}};

  int cols = sizeof(weights[0])/sizeof(int);
  int rows = (sizeof(weights)/cols)/sizeof(int);

  regulate(conc_count, concs, rows, cols, weights, 
	   tf_count, tf_genes, p_count, p_genes);

  printf("\nnormalised\n");
  for(int i = 0; i < conc_count; i++){
    printf("%f ", concs[i]);
  }

  return 0;
}

