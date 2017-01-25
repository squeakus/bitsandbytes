#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* # Training Cases */
#define TRAINING 2

/* Number of Nodes/Pattern Size */
#define PSIZE 5

/**
 * Print out the node values
 */
void printNodes(int nodeValues[PSIZE]) {
  for (int i = 0; i < PSIZE; i++)
    printf("%2d ", nodeValues[i]);
  printf("\n");
}

/**
 * Set the node values to a particular pattern
 */
void setNodeValues(int nodeValues[PSIZE], const int pattern[PSIZE]) {
  for (int i = 0; i < PSIZE; i++)
    nodeValues[i] = pattern[i];
}

/**
 * Iterates the network until stable.
 * Update is random asyncronous.
 * Stable means that all nodes have been updated but did not change.
 * Stable nodes are not retested.
 * All nodes are considered unstable until retested if a change is detected.
 */
void runAsync(int nodeValues[PSIZE], double weights[PSIZE][PSIZE]) {
  int i, stableIndexes[PSIZE] = { 0 };
  double new;

  /* When stable reaches PSIZE all nodes are stable */
  for (int iteration = 1, stable = 0; stable < PSIZE;) {

    /* Randomly get a non-stable node index */
    do {
      i = rand() % PSIZE;
    } while (stableIndexes[i] == iteration);

    /* Sets the node index in stableIndexes to the current iteration
       so it won't be chosen again */
    stableIndexes[i] = iteration;

    /* Calculate the update value for the node i */
    new = 0;
    for (int j = 0; j < PSIZE; j++)
      new += weights[i][j] * nodeValues[j];

    /* If the new value matches the old, consider it stable */
    if (nodeValues[i] == (new >= 0 ? 1 : -1))
      stable++;
    /* Otherwise, apply the new value, reset the stable counter and increment
       iterations. The result of this is that all nodes are now considered
       unstable */
    else {
      nodeValues[i] = new >= 0 ? 1 : -1;
      stable = 0;
      iteration++;
    }
  }
}

/**
 * Entry Point
 */
int main(int argc, char** argv) {
  const int training[TRAINING][PSIZE] = {{-1,1,1,-1,1}, {1,-1,1,-1,1}};
  //const int training[TRAINING][PSIZE] = {{-1,-1,-1,-1,-1}};
  const int test[PSIZE] = {1,1,1,1,1};

  /* Weight Matrix */
  static double weights[PSIZE][PSIZE];

  /* Node Values */
  static int nodeValues[PSIZE];

  /* Seed rand() */
  srand(time(NULL));

  /* One-shot training - Generate weights */
  for (int t = 0; t < TRAINING; t++)
    for (int i = 0; i < PSIZE; i++)
      for (int j = 0; j < PSIZE; j++)
        weights[i][j] += i == j ? 0 : (double)(training[t][i] * training[t][j]) / TRAINING;

  /* Run the test case */
  setNodeValues(nodeValues, test);
  printNodes(nodeValues);
  runAsync(nodeValues, weights);
  printNodes(nodeValues);

  return 0;
}
