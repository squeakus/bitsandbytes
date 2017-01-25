#include <stdio.h>
#include <math.h>

//Function Declarations

int print_array (int*, int*, int);
void test();
//Distance Function

void test(){
  printf("hey! im working fine!\n");
}

int print_array(int* array, int* result, int size){	
  int i;
  printf("array length %d\n", size);
  for(i = 0; i < size; i++){
    printf("idx %d = %d\n",i, array[i]);
    result[i] = array[i]+1;
  }
}
