#include<stdio.h>


int main()
{
	int counter;
	int val = 255;
	int val2 = 0;
	FILE *ptr_myfile;

	ptr_myfile=fopen("test.bin","wb");
	if (!ptr_myfile)
	{
		printf("Unable to open file!");
		return 1;
	}

	for(counter=0; counter < 8; counter++){
		if (counter < 2 || counter > 5){
			fwrite(&val, 1, 1, ptr_myfile);
		}
		else{
			fwrite(&val2, 1, 1, ptr_myfile);
		}
	}
	fclose(ptr_myfile);
}
