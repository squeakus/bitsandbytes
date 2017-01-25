#include<stdio.h>


int main()
{
	FILE *ptr_myfile;

	ptr_myfile=fopen("test.bin","wb");
	if (!ptr_myfile)
	{
		printf("Unable to open file!");
		return 1;
	}

	int x = 1;
	printf("Before %d\n", x);
	fwrite(&x, 1, 1, ptr_myfile);
	fclose(ptr_myfile);


	ptr_myfile=fopen("test.bin","rb");
	if (!ptr_myfile)
	{
		printf("Unable to open file!");
		return 1;
	}

	int y;
	fread(&y,1,1,ptr_myfile);
	printf("After %d\n",y);
	fclose(ptr_myfile);
}
