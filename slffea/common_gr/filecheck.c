/*
    This subroutine determines the existence and examines the
    names of input and output data files for every FEM GUI program.

                  Last Update 2/4/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int input_flag, post_flag, mode_choice, nmode;

int filecheck( char *name, char *name2, FILE **o1, FILE **o2, FILE **o3,
	char *file_exten, int exten_length )
{
	int i, j, check, name2_length, input_name_length,
		output_name_length, dum;
	char *ccheck, mod_exten[5], out[30], buf[ BUFSIZ ];

/* Initialize extensions*/

	memset(mod_exten,0,5*sizeof(char));
	memset(out,0,30*sizeof(char));

	ccheck = strncpy(mod_exten,".mod-",5);
	if(!ccheck) printf( " Problems with strncpy \n");

	name2_length = strlen(name2);

/*   *o1 contains all the structural data for input
     *o3 contains all the structural data for postprocessing
     *o2 is used to determine the existance of input and post files
*/

/* Every post processor file is named after the input file with an exten_length
   long character extension consisting of a period and the three letters
   identifying the element type */

	check = strncmp((name2 + name2_length - exten_length), file_exten, exten_length);

	if( !check )
	{
/* *o2 = *o3 equals post file */
		output_name_length = name2_length;
		ccheck = strncpy(out, name2, output_name_length);
		if(!ccheck) printf( " Problems with strncpy \n");
		/*printf( "out %30s\n ",out);*/
		*o3 = fopen( out,"r" );

/* Look for standard input file */

		input_name_length = name2_length - exten_length;
		ccheck = strncpy(name, name2, input_name_length);
		if(!ccheck) printf( " Problems with strncpy \n");
		printf( "name %30s\n ",name);
		*o1 = fopen( name,"r" );
		if(*o1 == NULL ) {

/* Look for modal analysis input file if nmode != 0.  For the case where
   (filename).mod-x.out and x < 10
 */
		    input_name_length -= 6;
		    if(nmode && input_name_length > 0)
		    {
			memset(name,0,30*sizeof(char));
			ccheck = strncpy(name, name2, input_name_length);
			if(!ccheck) printf( " Problems with strncpy \n");
			*o1 = fopen( name,"r" );
			if(*o1 == NULL ) {

/* Look again for modal analysis input file if nmode != 0.  For the case where 
   (filename).mod-x.out and 9 < x < 100 
 */

			    input_name_length -= 1;
			    if( input_name_length > 0)
			    {
				memset(name,0,30*sizeof(char));
				ccheck = strncpy(name, name2, input_name_length);
				if(!ccheck) printf( " Problems with strncpy \n");
				*o1 = fopen( name,"r" );
				if(*o1 == NULL ) {

/* Look again for modal analysis input file if nmode != 0.  For the case where  
   (filename).mod-x.out and 99 < x < 1000
 */
				    input_name_length -= 1;
				    if( input_name_length > 0)
				    {
					memset(name,0,30*sizeof(char));
					ccheck = strncpy(name, name2, input_name_length);
					if(!ccheck) printf( " Problems with strncpy \n");
					*o1 = fopen( name,"r" );
					if(*o1 == NULL ) {
					    printf("There is no input file %30s\n",name);
					    input_flag = 0;
					}
				    }
				    else
				    {
					printf("There is no input file %30s\n",name);
					input_flag = 0;
				    }
				}
			    }
			    else
			    {
				printf("There is no input file %30s\n",name);
				input_flag = 0;
			    }
			}
		    }
		    else
		    {
			printf("There is no input file %30s\n",name);
			input_flag = 0;
		    }
		}
	}
	else
	{
/* *o2 = *o1 equals input file */

		input_name_length = name2_length;
		ccheck = strncpy(name, name2, input_name_length);
		if(!ccheck) printf( " Problems with strncpy \n");
		*o1 = fopen( name,"r" );

		ccheck = strncpy(out, name, name2_length);
		if(!ccheck) printf( " Problems with strncpy \n");

/* Look for modal analysis ouput file if nmode != 0 */
		if(nmode)
		{
		    printf("This seems to be modal analysis data. \n");
		    printf("What mode would you like to see?(Choose number or \n");
		    printf("choose 0 for static analysis file) \n");
		    scanf( "%d",&mode_choice);
		    if(mode_choice)
		    {
			ccheck = strncpy(out+input_name_length, mod_exten, 5);
			if(!ccheck) printf( " Problems with strncpy \n");

			dum = 6;
			if(mode_choice > 9 ) dum = 7;
			if(mode_choice > 99 ) dum = 8;

			sprintf( (out+input_name_length+5), "%d", mode_choice);
			ccheck = strncpy((out+input_name_length+dum), file_exten, exten_length);
			if(!ccheck) printf( " Problems with strncpy \n");
			printf("This is the post file: %30s\n",out);

			*o3 = fopen( out,"r" );
			if(*o3 == NULL ) {
				printf("There is no post file %30s\n",out);
				post_flag = 0;
			}
		    }
		    else
		    {
/* If mode_choice = 0, Look for standard output file */
			ccheck = strncpy(out+input_name_length, file_exten, exten_length);
			if(!ccheck) printf( " Problems with strncpy \n");

			*o3 = fopen( out,"r" );

			if(*o3 == NULL ) {
				printf("There is no post file %30s\n",out);
				post_flag = 0;
			}
		    }
		}
		else
		{
/* Look for standard output file if nmode = 0 */

			ccheck = strncpy(out+input_name_length, file_exten, exten_length);
			if(!ccheck) printf( " Problems with strncpy \n");

			*o3 = fopen( out,"r" );

			if(*o3 == NULL ) {
				printf("There is no post file %30s\n",out);
				post_flag = 0;
			}
		}
	}

	return 1;
}

