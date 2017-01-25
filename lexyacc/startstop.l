/* to compile:
lex startstop.l
cc lex.yy.c -o startstrop -ll
*/

%{
#include <stdio.h>
%}

%%
stop    printf("Stop command received\n");
start   printf("Start command received\n");
%%
