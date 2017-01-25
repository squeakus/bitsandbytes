/*
to compile:
lex example4.l
yacc -d example4.y
cc lex.yy.c y.tab.c -o example4 

IO:
heat on
        Heater on!
heat off
        Heater off!
target temperature 22
        New temperature set!
*/

%{
#include <stdio.h>
#include "y.tab.h"
%}
%%
[0-9]+                  return NUMBER;
heat                    return TOKHEAT;
on|off                  return STATE;
target                  return TOKTARGET;
temperature             return TOKTEMPERATURE;
\n                      /* ignore end of line */;
[ \t]+                  /* ignore whitespace */;
%%
