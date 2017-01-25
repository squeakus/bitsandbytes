#!/usr/bin/env python
import sys
import os
import string

MERGE = "emacs"
QFLAG = "-Q"
EVAL  = "--eval"
EVALFUNC = ["ediff-merge-files-with-ancestor"]

# Order of arguments returned by Subversion
(base,theirs,mine,working) = tuple(['"' + e + '"' for e in sys.argv[1:5]])

# Order needed by Emacs Ediff
funcargs = [theirs,mine,base,"nil",working]
evalfuncstr = EVALFUNC.extend(funcargs)
evalfuncstr = "(" + string.join(EVALFUNC) + ")"

cmd = [MERGE,QFLAG,EVAL,evalfuncstr]
os.execvp(cmd[0],cmd)
