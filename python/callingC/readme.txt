gcc -fPIC -c dist.c <- flag position independent content
gcc -shared dist.o -o libdist.so  <- create lib
move to /usr/lib et voila!

