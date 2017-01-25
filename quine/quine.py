header = "\t0\n};\n\n"
d = header + open("bootstrap.c").read()
for c in d:
    print "\t%s," % repr(c)
print "\t0"
