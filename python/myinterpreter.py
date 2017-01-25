"A prompt-read-eval-print loop."
def repl():
    while True:
	inval = raw_input("mypython>")
        try:
            outval = eval(inval)
        except SyntaxError:
            exec(inval)
            outval = None
        if outval is not None: print str(outval)

repl()
