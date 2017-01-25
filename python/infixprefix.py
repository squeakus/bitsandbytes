def parse(s):
        for operator in ["+-", "*/"]:
            depth = 0
            for p in xrange(len(s) - 1, -1, -1):
                if s[p] == ')': depth += 1
                if s[p] == '(': depth -= 1
                if not depth and s[p] in operator:
                    return [s[p]] + parse(s[:p]) + parse(s[p+1:])
        s = s.strip()
        if s[0] == '(':
            return parse(s[1:-1])
        return [s]

prefix_list = parse("x0+(x0*x0)+(x0*x0*x0)+(x0*x0*x0*x0)")
prefix_string = ""
for elem in prefix_list:
    prefix_string = prefix_string + elem + ' '

print prefix_string 
