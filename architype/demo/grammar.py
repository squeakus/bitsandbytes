""" The grammar class generates output programs from an integer genome
and a BNF grammar file. It can generate trees and has BF and DF search
methods
Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
Hereby licensed under the GNU GPL v3."""

import re, copy


class Grammar(object):
    """parses BNF files, builds derivation trees and
    generates programs"""
    NT = "NT" # Non Terminal
    T = "T" # Terminal

    def __init__(self, file_name):
        self.rules = {}
        self.node_rules = []
        self.struct_rules = []
        self.non_terminals, self.terminals = set(), set()
        self.start_rule = None
        self.read_bnf_file(file_name)

    def read_bnf_file(self, file_name):
        """Read a grammar file in BNF format"""
        # <.+?> Non greedy match of anything between brackets
        non_terminal_pattern = "(<.+?>)"
        rule_separator = "::="
        prod_separator = "|"

        # Read the grammar file
        for line in open(file_name, 'r'):
            if not line.startswith("#") and line.strip() != "":
                # Split rules. Everything must be on one line
                if line.find(rule_separator):
                    nodal = True
                    lhs, productions = line.split(rule_separator)
                    lhs = lhs.strip()
                    if not re.search(non_terminal_pattern, lhs):
                        raise ValueError("lhs is not a NT:", lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule == None:
                        self.start_rule = (lhs, self.NT)
                    # Find terminals
                    tmp_productions = []
                    for production in [production.strip()
                                       for production
                                       in productions.split(prod_separator)]:
                        tmp_production = []
                        if not re.search(non_terminal_pattern, production):
                            self.terminals.add(production)
                            tmp_production.append((production, self.T))
                        else:
                            # Match non terminal or terminal pattern
                            for value in re.findall("<.+?>|[^<>]*",
                                                    production):
                                if value != '':
                                    if not re.search(non_terminal_pattern,
                                                     value):
                                        symbol = (value, self.T)
                                    else:
                                        symbol = (value, self.NT)
                                        nodal = False
                                    tmp_production.append(symbol)
                        tmp_productions.append(tmp_production)
                    # Create a rule
                    if not lhs in self.rules:
                        self.rules[lhs] = tmp_productions
                        if not nodal:
                            self.struct_rules.append(lhs)
                        else:
                            self.node_rules.append(lhs)
                    else:
                        raise ValueError("lhs should be unique", lhs)
                else:
                    raise ValueError("Each rule must be on one line")

        #print rule productions and arity
#         for rule in self.node_rules:
#             if len(self.rules[rule]) > 1:
#                 print ("nodal rule: ",str(rule),"productions:",
#                        len(self.rules[rule]))
# #                for prod in self.rules[rule]:
#  #                   print "arity:",len(prod)

#         for rule in self.struct_rules:
#             if len(self.rules[rule]) > 1:
#                 print ("struct rule: ",str(rule),"productions:",
#                        len(self.rules[rule]))

    def generate(self, genome, max_wraps=1):
        """Map genome via rules to output. Generates a derivation
        tree. Returns output and used_input"""
        used_input = 0
        wraps = 0
        struct_codons = []
        nodal_codons = []
        output = []
        derivation_tree = Tree(TreeNode(None, self.start_rule, 0, "ROOT"))
        unused_nodes = [derivation_tree.root]
        while (wraps < max_wraps) and (len(unused_nodes) > 0):
            if used_input > len(genome):
                print "using codon wrapping"
            # Wrap
            if used_input % len(genome) == 0 and used_input > 0:
                wraps += 1
            # Expand a production
            current_node = unused_nodes.pop(0)
            current_symbol = current_node.data
            #Avoiding adding the root as a child to itself
            if derivation_tree.root is not current_node:
                derivation_tree.add_child(current_node)
            derivation_tree.current_node = current_node
            # Set output if it is a terminal
            if current_symbol[1] != self.NT:
                output.append(current_symbol[0])
            else:
                production_choices = self.rules[current_symbol[0]]
                # Select a production
                current_production = self.select_production(used_input, genome,
                                                            production_choices)
                # Use an input if there was more then 1 choice
                if len(production_choices) > 1:
                    if current_symbol[0] in self.node_rules:
                        nodal_codons.append(used_input)
                    elif current_symbol[0] in self.struct_rules:
                        struct_codons.append(used_input)
                    used_input += 1
                # Set the current node to be the parent for the expansion
                for sym in reversed(production_choices[current_production]):
                    unused_nodes = ([TreeNode(derivation_tree.current_node,
                                              sym, current_node.depth + 1)]
                                    + unused_nodes)
            current_node.set_intput_data(nr=used_input - 1,
                                         value=genome[(used_input-1)
                                                      % len(genome)],
                                         choice=current_production,
                                         choices=len(production_choices))
        #Not completely expanded
        if len(unused_nodes) > 0:
            print "incomplete mapping, lengthen genome"
            values = {'phenotype': None, 'used_codons': None,
                      'nodal_codons': None, 'struct_codons': None,
                      'derivation_tree': None}
            return values
        output = "".join(output)
        #print "nodal:",len(nodal_codons),":",str(nodal_codons)
        #print "struct:",len(struct_codons),":",str(struct_codons)
        #print "used:",used_input
        values = {'phenotype': output, 'used_codons': used_input,
                 'nodal_codons': nodal_codons, 'struct_codons': struct_codons,
                 'derivation_tree': derivation_tree}
        return (values)

    def select_production(self, used_input, genome, production_choices):
        """ Factored out function for production selection to make mapping
        changes easier"""
        return genome[used_input % len(genome)] % len(production_choices)


class Tree(object):

    def __init__(self, root):
        self.root = root
        self.root._id = 0
        self.node_cnt = 1
        self.current_node = root
        self.max_depth = 1

    def add_child(self, node):
        node._id = self.node_cnt
        self.node_cnt += 1
        node.parent.append(node)
        if node.depth > self.max_depth:
            self.max_depth = node.depth

    def __str__(self):
        return self.root.__str__()

    def depth_first(self, root):
        nodes = [root]
        for child in root:
            nodes += (self.depth_first(child))
        return nodes

    def bread_first(self, root):
        nodes = [root]
        unvisited_nodes = copy.copy(root)
        while len(unvisited_nodes) > 0:
            nodes.append(unvisited_nodes.pop(0))
            for child in nodes[-1]:
                unvisited_nodes.append(child)
        return nodes

    def textual_tree_view(self):
        """String where each node is on a new line and each depth is
        indented equivialently structure more explicit,
        see DerivationNode.toString() in GEVA"""
        indent_level = -1
        tmp = self.__str__()
        i = 0
        while i < len(tmp):
            tok = tmp[i:i + 1]
            if tok == "[":
                indent_level += 1
            elif tok == "]":
                indent_level -= 1
            tabstr = "\n" + "  " * indent_level
            if tok == "[" or tok == "]":
                tmp = tmp.replace(tok, tabstr, 1)
            i += 1
        # Strip superfluous blank lines.
        txt = "\n".join([line for line in tmp.split("\n")
                         if line.strip() != ""])
        return txt


class TreeNode(list):

    def __init__(self, parent=None, data=None, depth=0,
                 _id=-1, input_nr=-1, input_value=-1,
                 input_choice=-1, input_choices=-1):
        super(TreeNode, self).__init__()
        self.parent = parent
        self.input_data = None
        self.data = data
        self._id = _id
        self.depth = depth
        self.set_intput_data(nr=input_nr, value=input_value,
                             choice=input_choices, choices=input_choice)

    def __str__(self):
        string = "[" + ":".join(map(str, ("ID", self._id, self.data,
                                      self.depth)))
        for node in self:
            string += node.__str__()
        string += "]"
        return string

    def set_intput_data(self, **kwargs):
        self.input_data = kwargs
