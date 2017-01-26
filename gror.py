#! /usr/bin/env python3

# Project: Grammar Translator
# Author: Karel Benes, xbenes20@stud.fit.vutbr.cz
# Purpose: Translate grammars between the W3C SRGS and Kaldi G-WFSA
#   and possibly other formats (STK?).

# Known deviations from W3C SRGS:
#   * <token> is treated as <item>, because OpenFST cannot handle whitespaces
#                   in i/o symbols

# A grammar class.
# In general a CFG is allowed at this level.
# Grammar knows all its terminals and nonterminals and the starting symbol.
# Grammar is able to spit out its FST equivalent.
class GrammarNotReady(ValueError):
    pass

class Grammar:
    """
    A context free grammar.

    Holds the 4-tuple (N,T,P,S), where P is relation N->(N+T)*.

    Provides information on whether it is right-linear or at least
    right-non-self-embeding, which is used for WFSA synthesis.
    """

    def __init__(self):
        self.starter = None
        self.terms = {}
        self.nonterms = {}

    def add_starting(self, starter):
        """Makes the grammar remember what symbol it starts with."""
        if not type(starter) == str:
            raise TypeError("Starter has to be specified by a name")
        self.starter = starter

    def add_term(self, terminal):
        """Adds a terminal to the T component of grammar."""
        if not isinstance(terminal,Term):
            raise TypeError("Term has to be an instance of Term")
        self.terms[terminal.word] = terminal

    def add_nonterm(self, nonterminal):
        """Adds a nonterminal to the component N of grammar"""
        if not isinstance(nonterminal,NonTerm):
            raise TypeError("Nonterminal has to be an instance of NonTerm")
        self.nonterms[nonterminal.label] = nonterminal

    def complete(self):
        """Decides whether all referenced terminal and nonterminals are present"""
        if not self.starter:
            sys.stderr.write("Starter not specified\n")
            return False

        if self.starter not in self.nonterms:
            sys.stderr.write("Starter not in nonterms\n")
            return False

        for nonterm in self.nonterms.values():
            for rhs in nonterm.rules:
                for elem in rhs[0]:
                    if (elem not in self.terms) and (elem not in self.nonterms):
                        print ("Missing " + str(elem))
                        return False

        return True

    def rescale_weights(self):
        """Rescales weights so that they sum to one"""
        for nt in self.nonterms.values():
            weight_sum = 0.0
            for rhs in nt.rules:
                weight_sum += rhs[1]

            for rhs in nt.rules:
                rhs[1] /= weight_sum


    def right_linear(self):
        """
        Decides whether the grammar is right linear.

        That means, rules have the form A->abc[A], or more specificaly P: N->T*[N].

        And there is a risk that if you ask a noncomplete grammar about its
        left linearity, a tyrannousaurus may come to bite thee.
        """
        for nonterm in self.nonterms.values():
            for rhs in nonterm.rules:
                for elem in rhs[:-1]:
                    if elem not in self.terms:
                        return False

        return True

    def d_star(self):
        """
        Returns the transitive closure of the D-set.

        The D-set is a relation on N, (A,B) \in D, if A -> uBv for any u,v is in P

        Therefore the D* is shows what nonterminals may be derived from the given one.
        In terms of definitions and similar: (A,B) \in D*, if A =>+ uBv for any u,v.
        """
        # here build the D-set
        d_set = set()
        for nonterm in self.nonterms:
            for rhs in self.nonterms[nonterm].rules:
                for elem in rhs[0]:
                    if elem in self.nonterms:
                        d_set.add((nonterm,elem))

        # and here we get its transitive closure
        d_star = d_set
        while True:
            new_rels = set((x,z) for (x,y) in d_star for (w,z) in d_star if y == w)

            new_d_star = d_star | new_rels
            if new_d_star == d_star:
                break
            d_star = new_d_star
        self.d_star_data = d_star
        return d_star

    def q_set(self):
        """
        Returns the Q-set of the grammar.

        Q-set is a relation on N, (A,B) \in Q, if A -> uBv for any u and any
        nonempty v is in P.

        This captures potentialy problematic derivations that might prevent us
        from turning the grammar into an equivalent WFSA.
        """
        # here build the Q-set
        q_set = set()
        for nonterm in self.nonterms:
            for rhs in self.nonterms[nonterm].rules:
                for elem in rhs[0][:-1]:
                    if elem in self.nonterms:
                        q_set.add((nonterm,elem))
        return q_set

    def _get_last_nonterms(self):
        """
        Finds all nonterminals such, that they only derive strings of terminals.

        These can be easily removed from the grammar by backsubstituting their
        expansions into their places.
        """
        return set([x
                for x in self.nonterms
                if not set([(x,y) for y in self.nonterms]).intersection(self.d_star_data)
                ])

    def back_substitute(self):
        """
        Reduces the number of nonterminals by doing backsubstitution (BS).

        There are two separate processes of BS: In the first round, terminals
        are pushed up as far possible. However, the starting symbols is never
        removed and alternative symbols (those with more expansions) removed
        neither as this would bring redundancy into their 'father' symbols.

        The other round then deals with situation where an alternative symbol
        occurs too deep in the DAG of grammar and therefore it would block the
        removal of its parent symbols. In this round, any nonterminal with only
        one expansion rule is removed and the content of its expansion is BSed
        into its parents. The exception are recursive rules which are left
        untouched.
        """
        self.d_star() # get the actual D* into the self.d_star_data

        # "terminal backsubtitution"
        change = True
        while change:
            change = False
            one_expansioned = set([x
                                for x in self._get_last_nonterms()
                                if len(self.nonterms[x].rules)==1])
            try:
                x = self.nonterms[one_expansioned.pop()]
            except KeyError: # if there is nothing left for term backsubstitution, cease the loop
                break

#            for nonterm
            nts_to_kill = []
            for nonterm in self.nonterms:
#                for rule
                for i in range(len(self.nonterms[nonterm].rules)):
                    try:
                        ind = self.nonterms[nonterm].rules[i][0].index(x.label) # here a ValueError may be risen
                        self.nonterms[nonterm].rules[i][0][ind:ind+1] = x.rules[0][0]
                        nts_to_kill.append(x.label)
                        change = True
#                        print("term-substituting " + x.label + " into " + nonterm)
                    except ValueError:
                        pass
            self.d_star_data = set([m for m in self.d_star_data if m[0] != x.label and m[1] != x.label])

            for victim in nts_to_kill:
                del self.nonterms[victim]

        # "nonterminal backsubstitution"
        nts_to_kill = []
        for nt in self.nonterms:
            if nt == self.starter:
                continue

            if len(self.nonterms[nt].rules) == 1:
                exp = self.nonterms[nt].rules[0][0]
                w = self.nonterms[nt].rules[0][1]
                if nt in exp:
                    continue # we cannot backsubstitute self recursive rules

                for ont in self.nonterms:                           # with all Other NonTerms
                    if ont == nt:
                        continue
                    for i in range(len(self.nonterms[ont].rules)):  # go through their rules
                        for j in range(len(self.nonterms[ont].rules[i][0])): # and search them completely
                            if self.nonterms[ont].rules[i][0][j] == nt:
                                self.nonterms[ont].rules[i][0][j:j+1] = exp
                                self.nonterms[ont].rules[i][1] *= w
#                                print("nonterm-substituting " + nt + " into " + ont)

                nts_to_kill.append(nt)

        for victim in nts_to_kill:
            del self.nonterms[victim]

    def fstisable(self):
        """
        Decides whether the grammar can be turned into an equivalent WFSA.

        Note that we are not strictly precise about when this is possible:
        Generaly, the sufficent (and exact) condition is that the grammar
        has to be non-self-embedding. We require a stronger condition: There
        shall be no nonterminal A such, that is possible to derive uAv for any
        nonempty v from it.
        """
        ds = self.d_star()
        dsaq = relcomp(ds, self.q_set())
        return not bool([x for x in dsaq if (x[1],x[0]) in ds])

    def __str__(self):
        string = "Starter: " + self.starter + "\n"
        for nonterm in self.nonterms:
            string += nonterm
            for rhs in self.nonterms[nonterm].rules:
                string += " " + str(rhs)
            string += "\n"

        string += "Terms: " + str([x for x in self.terms])

        return string

# A nonterminal element.
# It knows what it can expand into, each rule's RHS is a list of references
# to (non)terminals.
# Upon creation is empty and supports rule addition.
class WrongRHS(Exception):
    pass

class NonTerm:
    def __init__(self, label):
        self.label = label
        self.rules = []

    def add_rule(self, expansion, weight=1.0):
        """Adds a rule (list of strings) to the nonterminal"""
        if type(expansion) != list:
            raise WrongRHS("RHS of a rule has to be a list!")
        for elem in expansion:
            if not isinstance(elem, str):
                raise WrongRHS("RHS has to consist names only")

        self.rules.append([expansion,weight])


# A terminal element.
# Whatever it is, it is the same when it looks the same (same word or phn),
# therefore a string identifies it enough. We only have a class for it
# in order to force Python into some type controls.
# Is only created, compared and passed to writer.
class Term:
    def __init__(self, word):
        self.word = word

    def __eq__(self, other):
        return self.word == other.word

# A general (nondeterministic with epsilon rules) FST.
# Knows to be printed for OpenFST.
# Has always one starting and exactly one ending state.
# From maths:
#   Input alphabet  -- unnecessary
#   Output alphabet -- unnecessary
#   States          -- labeled by numbers - implicitly by index
#   Starter         -- number - zero
#   Final           -- number - last one
#   Arcs            -- hidden in states

import copy
class GeneralFST:
    def _fstize_nonterm(self, grammar, nonterm, pred_index, pred_indices):
        """
        Builds an WFST (actually WFSA) for the given nonterminal in the given grammar.

        The interesting info is passed in pred_indices. It contains indices of all
        _usable_ nonterminals already synthetised. In order not to lose the "return
        address" implicitly hidden in the (W)CFG, a single nonterminal may have
        several equivalent instances in the WFST. But only those which derive the actual
        nonterminal as its rightmost symbol (in arbitrary number of derivation steps)
        are accessible. Such an access then results in a recursion, or better, a loop
        in the automaton.

        Therefore, when synthetising a rightmost nonterminal of an expansion,
        the pred_indices is passed (with the actual nonterminal appended), while
        nonterminal at any other position is given an empty dictionary.
        """

        predecessor = self.content[pred_index]
        pred_indices[nonterm.label] = pred_index

        # prepare the ender state -- every path through this
        # nonterminal will end in it
        self.content.append([])
        ender_index = len(self.content)-1

        # synthetise each RHS independently of others
        for rhs in nonterm.rules:
            self.content.append([]) # prepare first state of the RHS
            w = rhs[1]  # get the weight of the current rule

            # eps-connect the predecesor to the first state of this RHS
            # this is THE spot, where weights get into the FST
            predecessor.append((len(self.content)-1,"","",w))

            #   prepare reference to the last state of rhs
            last_in_rhs = len(self.content)-1

            for no in range(len(rhs[0])): # for each element of the expansion
                elem = rhs[0][no]
                if elem not in grammar.terms: # handle nonterminals separately
                    if elem in pred_indices:
                        self.content[last_in_rhs] = ([(pred_indices[elem],
                                                        "","",None)])
                        last_in_rhs = None  # this expansion has no end
                                            # will be connected to ender

                    else: # if we do not have an instance ready to be used
                        if no == len(rhs[0]) -1: # to rightmost, pass the pred_indices
                            ppi = copy.copy(pred_indices)
                        else:
                            ppi = {} # others shall be given an empty dict
                        last_in_rhs = self._fstize_nonterm(grammar, 
                            grammar.nonterms[elem], last_in_rhs, ppi)

                else: # now for terminal do
                    self.content.append([]) # prepare new state
        #           tell the previous to term-move into the new one
                    self.content[last_in_rhs] = ([(len(self.content)-1,elem,elem,None)])
                    last_in_rhs = len(self.content)-1

        #   if there is an end of this rhs (see the recursion branch
        #       for counterexample)
            if last_in_rhs:
            #   tell the last state to eps-move into the ender
                self.content[last_in_rhs] = ([(ender_index,"","",None)])

        # the caller needs to know what is our last state
        return ender_index

    def __init__(self, grammar):
        if not isinstance(grammar, Grammar):
            raise TypeError("A GeneralFST instance may only be built upon a Grammar")
        if not grammar.complete():
            raise GrammarNotReady("Only complete grammars may be fstized")
        if not grammar.fstisable():
            raise GrammarNotReady("Only fstisable grammars may be fstized")

        # the _fstize_nonterm needs a predecessor, so we shall give him one
        self.content = [[]]
        self._fstize_nonterm(grammar, grammar.nonterms[grammar.starter],0,{})

    def to_att(self):
        """
        Turns the automaton into AT&T FSM syntax.

        Returns list of two elements, where the first element is a string describing
        arcs of the automaton and the other shows mapping symbol <->  number
        """
        symbol_number = [""] # symbol -> number (implicit by index)
        result = ["",""]

        for i in range(len(self.content)):
            if len(self.content[i])==0:
                result[0] += str(i) + "\n"
            for arc in self.content[i]:
                sym = arc[2] # the input symbol, so far equal to the output one
                if sym not in symbol_number:
                    symbol_number.append(sym)

                sym_text = sym
                if sym == "":
                    sym_text = "eps"

                if arc[3] == None:
                    result[0] += "\t".join([str(i),str(arc[0]),sym_text,
                                        sym_text]) + "\n"
                else:
                    from math import log
                    result[0] += "\t".join([str(i),str(arc[0]),sym_text,
                                        sym_text,str(-log(arc[3]))]) + "\n"

        for i in range(len(symbol_number)):
            if symbol_number[i] == "":
                result[1] += "eps\t"+str(i)+"\n"
            else:
                result[1] += "\t".join([symbol_number[i],str(i)]) + "\n"

        return result

# An XML reader.
import xml.etree.ElementTree as ET
import os.path as OP
class XMLTree:
    def __init__(self, filename):
        self.open = [OP.abspath(filename)]
        self.closed = []

    def _elem_expansion(self, name, elem):
        """
        Proccesses single element of the XML tree, works recursively.

        This method is the keypoint of turning an XML into a WCFG. For every
        element (<item>,<one-of> and so on and so forth) a rule is set on how
        it shall appear in the grammar. Note that with the etree internals,
        when processing a container element such as item or rule, every child's
        tail must be examined, because additional text may appear there.

        This function is expected to be first invoked upon a rule, then it goes
        deeper and deeper. To every child, this function passes a unique name,
        created by appending a dash-separated numerical suffix. This is "sure"
        to be a safe name as dash ('-') is a forbidden character in the rule name.
        """

        # create the new nonterm
        this_nonterm = NonTerm(name)

        if elem.tag.endswith("one-of"):
            for i in range(len(elem)): # for each alternative
                new_nonterm_name = name+"-"+str(i)
                self._elem_expansion(new_nonterm_name,elem[i]) # have a single nonterm
                try: # to find a weight of the rule
                    w = float(elem[i].attrib['weight'])
                except KeyError:    # if it is not there
                    w = 1.0         # set it to the default (see SRGS) 1.0
                this_nonterm.add_rule([new_nonterm_name], w)

        elif elem.tag.endswith("ruleref"):
            if 'uri' in elem.attrib:
                split_form = elem.attrib['uri'].split("#")
                if len(split_form) != 2:
                    raise ValueError("A ruleref URI has to have exactly one '#' (Got: '" +
                        str(elem) + "')")
                path, rulename = split_form
                if path != "":
                    full_name = OP.join(OP.dirname(self.active),path)
                    if full_name not in self.open + self.closed:
                        self.open.append(full_name)
                    rule = [full_name + ":" + rulename]
                else:
                    rule = [self.active + ":" + rulename]


            elif 'special' in elem.attrib:
                if elem.attrib['special'] == 'NULL':
                    rule = []
                elif elem.attrib['special'] == 'VOID':
                    rule = [name] # infinite recursion -> never matches anything
                else:
                    raise ValueError("Unsupported special rule: " + elem.attrib['special'])
            else:
                raise ValueError("Either a uri or a special MUST be given for a ruleref")

            this_nonterm.add_rule(rule, 1.0)

        elif elem.tag.endswith("item") or elem.tag.endswith("rule") or elem.tag.endswith("token"):
            if elem.tag.endswith("rule") and '-' in name.split(':')[1]:
                raise ValueError("Nested rules are not allowed! (rulename" +
                 name +")")
            
            ch_no = 0
            rule = []

            text = elem.text # text is the content till the first element
            if text and len(text) > 0:
                for word in text.split():
                    term = Term(word)
                    rule.append(word)
                    self.grammar.add_term(term)

            for child in elem:
                self._elem_expansion(name+"-"+str(ch_no), child)
                rule.append(name+"-"+str(ch_no))
                ch_no += 1

                text = child.tail
                if text and len(text) > 0:
                    for word in text.split():
                        term = Term(word)
                        rule.append(word)
                        self.grammar.add_term(term)

            if "repeat" in elem.attrib:
                if '-' in elem.attrib["repeat"]:
                    [lb, ub] = elem.attrib["repeat"].split("-")
                    lb = int(lb)

                    if ub != '': # so we have the form n-m
                        try:
                            rep_p = float(elem.attrib["repeat-prob"])
                        except KeyError:
                            rep_p = 1.0
                        ub = int(ub)
                        for i in range(lb, ub+1):
                            from math import pow
                            this_nonterm.add_rule(rule*i,pow(rep_p, i-lb))
                    else: # upper bound is unspecified ~ infinity
                        try:
                            rep_p = float(elem.attrib["repeat-prob"])
                        except KeyError:
                            rep_p = 0.5
                        repeater = NonTerm(name + ":")
                        repeater.add_rule(rule + [repeater.label],rep_p)
                        repeater.add_rule([],1-rep_p)
                        self.grammar.add_nonterm(repeater)
                        this_nonterm.add_rule(rule*lb+[repeater.label],1.0)

                else: # there is no '-' in the repeat specification
                    rep = int(elem.attrib["repeat"])
                    this_nonterm.add_rule(rule*rep,1.0)

            else: # there is no "repeat" attribute
                this_nonterm.add_rule(rule,1.0)

        # this effectively means ignoring the tags completely
        elif elem.tag.endswith("tag"):
            this_nonterm.add_rule([],1.0)

        elif elem.tag.endswith("example"):
            this_nonterm.add_rule([],1.0)

        else:
            raise ValueError("Unsupported " + elem.tag + " element.")

        # add the nonterm to the grammar
        self.grammar.add_nonterm(this_nonterm)

    def to_grammar(self):
        """
        Returns an WCFG matching the XML description.
       
        Following all the links in </ruleref>'s uri attribute, this procedure
        goes searchers a space (DAG) of linked XML files. This is done in
        the breadth-first manner, keeping OPEN and CLOSED list known from AI,
        together with implementation-required ACTIVE reference (filename) of
        the actualy processed node (file).

        Note that this process may crash severly for several reasons,
        as goes through all the needed files with grammars: it may be impossible
        to open those for reading or they may contain corrupted XML (in terms
        of ETree parsing). And, of course, these may contain unsupported
        constructions, which are otherwise well-formed.
        """
        self.grammar = Grammar()
        first_file = True
        while len(self.open) > 0:
            self.active = self.open[0]
            try:
                infile = open(self.active)
            except IOError:
                sys.stderr.write("Gror: Failed to open file: " + self.active + " with required grammar specification.\n")
                exit(1)
            self.root = ET.fromstring(infile.read())

            self.open = self.open[1:]
            self.closed.append(self.active)
            for rule in self.root:
                self._elem_expansion(self.active + ":" + rule.attrib["id"], rule)

            if first_file:
                self.grammar.add_starting(self.active + ":" + self.root.attrib["root"])
                first_file = False

        return self.grammar

def relcomp(a,b): # composition of a after b
    """Returns composition of a after b."""
    return set([(x[0],y[1]) for x in a for y in b if x[1] == y[0]])

import sys
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform SRGS XML into AT&T FST')
    parser.add_argument("--xml", dest='xml_f',
        help='file with xml grammar description')
    parser.add_argument("--att", dest='att_f',
        help='file to store the AT&T into')
    parser.add_argument("--isymbols", dest='isym_f',
        help='file to store input symbol table')
    parser.add_argument("--osymbols", dest='osym_f',
        help='file to store output symbol table')

    args = parser.parse_args()

    if not args.xml_f:
        sys.stderr.write("Gror: Error: stdin not allowed as input, I'm sorry\n")
        exit(1)

    try:
        tree = XMLTree(args.xml_f)
    except Exception as e:
        sys.stderr.write("Gror: Failed to parse the XML: " + str(e) + "\n")
        exit(1)

    grammar = tree.to_grammar()
    grammar.back_substitute()
    grammar.rescale_weights()
    fst = GeneralFST(grammar)
    textual = fst.to_att()

    if args.att_f:
        try:
            outfile = open(args.att_f,"w")
        except IOError as e:
            sys.stderr("Gror: Failed to open " + args.att_f + " for writing: " +
                str(e) + "\n")
            exit(1)
    else:
        outfile = sys.stdout


    outfile.write(textual[0])
    if args.isym_f:
        try:
            open(args.isym_f,"w").write(textual[1])
        except IOError() as e:
            sys.stderr("Gror: Failed to write isymbols into " + args.isym_f + ": " +
                str(e) + "\n")
    else:
        sys.stderr.write("Gror: Input symbol table not written anywhere.\n")

    if args.osym_f:
        try:
            open(args.osym_f,"w").write(textual[1])
        except IOError() as e:
            sys.stderr("Gror: Failed to write isymbols into " + args.osym_f + ": " +
                str(e) + "\n")
    else:
        sys.stderr.write("Gror: Output symbol table not written anywhere.\n")
