from knowledge_base import KnowledgeBase, parse_array
from utils import negate

def resolution(kb_file):
    kb, query = parse_kb(kb_file)
    negated_query = negate(query)
    kb.extend(negated_query)

    kb_size = len(kb.kb)
    i = 0
    while i < kb_size:
        for j in range(0, i):
            if kb.resolve(i, j):
                print(kb.clause_set)
                return True, kb.kb
        i += 1
        kb_size = len(kb.kb)
    
    print(kb.clause_set)
    return False, kb.kb

def parse_kb(kb_file):
    kb = list()
    with open(kb_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                kb.append(line)
    knowledge_base, query = parse_array(kb)
    kb = KnowledgeBase(knowledge_base)
    return kb, query