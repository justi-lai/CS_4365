from knowledge_base import KnowledgeBase, parse_array
from utils import negate, is_not, flip

def resolution(kb_file):
    kb, query = parse_kb(kb_file)
    print("Knowledge Base:")
    print(kb)
    print("Query:")
    print(query)
    negated_query = negate(query)
    print("Negated Query:")
    print(negated_query)

    kb.extend(negated_query)
    print("Updated Knowledge Base:")
    print(kb)

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