from utils import flip

class KnowledgeBase():
    def __init__(self, kb=None):
        if kb is None:
            kb = []
        else:
            if not isinstance(kb, list):
                raise ValueError("Knowledge base must be a list.")
        self.kb = kb

    def extend(self, sentence):
        if isinstance(sentence, list):
            if isinstance(sentence[0], tuple):
                self.kb.extend(sentence)
            elif isinstance(sentence[0], list):
                for item in sentence:
                    self.kb.append((item, []))
            else:
                raise ValueError("Invalid sentence format.")
        else:
            raise ValueError("Sentence must be a list.")

    def ask(self, query):
        for sentence in self.kb:
            if sentence == query:
                return True
        return False
    
    def resolve(self, idx1, idx2):
        if idx1 >= len(self.kb) or idx2 >= len(self.kb):
            raise IndexError("Index out of range.")
        clause1 = self.kb[idx1][0]
        clause2 = self.kb[idx2][0]

        for literal in clause1:
            negated_literal = flip(literal)
            if negated_literal in clause2:
                new_clause = []
                for lit in clause1 + clause2:
                    if lit not in new_clause and lit != negated_literal and lit != literal:
                        new_clause.append(lit)

                new_clause = [(new_clause, [idx1, idx2])]
                
                if self.tautology(new_clause[0][0]):
                    continue
                if not self.clause_exists(new_clause[0][0]):
                    self.kb.extend(new_clause)
                if new_clause[0][0] == []:
                    return True
        return False
    
    def tautology(self, clause):
        for literal in clause:
            negated_literal = flip(literal)
            if negated_literal in clause:
                return True
        return False

    def clause_exists(self, clause):
        for item in self.kb:
            set1 = set(item[0])
            set2 = set(clause)
            if set1 == set2:
                return True
        return False

    def __str__(self):
        return str(self.kb)
    
def parse_array(kb_array: list) -> list:
    kb = list()
    for line in kb_array:
        if not isinstance(line, str):
            raise ValueError("Each line in the knowledge base must be a string.")
        line = line.strip()
        if line:
            kb.append((line.split(), []))
    if not kb:
        raise ValueError("Knowledge base cannot be empty.")
    return kb[:-1], kb[-1]
        