from utils import flip

class KnowledgeBase():
    def __init__(self, kb=None):
        if kb is None:
            kb = []
        else:
            if not isinstance(kb, list):
                raise ValueError("Knowledge base must be a list.")
        self.kb = kb
        self.clause_set = set()
        for clause, _ in self.kb:  # Populate clause_set with initial kb
            self.clause_set.add(frozenset(clause))
        # print(f"clause_set: {self.clause_set}")

    def extend(self, sentence):
        if isinstance(sentence, list):
            if isinstance(sentence[0], tuple):
                self.kb.extend(sentence)
                for clause, _ in sentence:
                    self.clause_set.add(frozenset(clause))
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
                seen_literals = set()
                new_clause = []
                for lit in clause1 + clause2:
                    if lit != negated_literal and lit != literal and lit not in seen_literals:
                        new_clause.append(lit)
                        seen_literals.add(lit)

                new_clause = [(new_clause, [idx1, idx2])]
                
                if self.tautology(new_clause[0][0]):
                    continue
                if not self.clause_exists(new_clause[0][0]):
                    self.kb.extend(new_clause)
                if not new_clause[0][0]:
                    return True
        return False
    
    def tautology(self, clause):
        seen_literals = set()
        for literal in clause:
            negated_literal = flip(literal)
            if negated_literal in seen_literals:
                return True
            seen_literals.add(literal)
        return False

    def clause_exists(self, clause):
        return frozenset(clause) in self.clause_set

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
