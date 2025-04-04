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
            if isinstance(sentence[0], str):
                sentence = [sentence]
                self.kb.append(sentence)
            elif isinstance(sentence[0], list):
                self.kb.extend(sentence)
            else:
                raise ValueError("Each sentence must be a string or a list of strings.")
        elif isinstance(sentence, str):
            sentence = [sentence]
            self.kb.append(sentence)
        else:
            raise ValueError("Each sentence must be a string or a list of strings.")

    def ask(self, query):
        for sentence in self.kb:
            if sentence == query:
                return True
        return False
    
    def resolve(self, idx1, idx2):
        if idx1 >= len(self.kb) or idx2 >= len(self.kb):
            raise IndexError("Index out of range.")
        clause1 = self.kb[idx1]
        clause2 = self.kb[idx2]
        resolvents = []
        for literal in clause1:
            if literal in clause2:
                new_clause = list(set(clause1 + clause2) - {literal})
                resolvents.append(new_clause)
        return resolvents

    def __str__(self):
        return str(self.kb)
    
def parse_array(kb_array: list) -> list:
    kb = list()
    for line in kb_array:
        if not isinstance(line, str):
            raise ValueError("Each line in the knowledge base must be a string.")
        line = line.strip()
        if line:
            kb.append(line.split())
    if not kb:
        raise ValueError("Knowledge base cannot be empty.")
    return kb[:-1], kb[-1]
        