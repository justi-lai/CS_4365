def negate(expr):
    if isinstance(expr, tuple):
        if len(expr) == 1:
            return [(flip(expr[0][0]), [])]
        elif len(expr) >= 2:
            negated_expr = []
            clause, parents = expr
            for item in clause:
                if not isinstance(item, str):
                    raise ValueError("Each item in the expression list must be a string.")
                negated_item = flip(item)
                negated_expr.append(([negated_item], parents))
            return negated_expr
    elif isinstance(expr, str):
        return flip(expr)
    else:
        raise ValueError("Expression must be a string or a list of strings.")

def flip(var):
    if not isinstance(var, str):
        raise ValueError("Variable must be a string.")
    if is_not(var):
        return var[1:]
    else:
        return '~' + var[:]

def is_not(var):
    if not isinstance(var, str):
        raise ValueError("Variable must be a string.")
    if var[0] == '~':
        return True
    return False
    