def parse_var(var_filepath):
    try:
        with open(var_filepath, 'r') as f:
            variables = {}
            for line in f:
                var, domain = line.strip().split(':')
                variables[var] = sorted([int(x) for x in domain.split()])
        return variables
    except FileNotFoundError:
        raise FileNotFoundError(f"File {var_filepath} not found.")
    except ValueError as e:
        raise ValueError(f"Error parsing file {var_filepath}: {e}")

def parse_con(con_filepath):
    try:
        with open(con_filepath, 'r') as f:
            constraints = []
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid constraint format in line: {line.strip()}")
                var1, relation, var2 = parts
                if relation not in ['=', '!', '<', '>']:
                    raise ValueError(f"Invalid relation '{relation}' in line: {line.strip()}")
                constraints.append((var1.strip(), var2.strip(), relation.strip()))
        return constraints
    except FileNotFoundError:
        raise FileNotFoundError(f"File {con_filepath} not found.")
    except ValueError as e:
        raise ValueError(f"Error parsing file {con_filepath}: {e}")