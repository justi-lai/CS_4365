import utils
from copy import deepcopy

class Output():
    def __init__(self):
        self.output = list()

    def append(self, assignment, result):
        self.output.append((str(assignment), result))

    def clear(self):
        self.output.clear()

    def __str__(self):
        string = ""
        for num, x in enumerate(self.output):
            assignment, result = x
            status = "solution" if result else "failure"
            string += f"{num+1}. {assignment}  {status}\n"
        return string.strip()

output = Output()

def solve_csp(var_filepath, con_filepath, procedure):
    problem = parse_problem(var_filepath, con_filepath)
    initial_assignment = Assignment()
    if procedure == "none":
        recursive_backtracking(initial_assignment, problem)
        print(output)
        output.clear()
    elif procedure == "fc":
        forward_checking(initial_assignment, problem)
        print(output)
        output.clear()
    else:
        raise ValueError(f"Unknown consistency procedure: {procedure}")

def recursive_backtracking(assignment, problem):
    if assignment.complete(problem):
        output.append(str(assignment), True)
        return assignment

    unassigned_var = assignment.select_unassigned_variable(problem)
    for value in problem.order_domain_values(unassigned_var, assignment):
        new_assignment = Assignment(assignment.state.copy())
        new_assignment.state[unassigned_var] = value
        if problem.is_consistent(new_assignment):
            assignment.state[unassigned_var] = value
            result = recursive_backtracking(assignment, problem)
            if result is not None:
                return result
            del assignment.state[unassigned_var]
        else:
            output.append(str(new_assignment), False)
    return None

def forward_checking(assignment, problem):
    if assignment.complete(problem):
        output.append(str(assignment), True)
        return assignment

    unassigned_var = assignment.select_unassigned_variable(problem)
    new_assignment = Assignment(assignment.state.copy())
    for value in problem.order_domain_values(unassigned_var, assignment):
        new_assignment.state[unassigned_var] = value
        if problem.is_consistent(new_assignment):
            current_constraints = problem.get_current_constraints(unassigned_var, new_assignment)
            new_problem = remove_incorrect_values(problem, current_constraints, new_assignment)
            if new_problem is not None:
                result = forward_checking(new_assignment, new_problem)
                if result is not None:
                    return result
                del new_assignment.state[unassigned_var]
        else:
            output.append(str(new_assignment), False)
            del new_assignment.state[unassigned_var]
    output.append(str(new_assignment), False)
    return None

def remove_incorrect_values(problem, constraints, assignment):
    new_problem = Problem(deepcopy(problem.variables), deepcopy(problem.constraints))

    for c in constraints:
        var1, var2, relation = c
        if assignment.get_value(var1) is not None and assignment.get_value(var2) is not None:
            continue
        if relation == '=':
            if assignment.get_value(var1) is not None:
                new_problem.variables[var2] = [value for value in new_problem.variables[var2] if value == assignment.get_value(var1)]
            elif assignment.get_value(var2) is not None:
                new_problem.variables[var1] = [value for value in new_problem.variables[var1] if value == assignment.get_value(var2)]
        elif relation == '!':
            if assignment.get_value(var1) is not None:
                new_problem.variables[var2] = [value for value in new_problem.variables[var2] if value != assignment.get_value(var1)]
            elif assignment.get_value(var2) is not None:
                new_problem.variables[var1] = [value for value in new_problem.variables[var1] if value != assignment.get_value(var2)]
        elif relation == '<':
            if assignment.get_value(var1) is not None:
                new_problem.variables[var2] = [value for value in new_problem.variables[var2] if value > assignment.get_value(var1)]
            elif assignment.get_value(var2) is not None:
                new_problem.variables[var1] = [value for value in new_problem.variables[var1] if value < assignment.get_value(var2)]
        elif relation == '>':
            if assignment.get_value(var1) is not None:
                new_problem.variables[var2] = [value for value in new_problem.variables[var2] if value < assignment.get_value(var1)]
            elif assignment.get_value(var2) is not None:
                new_problem.variables[var1] = [value for value in new_problem.variables[var1] if value > assignment.get_value(var2)]
    return new_problem

def parse_problem(var_filepath, con_filepath):
    variables = utils.parse_var(var_filepath)
    constraints = utils.parse_con(con_filepath)
    return Problem(variables, constraints)

class Problem():
    def __init__(self, variables, constraints):
        self.variables = dict(sorted(variables.items()))
        self.constraints = constraints
        
    def __str__(self):
        return f"Problem(variables={self.variables} constraints={self.constraints})"
    
    def order_domain_values(self, var, assignment):

        if var not in self.variables:
            raise ValueError(f"Variable {var} not found in problem.")
        
        var_constraints = [c for c in self.constraints if c[0] == var or c[1] == var]
        var_constraints = [c for c in var_constraints if assignment.get_value(c[0]) is None and assignment.get_value(c[1]) is None]

        domain_values = self.variables[var]
        sorted_domain_values = sorted(domain_values, key=lambda value: (-self.constraining_value(var, value, var_constraints), value))
        return sorted_domain_values
    
    def is_consistent(self, assignment):
        temp_problem = Problem(deepcopy(self.variables), deepcopy(self.constraints))
        for var, value in assignment.get_assigned_variables().items():
            temp_problem.variables[var] = [value]

        for var, value in assignment.get_assigned_variables().items():
            var_constraints = temp_problem.get_constraints(var)
            for c in var_constraints:
                if c[0] == var:
                    other_var = c[1]
                else:
                    other_var = c[0]
                if assignment.get_value(other_var) is not None:
                    if not Problem.check_constraint(c[2], c[0], c[1], temp_problem):
                        return False
        return True
    
    def check_constraint(constraint, value1, value2, problem):
        if constraint == '=':
            for v1 in problem.variables[value1]:
                for v2 in problem.variables[value2]:
                    if v1 == v2:
                        return True
        elif constraint == '!':
            for v1 in problem.variables[value1]:
                for v2 in problem.variables[value2]:
                    if v1 != v2:
                        return True
        elif constraint == '<':
            for v1 in problem.variables[value1]:
                for v2 in problem.variables[value2]:
                    if v1 < v2:
                        return True
        elif constraint == '>':
            for v1 in problem.variables[value1]:
                for v2 in problem.variables[value2]:
                    if v1 > v2:
                        return True
        else:
            raise ValueError(f"Unknown constraint relation: {constraint[2]}")

    def constraining_value(self, var, value, var_constraints):
        count = 0
        for c in var_constraints:
            if c[0] == var:
                other_i = 1
            else:
                other_i = 0
            if c[2] == '=':
                if value in self.variables[c[other_i]]:
                    count += 1
            elif c[2] == '!':
                if value not in self.variables[c[other_i]]:
                    count += len(self.variables[c[other_i]])
                else:
                    count += len(self.variables[c[other_i]]) - 1
            elif c[2] == '<':
                if other_i == 0:
                    count += len([x for x in self.variables[c[other_i]] if x < value])
                else:
                    count += len([x for x in self.variables[c[other_i]] if x > value])
            elif c[2] == '>':
                if other_i == 0:
                    count += len([x for x in self.variables[c[other_i]] if x > value])
                else:
                    count += len([x for x in self.variables[c[other_i]] if x < value])
        return count
    
    def get_constraints(self, var):
        return [c for c in self.constraints if c[0] == var or c[1] == var]

    def get_current_constraints(self, var, assignment):
        var_constraints = [c for c in self.constraints if c[0] == var or c[1] == var]
        relevant_constraints = []
        for c in var_constraints:
            var1, var2, _ = c
            if (assignment.get_value(var1) is None and var1 != var) or (assignment.get_value(var2) is None and var2 != var):
                relevant_constraints.append(c)
        return relevant_constraints

class Assignment():
    def __init__(self, state={}):
        self.state = state

    def __str__(self):
        string = ""
        for var, value in self.state.items():
            if value is not None:
                string += f"{var}={value}, "
        return string.strip(", ")
    
    def get_value(self, var):
        return self.state.get(var, None)
    
    def get_assigned_variables(self):
        return {var: value for var, value in self.state.items() if value is not None}
    
    def complete(self, problem):
        return problem.is_consistent(self) and len(self.state) == len(problem.variables)
    
    def select_unassigned_variable(self, problem):
        most_constrained_var = list()
        for var in problem.variables:
            if self.get_value(var) is None:
                most_constrained_var.append((var, len(problem.variables[var])))
        if not most_constrained_var:
            return None
        most_constrained_var.sort(key=lambda x: x[1])

        least_constrained_length = most_constrained_var[0][1]
        count = 0
        for var, length in most_constrained_var:
            if length == least_constrained_length:
                count += 1
        if count == 1:
            return most_constrained_var[0][0]
        else:
            return min(most_constrained_var, key=lambda x: (x[1], -len(problem.get_current_constraints(x[0], self)), var))[0]