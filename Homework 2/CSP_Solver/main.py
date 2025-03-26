from solver import solve_csp

import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <path_to_var_file> <path_to_con_file> <none|fc>")
        return

    var_file = sys.argv[1]
    con_file = sys.argv[2]
    consistency_procedure = sys.argv[3].lower()
    
    solve_csp(var_file, con_file, consistency_procedure)


if __name__ == "__main__":
    main()