from resolution import resolution

import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_kb_file>")
        return
    
    kb_file = sys.argv[1]
    
    success, final_kb = resolution(kb_file)

    for i in range(len(final_kb)):
        clause, parents = final_kb[i]
        if clause:
            string = str(i+1) + ". " + " ".join([str(literal) for literal in clause]) + ' {' + ", ".join([str(p+1) for p in parents]) + '}'
            print(string)
        else:
            string = str(i+1) + ". Contradiction {" + ", ".join([str(p+1) for p in parents]) + '}'
            print(string)
    if success:
        print("Valid")
    else:
        print("Fail")

if __name__ == "__main__":
    main()