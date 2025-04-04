from resolution import resolution

import sys
import os

def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python main.py <path_to_kb_file>")
    #     return
    
    # kb_file = sys.argv[1]

    kb_file = os.path.join(os.path.dirname(__file__), 'input', 'demo.in')
    
    resolution(kb_file)

if __name__ == "__main__":
    main()