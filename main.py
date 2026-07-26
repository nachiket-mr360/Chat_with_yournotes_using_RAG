import os
import sys
import argparse 
import rag
def cmd_index(args):
    print(f"indexing '{args.docs}'..")
    rag.build_index(args.docs)
    print("done")
def cmd_chat(args):
    store = rag.load_index() 
    while True:
        q = input("you > ").strip()
        docs, token_gen = rag.answer(store, q)  
        print('\nbot > ', end="", flush=True)
        for t in token_gen:
            print(t, end="", flush=True)  
        print("\n")
def main():
    p = argparse.ArgumentParser(description="Chatwithfile CLI")
    sub = p.add_subparsers(dest="command", required=True)
    pi = sub.add_parser("index", help="build index from docs/")
    pi.add_argument("--docs", default="docs")
    pi.set_defaults(func=cmd_index)  
    pc = sub.add_parser("chat")
    pc.set_defaults(func=cmd_chat) 
    args = p.parse_args()
    args.func(args) 
if __name__ == "__main__":
    main()