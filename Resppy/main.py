#! /usr/bin/env python3

import sys
import argparse
from typing import *
from io import StringIO

from .repl import ResppyRepl
from .macro import *
from .compiler import *

if __name__ == "__main__":
    def main(args: List[str]):
        parser = argparse.ArgumentParser()
        parser.add_argument('filename', type=str, help='source file name')
        parser.add_argument('-c', '--compile',
                            action='store_true',
                            help='compile only')
        parser.add_argument('-r', '--repl',
                            action='store_true',
                            help='start repl after execution')

        args = parser.parse_args()

        if args.filename == "-":
            program = sys.stdin.read()
        else:
            with open(args.filename, encoding="utf-8") as stream:
                program = stream.read()

        context, env = generate_default_context()
        f = compile_stream(StringIO(program), context, env)

        if not args.compile:
            f()
        else:
            import uncompyle6
            uncompyle6.deparse_code2str(f.__code__)

        if args.repl:
            ResppyRepl(context_env=(context, env)).cmdloop_with_keyboard_interrupt()


    main(sys.argv)
