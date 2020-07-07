#! /usr/bin/env python3

from io import StringIO
from cmd import Cmd
import traceback

from .sexpr import SExprNodeBase, SExprSymbol, SExpr
from .macro import generate_default_context
from .compiler import parse_stream
from .asthelper import ASTHelper


class ResppyRepl(Cmd):
    intro = 'Welcome to the Resppy.\n'
    prompt = '> '
    buffer = []
    lparen = 0
    rparen = 0
    in_str = False

    def __init__(self, *args, context_env=None, **kwargs):
        super(ResppyRepl, self).__init__(*args, **kwargs)

        if context_env:
            self.context, self.env = context_env
        else:
            self.context, self.env = generate_default_context()

        self.init()

    def init(self):
        self.buffer = []
        self.lparen = 0
        self.rparen = 0
        self.in_str = False
        self.prompt = '>>> '

    def emptyline(self) -> bool:
        return False

    def default(self, line: str) -> bool:
        self.prompt = '... '
        self.buffer.append(line)

        it = iter(line)
        prev = ''

        for now in line:
            if self.in_str:
                if now == '"' and prev != '\\':
                    self.in_str = False
            else:
                if now == '(':
                    self.lparen += 1
                elif now == ')':
                    self.rparen += 1
                elif now == '"':
                    self.in_str = True
            prev = now

            if self.rparen > self.lparen:
                self.stdout.write("Unexpected ')'\n")
                self.init()
                return False

        if self.lparen == self.rparen:
            input = '\n'.join(self.buffer)
            self.init()

            sexprs = parse_stream(StringIO(input))
            contents = []

            for sexpr in sexprs:
                assert isinstance(sexpr, SExprNodeBase)

                sexpr = SExpr(SExprSymbol('print'), sexpr)
                contents.append(sexpr)

            target = SExpr(SExprSymbol('begin'), *contents)
            result = target.compile(self.context)
            result.drop_result(self.context)
            code = ASTHelper.compile(result, self.context, self.env)
            code()

        return False

    def cmdloop_with_keyboard_interrupt(self):
        while True:
            try:
                self.cmdloop()
            except KeyboardInterrupt as e:
                self.intro = None
                self.init()

                self.stdout.write("\n%s\n" % e.__class__.__name__)
            except Exception as e:
                self.intro = None
                self.init()

                self.stdout.write(traceback.format_exc())
                self.stdout.write("\n%s: %s\n" % (e.__class__.__name__, str(e)))


if __name__ == '__main__':
    ResppyRepl().cmdloop_with_keyboard_interrupt()
