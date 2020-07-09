#! /usr/bin/env python3

from __future__ import annotations

import ast
from itertools import chain
from types import FunctionType
from typing import *
from typing import TextIO
from enum import Enum, auto
from .sexpr import *

__all__ = ['compile_stream', 'compile_stream_to_code']

avail_special_chars = "~!@#$%^&*-_=+\\|:/?.>,<"


class SpecialToken:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return "SpecialToken(%s)" % repr(self.value)

    def __eq__(self, other):
        return str(self) == str(other)


class Tokenizer:
    READ_SIZE: int = 1024

    class TokenType(Enum):
        ConstLiteral = auto()
        Symbol = auto()
        Identifier = auto()

    def __init__(self, stream: TextIO):
        self.stream: TextIO = stream
        self.buffer: str = ""
        self.pos: int = 0
        self.lino = 1
        self.column = 1

    def read_from_stream(self):
        self.buffer = self.buffer[self.pos:] + self.stream.read(self.READ_SIZE)
        self.pos = 0
        if len(self.buffer) == 0:
            raise EOFError()

    def peek(self) -> str:
        ret = ' '
        try:
            if self.pos == len(self.buffer):
                self.read_from_stream()
            ret = self.buffer[self.pos]
        except EOFError:
            pass
        return ret

    def read(self) -> str:
        if self.pos == len(self.buffer):
            self.read_from_stream()
        ret = self.buffer[self.pos]
        self.pos += 1
        self.column += 1

        if ret == '\n':
            self.lino += 1
            self.column = 1
        return ret

    def eat_line(self):
        now_line = self.lino
        while self.lino == now_line:
            self.read()

    def parse_next_token(self):
        # TODO Make it more pretty
        now = ' '
        while now.isspace():
            now = self.read()

            if now == ';':
                self.eat_line()
                now = ' '

        if now in "(){}[]'`":
            return now, Tokenizer.TokenType.Symbol

        chars = [now]

        if now == '"':
            prev = now
            now = self.peek()

            while now != '"' or prev == '\\':
                chars.append(now)
                self.read()
                now = self.peek()

            chars.append(self.read())  # "

            return ast.literal_eval(''.join(chars)), Tokenizer.TokenType.ConstLiteral

        # identifier or num
        now = self.peek()
        while now.isalnum() or now in avail_special_chars:
            chars.append(now)
            self.read()
            now = self.peek()

        target = ''.join(chars)

        if target == "...":
            return ..., Tokenizer.TokenType.ConstLiteral

        try:
            return ast.literal_eval(target), Tokenizer.TokenType.ConstLiteral
        except Exception:
            pass

        return ''.join(chars), Tokenizer.TokenType.Identifier


def parse_sexpr(tokenizer: Tokenizer):
    param = []

    while True:
        next_value = parse_next(tokenizer)

        if isinstance(next_value, SpecialToken):
            if next_value == ')':
                break
            else:
                raise ValueError('Unexpexted %s' % str(next_value))

        param.append(next_value)

    return SExpr(*param)


def parse_list(tokenizer: Tokenizer):
    ret = []

    while True:
        next_value = parse_next(tokenizer)

        if isinstance(next_value, SpecialToken):
            if next_value == ']':
                break
            else:
                raise ValueError('Unexpexted %s' % str(next_value))

        ret.append(next_value)

    return SExpr(SExprSymbol('list*'), *ret)


def parse_quote(tokenizer: Tokenizer):
    value = parse_next(tokenizer)

    return SExpr(SExprSymbol('quote'), value)


def parse_quasiquote(tokenizer: Tokenizer):
    value = parse_next(tokenizer)

    return SExpr(SExprSymbol('quasiquote'), value)


def parse_next(tokenizer: Tokenizer) -> Union[SpecialToken, SExprNodeBase]:
    token_val, token_type = tokenizer.parse_next_token()

    try:
        if token_type == Tokenizer.TokenType.Symbol:
            if token_val == '(':
                return parse_sexpr(tokenizer)
            elif token_val == '[':
                return parse_list(tokenizer)
            elif token_val == '{':
                ...  # dict
            elif token_val == "'":
                return parse_quote(tokenizer)
            elif token_val == "`":
                return parse_quasiquote(tokenizer)
            elif token_val == ')':
                return SpecialToken(')')
            elif token_val == ']':
                return SpecialToken(']')
            elif token_val == '}':
                return SpecialToken('}')
            else:
                raise ValueError("Unknown symbol")
        elif token_type == Tokenizer.TokenType.ConstLiteral:
            return SExprLiteral(token_val)
        elif token_type == Tokenizer.TokenType.Identifier:
            if token_val[0] == ':':
                return SExprKeyword(token_val[1:])
            elif token_val[0] == '#':
                arg = parse_next(tokenizer)
                assert isinstance(arg, SExprNodeBase)
                return SExpr(SExprSymbol(token_val), arg)
            else:
                return SExprSymbol(token_val)
        else:
            raise ValueError('Unknown token type')
    except EOFError:
        raise ValueError("Unexpected end of file")


def parse_stream(stream: TextIO):
    ret = []
    tokenizer = Tokenizer(stream)
    try:
        while True:
            ret.append(parse_next(tokenizer))
    except EOFError:
        pass
    return ret


def compile_stream(stream: TextIO, context: SExprContextManager) -> FunctionType:
    sexprs = parse_stream(stream)
    blocks: List[ASTBlock] = []

    for sexpr in sexprs:
        # sexpr = SExpr(SExprSymbol("print"), sexpr)
        block = sexpr.compile(context)

        block.drop_result(context)

        blocks.append(block)

    all_blocks = ASTHelper.pack_block_stmts(blocks)

    return ASTHelper.compile(all_blocks, context)


def compile_stream_to_code(stream: TextIO, context: SExprContextManager) -> str:
    sexprs = parse_stream(stream)
    blocks: List[ASTBlock] = []

    for sexpr in sexprs:
        # sexpr = SExpr(SExprSymbol("print"), sexpr)
        block = sexpr.compile(context)

        block.drop_result(context)

        blocks.append(block)

    all_blocks = ASTHelper.pack_block_stmts(blocks)

    return ASTHelper.compile_to_code(all_blocks, context)


if __name__ == '__main__':
    def main():
        import uncompyle6
        from sys import stdin, argv
        from io import StringIO
        from .macro import generate_default_context

        if len(argv) == 2:
            target = open(argv[1], encoding='utf-8')
        else:
            target = stdin
        context, env = generate_default_context()
        f = compile_stream(StringIO(target.read()), context, env)

        uncompyle6.deparse_code2str(f.__code__)


    main()
