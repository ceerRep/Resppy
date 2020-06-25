#! /usr/bin/env python3

from __future__ import annotations

import ast
from itertools import chain
from types import FunctionType
from typing import *
from typing import TextIO
from enum import Enum, auto
from .sexpr import *

__all__ = ['compile_stream']

avail_special_chars = "~!@#$%^&*-_=+\\|;:/?.>,<"


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
        if self.pos == len(self.buffer):
            self.read_from_stream()
        return self.buffer[self.pos]

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

    def parse_next_token(self):
        # TODO Make it more pretty
        now = ' '
        while now.isspace():
            now = self.read()

        if now in "(){}[]'`":
            return now, Tokenizer.TokenType.Symbol

        chars = [now]
        if now.isdigit():
            now = self.peek()
            while now.isalnum() or now in avail_special_chars:
                chars.append(now)
                self.read()
                now = self.peek()

            return ast.literal_eval(''.join(chars)), Tokenizer.TokenType.ConstLiteral

        if now == '"':
            prev = now
            now = self.peek()

            while now != '"' or prev == '\\':
                chars.append(now)
                self.read()
                now = self.peek()

            chars.append(self.read())  # "

            return ast.literal_eval(''.join(chars)), Tokenizer.TokenType.ConstLiteral

        # identifier
        now = self.peek()
        while now.isalnum() or now in avail_special_chars:
            chars.append(now)
            self.read()
            now = self.peek()

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

    return SExpr(SExprSymbol('pylist'), *ret)


def parse_quote(tokenizer: Tokenizer):
    value = parse_next(tokenizer)

    return SExpr(SExprSymbol('quote'), value)


def parse_quasiquote(tokenizer: Tokenizer):
    value = parse_next(tokenizer)

    return SExpr(SExprSymbol('quasiquote'), value)


def parse_next(tokenizer: Tokenizer) -> Union[SpecialToken, SExprNodeBase]:
    token_val, token_type = tokenizer.parse_next_token()
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
        else:
            return SExprSymbol(token_val)
    else:
        raise ValueError('Unknown token type')


def parse_stream(stream: TextIO):
    ret = []
    tokenizer = Tokenizer(stream)
    try:
        while True:
            ret.append(parse_next(tokenizer))
    except EOFError:
        pass
    return ret


def compile_stream(stream: TextIO, context: SExprContextManager, global_dict: Optional[Dict] = None) -> FunctionType:
    sexprs = parse_stream(stream)
    blocks: List[ASTBlock] = []

    for sexpr in sexprs:
        sexpr = SExpr(SExprSymbol("print"), sexpr)
        block = sexpr.compile(context)

        block.drop_result(context)

        blocks.append(block)

    all_blocks = ASTHelper.pack_block_stmts(blocks)

    return ASTHelper.compile(all_blocks, context, global_dict)


if __name__ == '__main__':
    def main():
        from io import StringIO
        from sys import stdin

        context = SExprContextManager()

        compile_stream(StringIO("(print (min 1 2 3) (max 1 2 3))"), context)()

        compile_stream(stdin, context)()


    main()
