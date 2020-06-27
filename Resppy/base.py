#! /usr/bin/env python3

from __future__ import annotations

import ast
from typing import *
from copy import deepcopy
from keyword import iskeyword

__all__ = ["ASTBlock", "SExprMacro", "SExprContextManager", "sexpr_mangle"]

mangle_prefix = "_S_"
mangle_replace = [
    # ('_', '__'),
    ('~', '_SP_'),
    ('!', '_EX_'),
    ('@', '_AT_'),
    ('#', '_SH_'),
    ('$', '_DL_'),
    ('%', '_MO_'),
    ('^', '_XO_'),
    ('&', '_AN_'),
    ('*', '_ST_'),
    ('-', '_MI_'),
    ('+', '_AD_'),
    ('=', '_EQ_'),
    ('\\', '_LS_'),
    ('|', '_MD_'),
    (':', '_CN_'),
    ('?', '_QU_'),
    ('/', '_SL_'),
    (',', '_CO_'),
    # ('.', '_DO_'),
    ('<', '_LT_'),
    ('>', '_GT_')
]


class ASTBlock:
    def __init__(self):
        self.__temp_names = []

    @property
    def stmts(self) -> Iterable[ast.stmt]:
        raise NotImplementedError("ASTBlock.stmt")

    def free_temp(self, context: SExprContextManager) -> None:
        for name in self.__temp_names:
            context.free_temp(name)
        self.__temp_names = []

    def add_temp(self, name: str):
        self.__temp_names.append(name)
        return self

    def add_temps(self, temps: List[str]):
        self.__temp_names.extend(temps)
        return self

    def merge_temp(self, block: ASTBlock):
        self.add_temps(block.__temp_names)

    def get_result(self) -> ast.expr:
        raise NotImplementedError("ASTBlock.get_result")

    def drop_result(self, context: SExprContextManager):
        raise NotImplementedError("ASTBlock.drop_result")

    def apply_result(self, context: SExprContextManager):
        raise NotImplementedError("ASTBlock.apply_result")

    def get_store_result(self) -> ast.expr:
        ret = deepcopy(self.get_result())

        for node in ast.walk(ret):
            if hasattr(node, 'ctx'):
                node.ctx = ast.Store()

        return ret


class SExprContextManager:
    class SExprContext:
        TEMP_PREFIX = "_ST_"

        def __init__(self, parent):
            self.parent = parent
            self.macros: typing.Dict[str, SExprMacro] = {}
            self.free_temps = []
            self.next_temp_id = 1

        def register(self, name: str, body: SExprMacro):
            self.macros[sexpr_mangle(name)] = body

        def resolve(self, name) -> typing.Optional[SExprMacro]:
            if name in self.macros:
                return self.macros[name]
            elif self.parent:
                return self.parent.resolve[name]
            elif name in globals() and isinstance(globals()[name], SExprMacro):
                return globals()[name]
            else:
                return None

        def get_temp(self):
            if self.free_temps:
                ret = self.free_temps.pop()
            else:
                ret = self.TEMP_PREFIX + str(self.next_temp_id)
                self.next_temp_id += 1
            return ret

        def free_temp(self, name):
            self.free_temps.append(name)

    def __init__(self):
        self.contexts: typing.List[SExprContextManager.SExprContext] = [
            SExprContextManager.SExprContext(None)
        ]

    def register(self, name: str, body: SExprMacro):
        self.contexts[-1].register(name, body)

    def resolve(self, name):
        return self.contexts[-1].resolve(name)

    def get_temp(self):
        return self.contexts[-1].get_temp()

    def free_temp(self, name):
        return self.contexts[-1].free_temp(name)

    def push_context(self):
        self.contexts.append(
            SExprContextManager.SExprContext(self.contexts[-1])
        )

    def pop_context(self):
        assert len(self.contexts) > 1
        self.contexts.pop()


class SExprMacro:
    def expand(self, args: list, context) -> ASTBlock:
        raise NotImplementedError("SExprMacro.expand")


def sexpr_mangle(name: str) -> str:
    ret = name
    for src, tg in mangle_replace:
        ret = ret.replace(src, tg)

    if iskeyword(name):
        ret = mangle_prefix + ret

    if not ret.replace('.', '_DO_').isidentifier():
        raise ValueError("Invalid indentifier: %s -> %s" %
                         (name, ret))
    return ret
