#! /usr/bin/env python3

from __future__ import annotations

import ast
from itertools import chain

from types import FunctionType
from typing import *

from .base import *

__all__ = ['ASTBlock', 'ASTStmtBlock', 'ASTValues', 'ASTHelper']


class ASTStmtBlock(ASTBlock):
    __stmts: Iterable[ast.stmt]
    result: Optional[ast.expr]
    __temp_names: List[str]  # iterable?

    def __init__(self, stmts: Iterable[ast.stmt], result: Optional[ast.expr]):
        super().__init__()
        self.__stmts = stmts
        self.result = result

    def append_stmt(self, stmts: Iterable[ast.stmt]) -> None:
        self.__stmts = chain(self.__stmts, stmts)

    @property
    def stmts(self) -> Iterable[ast.stmt]:
        return self.__stmts

    def get_result(self) -> ast.expr:
        if self.result:
            return self.result
        else:
            return ast.Constant(None)

    def drop_result(self, context: SExprContextManager):
        if self.result:
            if not isinstance(self.result, ast.Constant) and \
                    not isinstance(self.result, ast.Name):
                # may have side effect
                self.__stmts = chain(self.__stmts, [ast.Expr(self.get_result())])
                self.result = None
            else:
                self.__stmts = chain(self.__stmts, [ast.Pass()])
        self.free_temp(context)

    def apply_result(self, context: SExprContextManager):
        # WARNING DO NOT USE IT IN THIS FILE

        if self.result:
            if not isinstance(self.result, ast.Constant) and \
                    not isinstance(self.result, ast.Name):
                # may have side effect

                name = context.get_temp()
                self.__stmts = ASTHelper.build_block_from_assign(
                    ASTHelper.build_block_from_symbol(name),
                    self,
                    context
                ).stmts
                self.add_temp(name)
                self.result = ast.Name(name, ast.Load())
            else:
                self.__stmts = chain(self.__stmts, [ast.Pass()])


class ASTValues(ASTBlock):
    values: ASTBlock

    def __init__(self, values: ASTBlock):
        super().__init__()
        self.values = values

    @property
    def stmts(self) -> Iterable[ast.stmt]:
        return self.values.stmts

    def get_result(self) -> ast.expr:
        return ast.Starred(self.values.get_result(), ast.Load())

    def drop_result(self, context: SExprContextManager):
        self.values.drop_result(context)

    def apply_result(self, context: SExprContextManager):
        self.values.apply_result(context)


class ASTHelper:
    ops = {
        '+': ast.Add,
        '-': ast.Sub,
        '*': ast.Mult,
        '/': ast.Div,
        '//': ast.FloorDiv,
        '%': ast.Mod,
        '@': ast.MatMult,
        '**': ast.Pow,
        '<<': ast.LShift,
        '>>': ast.RShift,
        '|': ast.BitOr,
        '^': ast.BitXor,
        '&': ast.BitAnd
    }

    cmpops = {
        '<': ast.Lt,
        '<=': ast.LtE,
        '==': ast.Eq,
        '!=': ast.NotEq,
        '>': ast.Gt,
        '>=': ast.GtE,
        'is': ast.Is,
        'isnot': ast.IsNot,
        'in': ast.In,
        'notin': ast.NotIn
    }

    @staticmethod
    def build_block_from_symbol(symbol: str) -> ASTStmtBlock:

        symbols = symbol.split(".")

        target = ast.Name(symbols[0], ast.Load())

        for attr in symbols[1:]:
            target = ast.Attribute(target, attr, ast.Load())

        return ASTStmtBlock([], target)

    @staticmethod
    def build_block_from_getattr(target: ASTBlock, attr: str) -> ASTBlock:
        assert isinstance(target, ASTStmtBlock), "target must be ASTStmtBlock"

        target.result = ast.Attribute(target.get_result(), attr, ast.Load())

        return target

    @staticmethod
    def build_block_from_literal(literal: Union[bool, int, float, complex, str, bytes, ..., None]) -> ASTStmtBlock:
        return ASTStmtBlock([], ast.Constant(literal))

    @staticmethod
    def build_block_from_pass() -> ASTStmtBlock:
        return ASTStmtBlock([ast.Pass()], None)

    @staticmethod
    def build_block_from_list(contents: List[ASTBlock]) -> ASTStmtBlock:
        results = []

        for content in contents:
            results.append(content.get_result())

        ret = ASTHelper.pack_block_stmts(contents)
        ret.result = ast.List(results, ast.Load())

        return ret

    @staticmethod
    def build_block_from_tuple(contents: List[ASTBlock]) -> ASTStmtBlock:
        results = []

        for content in contents:
            results.append(content.get_result())

        ret = ASTHelper.pack_block_stmts(contents)
        ret.result = ast.Tuple(results, ast.Load())

        return ret

    @staticmethod
    def build_block_from_assign(target: ASTBlock, value: ASTBlock,
                                context: SExprContextManager) -> ASTStmtBlock:
        stmts = [target, value]

        ret = ASTStmtBlock(
            chain(*map(lambda x: x.stmts, stmts), [ast.Assign([target.get_store_result()], value.get_result())]),
            None
        )

        ret.merge_temp(target)
        value.drop_result(context)

        return ret

    @staticmethod
    def build_block_from_for(target: ASTBlock,
                             iter: ASTBlock,
                             body: ASTBlock,
                             elbody: ASTBlock,
                             context: SExprContextManager) -> ASTStmtBlock:

        body.drop_result(context)
        elbody.drop_result(context)

        target.free_temp(context)
        iter.free_temp(context)

        stmts = [target,
                 iter,
                 ASTStmtBlock([
                     ast.For(target.get_store_result(),
                             iter.get_result(),
                             list(body.stmts),
                             list(elbody.stmts))
                 ], None)]

        return ASTHelper.pack_block_stmts(stmts)

    @staticmethod
    def build_block_from_while(test: ASTBlock,
                               body: ASTBlock,
                               elbody: ASTBlock,
                               context: SExprContextManager) -> ASTStmtBlock:

        body.drop_result(context)
        elbody.drop_result(context)

        body = ASTHelper.pack_block_stmts([
            ASTHelper.build_block_from_if(test,
                                          ASTHelper.build_block_from_literal(...),
                                          ASTHelper.build_block_from_break(),
                                          context),
            body
        ])

        ret = ASTStmtBlock([
            ast.While(ast.Constant(True),
                      list(body.stmts),
                      list(elbody.stmts))
        ], None)

        ret.free_temp(context)

        return ret

    @staticmethod
    def build_block_from_break() -> ASTStmtBlock:
        return ASTStmtBlock([ast.Break()], None)

    @staticmethod
    def build_block_from_if(test: ASTBlock,
                            body: ASTBlock,
                            elbody: ASTBlock,
                            context: SExprContextManager) -> ASTStmtBlock:

        body.drop_result(context)
        elbody.drop_result(context)

        stmts = [test,
                 ASTStmtBlock([
                     ast.If(test.get_result(),
                            list(body.stmts),
                            list(elbody.stmts))
                 ], None)]

        test.free_temp(context)

        return ASTHelper.pack_block_stmts(stmts)

    @staticmethod
    def build_block_from_import(modules: List[str]):
        # TODO: alias
        return ASTStmtBlock([ast.Import(list(map(lambda x: ast.alias(x), modules)))], None)

    @staticmethod
    def build_block_from_func_call(func: ASTBlock,
                                   args: List[Union[ASTBlock, Tuple[Optional[str], ASTBlock]]]) -> ASTStmtBlock:
        stmts = []
        vargs = []
        kwargs = []

        stmts.append(func)
        func_value = func.get_result()

        for arg in args:
            if isinstance(arg, tuple):
                arg: Tuple[Optional[str], ASTBlock]
                stmts.append(arg[1])
                kwargs.append(ast.keyword(arg[0], arg[1].get_result()))
            elif isinstance(arg, ASTBlock):
                arg: ASTBlock
                stmts.append(arg)
                vargs.append(arg.get_result())

        ret = ASTHelper.pack_block_stmts(stmts)
        ret.result = ast.Call(func_value, vargs, kwargs)

        return ret

    @staticmethod
    def build_block_from_func_decl(name: str, args: List[str], stmts: List[ASTBlock], context: SExprContextManager):
        for stmt in stmts[:-1]:
            stmt.drop_result(context)
        body = ASTHelper.pack_block_stmts(stmts)
        body.append_stmt([ast.Return(stmts[-1].get_result())])

        # TODO: py3.8 kwargs defaults

        return ASTStmtBlock([
            ast.FunctionDef(name=name,
                            args=ast.arguments(
                                args=list(map(lambda x: ast.arg(x), args)),
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[]
                            ),
                            body=list(body.stmts),
                            decorator_list=[])],
            ast.Name(name, ast.Load()))

    @staticmethod
    def build_block_from_op(op: str, left: ASTBlock, right: ASTBlock):

        ret = ASTHelper.pack_block_stmts([left, right])

        if op in ASTHelper.ops:  # binary
            ret.result = ast.BinOp(left.get_result(), ASTHelper.ops[op](), right.get_result())
        elif op in ASTHelper.cmpops:
            ret.result = ast.Compare(left.get_result(), [ASTHelper.cmpops[op]()], [right.get_result()])

        return ret

    @staticmethod
    def build_block_from_return(value: ASTBlock, context: SExprContextManager) -> ASTStmtBlock:
        ret_val = value.get_result()

        ret = ASTHelper.pack_block_stmts(
            chain(value,
                  [ASTStmtBlock([
                      ast.Return(ret_val)
                  ], None)]))
        ret.free_temp(context)

        return ret

    @staticmethod
    def pack_block_stmts(blocks: Iterable[ASTBlock]) -> ASTStmtBlock:
        stmts = []

        ret = ASTStmtBlock([], None)

        for block in blocks:
            stmts.append(block.stmts)
            ret.merge_temp(block)

        ret.append_stmt(chain(*stmts))

        return ret

    @staticmethod
    def compile(result: ASTBlock, context: SExprContextManager, globals_dict: Optional[Dict] = None) -> FunctionType:
        result.drop_result(context)

        stmts = list(result.stmts)
        naive = ast.fix_missing_locations(ast.Module(stmts))

        # import astpretty
        # astpretty.pprint(naive)

        ret = compile(
            naive,
            'none',
            'exec'
        )

        return FunctionType(ret, globals_dict if globals_dict else globals())


if __name__ == '__main__':
    def main():
        # TODO: right test
        context = SExprContextManager()
        code = ASTHelper.compile(
            ASTHelper.pack_block_stmts(
                [
                    ASTHelper.build_block_from_assign(
                        ASTHelper.build_block_from_symbol("x"),
                        ASTHelper.build_block_from_literal("x"),
                        context
                    ),
                    ASTHelper.build_block_from_func_call(
                        ASTHelper.build_block_from_symbol("print"),
                        [ASTHelper.build_block_from_literal(1),
                         ASTHelper.build_block_from_literal(1),
                         ASTHelper.build_block_from_literal(1),
                         ("sep", ASTHelper.build_block_from_symbol("x"))]
                    )
                ]
            ),
            context
        )
        code()


    main()
