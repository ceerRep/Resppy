#! /usr/bin/env python3

from __future__ import annotations

import ast

from types import FunctionType
from typing import *

from .base import *

__all__ = ['ASTBlock', 'ASTStmtBlock', 'ASTValues', 'ASTHelper']


class ASTStmtBlock(ASTBlock):
    __stmts: Iterable[ast.stmt]
    result: Optional[ast.expr]

    def __init__(self, stmts: Iterable[ast.stmt], result: Optional[ast.expr]):
        super().__init__()
        self.__stmts = stmts
        self.result = result

    def append_stmt(self, stmts: Iterable[ast.stmt]) -> None:
        self.__stmts = chain0(self.__stmts, stmts)

    @property
    def stmts(self) -> Iterable[ast.stmt]:
        return self.__stmts

    def get_result(self) -> ast.expr:
        if self.result:
            return self.result
        else:
            return ast.Constant(None)

    # def free_temp(self, context: SExprContextManager) -> List[str]:
    #     temps = super(ASTStmtBlock, self).free_temp(context)
    #     if temps:
    #         self.__stmts = chain0(self.__stmts, [ast.Delete([ast.Name(temp, ast.Del()) for temp in temps])])
    #     return temps

    def drop_result(self, context: SExprContextManager):
        if self.result:
            if not isinstance(self.result, ast.Constant) and \
                    not isinstance(self.result, ast.Name):
                # may have side effect
                self.__stmts = chain0(self.__stmts, [ast.Expr(self.get_result())])
                self.result = None
            else:
                self.__stmts = chain0(self.__stmts, [ast.Pass()])
        self.free_temp(context)

    def apply_result(self, context: SExprContextManager, name: Optional[str]):
        # WARNING DO NOT USE IT IN THIS FILE
        if self.result:
            if not isinstance(self.result, ast.Constant) and \
                    not isinstance(self.result, ast.Name):
                # may have side effect
                if not name:
                    name = context.get_temp()
                self.__stmts = ASTHelper.build_block_from_assign(
                    ASTHelper.build_block_from_symbol(name),
                    self,
                    context
                ).stmts
                self.add_temp(name)
                self.result = ast.Name(name, ast.Load())
            else:
                if name:
                    context.free_temp(name)
                self.__stmts = chain0(self.__stmts, [ast.Pass()])


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

    def apply_result(self, context: SExprContextManager, name: Optional[str]):
        self.values.apply_result(context, name)


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
    def build_block_from_nonlocal(symbols: List[str]) -> ASTStmtBlock:
        return ASTStmtBlock([ast.Nonlocal(symbols)], None)

    @staticmethod
    def build_block_from_global(symbols: List[str]) -> ASTStmtBlock:
        return ASTStmtBlock([ast.Global(symbols)], None)

    @staticmethod
    def build_block_from_symbol(symbol: str) -> ASTStmtBlock:

        symbols = symbol.split(".")

        if all(symbols):
            target = ast.Name(symbols[0], ast.Load())

            for attr in symbols[1:]:
                target = ast.Attribute(target, attr, ast.Load())
        else:
            target = ast.Name(symbol, ast.Load())
        return ASTStmtBlock([], target)

    @staticmethod
    def build_block_from_getattr(target: ASTBlock, attr: str) -> ASTBlock:
        assert isinstance(target, ASTStmtBlock), "target must be ASTStmtBlock"

        target.result = ast.Attribute(target.get_result(), attr, ast.Load())

        return target

    @staticmethod
    def build_block_from_subscr(target: ASTBlock, index: ASTBlock) -> ASTBlock:
        ret = ASTHelper.pack_block_stmts([target, index])
        ret.result = ast.Subscript(target.get_result(), ast.Index(index.get_result()), ast.Load())

        return ret

    @staticmethod
    def build_block_from_slice(lower: Optional[ASTBlock], upper: Optional[ASTBlock],
                               step: Optional[ASTBlock]) -> ASTBlock:
        stmts = []
        if lower:
            stmts.append(lower)
            lower = lower.get_result()
        if upper:
            stmts.append(upper)
            upper = upper.get_result()
        if step:
            stmts.append(step)
            step = step.get_result()
        ret = ASTHelper.pack_block_stmts(stmts)
        ret.result = ast.Slice(lower, upper, step)
        return ret

    @staticmethod
    def build_block_from_literal(literal: Union[bool, int, float, complex, str, bytes, ..., None]) -> ASTStmtBlock:
        return ASTStmtBlock([], ast.Constant(literal))

    @staticmethod
    def build_block_from_pass() -> ASTStmtBlock:
        return ASTStmtBlock([ast.Pass()], None)

    @staticmethod
    def build_block_from_with(items: List[Tuple[ASTBlock, Optional[ASTBlock]]],
                              body: ASTBlock,
                              context: SExprContextManager) -> ASTStmtBlock:
        stmts: List[ASTBlock] = []
        items_ast = []

        for value, alias in items:
            stmts.append(value)
            if alias is not None:
                stmts.append(alias)
                items_ast.append(ast.withitem(value.get_result(), alias.get_result_in_context(ast.Store())))
            else:
                items_ast.append(ast.withitem(value.get_result(), None))
        body.drop_result(context)
        stmts.append(ASTStmtBlock([
            ast.With(items_ast, list(body.stmts))
        ], None))

        ret = ASTHelper.pack_block_stmts(stmts)
        ret.free_temp(context)

        return ret

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
            chain0(*map(lambda x: x.stmts, stmts), [ast.Assign([target.get_result_in_context(ast.Store())],
                                                               value.get_result())]),
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

        stmts = [target,
                 iter,
                 ASTStmtBlock([
                     ast.For(target.get_result_in_context(ast.Store()),
                             iter.get_result(),
                             list(body.stmts),
                             list(elbody.stmts))
                 ], None)]
        ret = ASTHelper.pack_block_stmts(stmts)
        ret.free_temp(context)
        return ret

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
    def build_block_from_import(modules: List[Tuple[str, Optional[str]]]):
        return ASTStmtBlock([ast.Import(list(map(lambda x: ast.alias(*x), modules)))], None)

    @staticmethod
    def build_block_from_import_from(module: Optional[str],
                                     names: List[Tuple[str, Optional[str]]],
                                     level: Optional[int]):
        return ASTStmtBlock([
            ast.ImportFrom(module,
                           list(map(lambda x: ast.alias(*x), names)),
                           level)], None)

    @staticmethod
    def build_block_from_try(excepthandler: List[Tuple[Optional[ASTBlock],
                                                       Optional[str],
                                                       ASTBlock]],
                             body: ASTBlock,
                             orelse: ASTBlock,
                             finalbody: ASTBlock,
                             context: SExprContextManager) -> ASTStmtBlock:
        assert excepthandler or finalbody is not None

        stmts: List[ASTBlock] = []
        handlers = []

        for typestmt, name, bodystmt in excepthandler:
            if typestmt is not None:
                stmts.append(typestmt)
            bodystmt.drop_result(context=context)
            handlers.append(((typestmt.get_result() if typestmt is not None else None),
                             (name if name is not None else None),
                             list(bodystmt.stmts)))

        body.drop_result(context)
        body = list(body.stmts)
        orelse.drop_result(context)
        orelse = list(orelse.stmts)
        finalbody.drop_result(context)
        finalbody = list(finalbody.stmts)

        stmts.append(ASTStmtBlock([
            ast.Try(body,
                    [ast.ExceptHandler(*x) for x in handlers],
                    orelse,
                    finalbody)
        ], None))

        ret = ASTHelper.pack_block_stmts(stmts)
        ret.free_temp(context)

        return ret

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
    def build_block_from_class_decl(classname: str,
                                    baseclasses: List[Union[ASTBlock, Tuple[Optional[str], ASTBlock]]],
                                    classbody: ASTBlock,
                                    context: SExprContextManager):
        stmts = []
        bases = []
        keywords = []

        for base in baseclasses:
            if isinstance(base, tuple):
                base: Tuple[Optional[str], ASTBlock]
                stmts.append(base[1])
                keywords.append(ast.keyword(base[0], base[1].get_result()))
            elif isinstance(base, ASTBlock):
                arg: ASTBlock
                stmts.append(base)
                bases.append(base.get_result())

        classbody.drop_result(context)
        body = list(classbody.stmts)

        stmts.append(ASTStmtBlock([
            ast.ClassDef(name=classname,
                         bases=bases,
                         keywords=keywords,
                         body=body,
                         decorator_list=[])],
            None))
        ret = ASTHelper.pack_block_stmts(stmts)
        ret.result = ASTHelper.build_block_from_symbol(classname).get_result()

        return ret

    @staticmethod
    def build_arguments(
            arguments: List[Tuple[str, Union[ASTBlock, None, Ellipsis]]]
    ) -> Tuple[List[ASTBlock], ast.arguments]:
        declstmts: List[ASTBlock] = []
        args = []
        vararg = None
        kwonlyargs = []
        kw_defaults = []
        kwarg = None
        defaults = []

        for arg in arguments:
            name, value = arg

            if value is ...:
                assert not vararg
                vararg = ast.arg(name, annotation=None)
            elif vararg:  # kwonly
                kwonlyargs.append(ast.arg(name, annotation=None))

                if isinstance(value, ASTBlock):
                    declstmts.append(value)
                    kw_defaults.append(value.get_result())
                else:
                    kw_defaults.append(value)
            else:
                args.append(ast.arg(name, annotation=None))

                if defaults or isinstance(value, ASTBlock):
                    assert isinstance(value, ASTBlock)
                    declstmts.append(value)
                    defaults.append(value.get_result())
        return (declstmts,
                ast.arguments(
                    args=args,
                    vararg=vararg,
                    kwonlyargs=kwonlyargs,
                    kw_defaults=kw_defaults,
                    kwarg=kwarg,
                    defaults=defaults
                ))

    @staticmethod
    def build_block_from_func_decl(funcname: str,
                                   arguments: List[Tuple[str, Union[ASTBlock, None, Ellipsis]]],
                                   decorators: List[ASTBlock],
                                   body: ASTBlock,
                                   context: SExprContextManager) -> ASTStmtBlock:
        deco_exprs = [decorator.get_result() for decorator in decorators]
        declstmts, args = ASTHelper.build_arguments(arguments)
        declstmts = decorators + declstmts

        body = ASTHelper.build_block_from_return(body, context)

        # TODO: py3.8 kwargs

        declstmts.append(ASTStmtBlock([
            ast.FunctionDef(name=funcname,
                            args=args,
                            body=list(body.stmts),
                            decorator_list=deco_exprs,
                            returns=None)],
            None))
        ret = ASTHelper.pack_block_stmts(declstmts)
        ret.result = ASTHelper.build_block_from_symbol(funcname).get_result()

        return ret

    @staticmethod
    def build_block_from_lambda(arguments: List[Tuple[str, Union[ASTBlock, None, Ellipsis]]],
                                decorators: List[ASTBlock],
                                body: ASTBlock,
                                context: SExprContextManager) -> Optional[ASTStmtBlock]:
        """
        :return: a lambda expr if possible, otherwise return a normal func decl
        """

        declstmts, args = ASTHelper.build_arguments(arguments)

        if any(declstmts) or body:
            return None
        else:
            return ASTStmtBlock([], ast.Lambda(args, body.get_result()))

    @staticmethod
    def build_block_from_op(op: str, left: ASTBlock, right: ASTBlock) -> ASTStmtBlock:

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
            chain0([value],
                   [ASTStmtBlock([
                       ast.Return(ret_val)
                   ], None)]))
        ret.free_temp(context)

        return ret

    @staticmethod
    def build_block_from_delete(targets: List[ASTBlock], context: SExprContextManager) -> ASTStmtBlock:
        stmts = targets[:]
        stmts.append(
            ASTStmtBlock([ast.Delete([stmt.get_result_in_context(ast.Del()) for stmt in stmts])],
                         None))
        ret = ASTHelper.pack_block_stmts(stmts)
        ret.free_temp(context)
        return ret

    @staticmethod
    def pack_block_stmts(blocks: Iterable[ASTBlock]) -> ASTStmtBlock:
        stmts = []

        ret = ASTStmtBlock([], None)

        for block in blocks:
            stmts.append(block.stmts)
            ret.merge_temp(block)

        ret.append_stmt(chain0(*stmts))

        return ret

    @staticmethod
    def compile(result: ASTBlock, context: SExprContextManager) -> FunctionType:
        result.drop_result(context)

        stmts = list(result.stmts)
        node = ast.fix_missing_locations(ast.Module(stmts))

        # import astpretty
        # astpretty.pprint(naive)

        ret = compile(
            node,
            'none',
            'exec'
        )

        return FunctionType(ret, context.env)

    @staticmethod
    def compile_to_code(result: ASTBlock, context: SExprContextManager) -> str:
        from .unparse import Unparser
        from io import StringIO
        result.drop_result(context)

        stmts = list(result.stmts)
        node = ast.fix_missing_locations(ast.Module(stmts))

        output = StringIO()
        Unparser(node, output)

        return output.getvalue()
