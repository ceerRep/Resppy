#! /usr/bin/env python3

from __future__ import annotations

from .base import *
from .sexpr import *


class SystemMacro(SExprMacro):
    def __init__(self, func: Callable[..., ASTBlock]):
        self.func = func

    def expand(self, args: list, context) -> ASTBlock:
        return self.func(*args, context=context)


def quote_macro(body: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    return body.dump_to_ast(context)


def pylist_macro(*content: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    exprs = list(map(lambda x: x.compile(context), content))
    return ASTHelper.build_block_from_list(exprs)


def begin_macro(first: SExprNodeBase, *content: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    stmts = [first.compile(context)]
    stmts.extend(map(lambda x: x.compile(context), content))
    last = stmts[-1]

    for stmt in stmts[:-1]:
        stmt.drop_result(context)

    ret = ASTHelper.pack_block_stmts(stmts[:-1])

    ret = ASTHelper.pack_block_stmts([ret, last])
    ret.result = last.get_result()

    return ret


def func_decl_macro(name: SExprNodeBase,
                    args: SExprNodeBase,
                    *body: SExprNodeBase,
                    context: SExprContextManager) -> ASTBlock:
    if not isinstance(name, SExprSymbol):
        raise ValueError("Function name must be a symbol")

    # TODO: kwargs
    param_list: List[str] = []

    if not isinstance(args, SExpr):
        raise ValueError("Param list must be a list")

    for arg in args:
        if not isinstance(arg, SExprSymbol):
            raise ValueError("Function param must be a symbol")
        param_list.append(arg.get_mangled_name())

    stmts: List[ASTBlock] = [stmt.compile(context) for stmt in body]

    ret = ASTHelper.build_block_from_func_decl(name.get_mangled_name(),
                                               param_list,
                                               stmts,
                                               context)

    return ret


def lanbda_decl_macro(args: SExprNodeBase,
                      *body: SExprNodeBase,
                      context: SExprContextManager) -> ASTBlock:
    name = context.get_temp()

    # TODO: kwargs
    param_list: List[str] = []

    if not isinstance(args, SExpr):
        raise ValueError("Param list must be a list")

    for arg in args:
        if not isinstance(arg, SExprSymbol):
            raise ValueError("Function param must be a symbol")
        param_list.append(arg.get_mangled_name())

    stmts: List[ASTBlock] = [stmt.compile(context) for stmt in body]

    ret = ASTHelper.build_block_from_func_decl(name,
                                               param_list,
                                               stmts,
                                               context)
    ret.add_temp(name)
    return ret


def import_macro(*args: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    modules: List[str] = []

    # TODO alias
    for arg in args:
        if not isinstance(arg, SExprSymbol):
            raise ValueError("Module name must be an symbol")
        modules.append(arg.get_mangled_name())

    return ASTHelper.build_block_from_import(modules)


def for_macro(bindings: SExprNodeBase,
              first: SExprNodeBase,
              *content: SExprNodeBase,
              context: SExprContextManager) -> ASTBlock:
    if not isinstance(bindings, SExpr):
        raise ValueError("bindings must be an sexpr")

    names = []
    values = []

    for binding in bindings:
        if not isinstance(binding, SExpr) or len(binding) != 2:
            raise ValueError("bindings must be an sexpr")
        name, value = binding

        assert (isinstance(name, SExprSymbol))
        value: SExprNodeBase

        names.append(name.compile(context))
        values.append(value.compile(context))

    body = begin_macro(first, *content, context=context)

    return ASTHelper.build_block_from_for(
        ASTHelper.build_block_from_tuple(names),
        ASTHelper.build_block_from_func_call(
            ASTHelper.build_block_from_symbol("zip"),
            values),
        body,
        ASTHelper.build_block_from_literal(...),
        context
    )


def if_macro(test: SExprNodeBase,
             body: SExprNodeBase,
             elbody: SExprNodeBase,
             context: SExprContextManager) -> ASTBlock:
    result_name = context.get_temp()
    test = test.compile(context)
    body = ASTHelper.build_block_from_assign(ASTHelper.build_block_from_symbol(result_name),
                                             body.compile(context),
                                             context)
    elbody = ASTHelper.build_block_from_assign(ASTHelper.build_block_from_symbol(result_name),
                                               elbody.compile(context),
                                               context)

    ret = ASTHelper.build_block_from_if(test, body, elbody, context)
    ret.add_temp(result_name)
    ret.result = ASTHelper.build_block_from_symbol(result_name).get_result()
    return ret


def while_macro(test: SExprNodeBase,
                first: SExprNodeBase,
                *content: SExprNodeBase,
                context: SExprContextManager) -> ASTBlock:
    test = test.compile(context)
    body = begin_macro(first, *content, context=context)
    body.drop_result(context)

    ret = ASTHelper.build_block_from_while(test,
                                           body,
                                           ASTHelper.build_block_from_literal(...),
                                           context)
    return ret


def when_macro(test: SExprNodeBase,
               first: SExprNodeBase,
               *content: SExprNodeBase,
               context: SExprContextManager) -> ASTBlock:
    test = test.compile(context)
    body = begin_macro(first, *content, context=context)
    body.drop_result(context)

    ret = ASTHelper.build_block_from_if(test,
                                        body,
                                        ASTHelper.build_block_from_literal(...),
                                        context)
    return ret


def unless_macro(test: SExprNodeBase,
                 first: SExprNodeBase,
                 *content: SExprNodeBase,
                 context: SExprContextManager) -> ASTBlock:
    test = test.compile(context)
    body = begin_macro(first, *content, context=context)
    body.drop_result(context)

    ret = ASTHelper.build_block_from_if(test,
                                        ASTHelper.build_block_from_literal(...),
                                        body,
                                        context)
    return ret


def setv_macro(target: SExprNodeBase,
               value: SExprNodeBase,
               context: SExprContextManager) -> ASTBlock:
    return ASTHelper.build_block_from_assign(target.compile(context),
                                             value.compile(context),
                                             context)


def void_macro(context: SExprContextManager) -> ASTBlock:
    return ASTHelper.build_block_from_literal(None)


if __name__ == "__main__":
    def main():
        from io import StringIO
        import uncompyle6
        from .compiler import compile_stream

        context = SExprContextManager()
        context.register('quote', SystemMacro(quote_macro))
        context.register('pylist', SystemMacro(pylist_macro))
        context.register('begin', SystemMacro(begin_macro))
        context.register('defn', SystemMacro(func_decl_macro))
        context.register('fn', SystemMacro(lanbda_decl_macro))
        context.register('import', SystemMacro(import_macro))
        context.register('for', SystemMacro(for_macro))
        context.register('while', SystemMacro(while_macro))
        context.register('if', SystemMacro(if_macro))
        context.register('when', SystemMacro(when_macro))
        context.register('unless', SystemMacro(unless_macro))
        context.register('setv', SystemMacro(setv_macro))
        context.register('void', SystemMacro(void_macro))

        f = compile_stream(
            StringIO(
                """ (begin 
                      (setv i 0)
                      (print i)
                      (setv [i j] [2 3])
                      (print i)
                      (void))
                    """),
            context,
            globals())

        # import dis
        # dis.dis(f)

        uncompyle6.deparse_code2str(f.__code__)
        print()
        f()


    main()
