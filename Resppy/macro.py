#! /usr/bin/env python3

from __future__ import annotations

import os
import sys
import itertools

from .base import *
from .sexpr import *


class SystemMacro(SExprMacro):
    def __init__(self, func: Callable[..., ASTBlock]):
        self.func = func

    def expand(self, args: list, context) -> ASTBlock:
        return self.func(*args, context=context)


class UserMacro(SExprMacro):
    def __init__(self, func: Callable[..., SExprNodeBase]):
        self.func = func

    def expand(self, args: list, context) -> ASTBlock:
        sexpr = self.func(*args)
        try:
            ret = sexpr.compile(context)
            return ret
        except Exception as e:
            print("Macro expand error: ", sexpr)
            raise e


def quote_macro(body: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    return body.dump_to_ast(context)


def pylist_macro(*content: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    exprs = list(map(lambda x: x.compile(context), content))
    return ASTHelper.build_block_from_list(exprs)


def tuple_macro(*content: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    exprs = list(map(lambda x: x.compile(context), content))
    return ASTHelper.build_block_from_tuple(exprs)


def sharp_macro(content: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    if isinstance(content, SExpr):
        if not isinstance(content[0], SExprSymbol) or content[0] != SExprSymbol("list*"):
            # Tuple
            exprs = list(map(lambda x: x.compile(context), content))
            return ASTHelper.build_block_from_tuple(exprs)
        else:
            # Slice
            params = []

            for param in content[1:]:
                if isinstance(param, SExprSymbol) and param == SExprSymbol("$"):
                    params.append(SExprLiteral(None))
                else:
                    params.append(param)
            while len(params) < 3:
                params.append(SExprLiteral(None))
            assert len(params) == 3
            return SExpr(SExprSymbol("slice"), *params).compile(context)
    elif isinstance(content, SExprLiteral):
        if isinstance(content.value, str):
            return ASTHelper.build_block_from_literal(eval("b" + repr(content.value)))
    else:
        raise ValueError("Unknown arg")


def begin_macro(*content: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    if not content:
        return ASTHelper.build_block_from_literal(None)
    else:
        first, *content = content

    stmts = [first.compile(context)]
    stmts.extend(map(lambda x: x.compile(context), content))
    last = stmts[-1]

    for stmt in stmts[:-1]:
        stmt.drop_result(context)

    ret = ASTHelper.pack_block_stmts(stmts[:-1])

    ret = ASTHelper.pack_block_stmts([ret, last])
    ret.result = last.get_result()

    return ret


def begin_macro_with_new_locals(*content: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    target = lambda_decl_macro(SExpr(), *content, context=context)
    target = ASTHelper.build_block_from_func_call(target, [])

    return target


def with_macro(bindings: SExprNodeBase,
               *body: SExprNodeBase,
               context: SExprContextManager) -> ASTBlock:
    result_name = context.get_temp()

    assert isinstance(bindings, SExpr), "Bindings should be sexpr"

    values = []
    alias = []
    binding_iter = iter(bindings)

    for now in binding_iter:
        if isinstance(now, SExprKeyword):
            assert now == SExprKeyword("as")
            assert alias[-1] is None
            now = next(binding_iter).compile(context)
            assert not now, "binding should not have stmt"
            alias[-1] = now
        else:
            values.append(now)
            alias.append(None)

    values, _ = compile_sequence(*values, context=context)
    body = SExpr(SExprSymbol("set!"),
                 SExprSymbol(result_name),
                 SExpr(SExprSymbol("__begin"), *body)).compile(context)

    ret = ASTHelper.build_block_from_with(list(zip(values, alias)), body, context)
    ret.result = ASTHelper.build_block_from_symbol(result_name).get_result()
    ret.add_temp(result_name)

    return ret


def parse_decl_arg_list(
        args: SExprNodeBase,
        context: SExprContextManager) -> Tuple[List[Tuple[str, Union[None, Ellipsis, ASTBlock]]], bool]:
    param_list: List[Tuple[str, Union[None, Ellipsis, ASTBlock]]] = []

    sexprs = []
    name_pos = []

    if not isinstance(args, SExpr):
        raise ValueError("Param list must be a list")

    for arg in args:
        if isinstance(arg, SExpr):
            assert len(arg) == 2
            assert isinstance(arg[0], SExprSymbol), "Function param must be a symbol"

            if isinstance(arg[1], SExprLiteral) and arg[1].value is ...:
                param_list.append((arg[0].get_mangled_name(), ...))
            else:
                sexprs.append(arg[1])
                name_pos.append((arg[0].get_mangled_name(), len(param_list)))
                param_list.append((arg[0].get_mangled_name(), ASTBlock()))
        elif isinstance(arg, SExprSymbol):
            param_list.append((arg.get_mangled_name(), None))
        else:
            raise ValueError("Function param must be a symbol")

    blocks, applied = compile_sequence(*sexprs, context=context)
    for block, (name, pos) in zip(blocks, name_pos):
        param_list[pos] = (name, block)

    return param_list, applied


def class_decl_macro(name: SExprNodeBase,
                     bases: SExprNodeBase,
                     *body: SExprNodeBase,
                     context: SExprContextManager) -> ASTBlock:
    assert isinstance(name, SExprSymbol), "Class name must be a symbol"
    assert isinstance(bases, SExpr), "Class bases must be a sexpr"

    baseclasses = []
    sexprs = []
    actions = []

    base_iter = iter(bases)
    for sexpr in base_iter:
        if isinstance(sexpr, SExprKeyword):
            kwname = sexpr.get_mangled_name()
            sexpr = next(base_iter)
            sexprs.append(sexpr)
            actions.append(lambda kwval, kwname=kwname: baseclasses.append((kwname, kwval)))
        else:
            sexprs.append(sexpr)
            actions.append(lambda val: baseclasses.append(val))
    blocks, _ = compile_sequence(*sexprs, context=context)

    for block, action in zip(blocks, actions):
        action(block)

    bodystmt = begin_macro(*body, context=context)
    bodystmt.drop_result(context)

    return ASTHelper.build_block_from_class_decl(name.get_mangled_name(),
                                                 baseclasses,
                                                 bodystmt,
                                                 context)


def func_decl_macro(name: SExprNodeBase,
                    args: SExprNodeBase,
                    *body: SExprNodeBase,
                    context: SExprContextManager) -> ASTBlock:
    decorator_sexprs = []
    if isinstance(name, SExpr):  # decorators
        *decorator_sexprs, name = name
    assert isinstance(name, SExprSymbol), "Function name must be a symbol"

    with SExprTempAllocContext(decorator_sexprs, context) as deco_context:
        param_list, applied = parse_decl_arg_list(args, context)

        if applied or any(deco_context.blocks):
            deco_context.apply()

        decorators = deco_context.blocks

    context.push_context()
    body = begin_macro(*body, context=context)
    context.pop_context()

    ret = ASTHelper.build_block_from_func_decl(name.get_mangled_name(),
                                               param_list,
                                               decorators,
                                               body,
                                               context)
    return ret


def lambda_decl_macro(args: SExprNodeBase,
                      *body: SExprNodeBase,
                      context: SExprContextManager) -> ASTBlock:
    param_list, applied = parse_decl_arg_list(args, context)

    context.push_context()
    body = begin_macro(*body, context=context)
    context.pop_context()

    ret = ASTHelper.build_block_from_lambda(param_list,
                                            [],
                                            body,
                                            context)

    if ret is None:
        name = context.get_temp()
        ret = ASTHelper.build_block_from_func_decl(name,
                                                   param_list,
                                                   [],
                                                   body,
                                                   context)
        ret.add_temp(name)
    return ret


def import_macro(*args: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    stmts: List[ASTBlock] = []

    it = iter(args)

    while True:

        modules: List[Tuple[str, Optional[str]]] = []
        for arg in it:
            if isinstance(arg, SExprKeyword):
                assert arg == SExprKeyword("as")
                assert not modules[-1][1]
                arg = next(it)
                assert isinstance(arg, SExprSymbol)
                modules[-1] = (modules[-1][0], arg.get_mangled_name())
            elif isinstance(arg, SExprSymbol):
                modules.append((arg.get_mangled_name(), None))
            elif isinstance(arg, SExpr):
                break
            else:
                assert False, "Unknown argument type"
        else:
            if len(modules):
                stmts.append(ASTHelper.build_block_from_import(modules))
            break

        if len(modules):
            stmts.append(ASTHelper.build_block_from_import(modules))

        assert isinstance(arg, SExpr)

        itarg = iter(arg)

        module = next(itarg)
        assert isinstance(module, SExprSymbol)
        module_name = module.get_mangled_name()
        real_module_name = module_name.lstrip(".")
        floor = len(module_name) - len(real_module_name)
        names: List[Tuple[str, Optional[str]]] = []

        for now in itarg:
            if isinstance(now, SExprKeyword):
                assert now == SExprKeyword("as")
                assert not names[-1][1]
                now = next(itarg)
                assert isinstance(now, SExprSymbol)
                names[-1] = (names[-1][0], now.get_mangled_name())
            elif isinstance(now, SExprSymbol):
                names.append((now.get_mangled_name(), None))
            else:
                assert False, "Unknown argument type"

        stmts.append(ASTHelper.build_block_from_import_from(real_module_name if real_module_name else None,
                                                            names,
                                                            floor))

    return ASTHelper.pack_block_stmts(stmts)


def for_macro(bindings: SExprNodeBase,
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

        name: SExprNodeBase
        value: SExprNodeBase

        name: ASTBlock = name.compile(context)
        assert not name, "binding should not have stmts"
        names.append(name)
        values.append(value)

    values = SExpr(SExprSymbol("zip"), *values).compile(context)

    body = begin_macro(*content, context=context)

    return ASTHelper.build_block_from_for(
        ASTHelper.build_block_from_tuple(names),
        values,
        body,
        ASTHelper.build_block_from_pass(),
        context
    )


def for_list_macro(bindings: SExprNodeBase,
                   *content: SExprNodeBase,
                   context: SExprContextManager) -> ASTBlock:
    result_name = context.get_temp()
    append_name = context.get_temp()
    body = SExpr(SExprSymbol(append_name),
                 SExpr(SExprSymbol("__begin"), *content))
    ret = for_macro(bindings, body, context=context)
    ret.drop_result(context)
    ret = ASTHelper.pack_block_stmts([
        ASTHelper.build_block_from_assign(ASTHelper.build_block_from_symbol(result_name),
                                          ASTHelper.build_block_from_list([]),
                                          context),
        ASTHelper.build_block_from_assign(ASTHelper.build_block_from_symbol(append_name),
                                          ASTHelper.build_block_from_symbol(result_name + ".append"),
                                          context),
        ret
    ])
    ret.add_temp(append_name)
    ret.free_temp(context)
    ret.add_temp(result_name)
    ret.result = ASTHelper.build_block_from_symbol(result_name).get_result()

    return ret


def for_tuple_macro(bindings: SExprNodeBase,
                    *content: SExprNodeBase,
                    context: SExprContextManager) -> ASTBlock:
    return SExpr(SExprSymbol('tuple*'),
                 SExpr(SExprSymbol("#*"),
                       SExpr(SExprSymbol("__for/list"), bindings, *content))).compile(context)


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
                                           ASTHelper.build_block_from_pass(),
                                           context)
    return ret


def break_macro(context: SExprContextManager) -> ASTBlock:
    return ASTHelper.build_block_from_break()


def continue_macro(context: SExprContextManager) -> ASTBlock:
    return ASTHelper.build_block_from_continue()


def pass_macro(context: SExprContextManager) -> ASTBlock:
    return ASTHelper.build_block_from_pass()


def when_macro(test: SExprNodeBase,
               first: SExprNodeBase,
               *content: SExprNodeBase,
               context: SExprContextManager) -> ASTBlock:
    return if_macro(
        test,
        SExpr(SExprSymbol('__begin'), first, *content),
        SExprLiteral(None),
        context
    )


def when_macro_with_new_locals(test: SExprNodeBase,
                               first: SExprNodeBase,
                               *content: SExprNodeBase,
                               context: SExprContextManager) -> ASTBlock:
    return if_macro(
        test,
        SExpr(SExprSymbol('begin'), first, *content),
        SExprLiteral(None),
        context
    )


def unless_macro(test: SExprNodeBase,
                 first: SExprNodeBase,
                 *content: SExprNodeBase,
                 context: SExprContextManager) -> ASTBlock:
    return if_macro(
        test,
        SExprLiteral(None),
        SExpr(SExprSymbol('__begin'), first, *content),
        context
    )


def unless_macro_with_new_locals(test: SExprNodeBase,
                                 first: SExprNodeBase,
                                 *content: SExprNodeBase,
                                 context: SExprContextManager) -> ASTBlock:
    return if_macro(
        test,
        SExprLiteral(None),
        SExpr(SExprSymbol('begin'), first, *content),
        context
    )


def set_macro(target: SExprNodeBase,
              value: SExprNodeBase,
              context: SExprContextManager) -> ASTBlock:
    return ASTHelper.build_block_from_assign(target.compile(context),
                                             value.compile(context),
                                             context)


def subscr_macro(target: SExprNodeBase,
                 *indexes: SExprNodeBase,
                 context: SExprContextManager) -> ASTBlock:
    ret = target.compile(context)

    for index in indexes:
        ret = ASTHelper.build_block_from_subscr(ret, index.compile(context))

    return ret


def set_subscr_macro(target: SExprNodeBase,
                     *indexes: SExprNodeBase,
                     context: SExprContextManager) -> ASTBlock:
    ret = target.compile(context)

    value = indexes[-1]
    indexes = indexes[:-1]

    for index in indexes:
        ret = ASTHelper.build_block_from_subscr(ret, index.compile(context))

    return ASTHelper.build_block_from_assign(ret, value.compile(context), context)


def dot_macro(source: SExprNodeBase,
              *attrs: SExprNodeBase,
              context: SExprContextManager) -> ASTBlock:
    target = source.compile(context)

    for attr in attrs:
        assert isinstance(attr, SExprSymbol)
        target = ASTHelper.build_block_from_getattr(target, attr.get_mangled_name())

    return target


def void_macro(*args, context: SExprContextManager) -> ASTBlock:
    return ASTHelper.build_block_from_literal(None)


def unpack_iterable_macro(arg: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    return ASTValues(arg.compile(context))


def contains_macro(left: SExprNodeBase, right: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    left = left.compile(context)
    right = right.compile(context)

    return ASTHelper.build_block_from_op('in', right, left)


def quasiquote_macro(target: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
    if not isinstance(target, SExpr):
        return target.dump_to_ast(context)
    else:
        if len(target) == 0 or (target[0] != SExprSymbol("#~") and target[0] != SExprSymbol("#~*")):
            params = []

            for content in target:
                params.append(quasiquote_macro(content, context))

            return ASTHelper.build_block_from_func_call(
                ASTHelper.build_block_from_symbol(SExpr.__name__),
                params)
        elif target[0] == SExprSymbol("#~"):
            return target[1].compile(context)
        else:
            return unpack_iterable_macro(target[1], context=context)


def try_macro(body: SExprNodeBase,
              *others: SExprNodeBase,
              context: SExprContextManager) -> ASTBlock:
    result = context.get_temp()

    handlers: List[Tuple[Optional[ASTBlock],
                         Optional[str],
                         ASTBlock]] = []
    elsestmt = None
    finalstmt = None

    for head, *handlerbody in others:
        bodystmt = SExpr(SExprSymbol("set!"),
                         SExprSymbol(result),
                         SExpr(SExprSymbol("__begin"),
                               *handlerbody)).compile(context)
        if isinstance(head, SExpr):
            assert len(head) <= 2
            if len(head) == 0:
                handlers.append((None,
                                 None,
                                 bodystmt))
            else:
                typeexpr = head[0].compile(context)

                if len(head) == 2:
                    assert isinstance(head[1], SExprSymbol)
                    alias: Optional[str] = head[1].get_mangled_name()
                else:
                    alias = None

                if typeexpr:
                    typeexpr = ASTHelper.build_block_from_lambda([], [], typeexpr, context)
                    typeexpr = ASTHelper.build_block_from_func_call(typeexpr, [])
                handlers.append((typeexpr,
                                 alias,
                                 bodystmt))
        else:
            assert isinstance(head, SExprKeyword)
            if SExprKeyword("else") == head:
                assert elsestmt is None
                elsestmt = bodystmt
            elif SExprKeyword("finally") == head:
                assert finalstmt is None
                finalstmt = bodystmt
            else:
                assert False, "Keyword should be else or finally"

    if elsestmt is None:
        elsestmt = ASTHelper.build_block_from_pass()

    if finalstmt is None:
        finalstmt = ASTHelper.build_block_from_pass()

    body = SExpr(SExprSymbol("set!"),
                 SExprSymbol(result),
                 body).compile(context)

    ret = ASTHelper.build_block_from_try(handlers,
                                         body,
                                         elsestmt,
                                         finalstmt,
                                         context)

    ret.result = ASTHelper.build_block_from_symbol(result).get_result()
    ret.add_temp(result)
    return ret


def raise_macro(*body: SExprNodeBase,
                context: SExprContextManager) -> ASTBlock:
    assert len(body) <= 2
    body, _ = compile_sequence(*body, context=context)
    while len(body) < 2:
        body.append(None)
    return ASTHelper.build_block_from_raise(*body, context=context)


def defmacro_macro(name: SExprNodeBase,
                   args: SExprNodeBase,
                   *body: SExprNodeBase,
                   context: SExprContextManager) -> ASTBlock:
    assert isinstance(name, SExprSymbol)
    function = func_decl_macro(name, args, *body, context=context)

    exec(ASTHelper.compile(function, context), context.env)

    function = context.env[name.get_mangled_name()]
    context.register(str(name), UserMacro(function))

    return ASTHelper.build_block_from_literal(None)


def generate_default_context() -> SExprContextManager:
    def register_global_variable(name: str, func, context: SExprContextManager):
        context.env[sexpr_mangle(name)] = func

    def register_op(op: str, context: SExprContextManager, alias: Optional[str] = None):
        def op_macro(left: SExprNodeBase, right: SExprNodeBase, context: SExprContextManager) -> ASTBlock:
            left = left.compile(context)
            right = right.compile(context)

            return ASTHelper.build_block_from_op(op, left, right)

        context.register(alias if alias else op, SystemMacro(op_macro))

    context = SExprContextManager()

    context.register('#', SystemMacro(sharp_macro))
    context.register('#*', SystemMacro(unpack_iterable_macro))

    context.register('quote', SystemMacro(quote_macro))
    context.register('list*', SystemMacro(pylist_macro))
    context.register('tuple*', SystemMacro(tuple_macro))
    context.register('__begin', SystemMacro(begin_macro))
    context.register('begin', SystemMacro(begin_macro_with_new_locals))
    context.register('defn', SystemMacro(func_decl_macro))
    context.register('fn', SystemMacro(lambda_decl_macro))
    context.register('defmacro', SystemMacro(defmacro_macro))
    context.register('defclass', SystemMacro(class_decl_macro))
    context.register('import', SystemMacro(import_macro))
    context.register('__for', SystemMacro(for_macro))
    context.register('__for/list', SystemMacro(for_list_macro))
    context.register('__for/tuple', SystemMacro(for_tuple_macro))
    context.register('with', SystemMacro(with_macro))
    context.register('__while', SystemMacro(while_macro))
    context.register('__break', SystemMacro(break_macro))
    context.register('__continue', SystemMacro(continue_macro))
    context.register('pass', SystemMacro(pass_macro))
    context.register('try', SystemMacro(try_macro))
    context.register('raise', SystemMacro(raise_macro))
    context.register('if', SystemMacro(if_macro))
    context.register('when', SystemMacro(when_macro))
    context.register('unless', SystemMacro(unless_macro))
    context.register('set!', SystemMacro(set_macro))
    context.register('getscr', SystemMacro(subscr_macro))
    context.register('setscr!', SystemMacro(set_subscr_macro))
    context.register('void', SystemMacro(void_macro))
    context.register('unpack-iterable', SystemMacro(unpack_iterable_macro))
    context.register('quasiquote', SystemMacro(quasiquote_macro))
    context.register('.', SystemMacro(dot_macro))
    register_op('+', context)
    register_op('-', context)
    register_op('*', context)
    register_op('@', context)
    register_op('/', context)
    register_op('//', context)
    register_op('%', context)
    register_op('**', context)
    register_op('>>', context)
    register_op('<<', context)
    register_op('&', context)
    register_op('|', context)
    register_op('^', context)
    register_op('<', context)
    register_op('<=', context)
    register_op('==', context, 'eq?')
    register_op('!=', context)
    register_op('>', context)
    register_op('>=', context)
    context.register('contains?', SystemMacro(contains_macro))

    register_global_variable("void", lambda *args, **kwargs: None, context)
    register_global_variable("list*", lambda *args: [*args], context)
    register_global_variable("tuple*", lambda *args: (*args,), context)
    register_global_variable("+", lambda a, b: a + b, context)
    register_global_variable("-", lambda a, b: a - b, context)
    register_global_variable("*", lambda a, b: a * b, context)
    register_global_variable("/", lambda a, b: a / b, context)
    register_global_variable("//", lambda a, b: a // b, context)
    register_global_variable("%", lambda a, b: a % b, context)
    register_global_variable("<<", lambda a, b: a << b, context)
    register_global_variable(">>", lambda a, b: a >> b, context)
    register_global_variable("&", lambda a, b: a & b, context)
    register_global_variable("|", lambda a, b: a | b, context)
    register_global_variable("^", lambda a, b: a ^ b, context)
    register_global_variable("<", lambda a, b: a < b, context)
    register_global_variable("<=", lambda a, b: a <= b, context)
    register_global_variable(">", lambda a, b: a > b, context)
    register_global_variable(">=", lambda a, b: a >= b, context)
    register_global_variable("eq?", lambda a, b: a == b, context)
    register_global_variable("**", lambda a, b: a ** b, context)
    register_global_variable("@", lambda a, b: a @ b, context)
    register_global_variable("contains?", lambda a, b: b in a, context)

    def get_subscr_func(target, *indexes):
        for index in indexes:
            target = target[index]
        return target

    def set_subscr_func(target, *values):
        value = values[-1]
        last_ind = values[-2]
        values = values[:-2]

        for index in values:
            target = target[index]

        target[last_ind] = values

    def get_gensym():
        counter = itertools.count()

        def gensym():
            ret = next(counter)
            print(ret)
            return SExprSymbol("_TS_%d" % ret)

        return gensym

    register_global_variable("getscr", get_subscr_func, context)
    register_global_variable("setscr!", set_subscr_func, context)
    register_global_variable("gensym", get_gensym(), context)

    from .compiler import compile_stream_to_code
    with open(os.path.join(sys.modules[__package__].__path__[0], "macros.rsp"), encoding="utf-8") as fin:
        context.headers.append(compile_stream_to_code(fin, context))

    return context
