#! /usr/bin/env python3

from __future__ import annotations

from typing import *

from .base import *
from .asthelper import *


class SExprNodeBase:
    def __str__(self):
        raise NotImplementedError("SExprNodeBase.__str__")

    def __repr__(self):
        raise NotImplementedError("SExprNodeBase.__repr__")

    def dump_to_ast(self, context: SExprContextManager) -> ASTBlock:
        raise NotImplementedError("SExprNodeBase.dump_to_ast")

    def compile(self, context: SExprContextManager) -> ASTBlock:
        raise NotImplementedError("SExprNodeBase.compile")


class SExprLiteral(SExprNodeBase):
    def __init__(self, value: Union[bool, int, float, complex, str, bytes, None, ...]):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.value) + ")"

    def dump_to_ast(self, context: SExprContextManager) -> ASTBlock:
        return ASTHelper.build_block_from_func_call(
            ASTHelper.build_block_from_symbol(self.__class__.__name__),
            [ASTHelper.build_block_from_literal(self.value)]
        )

    def compile(self, context: SExprContextManager) -> ASTBlock:
        return ASTHelper.build_block_from_literal(self.value)


class SExprKeyword(SExprNodeBase):
    def __init__(self, key: str):
        self.key = SExprSymbol(key)

    def __str__(self) -> str:
        return ":" + str(self.key)

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.key) + ")"

    def dump_to_ast(self, context: SExprContextManager) -> ASTBlock:
        return ASTHelper.build_block_from_func_call(
            ASTHelper.build_block_from_symbol(self.__class__.__name__),
            [ASTHelper.build_block_from_literal(str(self.key))]
        )

    def compile(self, context: SExprContextManager) -> ASTBlock:
        raise NotImplementedError("SExprKeyword.compile")

    def get_mangled_name(self):
        return self.key.get_mangled_name()

    def __eq__(self, other: SExprKeyword):
        if not isinstance(other, SExprKeyword):
            return False
        return self.key == other.key

    def __ne__(self, other):
        return not self == other


class SExprSymbol(SExprNodeBase):
    def __init__(self, name: str):
        self.name = name
        self.mangled = self.mangle()

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return "SExprSymbol(" + repr(self.name) + ")"

    def mangle(self):
        return sexpr_mangle(self.name)

    def get_mangled_name(self):
        return self.mangled

    def dump_to_ast(self, context: SExprContextManager) -> ASTBlock:
        return ASTHelper.build_block_from_func_call(
            ASTHelper.build_block_from_symbol(self.__class__.__name__),
            [ASTHelper.build_block_from_literal(self.name)]
        )

    def compile(self, context: SExprContextManager) -> ASTBlock:
        return ASTHelper.build_block_from_symbol(self.get_mangled_name())

    def __eq__(self, other: SExprSymbol):
        if not isinstance(other, SExprSymbol):
            return False
        return self.name == other.name

    def __ne__(self, other):
        return not self == other


class SExpr(SExprNodeBase):
    def __init__(self, *args: SExprNodeBase):
        self.expr: List[SExprNodeBase] = list(args[::-1])

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = None
            stop = None
            step = -1
            if key.start:
                start = -key.start - 1
            if key.stop:
                stop = -key.stop - 1
            if key.step:
                step = -key.step
            key = slice(start, stop, step)
        else:
            key = - key - 1
        return self.expr[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start = None
            stop = None
            step = -1
            if key.start:
                start = -key.start - 1
            if key.stop:
                stop = -key.stop - 1
            if key.step:
                step = -key.step
            key = slice(start, stop, step)
        else:
            key = - key - 1
        self.expr[key] = value

    def __iter__(self):
        return reversed(self.expr)

    def compile(self, context: SExprContextManager) -> ASTBlock:
        target = self[0]

        if isinstance(target, SExprSymbol) and context.resolve(target.get_mangled_name()):
            return context.resolve(target.get_mangled_name()).expand(self[1:], context)
        else:
            sexprs = [target]
            actions = [lambda x: None]
            params: List[Union[ASTBlock, Tuple[Optional[str], ASTBlock]]] = []

            self_iter: Iterator[SExprNodeBase] = iter(self)
            next(self_iter)

            for item in self_iter:
                if isinstance(item, SExprKeyword):
                    try:
                        kwname = item.get_mangled_name()
                        sexprs.append(next(self_iter))
                        actions.append(lambda kwval, kwname=kwname: params.append((kwname, kwval)))
                    except StopIteration:
                        raise ValueError("Excepted value for keyword " + str(item))
                else:
                    sexprs.append(item)
                    actions.append(lambda arg: params.append(arg))

            blocks, _ = compile_sequence(*sexprs, context=context)

            func = blocks[0]
            for block, action in zip(blocks, actions):
                action(block)

            return ASTHelper.build_block_from_func_call(
                func,
                params
            )

    def dump_to_ast(self, context: SExprContextManager) -> ASTBlock:
        params = []

        for content in self:
            params.append(content.dump_to_ast(context))

        return ASTHelper.build_block_from_func_call(
            ASTHelper.build_block_from_symbol(self.__class__.__name__),
            params
        )

    def __str__(self):
        rets = ['(']
        for item in self:
            rets.append(str(item))
            rets.append(' ')
        if len(self) != 0:
            rets.pop()
        rets.append(')')
        return ''.join(rets)

    def __repr__(self):
        rets = [self.__class__.__name__, '(']
        for item in self:
            rets.append(repr(item))
            rets.append(', ')
        if len(self) != 0:
            rets[-1] = ','
        rets.append(')')
        return ''.join(rets)


class SExprTempAllocContext:
    def __init__(self, nodes: Iterable[SExprNodeBase], context: SExprContextManager):
        self.names = []
        self.nodes = nodes
        self.context = context
        self.applied = False

    def __enter__(self):
        self.names = [self.context.get_temp() for _ in self.nodes]
        self.blocks = [node.compile(self.context) for node in self.nodes]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        blocks = iter(self.blocks)
        names = iter(self.names)
        if self.applied:
            for block, name in zip(blocks, names):
                block.apply_result(self.context, name)
            self.names = []
        for name in names:
            self.context.free_temp(name)

    def compile(self, node: SExprNodeBase) -> ASTBlock:
        tmp = self.context.get_temp()
        self.names.append(tmp)
        ret = node.compile(self.context)
        self.blocks.append(ret)
        return ret

    def apply(self):
        self.applied = True


def compile_sequence(*nodes: SExprNodeBase, context: SExprContextManager) -> Tuple[List[ASTBlock], bool]:
    applied = False

    with SExprTempAllocContext(nodes, context) as temp_context:
        if any(temp_context.blocks):
            applied = True
            temp_context.apply()

        blocks = temp_context.blocks

    return blocks, applied


if __name__ == '__main__':
    def main():
        a = 3

        def naive(x, y):
            print(x + y)

        sexpr = SExpr(
            SExprSymbol('print'),
            SExprLiteral('1'),
            SExprLiteral(2)
        )

        context = SExprContextManager()

        result = sexpr.compile(context)
        code = ASTHelper.compile(result, context)

        dump = sexpr.dump_to_ast(context)

        import astpretty
        import uncompyle6
        code()
        uncompyle6.deparse_code2str(code.__code__)


    main()
