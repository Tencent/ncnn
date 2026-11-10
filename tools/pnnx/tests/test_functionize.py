# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Shared functionize regressions, tested without export's run_decompositions()."""

from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile

import torch
import torch.nn.functional as F
from packaging import version


class Aliases(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        base = x + 1
        if self.mode == 'full_slice':
            before = base.clone()
            result = base[:, :].add_(2)
            return before, base, result
        if self.mode == 'unbind_writeback':
            left, right = base[0], base[1]
            before = right.clone()
            left.add_(2)
            return before, base, left, right
        if self.mode == 'split_writeback':
            left, right = base[:1], base[1:]
            before = right.clone()
            left.add_(2)
            return before, base, left, right
        if self.mode == 'dynamic_bounds':
            base = torch.cat((base, base), 0)
            view = base[:x.size(0) * 2 - 1, 1:]
            before = view.clone()
            view.add_(2)
            return before, base, view
        if self.mode == 'chained_view':
            result = base[:, 1:].relu_()
            result[:, 1:].add_(1)
            return base, result
        if self.mode == 'chained_select':
            result = base.select(0, 1).relu_()
            result[1:].add_(2)
            return base, result
        if self.mode == 'repeated_operand':
            view = base[:, 1:]
            view.add_(view)
            return base, view, view + view
        if self.mode == 'unrelated_write':
            view = base.transpose(0, 1)
            before = view.clone()
            base.add_(2)
            middle = view + 1
            other = (x + 3).relu_()
            return before, middle, view + 2, other
        if self.mode == 'nested_slice':
            outer = base[:, 1:]
            inner = outer[:, 1:]
            before = outer.clone()
            result = inner.add_(3)
            return before, base, outer, inner, result
        if self.mode == 'negative_permute':
            base = base.unsqueeze(1)
            view = base.permute(-1, -3, -2)
            before = view.clone()
            result = view[1:3].add_(2)
            return before, base, view, result
        if self.mode == 'siblings':
            left, right = base[:, :3], base[:, 1:]
            before = left.clone()
            result = left.mul_(2)
            right.add_(3)
            return before, base, left, right, result
        if self.mode == 'nonfirst':
            alias = base[:, 1:]
            before = torch.sub(torch.ones_like(alias), alias)
            base.add_(2)
            after = torch.sub(torch.ones_like(alias), alias)
            return before, after, base, alias
        if self.mode == 'chained':
            result = base.relu_()
            alias = result[:, 1:]
            before = alias.clone()
            result.add_(2)
            result[:, 2:].mul_(3)
            return before, base, result, alias
        if self.mode == 'multiinput':
            left, right = base[:, :3], base[:, 1:]
            before = torch.stack((left, right))
            base.add_(2)
            return before, torch.stack((left, right)), base
        if self.mode in ('copy', 'fill'):
            alias = base[:, 1:]
            before = alias.clone()
            if self.mode == 'copy':
                result = alias.copy_(torch.full_like(alias, 7))
            else:
                result = alias.fill_(7)
            return before, base, alias, result
        if self.mode == 'select':
            alias = base.select(1, 1)
            before = alias.clone()
            result = alias.add_(4)
            return before, base, alias, result
        if self.mode in ('view_slice_base', 'view_select_base'):
            view = base.view(-1)
            sibling = base[:, 1:]
            before = sibling.clone()
            target = view[2:] if self.mode == 'view_slice_base' else view.select(0, 2)
            result = target.add_(2)
            return before, base, view, sibling, result
        if self.mode in ('view_base', 'transpose', 'permute', 'unsqueeze', 'squeeze'):
            if self.mode == 'view_base':
                view = base.view(-1)
            elif self.mode == 'transpose':
                view = base.transpose(0, 1)
            elif self.mode == 'permute':
                view = base.permute(1, 0)
            elif self.mode == 'unsqueeze':
                view = base.unsqueeze(1)
            else:
                view = base.unsqueeze(1).squeeze(1)
            sibling = base[:, 1:]
            before = sibling.clone()
            view.add_(2)
            return before, base, view, sibling
        view = base[:1].view(-1)
        alias = view[1:]
        before = alias.clone()
        if self.mode == 'view_slice':
            result = view[2:].add_(2)
        elif self.mode == 'view_select':
            result = view.select(0, 2).add_(2)
        else:
            result = view.add_(2)
            result.mul_(3)
        return before, alias, result


class ConditionalAlias(torch.nn.Module):
    __constants__ = ['operation', 'view', 'metadata']

    def __init__(self, operation, view='root', metadata=False):
        super().__init__()
        self.operation = operation
        self.view = view
        self.metadata = metadata

    def forward(self, x):
        base = x + 1
        if self.view == 'slice':
            target = base[:, :, 1:]
        elif self.view == 'select':
            target = base.select(0, 1)
        elif self.view == 'transpose':
            target = base.transpose(0, 1)
        else:
            target = base
        old = target.transpose(0, 1)
        child = old.select(0, 0)
        items = [target, target]
        before = base.clone()
        snapshot = old + 3
        if self.operation == 'dropout':
            result = F.dropout(target, 0.5, False)
        elif self.operation == 'dropout2d':
            result = F.dropout2d(target, 0.5, False)
        elif self.operation == 'dropout3d':
            result = F.dropout3d(target, 0.5, False)
        elif self.operation == 'alpha_dropout':
            result = F.alpha_dropout(target, 0.5, False)
        elif self.operation == 'feature_alpha_dropout':
            result = F.feature_alpha_dropout(target, 0.5, False)
        elif self.operation == 'type_as':
            result = target.type_as(x)
        elif self.operation == 'type_as_half':
            result = target.type_as(x.half())
        elif self.operation == 'to':
            result = target.to(dtype=torch.float32, copy=False)
        elif self.operation == 'to_copy':
            result = target.to(dtype=torch.float32, copy=True)
        elif self.operation == 'to_half':
            result = target.to(dtype=torch.float16)
        else:
            result = target.clone()
        if self.metadata:
            result.transpose_(0, 1)
        layout = target.clone()
        result.add_(2)
        middle = old + 5
        # The new view uses the current geometry; old and child keep theirs.
        new = result.select(0, 0)
        new.mul_(3)
        base.add_(1)
        return before, snapshot, layout, middle, base, target, result, old, child, torch.stack(items)


class CastPrecisionBoundary(torch.nn.Module):
    __constants__ = ['mode']

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        base = x + 1
        if self.mode == 'same':
            base = base.half()
        converted = base.half()
        if self.mode in ('source', 'source_metadata'):
            old = converted[:, 1:]
            if self.mode == 'source_metadata':
                base.transpose_(0, 1)
                base.add_(2)
            else:
                base[:, 1:].add_(2)
        else:
            old = base[:, 1:]
            converted.add_(2)
        # FP32 outputs still observe whether the earlier half cast copied.
        return base.float(), old.float(), converted.float()


class UnaryCastSource(torch.nn.Module):
    __constants__ = ['operation', 'source']

    def __init__(self, operation, source):
        super().__init__()
        self.operation = operation
        self.source = source

    def forward(self, x):
        base = x.exp() if self.operation == 'exp' else x.tanh()
        old = base.transpose(0, 1)
        converted = base.half()
        converted.transpose_(0, 1)
        if self.source:
            base.add_(2)
        else:
            converted.add_(2)
        return base.float(), old.float(), converted.float()


class ModuleCastSource(torch.nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.layer = (torch.nn.Conv2d(3, 3, 1) if convolution else torch.nn.Linear(3, 3)).double()
        with torch.no_grad():
            self.layer.weight.fill_(1)
            self.layer.bias.zero_()

    def forward(self, x):
        # Double kernels also run on older producer versions. Integer-valued
        # inputs exercise cast identity without requiring FP64 numerical parity.
        base = self.layer(x.double())
        old = base.transpose(0, 1)
        converted = base.double()
        converted.transpose_(0, 1)
        converted.add_(2)
        return base.float(), old.float(), converted.float()


class ComplexCastBoundary(torch.nn.Module):
    __constants__ = ['copy']

    def __init__(self, copy):
        super().__init__()
        self.copy = copy

    def forward(self, x):
        base = x.clone()
        old = base.transpose(0, 1)
        converted = base.to(torch.complex64 if self.copy else torch.complex128)
        converted.transpose_(0, 1)
        converted.add_(2)
        return base, old, converted

    def reference(self, x):
        base = x.clone()
        old = base.transpose(0, 1)
        converted = base.clone() if self.copy else base
        converted.transpose_(0, 1)
        converted.add_(2)
        return base, old, converted


class RankDependentCast(torch.nn.Module):
    def forward(self, x, source):
        base = x.clone().half() + source
        old = base.transpose(0, 1)
        converted = base.half()
        converted.add_(2)
        return base.float(), old.float(), converted.float()


def conditional_alias_cases():
    for operation in ('dropout', 'dropout2d', 'dropout3d', 'alpha_dropout',
                      'feature_alpha_dropout', 'type_as', 'type_as_half',
                      'to', 'to_copy', 'to_half', 'clone'):
        shape = (2, 3, 2, 4, 2) if operation == 'dropout3d' else (2, 3, 2, 4)
        inputs = (torch.arange(96 if operation == 'dropout3d' else 48).reshape(shape).float() - 4,)
        yield 'conditional_' + operation, ConditionalAlias(operation), inputs, ()
    # Pair alias and independent storage on a few different view geometries.
    for view, operations in [('slice', ('dropout2d', 'clone')),
                             ('select', ('type_as', 'type_as_half')),
                             ('transpose', ('to', 'to_copy'))]:
        inputs = (torch.arange(48).reshape(2, 3, 2, 4).float() - 4,)
        for operation in operations:
            yield 'conditional_' + operation + '_' + view, ConditionalAlias(operation, view), inputs, ()
    for operation in ('dropout2d', 'type_as', 'type_as_half', 'to', 'to_copy', 'to_half', 'clone'):
        inputs = (torch.arange(48).reshape(2, 3, 2, 4).float() - 4,)
        yield ('conditional_metadata_' + operation, ConditionalAlias(operation, metadata=True),
               inputs, ('transpose',))
    for mode in ('same', 'different', 'source', 'source_metadata'):
        metadata_ops = ('transpose',) if mode == 'source_metadata' else ()
        yield 'conditional_precision_' + mode, CastPrecisionBoundary(mode), None, metadata_ops
    for operation in ('exp', 'tanh'):
        for source in (False, True):
            name = 'conditional_' + operation + ('_source_write' if source else '_result_write')
            yield name, UnaryCastSource(operation, source), (torch.zeros(2, 3),), ('transpose',)
    for convolution in (False, True):
        shape = (2, 3, 2, 3) if convolution else (2, 3)
        name = 'conditional_module_cast_' + ('conv' if convolution else 'linear')
        yield name, ModuleCastSource(convolution), (torch.ones(shape),), ('transpose',)
    for copy in (False, True):
        inputs = (torch.arange(6).reshape(2, 3).to(torch.complex128) * (1 + 1j),)
        name = 'conditional_complex_cast_' + ('copy' if copy else 'alias')
        yield name, ComplexCastBoundary(copy), inputs, ('transpose',)


class ModuleDropoutAlias(torch.nn.Module):
    __constants__ = ['metadata']

    def __init__(self, dropout, metadata=False):
        super().__init__()
        self.dropout = dropout
        self.metadata = metadata

    def forward(self, x):
        base = x + 1
        old = base.transpose(0, 1)
        before = base.clone()
        snapshot = old + 3
        result = self.dropout(base)
        if self.metadata:
            result.transpose_(0, 1)
        layout = base.clone()
        result.add_(2)
        middle = old + 5
        result.select(0, 0).mul_(3)
        base.add_(1)
        return before, snapshot, layout, middle, base, result, old


def module_dropout_cases():
    inputs = (torch.arange(48).reshape(2, 3, 2, 4).float() - 4,)
    for name, dropout in [('dropout', torch.nn.Dropout()), ('dropout2d', torch.nn.Dropout2d()),
                          ('alpha_dropout', torch.nn.AlphaDropout())]:
        yield 'module_' + name, ModuleDropoutAlias(dropout), inputs, ()
    # Unbatched Dropout3d uses an independent view; batched eval is identity.
    for batched in (False, True):
        shape = (2, 3, 2, 4, 2) if batched else (2, 3, 2, 4)
        inputs = (torch.arange(96 if batched else 48).reshape(shape).float() - 4,)
        name = 'module_dropout3d_metadata_' + ('5d' if batched else '4d')
        yield name, ModuleDropoutAlias(torch.nn.Dropout3d(), metadata=True), inputs, ('transpose',)


class NestedContainers(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base.transpose(0, 1)
        aliases = [base, view]
        pair = (aliases, base)
        nested = (pair, view)
        before = torch.stack(aliases)
        base.add_(2)
        between = torch.stack(aliases)
        view.mul_(3)
        return before, between, nested, base, view


class UnpackedContainers(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base.transpose(0, 1)
        aliases = [base, view]
        left, right = aliases
        before = torch.stack(aliases)
        right.add_(2)
        between = torch.stack(aliases)
        left.mul_(3)
        return before, between, aliases, left, right, base, view


class DynamicPermutation(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        dim = x.size(0) - 2
        view = base.permute(dim, 1 - dim)
        view.add_(2)
        return base, view


class NestedUnpack(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base.transpose(0, 1)
        nested = ([base, view],)
        items, = nested
        left, right = items
        before = left.clone()
        right.add_(2)
        left.mul_(3)
        return before, base, view, left, right


class SharedShape(torch.nn.Module):
    def forward(self, x):
        shape = [2, 8]
        before = x.view(shape)
        y = (x + 1).relu_()
        return before, y.view(shape)


class ScalarAdd(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        y.add_(1)
        return y


class DynamicSlice(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        y.select(x.size(0) - 2, 1).add_(1)
        return y


class MetadataMutation(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        sibling = y.view(-1)
        y.transpose_(0, 1)
        y.add_(2)
        return y, sibling


class LocalMetadataMutation(torch.nn.Module):
    __constants__ = ['operation']

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        y = x + 1
        before = y.clone()
        if self.operation == 'transpose':
            y.transpose_(0, 1)
        elif self.operation == 't':
            y.t_()
        elif self.operation == 'squeeze':
            y.squeeze_(1)
        else:
            y.unsqueeze_(1)
        y.add_(2)
        return before, y, y * 3


class MetadataReplay(torch.nn.Module):
    __constants__ = ['operation']

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        y = x + 1
        previous = y.add_(2)
        items = [y, previous]
        # This old view is dead before the metadata write. The container and
        # the previous mutation result still refer to the same tensor object.
        before = y.transpose(0, 1).clone()
        if self.operation == 'transpose':
            result = previous.transpose_(0, 1)
        elif self.operation == 't':
            result = previous.t_()
        elif self.operation == 'squeeze':
            result = previous.squeeze_(1)
        else:
            result = previous.unsqueeze_(1)
        after = y + 3
        result.unsqueeze_(0)
        y.squeeze_(0)
        view = y.select(0, 0)
        view_before = view.clone()
        y.add_(2)
        middle = view + 3
        view.mul_(2)
        return before, after, view_before, middle, torch.stack(items), y, result, previous, view


class ObservedMetadataMutation(torch.nn.Module):
    __constants__ = ['operation', 'mode']

    def __init__(self, operation, mode):
        super().__init__()
        self.operation = operation
        self.mode = mode

    def forward(self, x):
        base = x + 1
        if self.mode == 'target':
            y = base[1:]
            sibling = y
        elif self.mode == 'untracked_target':
            y = base.reshape_as(base)
            sibling = y
        else:
            y = base
            sibling = base.transpose(0, 1)
            if self.mode == 'descendant':
                sibling = sibling.select(0, 0)
            elif self.mode == 'untracked_sibling':
                sibling = base.reshape(-1)
            elif self.mode == 'mutation_result':
                sibling = sibling.add_(2)
        stored = [sibling, sibling]
        if self.operation == 'transpose':
            y.transpose_(0, 1)
        elif self.operation == 't':
            y.t_()
        elif self.operation == 'squeeze':
            y.squeeze_(1)
        else:
            y.unsqueeze_(1)
        y.add_(2)
        # Observe values after the metadata change and the subsequent write.
        if self.mode == 'container':
            return y + 3, torch.stack(stored) * 2, base + 5
        return y + 3, sibling * 2, base + 5


class ViewMetadataMutation(torch.nn.Module):
    __constants__ = ['operation', 'view']

    def __init__(self, operation, view):
        super().__init__()
        self.operation = operation
        self.view = view

    def mutate(self, value):
        if self.operation == 'transpose':
            return value.transpose_(0, 1)
        if self.operation == 't':
            return value.t_()
        if self.operation == 'squeeze':
            return value.squeeze_(1)
        return value.unsqueeze_(1)

    def forward(self, x):
        base = x + 1
        if self.view == 'slice':
            target = base[1:3, 1:5]
        elif self.view == 'select':
            target = base.view(2, 2, 6).select(0, 1)
        elif self.view == 'view':
            target = base.view(3, 8)
        elif self.view == 'transpose':
            target = base.transpose(0, 1)
        elif self.view == 't':
            target = base.t()
        elif self.view == 'permute':
            target = base.permute(1, 0)
        elif self.view == 'squeeze':
            target = base.unsqueeze(1).squeeze(1)
        elif self.view == 'unsqueeze':
            target = base.select(0, 1).unsqueeze(0)
        elif self.view == 'detach':
            target = base.detach()
        else:
            target = torch.ops.aten.alias(base)
        if self.operation == 'squeeze':
            target = target.unsqueeze(1)
        sibling = base.transpose(0, 1)
        old = target.transpose(0, 1)
        child = old.select(0, 0)
        previous = target.add_(2)
        targets = [target, previous]
        old_views = [old, old]
        before_base = base.clone()
        before_target = target.clone()
        before_old = old.clone()
        before_child = child.clone()
        result = self.mutate(previous)
        after_target = target.clone()
        after_old = old.clone()
        after_container = torch.stack(targets)
        old_container = torch.stack(old_views)
        base.add_(x)
        root_write = target.clone()
        sibling.select(0, 1).mul_(3)
        sibling_write = target.clone()
        target.select(0, 0).add_(5)
        target_write = base.clone()
        # A view made between metadata writes retains that intermediate layout.
        middle = target.transpose(0, 1)
        middle_before = middle.clone()
        if self.operation == 'squeeze':
            result.unsqueeze_(1)
        repeated = self.mutate(result)
        repeated.mul_(2)
        return (before_base, before_target, before_old, before_child, after_target, after_old,
                after_container, old_container, root_write, sibling_write, target_write,
                middle_before, base, target, previous, result, repeated, old, child, middle,
                torch.stack(targets), torch.stack(old_views))


class InputViewMetadata(torch.nn.Module):
    def forward(self, x):
        view = x[0]
        old = view[:]
        items = [view, view]
        before = view.clone()
        result = view.unsqueeze_(0)
        return before, x, old, view, result, torch.stack(items), view + 3


class DetachMetadata(torch.nn.Module):
    __constants__ = ['operation']

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        base = x + 1
        if self.operation == 'detach':
            target = base.detach()
        elif self.operation == 'alias':
            target = torch.ops.aten.alias(base)
        else:
            target = base.detach_()
        items = [base, base]
        targets = [target, target]
        before = torch.stack(items)
        target.unsqueeze_(0)
        after = torch.stack(items)
        after_target = torch.stack(targets)
        base.add_(2)
        target.mul_(3)
        return before, after, after_target, base, target, torch.stack(items), torch.stack(targets)


class StorageMutation(torch.nn.Module):
    __constants__ = ['operation']

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        y = x + 1
        if self.operation == 'resize':
            y.resize_(4, 2)
        elif self.operation == 'resize_as':
            y.resize_as_(x.transpose(0, 1))
        elif self.operation == 'set':
            y.set_(x)
        else:
            y.as_strided_((4, 2), (1, 4))
        return y + 3


def metadata_cases():
    yield 'metadata_mutation', MetadataMutation(), None, ('transpose',), None
    for operation in ('transpose', 't', 'squeeze', 'unsqueeze'):
        shape = (2, 1, 4) if operation == 'squeeze' else (2, 4)
        inputs = (torch.arange(-4, 4).reshape(shape).float(),)
        yield 'metadata_' + operation + '_local', LocalMetadataMutation(operation), inputs, (operation,), None
        yield ('metadata_' + operation + '_replay', MetadataReplay(operation), inputs,
               (operation, 'unsqueeze', 'squeeze'), None)
        for mode in ('target', 'sibling', 'container', 'descendant', 'mutation_result',
                     'untracked_target', 'untracked_sibling'):
            rejection = 'unsupported observable alias' if mode.startswith('untracked_') else None
            yield ('metadata_' + operation + '_' + mode, ObservedMetadataMutation(operation, mode),
                   inputs, (operation,), rejection)
        for view in ('slice', 'select', 'view', 'transpose', 't', 'permute',
                     'squeeze', 'unsqueeze', 'detach', 'alias'):
            # Before 1.13, detached tensors forbid direct metadata changes.
            # squeeze modifies a newly created unsqueeze view and remains valid.
            if view == 'detach' and operation != 'squeeze' and version.parse(torch.__version__) < version.parse('1.13'):
                continue
            yield ('metadata_' + operation + '_view_' + view, ViewMetadataMutation(operation, view),
                   (torch.arange(-12, 12).reshape(4, 6).float(),), (operation, operation), None)
    yield 'metadata_input_view', InputViewMetadata(), None, ('unsqueeze',), None
    for operation in ('detach', 'detach_', 'alias'):
        if operation == 'detach' and version.parse(torch.__version__) < version.parse('1.13'):
            continue
        yield 'metadata_identity_' + operation, DetachMetadata(operation), None, ('unsqueeze',), None


class IndexedContainer(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base.transpose(0, 1)
        items = [base, view]
        selected = items[-1]
        selected.add_(2)
        return base, view, selected


class UntrackedView(torch.nn.Module):
    def __init__(self, mode, observe_base=True):
        super().__init__()
        self.mode = mode
        self.observe_base = observe_base

    def forward(self, x):
        base = x + 1
        if self.mode == 'reshape':
            view = base.reshape(-1)
        elif self.mode == 'flatten':
            view = base.flatten()
        else:
            view, _ = base.split(2, 1)
        view.add_(2)
        return (base, view) if self.observe_base else view


class UntrackedContainer(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base.reshape(-1)
        stored = (view,)
        base.add_(2)
        return stored, base


class FunctionalCopy(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base[:, 1:]
        result = torch.ops.aten.copy(view, torch.ones(1, 3))
        return base, result


class FullCopy(torch.nn.Module):
    def forward(self, x, source):
        base = x + 1
        base[:, :] = source
        return base


class MixedMutation(torch.nn.Module):
    def __init__(self, operation, transpose):
        super().__init__()
        self.operation = operation
        self.transpose = transpose

    def forward(self, x, source):
        base = x + 1
        view = base.transpose(0, 1) if self.transpose else base
        before = view.clone()
        if self.operation == 'add':
            result = view.add_(source)
        elif self.operation == 'div':
            result = view.div_(source)
        else:
            result = view.copy_(source)
        return before, base, view, result

    def reference(self, x, source):
        if self.operation != 'copy':
            return self(x, source)
        base = x + 1
        view = base.transpose(0, 1) if self.transpose else base
        before = view.clone()
        # Some older CPU copy_ kernels mishandle mixed-dtype broadcasting.
        # Keep copy_ in the exported model, but verify its documented value
        # semantics independently instead of copying that kernel's bad result.
        copied = source.to(view.dtype).expand_as(view).clone()
        updated = copied.transpose(0, 1) if self.transpose else copied
        return before, updated, copied, copied


def mixed_cases():
    for transpose in (False, True):
        where = 'transpose' if transpose else 'root'
        shape = (3, 2) if transpose else (2, 3)
        # Above half an f16 ULP at 1, but rounds to exactly half an ULP
        # if the source is incorrectly cast before performing the addition.
        yield 'add_tiny_' + where, MixedMutation('add', transpose), (
            torch.zeros(2, 3, dtype=torch.float16), torch.full(shape, 0.0004884))
        for dtype, label in ((torch.float16, 'half'), (torch.int32, 'int')):
            for broadcast in (False, True):
                source_shape = (1, shape[1]) if broadcast else shape
                source = torch.arange(1, 1 + source_shape[0] * source_shape[1]).reshape(source_shape).float() - 2.75
                name = 'copy_' + label + '_' + where + ('_broadcast' if broadcast else '')
                yield name, MixedMutation('copy', transpose), (torch.zeros(2, 3, dtype=dtype), source)
        yield 'copy_half_source_' + where, MixedMutation('copy', transpose), (
            torch.zeros(2, 3, dtype=torch.float16), torch.full((1, shape[1]), 2.5, dtype=torch.float16))
    yield 'copy_scalar', MixedMutation('copy', False), (torch.tensor(0, dtype=torch.int32), torch.tensor(-2.75))
    yield 'add_scalar', MixedMutation('add', False), (torch.tensor(0, dtype=torch.float16), torch.tensor(0.0004884))
    yield 'copy_empty', MixedMutation('copy', True), (torch.empty(0, 3, dtype=torch.int32), torch.empty(3, 0))
    yield 'add_empty', MixedMutation('add', True), (torch.empty(0, 3, dtype=torch.float16), torch.empty(3, 0))


def complex_cases():
    for operation in ('add', 'div'):
        for where in ('root', 'transpose', 'scalar'):
            scalar = where == 'scalar'
            transpose = where == 'transpose'
            target = torch.zeros((), dtype=torch.complex64) if scalar else (
                torch.arange(6).reshape(2, 3).to(torch.complex64) * (1 + 0.5j))
            kinds = ('f32_scalar', 'f64_scalar', 'c128_scalar') if scalar else (
                'f32_scalar', 'f32_vector', 'f64_scalar', 'f64_vector', 'c128_vector')
            for kind in kinds:
                dtype = {'f32': torch.float32, 'f64': torch.float64, 'c128': torch.complex128}[kind.split('_')[0]]
                shape = () if kind.endswith('scalar') else (2 if transpose else 3,)
                # f64/c128 values distinguish post-compute narrowing from
                # incorrectly rounding the source before arithmetic.
                value = 2.0 ** -24 + 2.0 ** -50
                if operation == 'div':
                    value += 1
                source = torch.full(shape, value, dtype=dtype)
                name = 'complex_' + operation + '_' + where + '_' + kind
                no_cast = torch.result_type(target, source) == target.dtype
                yield name, MixedMutation(operation, transpose), (target, source), no_cast


class Unsupported(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'input':
            return x.add_(1)
        if self.mode == 'detach':
            return x.detach().relu_()
        base = x + 1
        view = base.as_strided((2, 3), (4, 1))
        view.add_(2)
        return base, view


def check_outputs(expected, actual, path=()):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), (path, type(actual))
        assert expected.shape == actual.shape, (path, expected.shape, actual.shape)
        assert expected.dtype == actual.dtype, (path, expected.dtype, actual.dtype)
        assert torch.equal(expected, actual), (path, expected, actual)
        return
    assert isinstance(actual, (tuple, list)), (path, type(actual))
    assert len(expected) == len(actual), (path, len(expected), len(actual))
    for index, (a, b) in enumerate(zip(expected, actual)):
        check_outputs(a, b, path + (index,))


def flatten_outputs(values):
    if isinstance(values, torch.Tensor):
        return (values,)
    return tuple(item for value in values for item in flatten_outputs(value))


def float_outputs(values):
    if isinstance(values, torch.Tensor):
        if values.is_complex():
            return values.to(torch.complex64)
        return values.float() if values.is_floating_point() else values
    return tuple(float_outputs(value) for value in values)


def check_ncnn(name, model, value):
    # Opt-in because ncnn's Python module is not required for pnnx unit tests.
    import ncnn

    expected = model(value.clone())
    original = value.clone()
    actual = []
    with ncnn.Net() as net:
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        status = net.load_param(name + '.ncnn.param')
        assert status == 0, ('ncnn load_param', name, status)
        status = net.load_model(name + '.ncnn.bin')
        assert status == 0, ('ncnn load_model', name, status)
        with net.create_extractor() as ex:
            assert ex.input('in0', ncnn.Mat(value.numpy(), batch_index=233).clone()) == 0
            for index in range(len(expected)):
                status, output = ex.extract('out%d' % index)
                assert status == 0, ('ncnn extraction', index, status)
                actual.append(torch.from_numpy(output.numpy(batch_index=233)).clone())
    check_outputs(expected, actual)
    assert torch.equal(value, original), 'ncnn modified a caller input'


def check_dynamic_copy(binary):
    model = FullCopy().eval()
    x = torch.zeros(2, 4)
    source = torch.arange(8).reshape(2, 4).float()
    name = 'torchscript_dynamic_full_copy'
    torch.jit.trace(model, (x, source)).save(name + '.pt')
    result = subprocess.run([binary, name + '.pt', 'fp16=0', 'optlevel=2',
                             'inputshape=[2,4],[2,4]', 'inputshape2=[3,4],[3,4]'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    spec = importlib.util.spec_from_file_location(name, name + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    converted = module.Model().float().eval()
    # Both traced dimensions become -1, but valid runtime inputs may broadcast.
    for n, m in [(2, 2), (3, 3), (5, 1)]:
        x = torch.arange(n * 4).reshape(n, 4).float()
        source = torch.arange(m * 4).reshape(m, 4).float() - 3
        originals = (x.clone(), source.clone())
        check_outputs(model(x, source), converted(x, source))
        check_outputs(originals, (x, source))


def run_case(binary, model, mode, frontend, dtype=torch.float32, dynamic=False,
             rejection=None, inputs=None, forbid_type_as=False, slice_copies=None, metadata_ops=(),
             check_source=False):
    model.eval()
    # Exact integer-valued data exercises negative values and avoids tolerances
    # masking an alias update that was lost or performed at the wrong time.
    x = torch.arange(-4, 4).reshape(2, 4).to(dtype)
    samples = inputs if inputs is not None else (x,)
    name = '%s_%s_%s%s' % (frontend, mode, str(dtype).split('.')[-1], '_dynamic' if dynamic else '')
    path = name + ('.pt2' if frontend == 'pt2' else '.pt')
    if frontend == 'pt2':
        shapes = ({0: torch.export.Dim('batch', min=2, max=8)},) if dynamic else None
        program = torch.export.export(model, tuple(value.clone() for value in samples), dynamic_shapes=shapes)
        mutable = [node.target._schema.name for node in program.graph.nodes
                   if node.op == 'call_function' and getattr(node.target, '_schema', None) is not None
                   and node.target._schema.is_mutable]
        assert mutable, 'raw export lost the mutation under test'
        for operation in set(metadata_ops):
            assert mutable.count('aten::' + operation + '_') >= metadata_ops.count(operation), (
                'raw export lost metadata mutation', operation, mutable)
        torch.export.save(program, path)
        args = ['inputshape=[2,4]', 'inputshape2=[3,4]'] if dynamic else []
    else:
        if frontend == 'torchscript_script':
            torch.jit.script(model).save(path)
        else:
            torch.jit.trace(model, tuple(value.clone() for value in samples), check_trace=False).save(path)
        suffix = 'i32' if dtype == torch.int32 else ''
        types = {torch.float16: 'f16', torch.float32: '', torch.float64: 'f64',
                 torch.int32: 'i32', torch.complex64: 'c64', torch.complex128: 'c128'}
        args = ['inputshape=' + ','.join('[' + ','.join(str(d) for d in value.shape) + ']' +
                                      types[value.dtype] for value in samples)]
        if any(value.dim() == 0 for value in samples):
            # Older TorchScript archives omit traced_inputs.pkl, and the CLI
            # shape-list syntax cannot represent scalar entries. Supply exact
            # shapes and dtypes through the existing NumPy input interface.
            import numpy as np
            paths = []
            for index, value in enumerate(samples):
                sample_path = name + '_input_' + str(index) + '.npy'
                np.save(sample_path, value.numpy())
                paths.append(sample_path)
            args = ['input=' + ','.join(paths)]
        if dynamic:
            args.append('inputshape2=[3,4]' + suffix)
    if check_source:
        source = torch.export.load(path).module() if frontend == 'pt2' else torch.jit.load(path)
        for _ in range(2):
            values = [item.clone() for item in samples]
            expected = model(*[item.clone() for item in samples])
            try:
                # Legacy JIT execution optimizations can lose type_as aliases.
                # Validate the serialized graph before those runtime rewrites.
                with torch.jit.optimized_execution(False):
                    check_outputs(expected, source(*values))
                check_outputs(samples, values)
            except AssertionError as error:
                raise AssertionError('serialized source differs from eager: ' + name) from error
    # Retained diagnostic directories may be reused; stale output must never
    # turn a failed conversion into a passing regression.
    output_suffixes = ('_pnnx.py', '.pnnx.param', '.pnnx.bin', '.pnnx.onnx',
                       '_ncnn.py', '.ncnn.param', '.ncnn.bin')
    for suffix in output_suffixes:
        Path(name + suffix).unlink(missing_ok=True)
    result = subprocess.run([binary, path, 'fp16=0', 'optlevel=2'] + args,
                            capture_output=True, text=True, timeout=90)
    Path(name + '.conversion.log').write_text(result.stdout + result.stderr)
    if rejection is not None:
        assert result.returncode > 0, ('unsupported mutation did not reject cleanly', result.returncode)
        assert rejection in result.stderr, result.stderr
        assert not any(Path(name + suffix).exists() for suffix in output_suffixes), 'rejected graph emitted output'
        if frontend != 'pt2':
            # A rejected conversion must still originate from a valid model.
            # Check its values, including uses after the unsupported mutation.
            expected = model(*[item.clone() for item in samples])
            actual = torch.jit.load(path)(*[item.clone() for item in samples])
            check_outputs(expected, actual)
        return
    assert result.returncode == 0, result.stderr
    spec = importlib.util.spec_from_file_location(name, name + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    converted = module.Model().float().eval()
    for batch in ([2, 3, 5] if dynamic else [2]):
        value = torch.arange(-4, batch * 4 - 4).reshape(batch, 4).to(dtype)
        values = [item.clone() for item in samples] if inputs is not None else [value]
        values = [float_outputs(item) for item in values]
        originals = [item.clone() for item in values]
        reference = getattr(model, 'reference', model)
        expected = float_outputs(reference(*[item.clone() for item in values]))
        actual = converted(*values)
        if frontend == 'pt2' and mode in ('containers', 'containers_unpack'):
            # Raw export serializes the output pytree as flat graph outputs.
            expected, actual = flatten_outputs(expected), flatten_outputs(actual)
        check_outputs(expected, actual)
        assert all(torch.equal(a, b) for a, b in zip(values, originals)), 'modified a caller input'
        # Reusing a converted model must not retain a mutation from the last run.
        repeated = converted(*values)
        if frontend == 'pt2' and mode in ('containers', 'containers_unpack'):
            repeated = flatten_outputs(repeated)
        check_outputs(expected, repeated)
        assert all(torch.equal(a, b) for a, b in zip(values, originals)), 'modified a caller input on repeat'
    if forbid_type_as:
        assert 'Tensor.type_as' not in Path(name + '.pnnx.param').read_text(), 'unnecessary dtype cast'
    if slice_copies is not None:
        param = Path(name + '.pnnx.param').read_text()
        count = sum(line.split()[0] == 'Tensor.slice_copy' for line in param.splitlines()[2:])
        assert count == slice_copies, ('slice copies', count, slice_copies)
    # Raw pt2 ones_like is currently rejected by the ncnn backend.
    if (os.environ.get('PNNX_TEST_NCNN') == '1' and dtype == torch.float32 and
            not dynamic and mode in ('siblings', 'nonfirst', 'chained', 'select', 'view_base') and
            not (frontend == 'pt2' and mode == 'nonfirst')):
        check_ncnn(name, model, x.clone())
        print('PASS ncnn/' + name)
    return Path(name + '.pnnx.param').read_text()


def test():
    torch.set_num_threads(1)
    binary = str(Path(os.environ.get('PNNX_BIN', '../src/pnnx')).resolve())
    frontends = ['torchscript']
    if version.parse(torch.__version__) >= version.parse('2.11'):
        frontends.append('pt2')
    failures = []
    cwd = os.getcwd()
    artifacts = os.environ.get('PNNX_FUNCTIONIZE_ARTIFACTS')
    if artifacts:
        artifacts = str(Path(artifacts).resolve())
        Path(artifacts).mkdir(parents=True, exist_ok=True)
    context = nullcontext(artifacts) if artifacts else tempfile.TemporaryDirectory(prefix='test_functionize_')
    with context as work:
        os.chdir(work)
        try:
            check_dynamic_copy(binary)
            for frontend in frontends:
                for mode, model, inputs, metadata_ops in conditional_alias_cases():
                    label = frontend + '/' + mode
                    try:
                        # Old tracers can lose the old view of a type_as identity.
                        case_frontend = ('torchscript_script' if frontend == 'torchscript' and
                                         isinstance(model, ConditionalAlias) and model.operation == 'type_as' else frontend)
                        run_case(binary, model, mode, case_frontend, inputs=inputs,
                                 metadata_ops=metadata_ops, check_source=True)
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
                if frontend == 'torchscript':
                    # Legacy JIT inference cannot prove these original dtypes:
                    # integer exp promotes to float, and mixed tensor promotion
                    # depends on the rank of the half operand after its cast.
                    for mode, model, inputs in [
                            ('unknown_integer_exp', UnaryCastSource('exp', False), (torch.zeros(2, 3, dtype=torch.int32),)),
                            ('unknown_rank_cast', RankDependentCast(), (torch.ones(2, 3), torch.tensor(0.5)))]:
                        run_case(binary, model, mode, frontend, inputs=inputs, check_source=True,
                                 rejection='cannot determine original dtype')
                for mode in ('siblings', 'nonfirst', 'chained', 'select',
                             'view_slice', 'view_select', 'view_repeat', 'view_base',
                             'transpose', 'permute', 'unsqueeze', 'squeeze', 'multiinput',
                             'copy', 'fill', 'view_slice_base', 'view_select_base',
                             'nested_slice', 'negative_permute'):
                    for dtype in (torch.float32, torch.int32):
                        label = '%s/%s/%s' % (frontend, mode, dtype)
                        try:
                            run_case(binary, Aliases(mode), mode, frontend, dtype)
                            print('PASS', label)
                        except Exception as error:
                            failures.append(label)
                            print('FAIL', label, error)
                for mode in ('chained_view', 'chained_select', 'repeated_operand', 'unrelated_write'):
                    run_case(binary, Aliases(mode), mode, frontend)
                for mode, copies, read_op in [('full_slice', 0, None),
                                             ('unbind_writeback', 1, 'torch.unbind'),
                                             ('split_writeback', 1, 'torch.tensor_split')]:
                    param = run_case(binary, Aliases(mode), mode, frontend, slice_copies=copies)
                    if read_op is not None:
                        assert read_op + ' ' in param, ('missing read fusion', read_op)
                run_case(binary, Aliases('dynamic_bounds'), 'dynamic_bounds', frontend, dynamic=True)
                for dtype in (torch.float32, torch.int32):
                    run_case(binary, ScalarAdd(), 'scalar_add', frontend, dtype=dtype,
                             inputs=(torch.tensor(2, dtype=dtype),), forbid_type_as=True)
                for mode, model, inputs in mixed_cases():
                    label = frontend + '/' + mode
                    try:
                        run_case(binary, model, mode, frontend, inputs=inputs)
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
                for mode, model, inputs, no_cast in complex_cases():
                    label = frontend + '/' + mode
                    try:
                        run_case(binary, model, mode, frontend, inputs=inputs, forbid_type_as=no_cast)
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
                for mode in ('siblings', 'nonfirst', 'chained'):
                    label = frontend + '/' + mode + '/dynamic'
                    try:
                        run_case(binary, Aliases(mode), mode, frontend, dynamic=True)
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
            for frontend in ['torchscript_script'] + (['pt2'] if 'pt2' in frontends else []):
                for mode, model in [('containers', NestedContainers()), ('containers_unpack', UnpackedContainers())]:
                    label = frontend + '/' + mode
                    try:
                        run_case(binary, model, mode, frontend,
                                 inputs=(torch.arange(-4, 5).reshape(3, 3).float(),))
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
            label = 'torchscript_script/reject/dynamic_permutation'
            try:
                run_case(binary, DynamicPermutation(), 'dynamic_permutation', 'torchscript_script',
                         dynamic=True, rejection='constant valid permutation for mutation writeback')
                print('PASS', label)
            except Exception as error:
                failures.append(label)
                print('FAIL', label, error)
            run_case(binary, NestedUnpack(), 'nested_unpack', 'torchscript_script',
                     inputs=(torch.arange(9).reshape(3, 3).float(),))
            run_case(binary, SharedShape(), 'shared_shape', 'torchscript_script',
                     inputs=(torch.arange(16).reshape(4, 4).float(),))
            run_case(binary, DynamicSlice(), 'dynamic_slice', 'torchscript_script', dynamic=True,
                     rejection='constant dimension for mutation writeback')
            for frontend in ['torchscript', 'torchscript_script'] + (['pt2'] if 'pt2' in frontends else []):
                for mode, model, inputs, metadata_ops in module_dropout_cases():
                    label = frontend + '/' + mode
                    try:
                        run_case(binary, model, mode, frontend, inputs=inputs,
                                 metadata_ops=metadata_ops, check_source=True)
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
                run_case(binary, MetadataMutation(), 'metadata_mutation', frontend, dynamic=True,
                         metadata_ops=('transpose',))
                print('PASS', frontend + '/metadata_mutation/dynamic')
                for mode, model, inputs, metadata_ops, rejection in metadata_cases():
                    label = frontend + '/' + mode
                    try:
                        run_case(binary, model, mode, frontend, inputs=inputs,
                                 metadata_ops=metadata_ops, rejection=rejection)
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
            # Tracing can omit resize_ entirely; scripting retains all of
            # these storage mutations for the rejection check.
            for operation in ('resize', 'resize_as', 'set', 'as_strided'):
                run_case(binary, StorageMutation(operation), 'storage_' + operation, 'torchscript_script',
                         rejection='changes tensor metadata or storage')
            run_case(binary, IndexedContainer(), 'indexed_container', 'torchscript_script')
            for mode in ('reshape', 'flatten', 'split'):
                run_case(binary, UntrackedView(mode), 'untracked_' + mode, 'torchscript',
                         rejection='unsupported observable alias')
                run_case(binary, UntrackedView(mode, False), 'local_' + mode, 'torchscript')
            run_case(binary, UntrackedContainer(), 'untracked_container', 'torchscript_script',
                     rejection='unsupported observable alias')
            if version.parse(torch.__version__) >= version.parse('1.12'):
                run_case(binary, FunctionalCopy(), 'functional_copy', 'torchscript')
            if 'pt2' in frontends:
                for mode, message in [('input', 'mutable'), ('detach', 'external tensor'),
                                      ('as_strided', 'as_strided')]:
                    label = 'pt2/reject/' + mode
                    try:
                        run_case(binary, Unsupported(mode), 'reject_' + mode, 'pt2', rejection=message)
                        print('PASS', label)
                    except Exception as error:
                        failures.append(label)
                        print('FAIL', label, error)
        finally:
            os.chdir(cwd)
    print('functionize failures:', ', '.join(failures) if failures else 'none')
    return not failures


if __name__ == '__main__':
    assert test()
