# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import subprocess

import torch
from packaging import version


class MetadataQuery(torch.nn.Module):
    def __init__(self, query, repeat=False, independent=False):
        super().__init__()
        self.query = query
        self.repeat = repeat
        self.independent = independent

    def forward(self, x):
        base = x.clone(memory_format=torch.contiguous_format)
        old = base.detach()
        if self.query == 'size':
            before = x + base.size(0)
        elif self.query == 'stride':
            before = x + base.stride(0)
        else:
            before = base.as_strided((2, 3), (3, 1), 0).clone()
        result = base.transpose_(0, 1)
        if self.repeat:
            result.transpose_(0, 1)
        target = old if self.independent else base
        if self.query == 'size':
            after = x + target.size(0)
        elif self.query == 'stride':
            after = x + target.stride(0)
        else:
            after = target.as_strided((2, 3), (3, 1), 0)
        return before, after, old.clone()


def make_program(model, x, reference):
    program = torch.export.export(model, (x,))
    graph = program.graph
    source = next(n for n in graph.nodes if n.op == 'placeholder')
    base = next(n for n in graph.nodes if n.target == torch.ops.aten.clone.default)
    old = next(n for n in graph.nodes if n.target == torch.ops.aten.detach.default)
    mutations = [n for n in graph.nodes if n.target == torch.ops.aten.transpose_.default]
    target = base
    if reference == 'current':
        target = mutations[-1]
    elif reference == 'previous':
        target = mutations[0]
    elif reference == 'independent':
        target = old
    elif reference in ('identity_before', 'identity_after'):
        position = base if reference == 'identity_before' else mutations[-1]
        with graph.inserting_after(position):
            target = graph.call_function(torch.ops.aten.type_as.default, (base, source))
            target.meta['val'] = position.meta['val']
    elif reference == 'data_result':
        with graph.inserting_after(base):
            target = graph.call_function(torch.ops.aten.add_.Tensor, (base, 0))
            target.meta['val'] = base.meta['val']
    elif reference == 'identity_data':
        with graph.inserting_before(mutations[0]):
            target = graph.call_function(torch.ops.aten.type_as.default, (base, source))
            target.meta['val'] = base.meta['val']
            value = target
            for _ in range(2):
                value = graph.call_function(torch.ops.aten.add_.Tensor, (value, 0))
                value.meta['val'] = base.meta['val']
        mutations[0].args = (value, *mutations[0].args[1:])

    if model.query == 'as_strided':
        views = [n for n in graph.nodes if n.target == torch.ops.aten.as_strided.default]
        views[-1].args = (target, *views[-1].args[1:])
    else:
        # Export specializes static queries. Retain real query nodes and an old
        # reference to the same Tensor object, as permitted by raw Export IR.
        adds = [n for n in graph.nodes if n.target == torch.ops.aten.add.Tensor]
        query = torch.ops.aten.sym_size.int if model.query == 'size' else torch.ops.aten.sym_stride.int
        for i, add in enumerate(adds):
            changed = i == 1 and not model.repeat and not model.independent
            expected = (3 if changed else 2) if model.query == 'size' else (1 if changed else 3)
            with graph.inserting_before(add):
                value = graph.call_function(query, (base if i == 0 else target, 0))
                value.meta['val'] = expected
            add.args = (add.args[0], value)
    graph.lint()
    program.graph_module.recompile()
    program.validate()
    return program


def same_outputs(expected, actual):
    return len(expected) == len(actual) and all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        for a, b in zip(expected, actual))


class WrappedQuery(torch.nn.Module):
    def __init__(self, query, repeat):
        super().__init__()
        self.query = query
        self.repeat = repeat

    def forward(self, x):
        base = x.clone(memory_format=torch.contiguous_format).view(2, 3)
        old = base.detach()
        if self.repeat:
            base.transpose_(0, 1)
        with torch.no_grad():
            base.transpose_(0, 1)
            if self.query == 'size':
                inside = x + base.size(0)
            elif self.query == 'stride':
                inside = x + base.stride(0)
            else:
                inside = base.as_strided((2, 3), (3, 1), 0).clone()
        if self.query == 'size':
            after = x + base.size(0)
        elif self.query == 'stride':
            after = x + base.stride(0)
        else:
            after = base.as_strided((2, 3), (3, 1), 0)
        return inside, after, old.clone(), base.clone()


def make_wrapped_program(model, x):
    # Keep the no_grad scope in raw Export IR.
    with torch.enable_grad():
        program = torch.export.export(model, (x,))
    graph = program.graph
    base = next(n for n in graph.nodes if n.target == torch.ops.aten.view.default)
    wrapper = next(n for n in graph.nodes if str(n.target) == 'wrap_with_set_grad_enabled')
    child = program.graph_module.get_submodule(wrapper.args[1].target)
    mutation = next(n for n in child.graph.nodes if n.target == torch.ops.aten.transpose_.default)
    source = mutation.args[0]
    if model.repeat:
        previous = next(n for n in graph.nodes if n.target == torch.ops.aten.transpose_.default)
        wrapper.args = tuple(base if arg is previous else arg for arg in wrapper.args)
    for current, reference in ((child.graph, source), (graph, base)):
        if model.query == 'as_strided':
            view = next(n for n in current.nodes if n.target == torch.ops.aten.as_strided.default)
            view.args = (reference, *view.args[1:])
        else:
            add = next(n for n in current.nodes if n.target == torch.ops.aten.add.Tensor)
            query = torch.ops.aten.sym_size.int if model.query == 'size' else torch.ops.aten.sym_stride.int
            with current.inserting_before(add):
                value = current.call_function(query, (reference, 0))
                value.meta['val'] = (2 if model.repeat else 3) if model.query == 'size' else (3 if model.repeat else 1)
            add.args = (add.args[0], value)
        current.lint()
    child.recompile()
    program.graph_module.recompile()
    program.validate()
    return program


def check_program(binary, prefix, model, program, samples):
    torch.export.save(program, prefix + '.pt2')
    loaded = torch.export.load(prefix + '.pt2')
    loaded.validate()
    for sample in samples:
        expected = model(sample.clone())
        if not same_outputs(expected, program.module()(sample.clone())) or not same_outputs(expected, loaded.module()(sample.clone())):
            print(prefix, 'serialized source differs from eager')
            return False
    result = subprocess.run([binary, prefix + '.pt2', 'fp16=0'], capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        print(prefix, result.stderr)
        return False
    spec = importlib.util.spec_from_file_location(prefix, prefix + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    converted = module.Model().eval()
    for sample in samples:
        original = sample.clone()
        for _ in range(2):
            if not same_outputs(model(original.clone()), converted(sample)) or not torch.equal(sample, original):
                print(prefix, 'converted output differs from eager')
                return False
    return True


@torch.no_grad()
def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    samples = (torch.arange(6, dtype=torch.float32).reshape(2, 3),
               torch.tensor([[7., -1., 4.], [8., 3., -2.]]))
    binary = os.environ.get('PNNX_BIN', '../../src/pnnx')
    for query in ('size', 'stride', 'as_strided'):
        for reference in ('original', 'current', 'previous', 'independent',
                          'identity_before', 'identity_after', 'data_result', 'identity_data'):
            model = MetadataQuery(query, reference == 'previous', reference == 'independent').eval()
            program = make_program(model, samples[0], reference)
            prefix = 'test_pt2_layout_' + query + '_' + reference
            if not check_program(binary, prefix, model, program, samples):
                return False
        for repeat in (False, True):
            model = WrappedQuery(query, repeat).eval()
            program = make_wrapped_program(model, samples[0])
            prefix = 'test_pt2_layout_' + query + '_wrapped_' + str(repeat)
            if not check_program(binary, prefix, model, program, samples):
                return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
