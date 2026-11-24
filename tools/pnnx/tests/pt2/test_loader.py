# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

import torch
from packaging import version

from common import rewrite_archive


def export(name, model, args, **kwargs):
    path = 'test_pt2_loader_' + name + '.pt2'
    torch.export.save(torch.export.export(model.eval(), args, **kwargs), path)
    return path


def convert(path, options=(), error=None):
    result = subprocess.run([os.environ.get('PNNX', '../../src/pnnx'), path,
                             'fp16=0', *options],
                            capture_output=True, text=True, timeout=90)
    if error is not None:
        assert result.returncode != 0 and error in result.stderr, result.stderr
        assert 'pt2 load:' in result.stderr, result.stderr
        return result.stderr
    assert result.returncode == 0, result.stderr
    prefix = str(Path(path).with_suffix(''))
    spec = importlib.util.spec_from_file_location(prefix + '_pnnx', prefix + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def compare(actual, expected):
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    else:
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


class DynamicFill(torch.nn.Module):
    def __init__(self, inherited):
        super().__init__()
        self.inherited = inherited

    def forward(self, x):
        if self.inherited:
            return x + x.new_full((x.shape[0], 3), 1e-6)
        return x + torch.full((x.shape[0], 3), 0.1)


class ScalarComparison(torch.nn.Module):
    def forward(self, x):
        return x + torch.scalar_tensor(x[0, 0].item() > 0.1)


def test_float_parameters():
    for inherited in (False, True):
        model = DynamicFill(inherited)
        path = export('fill_' + str(inherited), model, (torch.ones(4, 3),),
                      dynamic_shapes=({0: torch.export.Dim('batch', min=2, max=8)},))
        converted = convert(path, ('inputshape=[4,3]', 'inputshape2=[6,3]'))
        for size in (4, 6):
            x = torch.zeros(size, 3)
            compare(converted(x), model(x))

    # This comparison is observable, so assertion pruning cannot remove it.
    path = export('scalar_comparison', ScalarComparison(), (torch.ones(2, 3),))
    convert(path, error='floating point parameter cannot be represented faithfully')


class Activation(torch.nn.Module):
    def __init__(self, name, source='local'):
        super().__init__()
        self.name = name
        self.source = source
        self.register_buffer('buffer', torch.ones(2, 3))

    def forward(self, x):
        if self.source == 'input':
            y = x.detach()
        elif self.source == 'buffer':
            y = self.buffer
        else:
            y = x + 0.25
        alias = y[:, 1:]
        before = y.clone()
        op = getattr(torch.ops.aten, self.name + '_').default
        if self.name == 'threshold':
            op(y, 0.1, -0.25)
        elif self.name == 'leaky_relu':
            op(y, 0.2)
        elif self.name in ('elu', 'celu'):
            op(y, 0.5)
        else:
            op(y)
        y.relu_()
        return y, alias, before


class AliasMutation(torch.nn.Module):
    def __init__(self, live):
        super().__init__()
        self.live = live

    def forward(self, x):
        y = x + 1
        alias = y.unsqueeze(0).expand(2, -1, -1)
        observed = alias.clone()
        y.sigmoid_()
        return y, alias if self.live else observed


class ManyMutations(torch.nn.Module):
    def forward(self, x):
        result = x
        for i in range(32):
            y = x + i
            alias = y.detach()
            y.sigmoid_()
            result = result + alias
        return result


def test_mutations():
    x = torch.linspace(-3, 2, 6).reshape(2, 3)
    for name in ('silu', 'leaky_relu', 'sigmoid', 'hardsigmoid', 'elu',
                 'tanh', 'celu', 'selu', 'threshold', 'mish'):
        model = Activation(name)
        path = export(name, model, (x,))
        compare(convert(path)(x), model(x))
        for source in ('input', 'buffer'):
            path = export(name + '_' + source, Activation(name, source), (x.clone(),))
            convert(path, error='mutable operator writes external tensor storage')

    path = export('live_alias', AliasMutation(True), (x,))
    convert(path, error='mutation has an unsupported observable alias')
    for name, model in (('dead_alias', AliasMutation(False)), ('many_mutations', ManyMutations())):
        path = export(name, model, (x,))
        compare(convert(path)(x), model(x))


class RangeCheck(torch.nn.Module):
    def forward(self, x):
        n = x[0, 0].item()
        torch._check(n >= 1)
        torch._check(n <= 8)
        return x + 1


def test_range_constraints():
    path = export('range_check', RangeCheck(), (torch.ones(2, 3, dtype=torch.int64),))

    def replace_checks(program, entries):
        nodes = program['graph_module']['graph']['nodes']
        item = next(n for n in nodes if n['target'] == 'torch.ops.aten.item.default')['outputs'][0]
        checks = [n for n in nodes if n['target'] == 'torch.ops.aten._assert_scalar.default']
        assert len(checks) == 2
        for node, target in zip(checks, ('sym_constrain_range', 'sym_constrain_range_for_size')):
            node['target'] = 'torch.ops.aten.' + target + '.default'
            node['inputs'] = [
                {'name': 'size', 'arg': item, 'kind': 1},
                {'name': 'min', 'arg': {'as_int': 1}, 'kind': 2},
                {'name': 'max', 'arg': {'as_int': 2**63 - 1}, 'kind': 2},
            ]
            node['outputs'] = []

    rewritten = 'test_pt2_loader_range_constraints.pt2'
    rewrite_archive(path, rewritten, replace_checks)
    converted = convert(rewritten)
    for value in (0, 3, 10):
        x = torch.full((2, 3), value, dtype=torch.int64)
        compare(converted(x), x + 1)
    text = Path(rewritten.replace('.pt2', '.pnnx.param')).read_text()
    assert 'constrain' not in text and 'Tensor.item' not in text

    def bad_outputs(program, entries):
        graph = program['graph_module']['graph']
        node = next(n for n in graph['nodes'] if 'sym_constrain_range' in n['target'])
        node['outputs'] = [graph['outputs'][0]]

    bad = 'test_pt2_loader_range_outputs.pt2'
    rewrite_archive(rewritten, bad, bad_outputs)
    convert(bad, error='unexpected assertion outputs')


class Scatter(torch.nn.Module):
    def forward(self, x, y):
        return torch.slice_scatter(x, y, dim=1, start=0, end=2**63 - 1), x[:, :2**63 - 1]


def test_scatter_and_names():
    model = Scatter()
    args = (torch.arange(6, dtype=torch.float32).reshape(2, 3), torch.ones(2, 3))
    path = export('scatter', model, args)
    compare(convert(path)(*args), model(*args))
    lines = Path(path.replace('.pt2', '.pnnx.param')).read_text().splitlines()
    for kind in ('Input', 'Output'):
        names = [line.split()[1] for line in lines if line.startswith('pnnx.' + kind + ' ')]
        assert names == ['pnnx_' + kind.lower() + '_' + str(i) for i in range(2)], names

    def bad_start(program, entries):
        node = next(n for n in program['graph_module']['graph']['nodes'] if 'slice_scatter' in n['target'])
        start = next(a for a in node['inputs'] if a['name'] == 'start')
        start['arg'] = {'as_int': 2**63 - 1}

    bad = 'test_pt2_loader_scatter_start.pt2'
    rewrite_archive(path, bad, bad_start)
    convert(bad, error='integer outside pnnx parameter range')


class Nested(torch.nn.Module):
    def forward(self, x):
        with torch.no_grad():
            return x.sin() + 1


def child_graph(program):
    node = next(n for n in program['graph_module']['graph']['nodes'] if 'wrap_with_' in n['target'])
    return next(a['arg']['as_graph']['graph'] for a in node['inputs'] if 'as_graph' in a['arg'])


def test_child_errors():
    path = export('nested', Nested(), (torch.ones(2, 3),))
    compare(convert(path)(torch.ones(2, 3)), Nested()(torch.ones(2, 3)))

    def bad_operator(program, entries):
        child_graph(program)['nodes'][0]['target'] = 'torch.ops.aten.no_such_op.default'

    bad = 'test_pt2_loader_nested_operator.pt2'
    rewrite_archive(path, bad, bad_operator)
    error = convert(bad, error='schema')
    assert error.count('pt2 load: node ') == 1 and 'wrap_with_' not in error, error

    def bad_tensor(program, entries):
        next(iter(child_graph(program)['tensor_values'].values())).pop('strides')

    bad = 'test_pt2_loader_nested_tensor.pt2'
    rewrite_archive(path, bad, bad_tensor)
    error = convert(bad, error='incomplete tensor metadata')
    assert 'wrap_with_set_grad_enabled' in error and ': tensor ' in error, error


class StoredViews(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(8 * 6 * 4, dtype=torch.float32).reshape(8, 6, 4)
        self.register_buffer('a', base[1:5:2, 1:4, :])
        self.register_buffer('b', base[2:6:2, 2:5, :])
        self.register_buffer('broadcast', base[1, 1, 1].expand(2, 3, 4))
        self.register_buffer('transposed', torch.arange(24, dtype=torch.float32).reshape(3, 4, 2).permute(2, 0, 1))
        self.register_buffer('singleton', base[1:5:2, 1:2, :].expand(2, 3, 4))
        self.register_buffer('contiguous', torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
        self.register_buffer('offset', torch.arange(30, dtype=torch.float32)[3:27].reshape(2, 3, 4), persistent=False)

    def forward(self, x):
        return (x + self.a, x + self.b, x + self.broadcast, x + self.transposed,
                x + self.singleton, x + self.contiguous, x + self.offset)


def test_storage():
    model = StoredViews()
    x = torch.zeros(2, 3, 4)
    path = export('storage', model, (x,))
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('_weights_config.json'))
        config = json.loads(archive.read(name))['config']
    assert config['a']['path_name'] == config['b']['path_name'] == config['broadcast']['path_name']
    compare(convert(path)(x), model(x))

    def bad_view(program, entries):
        name = next(n for n in entries if n.endswith('_weights_config.json'))
        config = json.loads(entries[name])
        config['config']['b']['tensor_meta']['storage_offset'] = {'as_int': 2**40}
        entries[name] = json.dumps(config).encode()

    bad = 'test_pt2_loader_storage_bounds.pt2'
    rewrite_archive(path, bad, bad_view)
    convert(bad, error='tensor view exceeds storage')


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    # Export outside no_grad so Nested retains its inference wrapper.
    test_float_parameters()
    test_mutations()
    test_range_constraints()
    test_scatter_and_names()
    test_child_errors()
    test_storage()
    return True


if __name__ == '__main__':
    test()
