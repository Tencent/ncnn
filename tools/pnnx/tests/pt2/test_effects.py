# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import subprocess

import torch
from packaging import version
from common import rewrite_archive


class Local(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        return y.relu_()


class Lifted(torch.nn.Module):
    def forward(self, x):
        y = torch.tensor([1., -2., 3.])
        return x + y.detach_()


class InputWrite(torch.nn.Module):
    def forward(self, x):
        return x.add_(1)


class BufferWrite(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.zeros(2, 3))

    def forward(self, x):
        self.b.add_(1)
        return x + self.b


class AliasWrite(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        alias = y.t()
        y.relu_()
        return alias + 1


class ViewWrite(torch.nn.Module):
    def __init__(self, sliced=False, return_base=True, observe_alias=False):
        super().__init__()
        self.sliced = sliced
        self.return_base = return_base
        self.observe_alias = observe_alias

    def forward(self, x):
        y = x.clone()
        v = (y[:1] if self.sliced else y).view(-1)
        alias = v[1:] if self.observe_alias else v
        v.add_(1)
        return y if self.return_base else alias


class LocalViewAliases(torch.nn.Module):
    def __init__(self, repeat, view=True):
        super().__init__()
        self.repeat = repeat
        self.view = view

    def forward(self, x):
        y = x.clone()
        v = y[:1].view(-1) if self.view else y
        alias = v[:]
        if self.repeat:
            v.add_(1)
            v.mul_(2)
            return alias
        v[1:].add_(1)
        return torch.stack((v, alias))


class DetachWrite(torch.nn.Module):
    def forward(self, x):
        return x.detach().relu_()


class EvalDropout(torch.nn.Module):
    def __init__(self, inplace=False, external=False):
        super().__init__()
        self.inplace = inplace
        self.external = external

    def forward(self, x):
        base = x if self.external else x.clone()
        view = base.view(-1)
        result = torch.nn.functional.dropout(base, 0.5, False, self.inplace)
        result.add_(2)
        return torch.stack((base.flatten(), view, result.flatten()))


class NumericCopy(torch.nn.Module):
    def __init__(self, value, view=False, inplace=True):
        super().__init__()
        self.value = value
        self.view = view
        self.inplace = inplace

    def forward(self, x):
        base = x.clone()
        target = base[:, 1:] if self.view else base
        if self.inplace:
            result = torch.ops.aten.copy_.default(target, self.value)
        else:
            result = torch.ops.aten.copy.default(target, self.value)
        return torch.cat((base.flatten(), result.flatten()))


class InstanceNorm(torch.nn.Module):
    def __init__(self, use_input_stats, running_stats):
        super().__init__()
        self.use_input_stats = use_input_stats
        self.register_buffer('mean', torch.zeros(2) if running_stats else None)
        self.register_buffer('var', torch.ones(2) if running_stats else None)

    def forward(self, x):
        return torch.nn.functional.instance_norm(x.unsqueeze(0), self.mean, self.var,
                                                use_input_stats=self.use_input_stats, momentum=0.5)


class AssertMetadata(torch.nn.Module):
    def forward(self, x):
        torch.ops.aten._assert_tensor_metadata.default(x, dtype=torch.float32, device=torch.device('cpu'), layout=torch.strided)
        return x + 1


class AssertTensor(torch.nn.Module):
    def forward(self, x):
        torch.ops.aten._assert_async.msg(x.sum() > 0, 'positive sum expected')
        return x + 1


class AssertNested(torch.nn.Module):
    def forward(self, x):
        with torch.no_grad():
            torch._check(x[0, 0].item() > 0)
            return x + 1


def bad_assert(p, entries):
    node = next(n for n in p['graph_module']['graph']['nodes'] if '_assert_tensor_metadata' in n['target'])
    next(a for a in node['inputs'] if a['name'] == 'dtype')['arg'] = {'as_scalar_type': 8}


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    for name, model in [('local', Local()), ('lifted', Lifted()),
                         ('view_local', ViewWrite(return_base=False)),
                         ('slice_view_local', ViewWrite(sliced=True, return_base=False)),
                         ('view_alias', ViewWrite(return_base=False, observe_alias=True)),
                         ('slice_view_alias', ViewWrite(sliced=True, return_base=False, observe_alias=True)),
                         ('view_repeat_alias', LocalViewAliases(True)),
                         ('alias_write', AliasWrite()),
                         ('view_write', ViewWrite()),
                         ('slice_view_write', ViewWrite(sliced=True)),
                         ('view_alias_list', LocalViewAliases(False)),
                         ('root_alias_list', LocalViewAliases(False, view=False)),
                         ('eval_dropout', EvalDropout()),
                         ('eval_dropout_inplace', EvalDropout(inplace=True)),
                         ('copy_number', NumericCopy(2)),
                         ('copy_float_view', NumericCopy(-2.75, view=True)),
                         ('copy_bool', NumericCopy(True)),
                         ('copy_functional', NumericCopy(2, inplace=False)),
                         ('copy_functional_view', NumericCopy(-2.75, view=True, inplace=False)),
                         ('instance_norm_stateless', InstanceNorm(True, False)),
                         ('instance_norm_running', InstanceNorm(False, True))]:
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 3)
        expected = model(x)
        prefix = 'test_pt2_effects_' + name
        program = torch.export.export(model, (x,))
        torch.export.save(program, prefix + '.pt2')
        os.system('../../src/pnnx ' + prefix + '.pt2 fp16=0')

        actual = importlib.import_module(prefix + '_pnnx').Model().eval()(x)
        if actual.shape != expected.shape or not torch.allclose(actual, expected, 2e-5, 2e-5):
            return False

    # Nested checks use SymInt conditions, including when the condition is false at inference.
    model = AssertNested().eval()
    x = torch.tensor([[2, 3, 4], [5, 6, 7]])
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_effects_nested_assert.pt2')
    os.system('../../src/pnnx test_pt2_effects_nested_assert.pt2 fp16=0')

    import test_pt2_effects_nested_assert_pnnx
    converted = test_pt2_effects_nested_assert_pnnx.Model().eval()
    for value in (x, -x):
        actual = converted(value)
        if actual.shape != value.shape or actual.dtype != value.dtype or not torch.equal(actual, value + 1):
            return False

    model = AssertMetadata().eval()
    torch.manual_seed(0)
    x = torch.rand(2, 3)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_effects_metadata.pt2')
    os.system('../../src/pnnx test_pt2_effects_metadata.pt2 fp16=0')

    import test_pt2_effects_metadata_pnnx
    actual = test_pt2_effects_metadata_pnnx.test_inference()
    if actual.shape != expected.shape or not torch.allclose(actual, expected, 2e-5, 2e-5):
        return False

    for name, model, message in [
        ('input_write', InputWrite(), 'mutable'),
        ('buffer_write', BufferWrite(), 'mutable'),
        ('detach_write', DetachWrite(), 'external tensor'),
        ('dropout_input_write', EvalDropout(external=True), 'external tensor'),
        ('instance_norm_update', InstanceNorm(True, True), 'running statistics update'),
    ]:
        # Mutation samples exercise rejection, so avoid running their stateful references.
        program = torch.export.export(model.eval(), (torch.rand(2, 3),))
        path = 'test_pt2_effects_' + name + '.pt2'
        torch.export.save(program, path)
        result = subprocess.run(['../../src/pnnx', path, 'fp16=0'], capture_output=True, text=True, timeout=90)
        if result.returncode == 0 or message not in result.stderr:
            print(result.stderr)
            return False

    rewrite_archive('test_pt2_effects_metadata.pt2', 'test_pt2_effects_ignored_assert.pt2', bad_assert)
    if os.system('../../src/pnnx test_pt2_effects_ignored_assert.pt2 fp16=0') != 0:
        return False
    import test_pt2_effects_ignored_assert_pnnx
    if not torch.equal(test_pt2_effects_ignored_assert_pnnx.test_inference(), expected):
        return False

    model = AssertTensor().eval()
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_effects_tensor_assert.pt2')
    if os.system('../../src/pnnx test_pt2_effects_tensor_assert.pt2 fp16=0') != 0:
        return False
    import test_pt2_effects_tensor_assert_pnnx
    converted = test_pt2_effects_tensor_assert_pnnx.Model().eval()
    for value in (x, -torch.ones(2, 3)):
        if not torch.equal(converted(value), value + 1):
            return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
