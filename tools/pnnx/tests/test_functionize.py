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
from packaging import version


class Aliases(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        base = x + 1
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


def run_case(binary, model, mode, frontend, dtype=torch.float32, dynamic=False,
             rejection=None, inputs=None, forbid_type_as=False):
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
        assert any(node.op == 'call_function' and
                   str(node.target).split('.')[-2].endswith('_')
                   for node in program.graph.nodes), 'raw export lost the mutation under test'
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
    # Retained diagnostic directories may be reused; stale output must never
    # turn a failed conversion into a passing regression.
    for suffix in ('_pnnx.py', '.pnnx.param', '.pnnx.bin',
                   '_ncnn.py', '.ncnn.param', '.ncnn.bin'):
        Path(name + suffix).unlink(missing_ok=True)
    result = subprocess.run([binary, path, 'fp16=0'] + args,
                            capture_output=True, text=True, timeout=90)
    Path(name + '.conversion.log').write_text(result.stdout + result.stderr)
    if rejection is not None:
        assert result.returncode > 0, ('unsupported mutation did not reject cleanly', result.returncode)
        assert rejection in result.stderr, result.stderr
        assert not Path(name + '_pnnx.py').exists(), 'rejected graph emitted executable output'
        return
    # ncnn has narrower coverage than pnnx. Its explicit unsupported diagnostic
    # may occur after successful pnnx conversion, but other errors must fail.
    assert result.returncode == 0 or ('pt2 ncnn: unsupported' in result.stderr and
                                    Path(name + '_pnnx.py').is_file()), result.stderr
    spec = importlib.util.spec_from_file_location(name, name + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    converted = module.Model().eval()
    for batch in ([2, 3, 5] if dynamic else [2]):
        value = torch.arange(-4, batch * 4 - 4).reshape(batch, 4).to(dtype)
        values = [item.clone() for item in samples] if inputs is not None else [value]
        originals = [item.clone() for item in values]
        reference = model.reference if isinstance(model, MixedMutation) else model
        expected = reference(*[item.clone() for item in values])
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
    # Raw pt2 ones_like is currently rejected by the ncnn backend.
    if (os.environ.get('PNNX_TEST_NCNN') == '1' and dtype == torch.float32 and
            not dynamic and mode in ('siblings', 'nonfirst', 'chained', 'select', 'view_base') and
            not (frontend == 'pt2' and mode == 'nonfirst')):
        check_ncnn(name, model, x.clone())
        print('PASS ncnn/' + name)


def test():
    torch.set_num_threads(1)
    binary = str(Path(os.environ.get('PNNX_BIN', '../src/pnnx')).resolve())
    frontends = ['torchscript']
    if version.parse(torch.__version__) >= version.parse('2.11') and os.environ.get('PNNX_BUILD_PT2', '1') != '0':
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
            for frontend in frontends:
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
