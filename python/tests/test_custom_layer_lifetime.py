# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import gc
import struct
import weakref

import pytest

import ncnn


class LayerFactory:
    def __init__(self, increment):
        self.increment = increment
        self.layers = []
        self.destroyed = 0

    def create(self):
        increment = self.increment

        class CustomLayer(ncnn.Layer):
            def __init__(self):
                super().__init__()
                self.one_blob_only = True

            def forward(self, bottom, top, opt):
                top.clone_from(bottom, opt.blob_allocator)
                top.fill(bottom[0] + increment)
                return 0

        layer = CustomLayer()
        self.layers.append(layer)
        return layer

    def destroy(self, layer):
        self.layers.remove(layer)
        self.destroyed += 1


def register(net, factory, kind, index=0):
    key = 'CustomLayer' + str(index) if kind == 'name' else 256 + index
    assert net.register_custom_layer(key, factory.create, factory.destroy) == 0


def load(net, kind, tmp_path, index=0):
    if kind == 'name':
        assert net.load_param_mem(
            '7767517\n2 2\nInput data 0 1 data\n'
            'CustomLayer{} custom 1 1 data output\n'.format(index)
        ) == 0
    else:
        # Binary Input (16), followed by the registered custom layer.
        values = [7767517, 2, 2, 16, 0, 1, 0, -233,
                  256 + index, 1, 1, 0, 1, -233]
        path = tmp_path / 'custom.param.bin'
        path.write_bytes(struct.pack('<{}i'.format(len(values)), *values))
        assert net.load_param_bin(str(path)) == 0
    assert net.load_model(ncnn.DataReaderFromEmpty()) == 0


def infer(net, expected):
    with net.create_extractor() as ex:
        value = ncnn.Mat(1)
        value.fill(1.0)
        assert ex.input(0, value) == 0
        ret, output = ex.extract(1)
        assert ret == 0
        assert output[0] == expected


@pytest.mark.parametrize('kind', ['name', 'index'])
def test_repeated_net_registration(kind, tmp_path):
    for i in range(20):
        factory = LayerFactory(i)
        net = ncnn.Net()
        register(net, factory, kind)
        load(net, kind, tmp_path)
        infer(net, i + 1)
        del net
        gc.collect()
        assert factory.destroyed == 1
        assert not factory.layers


@pytest.mark.parametrize('kind', ['name', 'index'])
def test_independent_live_nets(kind, tmp_path):
    nets = []
    factories = []
    for i in range(20):
        net = ncnn.Net()
        factory = LayerFactory(i)
        register(net, factory, kind)
        load(net, kind, tmp_path)
        nets.append(net)
        factories.append(factory)
    # Exercise old callbacks after all subsequent registrations.
    for i, net in enumerate(nets):
        infer(net, i + 1)
        net.clear()
        assert factories[i].destroyed == 1
    for i, net in enumerate(nets):
        load(net, kind, tmp_path)
        infer(net, i + 1)
        net.clear()
        assert factories[i].destroyed == 2


@pytest.mark.parametrize('kind', ['name', 'index'])
def test_many_layers_in_one_net(kind, tmp_path):
    net = ncnn.Net()
    factories = [LayerFactory(i) for i in range(20)]
    for i, factory in enumerate(factories):
        register(net, factory, kind, i)
    for i in range(20):
        load(net, kind, tmp_path, i)
        infer(net, i + 1)
    net.clear()
    assert all(factory.destroyed == 1 for factory in factories)


@pytest.mark.parametrize('kind', ['name', 'index'])
def test_callbacks_live_until_net_destruction(kind, tmp_path):
    net = ncnn.Net()
    factory = LayerFactory(3)
    ref = weakref.ref(factory)
    register(net, factory, kind)
    del factory
    gc.collect()
    assert ref() is not None
    load(net, kind, tmp_path)
    infer(net, 4)
    net.clear()
    # clear() keeps registrations available for loading another model.
    assert ref() is not None
    load(net, kind, tmp_path)
    infer(net, 4)
    del net
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize('kind', ['name', 'index'])
def test_extractor_keeps_callbacks_alive(kind, tmp_path):
    net = ncnn.Net()
    factory = LayerFactory(5)
    ref = weakref.ref(factory)
    register(net, factory, kind)
    load(net, kind, tmp_path)
    ex = net.create_extractor()
    del net, factory
    gc.collect()
    assert ref() is not None
    value = ncnn.Mat(1)
    value.fill(1.0)
    assert ex.input(0, value) == 0
    ret, output = ex.extract(1)
    assert ret == 0 and output[0] == 6
    del ex
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize('kind', ['name', 'index'])
def test_replace_registration_after_clear(kind, tmp_path):
    net = ncnn.Net()
    factories = []
    for i in range(20):
        factory = LayerFactory(i)
        factories.append(factory)
        register(net, factory, kind)
        load(net, kind, tmp_path)
        infer(net, i + 1)
        net.clear()
        assert factory.destroyed == 1
    assert all(not factory.layers for factory in factories)


def test_rejected_registration_releases_callbacks():
    net = ncnn.Net()
    factory = LayerFactory(1)
    ref = weakref.ref(factory)
    assert net.register_custom_layer(-1, factory.create, factory.destroy) == -1
    del factory
    gc.collect()
    assert ref() is None


def test_global_net_interpreter_shutdown():
    import os
    import subprocess
    import sys
    import textwrap

    # Smoke-test process exit with a loaded global Net. Python callback/global
    # reference cycles are outside this test's collection guarantees.
    script = textwrap.dedent('''
        import ncnn

        _state = []

        class CustomLayer(ncnn.Layer):
            def __init__(self):
                super().__init__()
                self.one_blob_only = True

        def create():
            layer = CustomLayer()
            held.append(layer)
            return layer

        def destroy(layer):
            _state.append('destroyed')
            held.remove(layer)

        net = ncnn.Net()
        held = []
        assert net.register_custom_layer('CustomLayer', create, destroy) == 0
        assert net.load_param_mem(
            '7767517\\n2 2\\nInput data 0 1 data\\n'
            'CustomLayer custom 1 1 data output\\n'
        ) == 0
        print('loaded')
    ''')
    # Use the binding under test, including when pytest was launched with a
    # temporary sys.path rather than an installed ncnn package.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(os.path.abspath(path) for path in sys.path)
    result = subprocess.run(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'loaded'
    assert not result.stderr


def test_net_subclass_keeps_long_type_name_stable():
    class DerivedNet(ncnn.Net):
        pass

    net = DerivedNet()
    factory = LayerFactory(7)
    ref = weakref.ref(factory)
    name = 'CustomLayerWithLongTypeNameBeyondSmallStringStorage_' * 2
    param = ('7767517\n2 2\nInput data 0 1 data\n'
             '{} custom 1 1 data output\n'.format(name))
    assert net.register_custom_layer(name, factory.create, factory.destroy) == 0
    del name
    # Grow the owner container after registration, then use the original name
    # and userdata. Both addresses must remain valid through reallocations.
    for i in range(35):
        assert net.register_custom_layer(
            'OtherLongCustomLayerTypeName_{}'.format(i),
            factory.create,
            factory.destroy,
        ) == 0
    assert net.load_param_mem(param) == 0
    assert net.load_model(ncnn.DataReaderFromEmpty()) == 0
    infer(net, 8)
    net.clear()
    assert factory.destroyed == 1
    del factory, net
    gc.collect()
    assert ref() is None
