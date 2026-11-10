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


@pytest.mark.parametrize('index', [-1, 1000000 + 256])
def test_rejected_registration_releases_callbacks(index):
    net = ncnn.Net()
    active = LayerFactory(2)
    active_ref = weakref.ref(active)
    register(net, active, 'name')
    del active
    factory = LayerFactory(1)
    ref = weakref.ref(factory)
    assert net.register_custom_layer(index, factory.create, factory.destroy) == -1
    del factory
    gc.collect()
    assert ref() is None
    assert active_ref() is not None
    del net
    gc.collect()
    assert active_ref() is None


@pytest.mark.parametrize('first,second,type_name', [
    ('CustomLayer0', 'CustomLayer0', 'CustomLayer0'),
    (256, 256, None),
    ('CustomLayer0', 256, 'CustomLayer0'),
    ('Input', 16, 'Input'),
    (16, 'Input', 'Input'),
])
def test_re_registration_releases_superseded_callbacks(first, second, type_name, tmp_path):
    net = ncnn.Net()
    previous = LayerFactory(1)
    ref = weakref.ref(previous)
    assert net.register_custom_layer(first, previous.create, previous.destroy) == 0
    del previous
    current = LayerFactory(2)
    assert net.register_custom_layer(second, current.create, current.destroy) == 0
    gc.collect()
    assert ref() is None

    if type_name == 'Input':
        assert net.load_param_mem('7767517\n1 1\nInput data 0 1 data\n') == 0
        assert net.load_model(ncnn.DataReaderFromEmpty()) == 0
    else:
        # Name->index replacement must retain the original native name storage.
        load(net, 'name' if type_name else 'index', tmp_path)
        infer(net, 3)
    net.clear()
    assert current.destroyed == 1


def test_new_named_slot_with_custom_tag_is_rejected(tmp_path):
    net = ncnn.Net()
    refs = []
    for i in range(256):
        factory = LayerFactory(i)
        refs.append(weakref.ref(factory))
        register(net, factory, 'name', i)
    del factory
    gc.collect()
    assert all(ref() is not None for ref in refs)

    rejected = LayerFactory(300)
    rejected_ref = weakref.ref(rejected)
    assert net.register_custom_layer('CustomLayer256', rejected.create, rejected.destroy) == -1
    del rejected
    gc.collect()
    assert rejected_ref() is None
    assert all(ref() is not None for ref in refs)
    # Rejecting the unencodable next slot must preserve the original routing.
    load(net, 'name', tmp_path)
    infer(net, 1)
    net.clear()
    assert refs[0]().destroyed == 1


@pytest.mark.parametrize('index', [253, 254, 255])
@pytest.mark.parametrize('replacement', ['name', 'index'])
def test_named_boundary_slots_load_after_replacement(index, replacement, tmp_path):
    net = ncnn.Net()
    refs = []
    for i in range(index + 1):
        factory = LayerFactory(i)
        refs.append(weakref.ref(factory))
        register(net, factory, 'name', i)
    del factory
    current = LayerFactory(300)
    register(net, current, replacement, index)
    gc.collect()
    assert refs[index]() is None
    assert all(ref() is not None for ref in refs[:index])
    load(net, 'name', tmp_path, index)
    infer(net, 301)
    net.clear()
    assert current.destroyed == 1 and not current.layers
    load(net, 'name', tmp_path, index)
    infer(net, 301)
    net.clear()
    assert current.destroyed == 2


@pytest.mark.parametrize('index', [0, 254, 255, 512])
def test_index_registration_tracks_native_extent_for_new_names(index, tmp_path):
    net = ncnn.Net()
    indexed = LayerFactory(1)
    register(net, indexed, 'index', index)
    named = LayerFactory(2)
    ref = weakref.ref(named)
    next_index = index + 1
    result = net.register_custom_layer('FollowingNumeric', named.create, named.destroy)
    if next_index & 256:
        assert result == -1
        del named
        gc.collect()
        assert ref() is None
    else:
        assert result == 0
        assert net.load_param_mem(
            '7767517\n2 2\nInput data 0 1 data\n'
            'FollowingNumeric custom 1 1 data output\n'
        ) == 0
        assert net.load_model(ncnn.DataReaderFromEmpty()) == 0
        infer(net, 3)
        net.clear()
        assert named.destroyed == 1 and not named.layers
    load(net, 'index', tmp_path, index)
    infer(net, 2)
    net.clear()
    assert indexed.destroyed == 1 and not indexed.layers


def test_replacement_failed_and_builtin_registration_preserve_named_extent(tmp_path):
    net = ncnn.Net()
    for i in range(260):
        factory = LayerFactory(i)
        register(net, factory, 'name')
    del factory
    builtin = LayerFactory(0)
    assert net.register_custom_layer('Input', builtin.create, builtin.destroy) == 0
    assert net.register_custom_layer(16, builtin.create, builtin.destroy) == 0
    rejected = LayerFactory(1)
    rejected_ref = weakref.ref(rejected)
    for index in [-1, 1000000 + 256]:
        assert net.register_custom_layer(index, rejected.create, rejected.destroy) == -1
    del rejected
    gc.collect()
    assert rejected_ref() is None
    for i in range(1, 256):
        factory = LayerFactory(i)
        register(net, factory, 'name', i)
    del factory
    load(net, 'name', tmp_path, 255)
    infer(net, 256)
    net.clear()
    assert builtin.destroyed == 1 and not builtin.layers


@pytest.mark.parametrize('kind', ['name', 'index'])
def test_re_registration_retains_only_loaded_callback_owners(kind, tmp_path):
    class CompatibleFactory(LayerFactory):
        def destroy(self, layer):
            # Native registration uses the current destroyer, including for a
            # layer created before the replacement.
            if layer in self.layers:
                self.layers.remove(layer)
            self.destroyed += 1

    net = ncnn.Net()
    previous = CompatibleFactory(1)
    previous_ref = weakref.ref(previous)
    register(net, previous, kind)
    load(net, kind, tmp_path)
    del previous
    for increment in range(2, 9):
        current = CompatibleFactory(increment)
        current_ref = weakref.ref(current)
        register(net, current, kind)
        del current
        gc.collect()
        assert previous_ref() is not None
        if increment > 2:
            assert intermediate_ref() is None
        intermediate_ref = current_ref

    infer(net, 2)
    net.clear()
    gc.collect()
    assert previous_ref() is None
    assert current_ref().destroyed == 1
    load(net, kind, tmp_path)
    infer(net, 9)
    net.clear()
    assert not current_ref().layers
    assert current_ref().destroyed == 2
    del net
    gc.collect()
    assert current_ref() is None


def test_null_creator_does_not_retain_replaced_callbacks():
    class NullFactory:
        def create(self):
            return None

        def destroy(self, layer):
            raise AssertionError('no layer was created')

    net = ncnn.Net()
    previous = NullFactory()
    ref = weakref.ref(previous)
    assert net.register_custom_layer('CustomLayer0', previous.create, previous.destroy) == 0
    assert net.load_param_mem('7767517\n1 1\nCustomLayer0 custom 0 1 output\n') != 0
    del previous
    current = LayerFactory(1)
    register(net, current, 'name')
    gc.collect()
    assert ref() is None


def test_global_net_interpreter_shutdown(tmp_path):
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
        cwd=tmp_path,
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
