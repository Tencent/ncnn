# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import io
import json
import os
import pathlib
import shutil
import struct
import subprocess
import zipfile

import torch
from common import rewrite_archive


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)


class Unseekable(io.BytesIO):
    def seek(self, *args):
        raise OSError('not seekable')


def check_reject(path, message):
    result = subprocess.run(['../../src/pnnx', str(path), 'fp16=0'], capture_output=True, text=True)
    if result.returncode == 0 or message not in result.stderr:
        print(result.stderr)
        return False
    return True


def multiple(p, entries):
    name = next(n for n in entries if '/models/' in n)
    entries[name.replace('/models/model.json', '/models/other.json')] = entries[name]


def test():
    net = Model().eval()
    torch.manual_seed(0)
    x = torch.rand(2, 3)
    a = net(x)
    ep = torch.export.export(net, (x,))
    path = pathlib.Path('test_pt2_archive.pt2')
    torch.export.save(ep, path)
    os.system('../../src/pnnx test_pt2_archive.pt2 fp16=0')

    import test_pt2_archive_pnnx
    b = test_pt2_archive_pnnx.test_inference()
    if not torch.equal(a, b):
        return False

    shutil.copyfile(path, 'test_pt2_archive_renamed.bin')
    if os.system('../../src/pnnx test_pt2_archive_renamed.bin fp16=0') != 0:
        return False
    for minor in (17, 18, 19, 20):
        dest = 'test_pt2_archive_schema_8_%d.pt2' % minor
        rewrite_archive(path, dest, lambda p, e: p['schema_version'].update(minor=minor))
        if os.system('../../src/pnnx %s fp16=0' % dest) != 0:
            return False

    # Guards are metadata for the original input contract, not runtime operators.
    rewrite_archive(path, 'test_pt2_archive_guards.pt2', lambda p, e: p.update(guards_code=['x.size()[0] > 1']))
    if os.system('../../src/pnnx test_pt2_archive_guards.pt2 fp16=0') != 0:
        return False
    import test_pt2_archive_guards_pnnx
    if not torch.equal(test_pt2_archive_guards_pnnx.test_inference(), a):
        return False

    for name, change, message in (
        ('schema_old', lambda p, e: p['schema_version'].update(minor=16), 'schema version'),
        ('schema_major', lambda p, e: p['schema_version'].update(major=9), 'schema version'),
        ('schema_version', lambda p, e: p['schema_version'].update(minor=99), 'schema version'),
        ('opset_version', lambda p, e: p['opset_version'].update(aten=999), 'opset'),
        ('archive_version', lambda p, e: e.update({next(n for n in e if n.endswith('/archive_version')): b'99'}), 'archive version'),
        ('missing_graph', lambda p, e: p['graph_module'].pop('graph'), 'missing or invalid graph'),
        ('invalid_guards', lambda p, e: p.update(guards_code=None), 'missing or invalid guards_code'),
        ('missing_node_arg', lambda p, e: p['graph_module']['graph']['nodes'][0]['inputs'][0].pop('arg'), 'missing or invalid arg'),
        ('multiple', multiple, 'exactly one'),
    ):
        dest = 'test_pt2_archive_%s.pt2' % name
        rewrite_archive(path, dest, change)
        if not check_reject(dest, message):
            return False

    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
        directory_offset = archive.start_dir

    # Exactly 65535 entries still fit in an ordinary ZIP without a ZIP64 locator.
    root = next(iter(entries)).split('/')[0]
    with zipfile.ZipFile('test_pt2_archive_65535.pt2', 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for key, data in entries.items():
            archive.writestr(key, data)
        for i in range(65535 - len(entries)):
            archive.writestr('%s/extra/%d' % (root, i), b'')
    os.system('../../src/pnnx test_pt2_archive_65535.pt2 fp16=0')
    import test_pt2_archive_65535_pnnx
    if not torch.equal(test_pt2_archive_65535_pnnx.test_inference(), a):
        return False

    # Exercise ZIP64 entries and streaming descriptors without large files.
    stream = Unseekable()
    zip64_limit = zipfile.ZIP64_LIMIT
    zipfile.ZIP64_LIMIT = 0
    try:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
            for key, data in entries.items():
                archive.writestr(key, data)
    finally:
        zipfile.ZIP64_LIMIT = zip64_limit
    pathlib.Path('test_pt2_archive_zip64.pt2').write_bytes(stream.getvalue())
    if os.system('../../src/pnnx test_pt2_archive_zip64.pt2 fp16=0') != 0:
        return False
    import test_pt2_archive_zip64_pnnx
    if not torch.equal(test_pt2_archive_zip64_pnnx.test_inference(), a):
        return False

    graph_name = next(n for n in entries if '/models/' in n)
    for name, raw in [('duplicate', b'{"a":1,"a":2}'), ('number', b'{"a":01}'), ('trailing', entries[graph_name] + b'!'), ('unknown_union', None)]:
        if raw is None:
            program = json.loads(entries[graph_name])
            program['graph_module']['graph']['nodes'][0]['inputs'][1]['arg'] = {'as_future_kind': 1}
            raw = json.dumps(program).encode()
        dest = 'test_pt2_archive_%s.pt2' % name
        with zipfile.ZipFile(dest, 'w') as archive:
            for key, data in entries.items():
                archive.writestr(key, raw if key == graph_name else data)
        if not check_reject(dest, 'pt2 load:'):
            return False

    with zipfile.ZipFile('test_pt2_archive_compressed.pt2', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, data in entries.items():
            archive.writestr(key, data)
    if not check_reject('test_pt2_archive_compressed.pt2', 'unsupported zip entry'):
        return False

    raw = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        offset = archive.getinfo(graph_name).header_offset
    name_length, extra_length = struct.unpack_from('<HH', raw, offset + 26)
    raw[offset + 30 + name_length + extra_length] ^= 1
    pathlib.Path('test_pt2_archive_checksum.pt2').write_bytes(raw)
    if not check_reject('test_pt2_archive_checksum.pt2', 'CRC mismatch'):
        return False

    pathlib.Path('test_pt2_archive_truncated.pt2').write_bytes(path.read_bytes()[:-22])
    if not check_reject('test_pt2_archive_truncated.pt2', 'invalid zip'):
        return False

    raw = bytearray(path.read_bytes())
    struct.pack_into('<I', raw, directory_offset + 42, 0xfffffffe)
    pathlib.Path('test_pt2_archive_bad_offset.pt2').write_bytes(raw)
    if not check_reject('test_pt2_archive_bad_offset.pt2', 'invalid local file header'):
        return False

    raw = bytearray(path.read_bytes())
    struct.pack_into('<II', raw, directory_offset + 20, 0xfffffffe, 0xfffffffe)
    pathlib.Path('test_pt2_archive_bad_size.pt2').write_bytes(raw)
    if not check_reject('test_pt2_archive_bad_size.pt2', 'entry data outside file'):
        return False

    # Names under extra/ do not turn a TorchScript archive into a pt2 archive.
    traced = torch.jit.trace(Model(), x)
    traced.save('test_pt2_archive_extra_marker.pt', _extra_files={'archive_format': 'pt2', 'models/model.json': '{}'})
    return os.system('../../src/pnnx test_pt2_archive_extra_marker.pt inputshape=[2,3] fp16=0') == 0


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
