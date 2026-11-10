# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import json
import zipfile


def rewrite_archive(src, dst, change):
    with zipfile.ZipFile(src) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    graph_name = next(name for name in entries if '/models/' in name and name.endswith('.json'))
    program = json.loads(entries[graph_name])
    change(program, entries)
    entries[graph_name] = json.dumps(program, ensure_ascii=False, allow_nan=False).encode()
    with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
