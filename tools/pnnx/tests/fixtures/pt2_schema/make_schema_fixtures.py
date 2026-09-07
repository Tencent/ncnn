# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# Reproducer for fixtures/pt2_schema/*.pt2 : a fixed conv model exported with
# torch 2.13 (raw-payload schema 8.20), then downgraded in place to schema
# 8.17 / 8.15 / 8.14. The loader is field-presence driven (no schema_version
# gate), so these fixtures pin two things:
#   - the schema_version.minor value itself is not used to accept/reject
#   - archives with the pre-8.15 argument-variant field set still convert
#     identically (8.15+ added as_nested_tensors / as_int_lists /
#     as_string_to_argument / as_float_lists; 8.14/8.15 fixtures drop them)
#
# run with a torch >= 2.13 (uses torch.export.save) from this directory:
#   python make_schema_fixtures.py

import json
import os
import zipfile

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


def make_base(path):
    torch.manual_seed(42)

    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.c = nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            return self.c(x).relu() + 1

    m = M().eval()
    x = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        ep = torch.export.export(m, (x,))
        torch.export.save(ep, path)


def _strip_8_15_fields(obj):
    if isinstance(obj, dict):
        for k in ("as_nested_tensors", "as_int_lists", "as_string_to_argument", "as_float_lists"):
            obj.pop(k, None)
        for v in obj.values():
            _strip_8_15_fields(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_8_15_fields(v)


def downgrade(src, dst, minor, strip_new_fields):
    zin = zipfile.ZipFile(src)
    model_name = [n for n in zin.namelist() if n.endswith("models/model.json")][0]
    model = json.loads(zin.read(model_name))
    model["schema_version"]["minor"] = minor
    if strip_new_fields:
        _strip_8_15_fields(model)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for i in zin.infolist():
            if i.filename == model_name:
                z.writestr(model_name, json.dumps(model))
            else:
                z.writestr(i.filename, zin.read(i.filename))
    zin.close()


if __name__ == "__main__":
    base = os.path.join(HERE, "schema_8_20.pt2")
    make_base(base)
    downgrade(base, os.path.join(HERE, "schema_8_17.pt2"), 17, False)
    downgrade(base, os.path.join(HERE, "schema_8_15.pt2"), 15, True)
    downgrade(base, os.path.join(HERE, "schema_8_14.pt2"), 14, True)
    print("wrote schema_8_20/8_17/8_15/8_14.pt2 fixtures")
