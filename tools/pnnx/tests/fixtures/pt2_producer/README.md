# pt2 producer-version fixtures

These `.pt2` files are **real ExportedProgram archives** written by several
torch releases at and after the 2.8 pt2-archive switch, checked in so CI (which
only installs one torch) can still exercise the producer-compatibility surface.

Unlike `../pt2_schema` — which takes a single 2.13 export and downgrades the
`schema_version.minor` / strips newer argument-variant fields in place — these
were produced by the actual producer releases. They pin that the loader is
**producer-agnostic**: neither the container layout nor the raw-payload schema
field set of a given release gates acceptance.

| fixture | producer | covers |
|---|---|---|
| `producer_2_9_0.pt2` | torch 2.9.0+cpu | earliest release of the 2.8+ pt2-archive era |
| `producer_2_11_0.pt2` | torch 2.11.0+cpu | mid-range schema minor |
| `producer_2_12_1.pt2` | torch 2.12.1+cpu | schema minor immediately before the development version |
| `producer_2_14_0.pt2` | torch 2.14.0+cpu | producer newer than the linked LibTorch (forward compatibility) |

All four encode the **same model and the same weights**, so
`test_pt2_producer_versions.py` checks each converted model against one eager
reference computed with the CI's own torch.

## model

`generate.py` defines the model; it is the single source of truth because the
test imports it to build the reference:

- `nn.Conv2d(3, 4, 3, padding=1)` — weight + bias (state dict payload)
- `register_buffer("scale", ...)` — persistent buffer (state dict payload)
- `relu(y + 1.0)` — literal constant folded into the graph
- fixed `[1, 3, 8, 8]` input

Parameters are **deterministic by construction** (derived from `torch.arange`,
not from a seeded RNG), so the reference does not depend on Conv2d init
changing between releases.

## reproducer

```bash
pip install --target /tmp/pp290 \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.9.0"

# the tag argument also names the fixture: producer_<tag>.pt2
PYTHONPATH=/tmp/pp290 python generate.py 2.9.0
```

`torch.export.save` requires torch >= 2.8; the fixtures are read back by the
CI's own torch (`torch.load` is backward compatible).
