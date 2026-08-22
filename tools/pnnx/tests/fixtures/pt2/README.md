# PT2 compatibility fixtures

This directory implements roadmap phase P0 and supplies the frozen archives used by the P1 archive-reader tests. The frozen MVP contract is in `pt2_capabilities.json`; `required_producers.json` defines the producer/container boundaries that must have real fixtures before the loader is merged.

## Required producer matrix

| PyTorch | Expected container | Why it is required |
|---|---|---|
| 2.6.0 | legacy ExportedProgram ZIP | first supported schema-major-8 producer |
| 2.7.0 | legacy ExportedProgram ZIP | final legacy container boundary |
| 2.8.0 | PT2 Archive | first new-container boundary |
| 2.9.0 | PT2 Archive | schema 8.14 producer |
| 2.10.0 | PT2 Archive | schema 8.15 producer |
| 2.11.0 | PT2 Archive | schema 8.17 producer |
| 2.12.1 | PT2 Archive | current pnnx consumer-libtorch baseline |

Each producer generates the same three deterministic cases:

- `state_and_constants`: parameters, persistent/non-persistent buffers and a lifted tensor constant;
- `strided_tensors`: non-contiguous parameter plus aliased buffer views with storage offsets;
- `structured_io`: multiple positional/keyword inputs and nested user outputs.

The generated binary archives are intentionally not fabricated by the inspection tests. They must come from the exact PyTorch producer named in their manifest.

## Generate a fixture set

Create an isolated Python environment for the exact producer, install its CPU wheel, and run:

```sh
python generate_pt2_fixtures.py \
    --expected-torch 2.8.0 \
    --output data/torch_2_8_0
```

Use `--force` only when intentionally replacing an existing set. Use `--check-determinism` to export every case twice and require identical semantic records. The semantic hash excludes the archive `.data/serialization_id`, archive root prefix, and JSON `metadata` debug fields; PyTorch 2.12 embeds process-specific graph ids there. All graph/signature/schema/opset/payload records remain covered. The full archive SHA is still pinned in the manifest.

The generator loads every archive back with the same producer and requires exact eager/exported output equality. It then writes `manifest.json` containing the exact producer/Python versions, archive kind, normalized record list, compression/flag metadata, schema/opset versions, graph summary, byte size and SHA-256 for each `.pt2` file.

## Inspect and verify without PyTorch

The fixture inspector only uses the Python standard library:

```sh
python pt2_fixture_tools.py inspect data/torch_2_8_0/state_and_constants.pt2
python pt2_fixture_tools.py verify data/torch_2_8_0/manifest.json
python pt2_fixture_tools.py verify-matrix .
```

`verify-matrix` checks `required_producers.json`, requires every producer/case, verifies hashes and confirms the expected container/schema boundary. This is the P0 completion gate.

## Exercise the P1 C++ archive reader

The default CMake build now creates `pnnx_pt2_archive_test`. CTest registers every frozen `.pt2` archive independently, plus a standard-library-only malformed corpus covering truncated central directories, unsafe paths, duplicate normalized records, unsupported compression, bad offsets and CRC failures:

```sh
ctest --output-on-failure -R 'test_pt2_(archive|storezip|cli)'
```

The CLI probe test requires both legacy and PT2 Archive inputs to stop before TorchScript loading and report the temporary `recognized ... loader is not enabled yet` diagnostic. This error is intentional until the graph/schema loader lands in later roadmap phases.

## Acceptance policy

- Fixtures are immutable once reviewed. A changed SHA requires a manifest update explaining the producer or generator change.
- A fixture directory name uses the normalized exact version, for example `torch_2_12_1`.
- `manifest.json` is always generated, never hand-edited.
- Archives from a different producer version are rejected even if their schema happens to match.
- Pickled payloads are trusted test data only. Do not use the spike to load untrusted files.
