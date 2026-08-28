# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

PASS = "PASS"
EXPORT_UNSUPPORTED = "EXPORT_UNSUPPORTED"
PT2_FRONTEND_UNSUPPORTED = "PT2_FRONTEND_UNSUPPORTED"
PNNX_LOWERING_UNSUPPORTED = "PNNX_LOWERING_UNSUPPORTED"
UNCLASSIFIED = "UNCLASSIFIED"


# PT2 tests pass by default. Expected failures must name the exact failing
# boundary and match a stable diagnostic so that unrelated failures cannot
# become expected accidentally.
PT2_EXPECTED_FAILURES = {
    "test_pnnx_input_npy": (
        PT2_FRONTEND_UNSUPPORTED,
        "dynamic tensor shapes are unsupported",
    ),
    "test_Tensor_index": (
        EXPORT_UNSUPPORTED,
        "PendingUnbackedSymbolNotFound: Pending unbacked symbols",
    ),
    "test_torch_masked_select": (
        PT2_FRONTEND_UNSUPPORTED,
        "dynamic tensor shapes are unsupported",
    ),
    "test_torch_arange": (
        EXPORT_UNSUPPORTED,
        "GuardOnDataDependentSymNode: Could not guard on data-dependent expression",
    ),
    "test_transformers_funnel_attention": (
        PT2_FRONTEND_UNSUPPORTED,
        "dynamic tensor shapes are unsupported",
    ),
    "test_quantization_shufflenet_v2_x1_0": (EXPORT_UNSUPPORTED, "Conv2dPackedParamsBase"),
}


def pt2_expectation(basename):
    return PT2_EXPECTED_FAILURES.get(basename, (PASS, ""))
