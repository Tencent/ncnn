# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# Expected pt2 (torch.export) outcome per pnnx test tag.
#
# Every call to test_pnnx() is audited against this table so a test that
# silently drops off the pt2 channel (returns None) is a visible regression
# instead of a quiet skip.
#
#   unlisted tag => "pass"        : torch.export must succeed, conversion must
#                                   match (test_pnnx -> True)
#   "export-skip"                 : torch.export is pinned to reject the model
#                                   (test_pnnx -> None, e.g. dynamic shapes /
#                                   control flow the exporter cannot handle)
#   {"outcome": "skip", "needle"} : as above, plus the exporter error text must
#                                   contain the pinned diagnostic substring
#
# Regeneration workflow (after a torch version / suite change):
#   PNNX_PT2_RESULT_LOG=/tmp/r.log PNNX_PT2_RECORD_ONLY=1 \
#       python ../../tests/test_<tag>.py
# then move every None row of /tmp/r.log into EXPECT as an
# {"outcome": "skip", "needle": ...} entry pinning the torch.export diagnostic,
# and move every row that changed to True back to "pass".
#
# Full-suite audit (2026-09-06, torch 2.13): of the 393 test_pnnx() calls in the
# suite, 390 convert and match; exactly 3 are torch.export skips, all recorded
# below with a pinned diagnostic. PNNX_PT2_RESULT_LOG appends the exporter error
# for each skip, so a regeneration run yields the needles directly.

# Audited pt2-ncnn skips: the pt2 (torch.export) channel of these tags produces
# a graph whose operators have no ncnn lowering. pnnx still exits 0 and writes a
# .ncnn.param that keeps the ATen op, which ncnn then refuses to load, so
# test_pnnx_ncnn() reports a skip for the tags listed here and a *failure* for
# any other tag that ends up with a leftover op.
#
# All four are torchaudio spectrogram models: torch.export lifts the frontend to
# torchaudio.functional.spectrogram / .inverse_spectrogram, whose graphs keep the
# plain normalization ops (torch.stft + torch.view_as_real, torch.istft +
# torch.view_as_complex). pass_ncnn/torch_stft.cpp deliberately declines those
# forms because the surrounding abs()/sqrt(norm) scaling cannot be fused safely,
# so there is nothing to enable here. The torchscript channel of the same tests
# is unaffected and still converts to a plain Spectrogram layer.
EXPECT_NCNN_NO_LOWERING = {
    "test_torchaudio_F_spectrogram",
    "test_torchaudio_Spectrogram",
    "test_torchaudio_F_inverse_spectrogram",
    "test_torchaudio_InverseSpectrogram",
}

EXPECT = {
    # torch.export rejects these models (dynamic indexing / control flow /
    # dynamic-arange shapes); recorded so a future torch that starts exporting
    # them is noticed instead of silently re-covering them. Each entry pins the
    # diagnostic torch.export raises, so a change in the rejection reason is a
    # visible regression rather than a quiet skip.
    "test_Tensor_index": {
        "outcome": "skip",
        # unbacked (data-dependent) index shape leaked past the exported graph
        "needle": "Pending unbacked symbols",
    },
    "test_quantization_shufflenet_v2_x1_0": {
        "outcome": "skip",
        # torch.export cannot serialize the quantized packed-params object
        "needle": "__obj_flatten__",
    },
    "test_torch_arange": {
        "outcome": "skip",
        # torch.export refuses to guard on the data-dependent arange bound
        "needle": "Could not guard on data-dependent expression",
    },
}
