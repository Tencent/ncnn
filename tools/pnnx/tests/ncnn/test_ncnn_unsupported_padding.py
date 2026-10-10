# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import subprocess

import ncnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class Model(nn.Module):
    def __init__(self, mode, center=True):
        super(Model, self).__init__()
        self.mode = mode
        self.center = center

    def forward(self, x, y, z):
        a = torch.stft(x, n_fft=32, hop_length=8, center=self.center, pad_mode=self.mode, return_complex=True)
        b = torch.stft(x, n_fft=32, hop_length=8, window=torch.hann_window(32), center=self.center, pad_mode=self.mode, return_complex=True)
        c = torchaudio.functional.spectrogram(x, pad=0, window=torch.hann_window(32), n_fft=32, hop_length=8, win_length=32, power=None, normalized=False, center=self.center, pad_mode=self.mode)
        d = torchaudio.functional.spectrogram(x, pad=0, window=torch.hann_window(32), n_fft=32, hop_length=8, win_length=32, power=2, normalized=False, center=self.center, pad_mode=self.mode)
        e = torchaudio.functional.spectrogram(z, pad=0, window=torch.hann_window(32), n_fft=32, hop_length=8, win_length=32, power=None, normalized=False, center=self.center, pad_mode=self.mode)
        f = torchaudio.functional.spectrogram(z, pad=0, window=torch.hann_window(32), n_fft=32, hop_length=8, win_length=32, power=2, normalized=False, center=self.center, pad_mode=self.mode)
        g = F.pad(y, (3, 4), mode=self.mode if self.center else "reflect")
        return torch.view_as_real(a), torch.view_as_real(b), torch.view_as_real(c), d, torch.view_as_real(e), f, g


class PadModel(nn.Module):
    def __init__(self, mode, pad):
        super(PadModel, self).__init__()
        self.mode = mode
        self.pad = pad

    def forward(self, x):
        return F.pad(x, self.pad, mode=self.mode) + 1


def convert(net, name, shapes):
    net.eval()
    torch.manual_seed(0)
    inputs = tuple(torch.rand(shape) for shape in shapes)
    torch.jit.trace(net, inputs).save(name + ".pt")
    shape_arg = ",".join("[" + ",".join(str(d) for d in shape) + "]" for shape in shapes)
    proc = subprocess.run(["../../src/pnnx", name + ".pt", "inputshape=" + shape_arg, "fp16=0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return None
    with open(name + ".ncnn.param") as f:
        layers = [line.split()[0] for line in f.read().splitlines()[2:]]
    return inputs, layers, proc.stderr.decode()


def check_inference(net, name, inputs):
    spec = importlib.util.spec_from_file_location(name, name + "_ncnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = net(*inputs)
    actual = module.test_inference()
    if isinstance(expected, torch.Tensor):
        expected, actual = (expected,), (actual,)
    return len(expected) == len(actual) and all(a.shape == b.shape and torch.allclose(a, b, 1e-3, 1e-3) for a, b in zip(expected, actual))


def test():
    shapes = ((128,), (2, 128), (2, 3, 128))
    for mode, center in (("constant", True), ("replicate", True), ("reflect", True), ("circular", False)):
        net = Model(mode, center)
        name = "test_ncnn_padding_" + mode + str(center)
        result = convert(net, name, shapes)
        if result is None:
            return False
        inputs, layers, _ = result
        if any(layer in layers for layer in ("torch.stft", "torchaudio.functional.spectrogram", "F.pad")):
            return False
        if not check_inference(net, name, inputs):
            return False

    name = "test_ncnn_padding_circular"
    result = convert(Model("circular"), name, shapes)
    if result is None:
        return False
    _, layers, diagnostics = result
    # Each unsupported frontend must survive lowering, rather than silently
    # becoming a Spectrogram or Padding layer with another padding mode.
    if layers.count("torch.stft") != 2 or layers.count("torchaudio.functional.spectrogram") != 4 or layers.count("F.pad") != 1:
        return False
    if "Spectrogram" in layers or "Padding" in layers:
        return False
    for message in ("unsupported stft pad_mode circular", "unsupported spectrogram pad_mode circular", "unsupported pad mode circular"):
        if message not in diagnostics:
            return False
    with ncnn.Net() as net:
        if net.load_param(name + ".ncnn.param") == 0:
            return False

    # No-op padding is eliminated, and constant negative padding remains a crop.
    for mode, pad in (("circular", (0, 0)), ("constant", (-3, -4))):
        net = PadModel(mode, pad)
        name = "test_ncnn_padding_control_" + mode
        result = convert(net, name, ((2, 128),))
        if result is None or not check_inference(net, name, result[0]):
            return False
    return True


if __name__ == "__main__":
    exit(0 if test() else 1)
