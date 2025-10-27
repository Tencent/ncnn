
# operator fuse priority

## 10 info

* Tensor.size

## 20 creation

* Tensor.new_empty
* Tensor.new_ones
* Tensor.new_zeros
* torch.arange
* torch.clone
* torch.diag
* torch.empty
* torch.empty_like
* torch.full
* torch.full_like
* torch.normal
* torch.ones
* torch.ones_like
* torch.randn
* torch.randn_like
* torch.zeros
* torch.zeros_like

## 30 compare

* torch.eq
* torch.ge
* torch.gt
* torch.le
* torch.lt
* torch.ne

## 40 arithmetic

* torch.bitwise_and
* torch.bitwise_left_shift
* torch.bitwise_not
* torch.bitwise_or
* torch.bitwise_right_shift
* torch.bitwise_xor
* torch.clamp
* torch.dequantize
* torch.lgamma
* torch.positive
* torch.quantize_per_tensor

## 50 statistical / reduce

* torch.amax
* torch.amin
* torch.argmax
* torch.argmin
* torch.logsumexp
* torch.max
* torch.mean
* torch.min
* torch.prod
* torch.std
* torch.sum
* torch.topk
* torch.var

## 60 relayout / reshape

* Tensor.copy
* Tensor.expand
* Tensor.expand_as
* Tensor.permute
* Tensor.repeat
* Tensor.reshape
* Tensor.to
* Tensor.type_as
* Tensor.unflatten
* torch.cat
* torch.chunk
* torch.complex
* torch.flatten
* torch.flip
* torch.imag
* torch.narrow
* torch.real
* torch.repeat_interleave
* torch.roll
* torch.split
* torch.squeeze
* torch.stack
* torch.t
* torch.tensor_split
* torch.tile
* torch.transpose
* torch.unbind
* torch.unsqueeze
* torch.view_as_complex
* torch.view_as_real

## 70 gather / scatter

* Tensor.fill
* Tensor.index
* Tensor.index_put
* Tensor.masked_fill
* Tensor.select
* Tensor.slice
* torch.gather
* torch.index_select
* torch.masked_select
* torch.scatter_add
* torch.slice_scatter
* torch.where

## 80 fft

* torch.fft_fft
* torch.fft_fft2
* torch.fft_fftn
* torch.fft_hfft
* torch.fft_hfft2
* torch.fft_hfftn
* torch.fft_ifft
* torch.fft_ifft2
* torch.fft_ifftn
* torch.fft_ihfft
* torch.fft_ihfft2
* torch.fft_ihfftn
* torch.fft_irfft
* torch.fft_irfft2
* torch.fft_irfftn
* torch.fft_rfft
* torch.fft_rfft2
* torch.fft_rfftn
* torch.istft
* torch.stft

## 90 matrix arithmetic

* torch.addmm
* torch.baddbmm
* torch.bmm
* torch.cross
* torch.cumprod
* torch.cumsum
* torch.einsum
* torch.matmul
* torch.mm
* torch.mv
* torch.norm

## 100 activation

* F.alpha_dropout
* F.celu
* F.dropout
* F.dropout23d
* F.elu
* F.feature_alpha_dropout
* F.gelu
* F.glu
* F.hardshrink
* F.hardsigmoid
* F.hardswish
* F.hardtanh
* F.leaky_relu
* F.log_softmax
* F.logsigmoid
* F.mish
* F.prelu
* F.relu
* F.relu6
* F.rrelu
* F.selu
* F.sigmoid
* F.silu
* F.softmax
* F.softmin
* F.softplus
* F.softshrink
* F.softsign
* F.tanh
* F.tanhshrink
* F.threshold

## 110 low utility

* F.affine_grid
* F.fold
* F.interpolate
* F.linear
* F.pad
* F.pairwise_distance
* F.pixel_shuffle
* F.pixel_unshuffle
* F.unfold
* F.upsample
* F.upsample_bilinear
* F.upsample_nearest

## 120 pooling

* F.adaptive_avg_pool1d
* F.adaptive_avg_pool2d
* F.adaptive_avg_pool3d
* F.adaptive_max_pool1d
* F.adaptive_max_pool2d
* F.adaptive_max_pool3d
* F.avg_pool1d
* F.avg_pool2d
* F.avg_pool3d
* F.lp_pool1d
* F.lp_pool2d
* F.max_pool1d
* F.max_pool2d
* F.max_pool3d

## 130 normalization

* F.batch_norm
* F.group_norm
* F.instance_norm
* F.layer_norm
* F.local_response_norm
* F.normalize
* F.rms_norm

## 140 high level

* F.conv1d
* F.conv2d
* F.conv3d
* F.conv_transpose123d
* F.conv_transpose1d
* F.conv_transpose2d
* F.conv_transpose3d
* F.embedding
* F.grid_sample
* F.scaled_dot_product_attention
* torchaudio_F.inverse_spectrogram
* torchaudio_F.spectrogram



