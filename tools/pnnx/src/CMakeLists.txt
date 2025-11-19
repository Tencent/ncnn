
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(pnnx_pass_level0_SRCS
    pass_level0/constant_unpooling.cpp
    pass_level0/convert_half_to_float.cpp
    pass_level0/inline_block.cpp
    pass_level0/reset_device.cpp
    pass_level0/flatten_input.cpp
    pass_level0/shape_inference.cpp
)

set(pnnx_pass_level1_SRCS
    pass_level1/fuse_module_pass.cpp

    pass_level1/nn_AdaptiveAvgPool1d.cpp
    pass_level1/nn_AdaptiveAvgPool2d.cpp
    pass_level1/nn_AdaptiveAvgPool3d.cpp
    pass_level1/nn_AdaptiveMaxPool1d.cpp
    pass_level1/nn_AdaptiveMaxPool2d.cpp
    pass_level1/nn_AdaptiveMaxPool3d.cpp
    pass_level1/nn_AlphaDropout.cpp
    pass_level1/nn_AvgPool1d.cpp
    pass_level1/nn_AvgPool2d.cpp
    pass_level1/nn_AvgPool3d.cpp
    pass_level1/nn_BatchNorm1d.cpp
    pass_level1/nn_BatchNorm2d.cpp
    pass_level1/nn_BatchNorm3d.cpp
    pass_level1/nn_CELU.cpp
    pass_level1/nn_ChannelShuffle.cpp
    pass_level1/nn_ConstantPad1d.cpp
    pass_level1/nn_ConstantPad2d.cpp
    pass_level1/nn_ConstantPad3d.cpp
    pass_level1/nn_Conv1d.cpp
    pass_level1/nn_Conv2d.cpp
    pass_level1/nn_Conv3d.cpp
    pass_level1/nn_ConvTranspose1d.cpp
    pass_level1/nn_ConvTranspose2d.cpp
    pass_level1/nn_ConvTranspose3d.cpp
    pass_level1/nn_Dropout.cpp
    pass_level1/nn_Dropout2d.cpp
    pass_level1/nn_Dropout3d.cpp
    pass_level1/nn_ELU.cpp
    pass_level1/nn_Embedding.cpp
    pass_level1/nn_Fold.cpp
    pass_level1/nn_GELU.cpp
    pass_level1/nn_GLU.cpp
    pass_level1/nn_GroupNorm.cpp
    pass_level1/nn_GRU.cpp
    pass_level1/nn_Hardshrink.cpp
    pass_level1/nn_Hardsigmoid.cpp
    pass_level1/nn_Hardswish.cpp
    pass_level1/nn_Hardtanh.cpp
    pass_level1/nn_InstanceNorm1d.cpp
    pass_level1/nn_InstanceNorm2d.cpp
    pass_level1/nn_InstanceNorm3d.cpp
    pass_level1/nn_LayerNorm.cpp
    pass_level1/nn_LeakyReLU.cpp
    pass_level1/nn_Linear.cpp
    pass_level1/nn_LocalResponseNorm.cpp
    pass_level1/nn_LogSigmoid.cpp
    pass_level1/nn_LogSoftmax.cpp
    pass_level1/nn_LPPool1d.cpp
    pass_level1/nn_LPPool2d.cpp
    pass_level1/nn_LSTM.cpp
    pass_level1/nn_MaxPool1d.cpp
    pass_level1/nn_MaxPool2d.cpp
    pass_level1/nn_MaxPool3d.cpp
    #pass_level1/nn_maxunpool2d.cpp
    pass_level1/nn_Mish.cpp
    pass_level1/nn_MultiheadAttention.cpp
    pass_level1/nn_PixelShuffle.cpp
    pass_level1/nn_PixelUnshuffle.cpp
    pass_level1/nn_PReLU.cpp
    pass_level1/nn_ReflectionPad1d.cpp
    pass_level1/nn_ReflectionPad2d.cpp
    pass_level1/nn_ReLU.cpp
    pass_level1/nn_ReLU6.cpp
    pass_level1/nn_ReplicationPad1d.cpp
    pass_level1/nn_ReplicationPad2d.cpp
    pass_level1/nn_ReplicationPad3d.cpp
    pass_level1/nn_RMSNorm.cpp
    pass_level1/nn_RNN.cpp
    pass_level1/nn_RReLU.cpp
    pass_level1/nn_SELU.cpp
    pass_level1/nn_Sigmoid.cpp
    pass_level1/nn_SiLU.cpp
    pass_level1/nn_Softmax.cpp
    pass_level1/nn_Softmax2d.cpp
    pass_level1/nn_Softmin.cpp
    pass_level1/nn_Softplus.cpp
    pass_level1/nn_Softshrink.cpp
    pass_level1/nn_Softsign.cpp
    pass_level1/nn_Tanh.cpp
    pass_level1/nn_Tanhshrink.cpp
    pass_level1/nn_Threshold.cpp
    pass_level1/nn_Unfold.cpp
    pass_level1/nn_Upsample.cpp
    pass_level1/nn_UpsamplingBilinear2d.cpp
    pass_level1/nn_UpsamplingNearest2d.cpp
    pass_level1/nn_ZeroPad2d.cpp

    pass_level1/nn_quantized_Conv2d.cpp
    pass_level1/nn_quantized_DeQuantize.cpp
    pass_level1/nn_quantized_Linear.cpp
    pass_level1/nn_quantized_Quantize.cpp

    pass_level1/torchvision_DeformConv2d.cpp
    pass_level1/torchvision_RoIAlign.cpp
)

set(pnnx_pass_level2_SRCS
    pass_level2/eliminate_contiguous.cpp
    pass_level2/eliminate_size_numtotensor_int.cpp
    pass_level2/functionize.cpp
    pass_level2/fuse_constantlist.cpp

    pass_level2/F_adaptive_avg_pool1d.cpp
    pass_level2/F_adaptive_avg_pool2d.cpp
    pass_level2/F_adaptive_avg_pool3d.cpp
    pass_level2/F_adaptive_max_pool1d.cpp
    pass_level2/F_adaptive_max_pool2d.cpp
    pass_level2/F_adaptive_max_pool3d.cpp
    pass_level2/F_alpha_dropout.cpp
    pass_level2/F_affine_grid.cpp
    pass_level2/F_avg_pool1d.cpp
    pass_level2/F_avg_pool2d.cpp
    pass_level2/F_avg_pool3d.cpp
    pass_level2/F_batch_norm.cpp
    pass_level2/F_celu.cpp
    pass_level2/F_conv1d.cpp
    pass_level2/F_conv2d.cpp
    pass_level2/F_conv3d.cpp
    pass_level2/F_conv_transpose1d.cpp
    pass_level2/F_conv_transpose2d.cpp
    pass_level2/F_conv_transpose3d.cpp
    pass_level2/F_dropout.cpp
    pass_level2/F_dropout23d.cpp
    pass_level2/F_elu.cpp
    pass_level2/F_embedding.cpp
    pass_level2/F_feature_alpha_dropout.cpp
    pass_level2/F_fold.cpp
    pass_level2/F_gelu.cpp
    pass_level2/F_glu.cpp
    pass_level2/F_grid_sample.cpp
    pass_level2/F_group_norm.cpp
    pass_level2/F_hardshrink.cpp
    pass_level2/F_hardsigmoid.cpp
    pass_level2/F_hardswish.cpp
    pass_level2/F_hardtanh.cpp
    pass_level2/F_instance_norm.cpp
    pass_level2/F_interpolate.cpp
    pass_level2/F_layer_norm.cpp
    pass_level2/F_leaky_relu.cpp
    pass_level2/F_linear.cpp
    pass_level2/F_local_response_norm.cpp
    pass_level2/F_log_softmax.cpp
    pass_level2/F_logsigmoid.cpp
    pass_level2/F_lp_pool1d.cpp
    pass_level2/F_lp_pool2d.cpp
    pass_level2/F_max_pool1d.cpp
    pass_level2/F_max_pool2d.cpp
    pass_level2/F_max_pool3d.cpp
    pass_level2/F_mish.cpp
    pass_level2/F_normalize.cpp
    pass_level2/F_pad.cpp
    pass_level2/F_pairwise_distance.cpp
    pass_level2/F_pixel_shuffle.cpp
    pass_level2/F_pixel_unshuffle.cpp
    pass_level2/F_prelu.cpp
    pass_level2/F_relu.cpp
    pass_level2/F_relu6.cpp
    pass_level2/F_rms_norm.cpp
    pass_level2/F_rrelu.cpp
    pass_level2/F_scaled_dot_product_attention.cpp
    pass_level2/F_selu.cpp
    pass_level2/F_sigmoid.cpp
    pass_level2/F_silu.cpp
    pass_level2/F_softmax.cpp
    pass_level2/F_softmin.cpp
    pass_level2/F_softplus.cpp
    pass_level2/F_softshrink.cpp
    pass_level2/F_softsign.cpp
    pass_level2/F_tanh.cpp
    pass_level2/F_tanhshrink.cpp
    pass_level2/F_threshold.cpp
    pass_level2/F_unfold.cpp
    pass_level2/F_upsample_bilinear.cpp
    pass_level2/F_upsample_nearest.cpp
    pass_level2/F_upsample.cpp
    pass_level2/Tensor_copy.cpp
    pass_level2/Tensor_expand.cpp
    pass_level2/Tensor_expand_as.cpp
    pass_level2/Tensor_fill.cpp
    pass_level2/Tensor_index.cpp
    pass_level2/Tensor_index_put.cpp
    pass_level2/Tensor_masked_fill.cpp
    pass_level2/Tensor_new_empty.cpp
    pass_level2/Tensor_new_ones.cpp
    pass_level2/Tensor_new_zeros.cpp
    pass_level2/Tensor_permute.cpp
    pass_level2/Tensor_repeat.cpp
    pass_level2/Tensor_reshape.cpp
    pass_level2/Tensor_reshape_as.cpp
    pass_level2/Tensor_select.cpp
    pass_level2/Tensor_size.cpp
    pass_level2/Tensor_slice.cpp
    pass_level2/Tensor_to.cpp
    pass_level2/Tensor_type_as.cpp
    pass_level2/Tensor_unflatten.cpp
    pass_level2/torch_addmm.cpp
    pass_level2/torch_amax.cpp
    pass_level2/torch_amin.cpp
    pass_level2/torch_arange.cpp
    pass_level2/torch_argmax.cpp
    pass_level2/torch_argmin.cpp
    pass_level2/torch_as_strided.cpp
    pass_level2/torch_baddbmm.cpp
    pass_level2/torch_bmm.cpp
    pass_level2/torch_bitwise_not.cpp
    pass_level2/torch_bitwise_and.cpp
    pass_level2/torch_bitwise_or.cpp
    pass_level2/torch_bitwise_xor.cpp
    pass_level2/torch_bitwise_left_shift.cpp
    pass_level2/torch_bitwise_right_shift.cpp
    pass_level2/torch_cat.cpp
    pass_level2/torch_chunk.cpp
    pass_level2/torch_clamp.cpp
    pass_level2/torch_clone.cpp
    pass_level2/torch_complex.cpp
    pass_level2/torch_cross.cpp
    pass_level2/torch_cumprod.cpp
    pass_level2/torch_cumsum.cpp
    pass_level2/torch_dequantize.cpp
    pass_level2/torch_diag.cpp
    pass_level2/torch_einsum.cpp
    pass_level2/torch_empty.cpp
    pass_level2/torch_empty_like.cpp
    pass_level2/torch_eq.cpp
    pass_level2/torch_flatten.cpp
    pass_level2/torch_flip.cpp
    pass_level2/torch_full.cpp
    pass_level2/torch_full_like.cpp
    pass_level2/torch_gather.cpp
    pass_level2/torch_ge.cpp
    pass_level2/torch_gt.cpp
    pass_level2/torch_imag.cpp
    pass_level2/torch_index_select.cpp
    pass_level2/torch_le.cpp
    pass_level2/torch_lgamma.cpp
    pass_level2/torch_logical_not.cpp
    pass_level2/torch_logical_and.cpp
    pass_level2/torch_logical_or.cpp
    pass_level2/torch_logical_xor.cpp
    pass_level2/torch_logsumexp.cpp
    pass_level2/torch_lt.cpp
    pass_level2/torch_masked_select.cpp
    pass_level2/torch_matmul.cpp
    pass_level2/torch_max.cpp
    pass_level2/torch_mean.cpp
    pass_level2/torch_min.cpp
    pass_level2/torch_mm.cpp
    pass_level2/torch_mv.cpp
    pass_level2/torch_narrow.cpp
    pass_level2/torch_ne.cpp
    pass_level2/torch_norm.cpp
    pass_level2/torch_normal.cpp
    pass_level2/torch_ones.cpp
    pass_level2/torch_ones_like.cpp
    pass_level2/torch_positive.cpp
    pass_level2/torch_prod.cpp
    pass_level2/torch_quantize_per_tensor.cpp
    pass_level2/torch_randn.cpp
    pass_level2/torch_randn_like.cpp
    pass_level2/torch_real.cpp
    pass_level2/torch_repeat_interleave.cpp
    pass_level2/torch_roll.cpp
    pass_level2/torch_scatter_add.cpp
    pass_level2/torch_slice_scatter.cpp
    pass_level2/torch_split.cpp
    pass_level2/torch_squeeze.cpp
    pass_level2/torch_stack.cpp
    pass_level2/torch_std.cpp
    pass_level2/torch_sum.cpp
    pass_level2/torch_t.cpp
    pass_level2/torch_tensor_split.cpp
    pass_level2/torch_tile.cpp
    pass_level2/torch_topk.cpp
    pass_level2/torch_transpose.cpp
    pass_level2/torch_unbind.cpp
    pass_level2/torch_unsqueeze.cpp
    pass_level2/torch_var.cpp
    pass_level2/torch_view_as_complex.cpp
    pass_level2/torch_view_as_real.cpp
    pass_level2/torch_where.cpp
    pass_level2/torch_zeros.cpp
    pass_level2/torch_zeros_like.cpp
    pass_level2/torch_stft.cpp
    pass_level2/torch_istft.cpp
    pass_level2/torch_fft_irfft.cpp
    pass_level2/torch_fft_irfft2.cpp
    pass_level2/torch_fft_irfftn.cpp
    pass_level2/torch_fft_rfft.cpp
    pass_level2/torch_fft_rfft2.cpp
    pass_level2/torch_fft_rfftn.cpp
    pass_level2/torch_fft_ihfft.cpp
    pass_level2/torch_fft_ihfft2.cpp
    pass_level2/torch_fft_ihfftn.cpp
    pass_level2/torch_fft_hfft.cpp
    pass_level2/torch_fft_hfft2.cpp
    pass_level2/torch_fft_hfftn.cpp
    pass_level2/torch_fft_ifft.cpp
    pass_level2/torch_fft_ifft2.cpp
    pass_level2/torch_fft_ifftn.cpp
    pass_level2/torch_fft_fft.cpp
    pass_level2/torch_fft_fft2.cpp
    pass_level2/torch_fft_fftn.cpp

    pass_level2/nn_quantized_FloatFunctional.cpp

    pass_level2/torchaudio_F_inverse_spectrogram.cpp
    pass_level2/torchaudio_F_spectrogram.cpp

    pass_level2/nn_GRU.cpp
    pass_level2/nn_LSTM.cpp
    pass_level2/nn_RNN.cpp
)

set(pnnx_pass_level3_SRCS
    pass_level3/assign_unique_name.cpp
    pass_level3/eliminate_noop_math.cpp
    pass_level3/eliminate_squeeze_unsqueeze_pair.cpp
    pass_level3/eliminate_tuple_pair.cpp
    pass_level3/expand_quantization_modules.cpp
    pass_level3/fuse_opnto1_tensors.cpp
    pass_level3/fuse_op1ton_unpack.cpp
    pass_level3/fuse_dynamic_adaptive_pool.cpp
    pass_level3/fuse_einsum_operands.cpp
    pass_level3/fuse_expression.cpp
    pass_level3/fuse_index_expression.cpp
    pass_level3/fuse_maxpool_unpack.cpp
    pass_level3/fuse_multiheadattention_unpack.cpp
    pass_level3/fuse_rnn_unpack.cpp
    pass_level3/rename_F_dropoutnd.cpp
)

set(pnnx_pass_level4_SRCS
    pass_level4/attribute_pooling.cpp
    pass_level4/canonicalize.cpp
    pass_level4/dead_code_elimination.cpp
    pass_level4/fuse_custom_op.cpp
)

set(pnnx_pass_level5_SRCS
    pass_level5/attribute_unpooling.cpp
    pass_level5/eliminate_dropout.cpp
    pass_level5/eliminate_identity_operator.cpp
    pass_level5/eliminate_maxpool_indices.cpp
    pass_level5/eliminate_noop_cat.cpp
    pass_level5/eliminate_noop_einsum.cpp
    pass_level5/eliminate_noop_expand.cpp
    pass_level5/eliminate_noop_expression.cpp
    pass_level5/eliminate_noop_pad.cpp
    pass_level5/eliminate_noop_upsample.cpp
    pass_level5/eliminate_noop_slice.cpp
    pass_level5/eliminate_noop_reshape.cpp
    pass_level5/eliminate_reshape_shape_expression.cpp
    pass_level5/eliminate_type_as.cpp
    pass_level5/eval_expression.cpp
    pass_level5/fold_constants.cpp
    pass_level5/fuse_adjacent_reshape.cpp
    pass_level5/fuse_channel_shuffle.cpp
    pass_level5/fuse_constant_expression.cpp
    pass_level5/fuse_conv1d_batchnorm1d.cpp
    pass_level5/fuse_conv2d_batchnorm2d.cpp
    pass_level5/fuse_conv3d_batchnorm3d.cpp
    pass_level5/fuse_convtranspose1d_batchnorm1d.cpp
    pass_level5/fuse_convtranspose2d_batchnorm2d.cpp
    pass_level5/fuse_convtranspose3d_batchnorm3d.cpp
    pass_level5/fuse_linear_batchnorm1d.cpp
    pass_level5/fuse_pad_conv1d.cpp
    pass_level5/fuse_pad_conv2d.cpp
    pass_level5/fuse_pixel_shuffle.cpp
    pass_level5/fuse_pixel_unshuffle.cpp
    pass_level5/fuse_layernorm.cpp
    pass_level5/fuse_multiheadattention.cpp
    pass_level5/fuse_multiheadattention_sameqkv.cpp
    pass_level5/fuse_rmsnorm.cpp
    pass_level5/fuse_scaled_dot_product_attention.cpp
    pass_level5/fuse_select_to_unbind.cpp
    pass_level5/fuse_silu.cpp
    pass_level5/fuse_slice_copy.cpp
    pass_level5/fuse_slice_indices.cpp
    pass_level5/fuse_slice_to_tensor_split.cpp
    pass_level5/fuse_slice_squeeze_to_select.cpp
    pass_level5/fuse_static_batchnorm.cpp
    pass_level5/fuse_static_conv.cpp
    pass_level5/fuse_static_convtranspose.cpp
    pass_level5/fuse_static_embedding.cpp
    pass_level5/fuse_static_groupnorm.cpp
    pass_level5/fuse_static_instancenorm.cpp
    pass_level5/fuse_static_layernorm.cpp
    pass_level5/fuse_static_linear.cpp
    pass_level5/fuse_static_prelu.cpp
    pass_level5/fuse_static_rmsnorm.cpp
    pass_level5/fuse_transformers_multiheadattention.cpp
    pass_level5/fuse_transformers_scaled_dot_product_attention.cpp
    pass_level5/normalize_einsum_equation.cpp
    pass_level5/unroll_rnn_op.cpp
)

set(pnnx_pass_ncnn_SRCS
    pass_ncnn/convert_attribute.cpp
    pass_ncnn/convert_custom_op.cpp
    pass_ncnn/convert_module_op.cpp
    pass_ncnn/convert_half_to_float.cpp
    pass_ncnn/convert_input.cpp
    pass_ncnn/convert_reshape_interp_expression.cpp
    pass_ncnn/convert_slice_expression.cpp
    pass_ncnn/convert_torch_cat.cpp
    pass_ncnn/convert_torch_chunk.cpp
    pass_ncnn/convert_torch_einsum.cpp
    pass_ncnn/convert_torch_split.cpp
    pass_ncnn/convert_torch_stack.cpp
    pass_ncnn/convert_torch_tensor_split.cpp
    pass_ncnn/convert_torch_unbind.cpp
    pass_ncnn/convert_Tensor_select.cpp
    pass_ncnn/convert_Tensor_slice.cpp
    pass_ncnn/convert_Tensor_slice_copy.cpp
    pass_ncnn/eliminate_output.cpp
    pass_ncnn/expand_expression.cpp
    pass_ncnn/fuse_convert_shufflechannel_slice.cpp
    pass_ncnn/fuse_convert_rotaryembed.cpp
    pass_ncnn/insert_split.cpp
    pass_ncnn/chain_multi_output.cpp
    pass_ncnn/solve_batch_index.cpp

    pass_ncnn/eliminate_noop.cpp
    pass_ncnn/fuse_convolution_activation.cpp
    pass_ncnn/fuse_convolution1d_activation.cpp
    pass_ncnn/fuse_convolutiondepthwise_activation.cpp
    pass_ncnn/fuse_convolutiondepthwise1d_activation.cpp
    pass_ncnn/fuse_deconvolution_activation.cpp
    pass_ncnn/fuse_deconvolutiondepthwise_activation.cpp
    pass_ncnn/fuse_innerproduct_activation.cpp
    pass_ncnn/fuse_padding_convolution.cpp
    pass_ncnn/fuse_padding_convolutiondepthwise.cpp
    pass_ncnn/fuse_transpose_matmul.cpp
    pass_ncnn/fuse_binaryop_eltwise.cpp
    pass_ncnn/insert_reshape_numpy_binaryop_broadcast.cpp
    pass_ncnn/insert_reshape_linear.cpp
    pass_ncnn/insert_reshape_pooling.cpp
    pass_ncnn/insert_reshape_global_pooling.cpp

    pass_ncnn/F_adaptive_avg_pool1d.cpp
    pass_ncnn/F_adaptive_avg_pool2d.cpp
    pass_ncnn/F_adaptive_avg_pool3d.cpp
    pass_ncnn/F_adaptive_max_pool1d.cpp
    pass_ncnn/F_adaptive_max_pool2d.cpp
    pass_ncnn/F_adaptive_max_pool3d.cpp
    pass_ncnn/F_avg_pool1d.cpp
    pass_ncnn/F_avg_pool2d.cpp
    pass_ncnn/F_avg_pool3d.cpp
    pass_ncnn/F_batch_norm.cpp
    pass_ncnn/F_celu.cpp
    pass_ncnn/F_conv_transpose1d.cpp
    pass_ncnn/F_conv_transpose2d.cpp
    pass_ncnn/F_conv_transpose3d.cpp
    pass_ncnn/F_conv1d.cpp
    pass_ncnn/F_conv2d.cpp
    pass_ncnn/F_conv3d.cpp
    pass_ncnn/F_elu.cpp
    pass_ncnn/F_embedding.cpp
    pass_ncnn/F_fold.cpp
    pass_ncnn/F_gelu.cpp
    pass_ncnn/F_glu.cpp
    pass_ncnn/F_grid_sample.cpp
    pass_ncnn/F_group_norm.cpp
    pass_ncnn/F_hardshrink.cpp
    pass_ncnn/F_hardsigmoid.cpp
    pass_ncnn/F_hardswish.cpp
    pass_ncnn/F_hardtanh.cpp
    pass_ncnn/F_instance_norm.cpp
    pass_ncnn/F_interpolate.cpp
    pass_ncnn/F_layer_norm.cpp
    pass_ncnn/F_leaky_relu.cpp
    pass_ncnn/F_linear.cpp
    pass_ncnn/F_local_response_norm.cpp
    pass_ncnn/F_log_softmax.cpp
    pass_ncnn/F_logsigmoid.cpp
    pass_ncnn/F_max_pool1d.cpp
    pass_ncnn/F_max_pool2d.cpp
    pass_ncnn/F_max_pool3d.cpp
    pass_ncnn/F_mish.cpp
    pass_ncnn/F_normalize.cpp
    pass_ncnn/F_pad.cpp
    pass_ncnn/F_pixel_shuffle.cpp
    pass_ncnn/F_pixel_unshuffle.cpp
    pass_ncnn/F_prelu.cpp
    pass_ncnn/F_relu.cpp
    pass_ncnn/F_relu6.cpp
    pass_ncnn/F_rms_norm.cpp
    pass_ncnn/F_scaled_dot_product_attention.cpp
    pass_ncnn/F_selu.cpp
    pass_ncnn/F_sigmoid.cpp
    pass_ncnn/F_silu.cpp
    pass_ncnn/F_softmax.cpp
    pass_ncnn/F_softplus.cpp
    pass_ncnn/F_softshrink.cpp
    pass_ncnn/F_tanh.cpp
    pass_ncnn/F_unfold.cpp
    pass_ncnn/F_upsample_bilinear.cpp
    pass_ncnn/F_upsample_nearest.cpp
    pass_ncnn/F_upsample.cpp
    pass_ncnn/nn_AdaptiveAvgPool1d.cpp
    pass_ncnn/nn_AdaptiveAvgPool2d.cpp
    pass_ncnn/nn_AdaptiveAvgPool3d.cpp
    pass_ncnn/nn_AdaptiveMaxPool1d.cpp
    pass_ncnn/nn_AdaptiveMaxPool2d.cpp
    pass_ncnn/nn_AdaptiveMaxPool3d.cpp
    pass_ncnn/nn_AvgPool1d.cpp
    pass_ncnn/nn_AvgPool2d.cpp
    pass_ncnn/nn_AvgPool3d.cpp
    pass_ncnn/nn_BatchNorm1d.cpp
    pass_ncnn/nn_BatchNorm2d.cpp
    pass_ncnn/nn_BatchNorm3d.cpp
    pass_ncnn/nn_CELU.cpp
    pass_ncnn/nn_ChannelShuffle.cpp
    pass_ncnn/nn_ConstantPad1d.cpp
    pass_ncnn/nn_ConstantPad2d.cpp
    pass_ncnn/nn_ConstantPad3d.cpp
    pass_ncnn/nn_Conv1d.cpp
    pass_ncnn/nn_Conv2d.cpp
    pass_ncnn/nn_Conv3d.cpp
    pass_ncnn/nn_ConvTranspose1d.cpp
    pass_ncnn/nn_ConvTranspose2d.cpp
    pass_ncnn/nn_ConvTranspose3d.cpp
    pass_ncnn/nn_ELU.cpp
    pass_ncnn/nn_Embedding.cpp
    pass_ncnn/nn_Fold.cpp
    pass_ncnn/nn_GELU.cpp
    pass_ncnn/nn_GLU.cpp
    pass_ncnn/nn_GroupNorm.cpp
    pass_ncnn/nn_GRU.cpp
    pass_ncnn/nn_Hardshrink.cpp
    pass_ncnn/nn_Hardsigmoid.cpp
    pass_ncnn/nn_Hardswish.cpp
    pass_ncnn/nn_Hardtanh.cpp
    pass_ncnn/nn_InstanceNorm2d.cpp
    pass_ncnn/nn_LayerNorm.cpp
    pass_ncnn/nn_LeakyReLU.cpp
    pass_ncnn/nn_Linear.cpp
    pass_ncnn/nn_LocalResponseNorm.cpp
    pass_ncnn/nn_LogSigmoid.cpp
    pass_ncnn/nn_LogSoftmax.cpp
    pass_ncnn/nn_LSTM.cpp
    pass_ncnn/nn_MaxPool1d.cpp
    pass_ncnn/nn_MaxPool2d.cpp
    pass_ncnn/nn_MaxPool3d.cpp
    pass_ncnn/nn_Mish.cpp
    pass_ncnn/nn_MultiheadAttention.cpp
    pass_ncnn/nn_PixelShuffle.cpp
    pass_ncnn/nn_PixelUnshuffle.cpp
    pass_ncnn/nn_PReLU.cpp
    pass_ncnn/nn_ReflectionPad1d.cpp
    pass_ncnn/nn_ReflectionPad2d.cpp
    pass_ncnn/nn_ReLU.cpp
    pass_ncnn/nn_ReLU6.cpp
    pass_ncnn/nn_ReplicationPad1d.cpp
    pass_ncnn/nn_ReplicationPad2d.cpp
    pass_ncnn/nn_ReplicationPad3d.cpp
    pass_ncnn/nn_RMSNorm.cpp
    pass_ncnn/nn_RNN.cpp
    pass_ncnn/nn_SELU.cpp
    pass_ncnn/nn_Sigmoid.cpp
    pass_ncnn/nn_SiLU.cpp
    pass_ncnn/nn_Softmax.cpp
    pass_ncnn/nn_Softmax2d.cpp
    pass_ncnn/nn_Softshrink.cpp
    pass_ncnn/nn_Softplus.cpp
    pass_ncnn/nn_Tanh.cpp
    pass_ncnn/nn_Unfold.cpp
    pass_ncnn/nn_Upsample.cpp
    pass_ncnn/nn_UpsamplingBilinear2d.cpp
    pass_ncnn/nn_UpsamplingNearest2d.cpp
    pass_ncnn/nn_ZeroPad2d.cpp
    pass_ncnn/Tensor_expand.cpp
    pass_ncnn/Tensor_permute.cpp
    pass_ncnn/Tensor_reshape.cpp
    pass_ncnn/Tensor_reshape_as.cpp
    pass_ncnn/Tensor_repeat.cpp
    pass_ncnn/Tensor_unflatten.cpp
    pass_ncnn/torch_addmm.cpp
    pass_ncnn/torch_amax.cpp
    pass_ncnn/torch_amin.cpp
    pass_ncnn/torch_bmm.cpp
    pass_ncnn/torch_clamp.cpp
    pass_ncnn/torch_clone.cpp
    pass_ncnn/torch_cumsum.cpp
    pass_ncnn/torch_diag.cpp
    pass_ncnn/torch_flatten.cpp
    pass_ncnn/torch_flip.cpp
    pass_ncnn/torch_istft.cpp
    pass_ncnn/torch_logsumexp.cpp
    pass_ncnn/torch_matmul.cpp
    pass_ncnn/torch_max.cpp
    pass_ncnn/torch_mean.cpp
    pass_ncnn/torch_min.cpp
    pass_ncnn/torch_mm.cpp
    pass_ncnn/torch_norm.cpp
    pass_ncnn/torch_prod.cpp
    pass_ncnn/torch_roll.cpp
    pass_ncnn/torch_slice_scatter.cpp
    pass_ncnn/torch_squeeze.cpp
    pass_ncnn/torch_sum.cpp
    pass_ncnn/torch_stft.cpp
    pass_ncnn/torch_t.cpp
    pass_ncnn/torch_transpose.cpp
    pass_ncnn/torch_unsqueeze.cpp
    pass_ncnn/torchaudio_F_inverse_spectrogram.cpp
    pass_ncnn/torchaudio_F_spectrogram.cpp
    pass_ncnn/torchvision_DeformConv2d.cpp
)

if(PROTOBUF_FOUND)
    if(DEFINED PROTOBUF_VERSION AND PROTOBUF_VERSION VERSION_GREATER_EQUAL 3.22)
        set(CMAKE_CXX_STANDARD 17)
    endif()

    if(Protobuf_FOUND OR protobuf_MODULE_COMPATIBLE)
        protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx-data.proto onnx-ml.proto onnx-operators-ml.proto)
        add_library(onnxproto STATIC ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
        target_include_directories(onnxproto PUBLIC ${PROTOBUF_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(onnxproto PUBLIC ${PROTOBUF_LIBRARIES})
    else()
        add_library(onnxproto STATIC onnx-data.proto onnx-ml.proto onnx-operators-ml.proto)
        target_include_directories(onnxproto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
        protobuf_generate(TARGET onnxproto)
        target_link_libraries(onnxproto PUBLIC protobuf::libprotobuf)
    endif()

    # use onnxruntime onnx proto if found
    if(onnxruntime_FOUND)
        add_dependencies(onnxruntime::onnxruntime onnxproto)

        if(Protobuf_FOUND OR protobuf_MODULE_COMPATIBLE)
            set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PROTOBUF_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
            set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${PROTOBUF_LIBRARIES})
        else()
            set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})
            set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY INTERFACE_LINK_LIBRARIES protobuf::libprotobuf)
        endif()

        if(APPLE)
            set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY INTERFACE_LINK_LIBRARIES "-framework CoreFoundation")
        endif()
    endif()
endif()

set(torch2pnnx_SRCS

    pass_level0.cpp
    pass_level1.cpp

    ${pnnx_pass_level0_SRCS}
    ${pnnx_pass_level1_SRCS}

    load_torchscript.cpp
)

add_library(torch2pnnx OBJECT ${torch2pnnx_SRCS})
target_compile_definitions(torch2pnnx PRIVATE BUILD_TORCH2PNNX)
target_compile_options(torch2pnnx PUBLIC "${TORCH_CXX_FLAGS}")

if(WIN32)
    target_compile_definitions(torch2pnnx PUBLIC NOMINMAX)
endif()

if(TorchVision_FOUND)
    set_property(SOURCE load_torchscript.cpp APPEND PROPERTY COMPILE_DEFINITIONS PNNX_TORCHVISION)
endif()

if(PROTOBUF_FOUND)
    add_library(pnnx2onnx STATIC
        save_onnx.cpp
    )
    if(onnxruntime_FOUND)
        target_link_libraries(pnnx2onnx PRIVATE onnxruntime::onnxruntime)
    else()
        target_link_libraries(pnnx2onnx PRIVATE onnxproto)
    endif()

    message(STATUS "Building with onnx-zero")
else()
    message(STATUS "Building without onnx-zero")
endif()

if(onnxruntime_FOUND)

    set(pnnx_pass_onnx_SRCS
        pass_onnx/canonicalize.cpp
        pass_onnx/dead_code_elimination.cpp
        pass_onnx/eliminate_initializer_input.cpp
        pass_onnx/eliminate_noop.cpp
        pass_onnx/fold_constants.cpp
        pass_onnx/inline_containers.cpp
        pass_onnx/inline_if_graph.cpp
        pass_onnx/model_stat.cpp
        pass_onnx/shape_inference.cpp
        pass_onnx/fuse_constant_as_attribute.cpp
    )

    set(onnx2pnnx_SRCS
        pass_onnx.cpp
        ${pnnx_pass_onnx_SRCS}
        load_onnx.cpp
    )

    add_library(onnx2pnnx OBJECT ${onnx2pnnx_SRCS})
    target_link_libraries(onnx2pnnx PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(onnx2pnnx PRIVATE BUILD_ONNX2PNNX)

    message(STATUS "Building with onnx2pnnx")
else()
    message(STATUS "Building without onnx2pnnx")
endif()

if(PNNX_TNN2PNNX)
    set(pnnx_pass_tnn_SRCS
        pass_tnn/fuse_shape_size.cpp
        pass_tnn/fuse_shape_list_construct.cpp
        pass_tnn/lower_concat.cpp
        pass_tnn/lower_convolution_activation.cpp
        pass_tnn/lower_power.cpp
    )

    set(tnn2pnnx_SRCS
        ${pnnx_pass_tnn_SRCS}
        load_tnn.cpp
    )

    add_library(tnn2pnnx OBJECT ${tnn2pnnx_SRCS})
    target_compile_definitions(tnn2pnnx PRIVATE BUILD_TNN2PNNX)
    target_compile_options(tnn2pnnx PUBLIC "${TORCH_CXX_FLAGS}")

    message(STATUS "Building with tnn2pnnx")
else()
    message(STATUS "Building without tnn2pnnx")
endif()

if(NOT MSVC)
    add_definitions(-Wall -Wextra)
endif()

set(pnnx_SRCS
    main.cpp
    ir.cpp
    storezip.cpp
    utils.cpp

    pass_level2.cpp
    pass_level3.cpp
    pass_level4.cpp
    pass_level5.cpp

    ${pnnx_pass_level2_SRCS}
    ${pnnx_pass_level3_SRCS}
    ${pnnx_pass_level4_SRCS}
    ${pnnx_pass_level5_SRCS}

    pass_ncnn.cpp
    save_ncnn.cpp
    ${pnnx_pass_ncnn_SRCS}
)

add_executable(pnnx ${pnnx_SRCS})

set_property(SOURCE main.cpp APPEND PROPERTY COMPILE_DEFINITIONS BUILD_TORCH2PNNX)
target_link_libraries(pnnx PRIVATE torch2pnnx)

if(TorchVision_FOUND)
    target_link_libraries(pnnx PRIVATE ${TORCHVISION_LIBRARY})
endif()

if(WIN32)
    target_link_libraries(pnnx PRIVATE ${TORCH_LIBRARIES})
else()
    target_link_libraries(pnnx PRIVATE ${TORCH_LIBRARIES} pthread dl)
endif()

if(PROTOBUF_FOUND)
    set_property(SOURCE main.cpp APPEND PROPERTY COMPILE_DEFINITIONS BUILD_PNNX2ONNX)
    target_link_libraries(pnnx PRIVATE pnnx2onnx)
endif()

if(onnxruntime_FOUND)
    set_property(SOURCE main.cpp APPEND PROPERTY COMPILE_DEFINITIONS BUILD_ONNX2PNNX)
    target_link_libraries(pnnx PRIVATE onnx2pnnx)
endif()

if(PNNX_TNN2PNNX)
    set_property(SOURCE main.cpp APPEND PROPERTY COMPILE_DEFINITIONS BUILD_TNN2PNNX)
    target_link_libraries(pnnx PRIVATE tnn2pnnx)
endif()

if(PNNX_COVERAGE)
    target_compile_options(pnnx PUBLIC -coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(pnnx PUBLIC -coverage -lgcov)
endif()

# set_target_properties(pnnx PROPERTIES COMPILE_FLAGS -fsanitize=address)
# set_target_properties(pnnx PROPERTIES LINK_FLAGS -fsanitize=address)

if(APPLE)
    set_target_properties(pnnx PROPERTIES INSTALL_RPATH "@executable_path/")
else()
    set_target_properties(pnnx PROPERTIES INSTALL_RPATH "$ORIGIN/")
endif()
set_target_properties(pnnx PROPERTIES MACOSX_RPATH TRUE)

include(GNUInstallDirs)
install(TARGETS pnnx RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (WIN32)
    file(GLOB TORCH_DLL "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    install(FILES ${TORCH_DLL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
