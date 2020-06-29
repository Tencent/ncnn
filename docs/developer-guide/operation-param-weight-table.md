
|operation|param id|param phase|default value|weight order|
|:---:|:---:|:---:|:---:|:---:|
|AbsVal|||
|ArgMax TODO|||
|BatchNorm|0|channels|0|slope mean variance bias|
||1|eps|0.f|
|Bias|0|bias_data_size|0|
|BinaryOp|0|op_type|0|
||1|with_scalar|0|
||2|b|0.f|
|BNLL|||
|Cast|0|type_from|0|
||1|type_to|0|
|Clip|0|min|-FLT_MAX|
||1|max|FLT_MAX|
|Concat|0|axis|0|
|Convolution|0|num_output|0|weight bias|
||1|kernel_w|0|
||2|dilation_w|1|
||3|stride_w|1|
||4|pad_left|0|
||5|bias_term|0|
||6|weight_data_size|0|
||8|int8_scale_term|0|
||9|activation_type|0|
||10|activation_params|[ ]|
||11|kernel_h|kernel_w|
||12|dilation_h|dilation_w|
||13|stride_h|stride_w|
||15|pad_right|pad_left|
||14|pad_top|pad_left|
||16|pad_bottom|pad_top|
||17|impl_type|0|
||18|pad_value|0.f|
|ConvolutionDepthWise|0|num_output|0|weight bias|
||1|kernel_w|0|
||2|dilation_w|1|
||3|stride_w|1|
||4|pad_left|0|
||5|bias_term|0|
||6|weight_data_size|0|
||7|group|1|
||8|int8_scale_term|0|
||9|activation_type|0|
||10|activation_params|[ ]|
||11|kernel_h|kernel_w|
||12|dilation_h|dilation_w|
||13|stride_h|stride_w|
||15|pad_right|pad_left|
||14|pad_top|pad_left|
||16|pad_bottom|pad_top|
||18|pad_value|0.f|
|Crop|0|woffset|0|
||1|hoffset|0|
||2|coffset|0|
||3|outw|0|
||4|outh|0|
||5|outc|0|
||6|woffset2|0|
||7|hoffset2|0|
||8|coffset2|0|
||9|starts|[ ]|
||10|ends|[ ]|
||11|axes|[ ]|
|Deconvolution|0|num_output|0|weight bias|
||1|kernel_w|0|
||2|dilation_w|1|
||3|stride_w|1|
||4|pad_left|0|
||5|bias_term|0|
||6|weight_data_size|0|
||9|activation_type|0|
||10|activation_params|[ ]|
||11|kernel_h|kernel_w|
||12|dilation_h|dilation_w|
||13|stride_h|stride_w|
||15|pad_right|pad_left|
||14|pad_top|pad_left|
||16|pad_bottom|pad_top|
||18|output_pad_right|0|
||19|output_pad_bottom|output_pad_right|
||20|output_w|0|
||21|output_h|output_w|
|DeconvolutionDepthWise|0|num_output|0|weight bias|
||1|kernel_w|0|
||2|dilation_w|1|
||3|stride_w|1|
||4|pad_left|0|
||5|bias_term|0|
||6|weight_data_size|0|
||7|group|1|
||9|activation_type|0|
||10|activation_params|[ ]|
||11|kernel_h|kernel_w|
||12|dilation_h|dilation_w|
||13|stride_h|stride_w|
||15|pad_right|pad_left|
||14|pad_top|pad_left|
||16|pad_bottom|pad_top|
||18|output_pad_right|0|
||19|output_pad_bottom|output_pad_right|
||20|output_w|0|
||21|output_h|output_w|
|Dequantize|0|scale|1.f|bias|
||1|bias_term|0|
||2|bias_data_size|0|
|DetectionOutput|0|num_class|0|
||1|nms_threshold|0.05f|
||2|nms_top_k|300|
||3|keep_top_k|100|
||4|confidence_threshold|0.5f|
||5|variances[0]|0.1f|
||6|variances[1]|0.1f|
||7|variances[2]|0.2f|
||8|variances[3]|0.2f|
|Dropout|0|scale|1.f|
|Eltwise|0|op_type|0|
||1|coeffs|[ ]|
|ELU|0|alpha|0.1f|
|Embed|0|num_output|0|weight bias|
||1|input_dim|0|
||2|bias_term|0|
||3|weight_data_size|0|
|Exp|0|base|-1.f|
||1|scale|1.f|
||2|shift|0.f|
|ExpandDims|0|expand_w|0|
||1|expand_h|0|
||2|expand_c|0|
||3|axes|[ ]|
|Flatten|||
|HardSigmoid|0|alpha|0.2f||
||1|beta|0.5f|
|HardSwish|0|alpha|0.2f||
||1|beta|0.5f|
|InnerProduct|0|num_output|0|weight bias|
||1|bias_term|0|
||2|weight_data_size|0|
||8|int8_scale_term|0|
||9|activation_type|0|
||10|activation_params|[ ]|
|Input|0|w|0|
||1|h|0|
||2|c|0|
|InstanceNorm|0|channels|0|gamma bias|
||1|eps|0.001f|
|Interp|0|resize_type|0|
||1|height_scale|1.f|
||2|width_scale|1.f|
||3|output_height|0|
||4|output_width|0|
|Log|0|base|-1.f|
||1|scale|1.f|
||2|shift|0.f|
|LRN|0|region_type|0|
||1|local_size|5|
||2|alpha|1.f|
||3|beta|0.75f|
||4|bias|1.f|
|LSTM|0|num_output|0|
||1|weight_data_size|1|
||2|direction|0|
|MemoryData|0|w|0|
||1|h|0|
||2|c|0|
|Mish|||
|MVN|0|normalize_variance|0|
||1|across_channels|0|
||2|eps|0.0001f|
|Noop|||
|Normalize|0|across_spatial|0|scale|
||4|across_channel|0|
||1|channel_shared|0|
||2|eps|0.0001f|
||9|eps_mode|0|
||3|scale_data_size|0|
|Packing|0|out_packing|1|
||1|use_padding|0|
||2|cast_type_from|0|
||3|cast_type_to|0|
||4|storage_type_from|0|
||5|storage_type_to|0|
|Padding|0|top|0|per_channel_pad_data|
||1|bottom|0|
||2|left|0|
||3|right|0|
||4|type|0|
||5|value|0.f|
||6|per_channel_pad_data_size|0|
||7|front|0|
||8|behind|0|
|Permute|0|order_type|0|
|PixelShuffle|0|upscale_factor|1|
|Pooling|0|pooling_type|0|
||1|kernel_w|0|
||11|kernel_h|kernel_w|
||2|stride_w|1|
||12|stride_h|stride_w|
||3|pad_left|0|
||14|pad_right|pad_left|
||13|pad_top|pad_left|
||15|pad_bottom|pad_top|
||4|global_pooling|0|
||5|pad_mode|0|
|Power|0|power|1.f|
||1|scale|1.f|
||2|shift|0.f|
|PReLU|0|num_slope|0|slope|
|PriorBox|0|min_sizes|[ ]|
||1|max_sizes|[ ]|
||2|aspect_ratios|[ ]|
||3|varainces[0]|0.f|
||4|varainces[1]|0.f|
||5|varainces[2]|0.f|
||6|varainces[3]|0.f|
||7|flip|1|
||8|clip|0|
||9|image_width|0|
||10|image_height|0|
||11|step_width|-233.f|
||12|step_height|-233.f|
||13|offset|0.f|
||14|step_mmdetection|0|
||15|center_mmdetection|0|
|Proposal|0|feat_stride|16|
||1|base_size|16|
||2|pre_nms_topN|6000|
||3|after_nms_topN|300|
||4|num_thresh|0.7f|
||5|min_size|16|
|PSROIPooling|0|pooled_width|7|
||1|pooled_height|7|
||2|spatial_scale|0.0625f|
||3|output_dim|0|
|Quantize|0|scale|1.f|
|Reduction|0|operation|0|
||1|dim|0|
||2|coeff|1.f|
||3|axes|[ ]|
||4|keepdims|0|
|ReLU|0|slope|0.f|
|Reorg|0|stride|0|
|Requantize|0|scale_in|1.f|bias|
||1|scale_out|1.f|
||2|bias_term|0|
||3|bias_data_size|0|
||4|fusion_relu|0|
|Reshape|0|w|-233|
||1|h|-233|
||2|c|-233|
||3|permute|0|
|ROIAlign|0|pooled_width|0|
||1|pooled_height|0|
||2|spatial_scale|1.f|
||3|sampling_ratio|0|
||4|aligned|0|
||5|version|0|
|ROIPooling|0|pooled_width|0|
||1|pooled_height|0|
||2|spatial_scale|1.f|
|Scale|0|scale_data_size|0|scale bias|
||1|bias_term|0|
|SELU|0|alpha|1.67326324f||
||1|lambda|1.050700987f|
|ShuffleChannel|0|group|1|
|Sigmoid|||
|Slice|0|slices|[ ]|
||1|axis|0|
|Softmax|0|axis|0|
|Split|||
|SPP TODO|||
|Squeeze|0|squeeze_w|0|
||1|squeeze_h|0|
||2|squeeze_c|0|
||3|axes|[ ]|
|StatisticsPooling|0|include_stddev|0|
|Swish|||
|TanH|||
|Threshold|0|threshold|0.f|
|Tile TODO|||
|UnaryOp|0|op_type|0|
|YoloDetectionOutput|0|num_class|20|
||1|num_box|5|
||2|confidence_threshold|0.01f|
||3|num_threshold|0.45f|
||4|biases|[]|
|Yolov3DetectionOutput|0|num_class|20|
||1|num_box|5|
||2|confidence_threshold|0.01f|
||3|num_threshold|0.45f|
||4|biases|[]|
||5|mask|[]|
||6|anchors_scale|[]|
|RNN TODO|||
