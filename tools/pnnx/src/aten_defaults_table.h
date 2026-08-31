// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// aten 参数默认值静态表(离线生成,勿手改)。
//
// torch.export 会把等于默认值的实参从图里省略(cat 的 dim=0、flatten 的
// end_dim=-1、conv2d 的 dilation/groups 等);pt2 builder 据本表把省略的
// 实参补全为完整 schema 形态,使 pt2 图与 torchscript 图同构、下游
// pass_level2 形态分支(torch_cat / torch_flatten / F_conv2d_1 ...)零改动复用。
//
// 再生成:python scripts/dump_aten_defaults.py --scan tests/ncnn ../../../pt2-dump --ops aten::add.Tensor aten::batch_norm aten::cat aten::clamp aten::conv2d aten::embedding aten::flatten.using_ints aten::gelu aten::hardsigmoid aten::hardswish aten::layer_norm aten::leaky_relu aten::linear aten::max aten::mean aten::min aten::mish aten::pad aten::permute aten::relu aten::reshape aten::select.int aten::sigmoid aten::silu aten::slice.Tensor aten::softmax.int aten::squeeze aten::stack aten::sum aten::tanh aten::unsqueeze aten::view
// 来源:torch 2.13.0+cpu 的 torch._C._jit_get_all_schemas()(4362 个 schema)
// 生成时间:2026-08-31 18:50
// 收录算子:180 个
//
// 值编码(type 标签 + 字符串值):
//   NO_DEFAULT=-1  无默认值的必填参数(占位,保证形参顺序)
//   NONE=0         ""
//   INT=1          十进制整数
//   FLOAT=2        strtod 可解析(含 inf/-inf/nan)
//   BOOL=3         "0"/"1"
//   STRING=4       原文
//   INTS/FLOATS/STRINGS=5/6/7  逗号分隔平铺;值为 "" 表示空列表,builder 转
//                  type 0(None)—— ts 侧空列表实参物化为 None 常量(如
//                  max_pool2d 的 stride=()),下游转换器按 type 0 解释
//   DEVICE=8       ""=None,否则 "cpu"/"cuda:0" 形态(builder 转 STRING)
//   UNSUPPORTED=9  bool 列表/嵌套列表/Tensor 等 builder 无法表达的默认值,
//                  不参与补全(生成时告警留痕)
//
// 限制:覆盖随测试语料增长按需重跑扩充;表未收录的算子 builder 保持
// torch.export 原样转写(缺参不补,stderr 告警)。

#ifndef PNNX_ATEN_DEFAULTS_TABLE_H
#define PNNX_ATEN_DEFAULTS_TABLE_H

#include <stddef.h>
#include <string.h>

namespace pnnx {

enum Pt2DefaultType
{
    PT2_D_NO_DEFAULT = -1,
    PT2_D_NONE = 0,
    PT2_D_INT = 1,
    PT2_D_FLOAT = 2,
    PT2_D_BOOL = 3,
    PT2_D_STRING = 4,
    PT2_D_INTS = 5,
    PT2_D_FLOATS = 6,
    PT2_D_STRINGS = 7,
    PT2_D_DEVICE = 8,
    PT2_D_UNSUPPORTED = 9
};

struct Pt2ArgDefault
{
    const char* name;
    Pt2DefaultType type;
    const char* value;
};

struct Pt2DefaultsEntry
{
    const char* op; // 全名含 overload,如 "aten::conv2d.default"
    const Pt2ArgDefault* args;
    size_t arg_count;
};

// 按 pt2 target 全名(如 "aten::flatten.using_ints")查参数默认值表。
// 未收录返回 0。
inline const Pt2DefaultsEntry* find_pt2_aten_defaults(const char* op)
{
    static const Pt2ArgDefault args_aten__weight_norm_default[] = {
        {"v", PT2_D_NO_DEFAULT, ""},
        {"g", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten__weight_norm_default = {"aten::_weight_norm.default", args_aten__weight_norm_default, 3};
    if (strcmp(op, entry_aten__weight_norm_default.op) == 0) return &entry_aten__weight_norm_default;

    static const Pt2ArgDefault args_aten_abs_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_abs_default = {"aten::abs.default", args_aten_abs_default, 1};
    if (strcmp(op, entry_aten_abs_default.op) == 0) return &entry_aten_abs_default;

    static const Pt2ArgDefault args_aten_acos_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_acos_default = {"aten::acos.default", args_aten_acos_default, 1};
    if (strcmp(op, entry_aten_acos_default.op) == 0) return &entry_aten_acos_default;

    static const Pt2ArgDefault args_aten_acosh_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_acosh_default = {"aten::acosh.default", args_aten_acosh_default, 1};
    if (strcmp(op, entry_aten_acosh_default.op) == 0) return &entry_aten_acosh_default;

    static const Pt2ArgDefault args_aten_adaptive_avg_pool1d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_adaptive_avg_pool1d_default = {"aten::adaptive_avg_pool1d.default", args_aten_adaptive_avg_pool1d_default, 2};
    if (strcmp(op, entry_aten_adaptive_avg_pool1d_default.op) == 0) return &entry_aten_adaptive_avg_pool1d_default;

    static const Pt2ArgDefault args_aten_adaptive_avg_pool2d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_adaptive_avg_pool2d_default = {"aten::adaptive_avg_pool2d.default", args_aten_adaptive_avg_pool2d_default, 2};
    if (strcmp(op, entry_aten_adaptive_avg_pool2d_default.op) == 0) return &entry_aten_adaptive_avg_pool2d_default;

    static const Pt2ArgDefault args_aten_adaptive_avg_pool3d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_adaptive_avg_pool3d_default = {"aten::adaptive_avg_pool3d.default", args_aten_adaptive_avg_pool3d_default, 2};
    if (strcmp(op, entry_aten_adaptive_avg_pool3d_default.op) == 0) return &entry_aten_adaptive_avg_pool3d_default;

    static const Pt2ArgDefault args_aten_adaptive_max_pool1d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_adaptive_max_pool1d_default = {"aten::adaptive_max_pool1d.default", args_aten_adaptive_max_pool1d_default, 2};
    if (strcmp(op, entry_aten_adaptive_max_pool1d_default.op) == 0) return &entry_aten_adaptive_max_pool1d_default;

    static const Pt2ArgDefault args_aten_adaptive_max_pool2d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_adaptive_max_pool2d_default = {"aten::adaptive_max_pool2d.default", args_aten_adaptive_max_pool2d_default, 2};
    if (strcmp(op, entry_aten_adaptive_max_pool2d_default.op) == 0) return &entry_aten_adaptive_max_pool2d_default;

    static const Pt2ArgDefault args_aten_adaptive_max_pool3d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_adaptive_max_pool3d_default = {"aten::adaptive_max_pool3d.default", args_aten_adaptive_max_pool3d_default, 2};
    if (strcmp(op, entry_aten_adaptive_max_pool3d_default.op) == 0) return &entry_aten_adaptive_max_pool3d_default;

    static const Pt2ArgDefault args_aten_add_Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
        {"alpha", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_add_Tensor = {"aten::add.Tensor", args_aten_add_Tensor, 3};
    if (strcmp(op, entry_aten_add_Tensor.op) == 0) return &entry_aten_add_Tensor;

    static const Pt2ArgDefault args_aten_add__Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
        {"alpha", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_add__Tensor = {"aten::add_.Tensor", args_aten_add__Tensor, 3};
    if (strcmp(op, entry_aten_add__Tensor.op) == 0) return &entry_aten_add__Tensor;

    static const Pt2ArgDefault args_aten_addmm_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"mat1", PT2_D_NO_DEFAULT, ""},
        {"mat2", PT2_D_NO_DEFAULT, ""},
        {"beta", PT2_D_INT, "1"},
        {"alpha", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_addmm_default = {"aten::addmm.default", args_aten_addmm_default, 5};
    if (strcmp(op, entry_aten_addmm_default.op) == 0) return &entry_aten_addmm_default;

    static const Pt2ArgDefault args_aten_alias_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_alias_default = {"aten::alias.default", args_aten_alias_default, 1};
    if (strcmp(op, entry_aten_alias_default.op) == 0) return &entry_aten_alias_default;

    static const Pt2ArgDefault args_aten_alpha_dropout_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"p", PT2_D_NO_DEFAULT, ""},
        {"train", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_alpha_dropout_default = {"aten::alpha_dropout.default", args_aten_alpha_dropout_default, 3};
    if (strcmp(op, entry_aten_alpha_dropout_default.op) == 0) return &entry_aten_alpha_dropout_default;

    static const Pt2ArgDefault args_aten_amax_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INTS, ""},
        {"keepdim", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_amax_default = {"aten::amax.default", args_aten_amax_default, 3};
    if (strcmp(op, entry_aten_amax_default.op) == 0) return &entry_aten_amax_default;

    static const Pt2ArgDefault args_aten_amin_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INTS, ""},
        {"keepdim", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_amin_default = {"aten::amin.default", args_aten_amin_default, 3};
    if (strcmp(op, entry_aten_amin_default.op) == 0) return &entry_aten_amin_default;

    static const Pt2ArgDefault args_aten_asin_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_asin_default = {"aten::asin.default", args_aten_asin_default, 1};
    if (strcmp(op, entry_aten_asin_default.op) == 0) return &entry_aten_asin_default;

    static const Pt2ArgDefault args_aten_asinh_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_asinh_default = {"aten::asinh.default", args_aten_asinh_default, 1};
    if (strcmp(op, entry_aten_asinh_default.op) == 0) return &entry_aten_asinh_default;

    static const Pt2ArgDefault args_aten_atan_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_atan_default = {"aten::atan.default", args_aten_atan_default, 1};
    if (strcmp(op, entry_aten_atan_default.op) == 0) return &entry_aten_atan_default;

    static const Pt2ArgDefault args_aten_atan2_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_atan2_default = {"aten::atan2.default", args_aten_atan2_default, 2};
    if (strcmp(op, entry_aten_atan2_default.op) == 0) return &entry_aten_atan2_default;

    static const Pt2ArgDefault args_aten_atanh_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_atanh_default = {"aten::atanh.default", args_aten_atanh_default, 1};
    if (strcmp(op, entry_aten_atanh_default.op) == 0) return &entry_aten_atanh_default;

    static const Pt2ArgDefault args_aten_avg_pool1d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0"},
        {"ceil_mode", PT2_D_BOOL, "0"},
        {"count_include_pad", PT2_D_BOOL, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_avg_pool1d_default = {"aten::avg_pool1d.default", args_aten_avg_pool1d_default, 6};
    if (strcmp(op, entry_aten_avg_pool1d_default.op) == 0) return &entry_aten_avg_pool1d_default;

    static const Pt2ArgDefault args_aten_avg_pool2d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0,0"},
        {"ceil_mode", PT2_D_BOOL, "0"},
        {"count_include_pad", PT2_D_BOOL, "1"},
        {"divisor_override", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_avg_pool2d_default = {"aten::avg_pool2d.default", args_aten_avg_pool2d_default, 7};
    if (strcmp(op, entry_aten_avg_pool2d_default.op) == 0) return &entry_aten_avg_pool2d_default;

    static const Pt2ArgDefault args_aten_avg_pool3d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0,0,0"},
        {"ceil_mode", PT2_D_BOOL, "0"},
        {"count_include_pad", PT2_D_BOOL, "1"},
        {"divisor_override", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_avg_pool3d_default = {"aten::avg_pool3d.default", args_aten_avg_pool3d_default, 7};
    if (strcmp(op, entry_aten_avg_pool3d_default.op) == 0) return &entry_aten_avg_pool3d_default;

    static const Pt2ArgDefault args_aten_baddbmm_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"batch1", PT2_D_NO_DEFAULT, ""},
        {"batch2", PT2_D_NO_DEFAULT, ""},
        {"beta", PT2_D_INT, "1"},
        {"alpha", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_baddbmm_default = {"aten::baddbmm.default", args_aten_baddbmm_default, 5};
    if (strcmp(op, entry_aten_baddbmm_default.op) == 0) return &entry_aten_baddbmm_default;

    static const Pt2ArgDefault args_aten_batch_norm_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NO_DEFAULT, ""},
        {"running_mean", PT2_D_NO_DEFAULT, ""},
        {"running_var", PT2_D_NO_DEFAULT, ""},
        {"training", PT2_D_NO_DEFAULT, ""},
        {"momentum", PT2_D_NO_DEFAULT, ""},
        {"eps", PT2_D_NO_DEFAULT, ""},
        {"cudnn_enabled", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_batch_norm_default = {"aten::batch_norm.default", args_aten_batch_norm_default, 9};
    if (strcmp(op, entry_aten_batch_norm_default.op) == 0) return &entry_aten_batch_norm_default;

    static const Pt2ArgDefault args_aten_bmm_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"mat2", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_bmm_default = {"aten::bmm.default", args_aten_bmm_default, 2};
    if (strcmp(op, entry_aten_bmm_default.op) == 0) return &entry_aten_bmm_default;

    static const Pt2ArgDefault args_aten_cat_default[] = {
        {"tensors", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_cat_default = {"aten::cat.default", args_aten_cat_default, 2};
    if (strcmp(op, entry_aten_cat_default.op) == 0) return &entry_aten_cat_default;

    static const Pt2ArgDefault args_aten_ceil_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_ceil_default = {"aten::ceil.default", args_aten_ceil_default, 1};
    if (strcmp(op, entry_aten_ceil_default.op) == 0) return &entry_aten_ceil_default;

    static const Pt2ArgDefault args_aten_celu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"alpha", PT2_D_FLOAT, "1.0"},
    };
    static const Pt2DefaultsEntry entry_aten_celu_default = {"aten::celu.default", args_aten_celu_default, 2};
    if (strcmp(op, entry_aten_celu_default.op) == 0) return &entry_aten_celu_default;

    static const Pt2ArgDefault args_aten_channel_shuffle_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"groups", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_channel_shuffle_default = {"aten::channel_shuffle.default", args_aten_channel_shuffle_default, 2};
    if (strcmp(op, entry_aten_channel_shuffle_default.op) == 0) return &entry_aten_channel_shuffle_default;

    static const Pt2ArgDefault args_aten_chunk_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"chunks", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_chunk_default = {"aten::chunk.default", args_aten_chunk_default, 3};
    if (strcmp(op, entry_aten_chunk_default.op) == 0) return &entry_aten_chunk_default;

    static const Pt2ArgDefault args_aten_clamp_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"min", PT2_D_NONE, ""},
        {"max", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_clamp_default = {"aten::clamp.default", args_aten_clamp_default, 3};
    if (strcmp(op, entry_aten_clamp_default.op) == 0) return &entry_aten_clamp_default;

    static const Pt2ArgDefault args_aten_clamp__default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"min", PT2_D_NONE, ""},
        {"max", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_clamp__default = {"aten::clamp_.default", args_aten_clamp__default, 3};
    if (strcmp(op, entry_aten_clamp__default.op) == 0) return &entry_aten_clamp__default;

    static const Pt2ArgDefault args_aten_clamp_min_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"min", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_clamp_min_default = {"aten::clamp_min.default", args_aten_clamp_min_default, 2};
    if (strcmp(op, entry_aten_clamp_min_default.op) == 0) return &entry_aten_clamp_min_default;

    static const Pt2ArgDefault args_aten_clone_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"memory_format", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_clone_default = {"aten::clone.default", args_aten_clone_default, 2};
    if (strcmp(op, entry_aten_clone_default.op) == 0) return &entry_aten_clone_default;

    static const Pt2ArgDefault args_aten_col2im_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"dilation", PT2_D_NO_DEFAULT, ""},
        {"padding", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_col2im_default = {"aten::col2im.default", args_aten_col2im_default, 6};
    if (strcmp(op, entry_aten_col2im_default.op) == 0) return &entry_aten_col2im_default;

    static const Pt2ArgDefault args_aten_contiguous_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"memory_format", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_contiguous_default = {"aten::contiguous.default", args_aten_contiguous_default, 2};
    if (strcmp(op, entry_aten_contiguous_default.op) == 0) return &entry_aten_contiguous_default;

    static const Pt2ArgDefault args_aten_conv1d_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1"},
        {"padding", PT2_D_INTS, "0"},
        {"dilation", PT2_D_INTS, "1"},
        {"groups", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv1d_default = {"aten::conv1d.default", args_aten_conv1d_default, 7};
    if (strcmp(op, entry_aten_conv1d_default.op) == 0) return &entry_aten_conv1d_default;

    static const Pt2ArgDefault args_aten_conv1d_padding[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1"},
        {"padding", PT2_D_STRING, "valid"},
        {"dilation", PT2_D_INTS, "1"},
        {"groups", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv1d_padding = {"aten::conv1d.padding", args_aten_conv1d_padding, 7};
    if (strcmp(op, entry_aten_conv1d_padding.op) == 0) return &entry_aten_conv1d_padding;

    static const Pt2ArgDefault args_aten_conv2d_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1,1"},
        {"padding", PT2_D_INTS, "0,0"},
        {"dilation", PT2_D_INTS, "1,1"},
        {"groups", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv2d_default = {"aten::conv2d.default", args_aten_conv2d_default, 7};
    if (strcmp(op, entry_aten_conv2d_default.op) == 0) return &entry_aten_conv2d_default;

    static const Pt2ArgDefault args_aten_conv2d_padding[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1,1"},
        {"padding", PT2_D_STRING, "valid"},
        {"dilation", PT2_D_INTS, "1,1"},
        {"groups", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv2d_padding = {"aten::conv2d.padding", args_aten_conv2d_padding, 7};
    if (strcmp(op, entry_aten_conv2d_padding.op) == 0) return &entry_aten_conv2d_padding;

    static const Pt2ArgDefault args_aten_conv3d_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1,1,1"},
        {"padding", PT2_D_INTS, "0,0,0"},
        {"dilation", PT2_D_INTS, "1,1,1"},
        {"groups", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv3d_default = {"aten::conv3d.default", args_aten_conv3d_default, 7};
    if (strcmp(op, entry_aten_conv3d_default.op) == 0) return &entry_aten_conv3d_default;

    static const Pt2ArgDefault args_aten_conv3d_padding[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1,1,1"},
        {"padding", PT2_D_STRING, "valid"},
        {"dilation", PT2_D_INTS, "1,1,1"},
        {"groups", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv3d_padding = {"aten::conv3d.padding", args_aten_conv3d_padding, 7};
    if (strcmp(op, entry_aten_conv3d_padding.op) == 0) return &entry_aten_conv3d_padding;

    static const Pt2ArgDefault args_aten_conv_transpose1d_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1"},
        {"padding", PT2_D_INTS, "0"},
        {"output_padding", PT2_D_INTS, "0"},
        {"groups", PT2_D_INT, "1"},
        {"dilation", PT2_D_INTS, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv_transpose1d_default = {"aten::conv_transpose1d.default", args_aten_conv_transpose1d_default, 8};
    if (strcmp(op, entry_aten_conv_transpose1d_default.op) == 0) return &entry_aten_conv_transpose1d_default;

    static const Pt2ArgDefault args_aten_conv_transpose2d_input[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1,1"},
        {"padding", PT2_D_INTS, "0,0"},
        {"output_padding", PT2_D_INTS, "0,0"},
        {"groups", PT2_D_INT, "1"},
        {"dilation", PT2_D_INTS, "1,1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv_transpose2d_input = {"aten::conv_transpose2d.input", args_aten_conv_transpose2d_input, 8};
    if (strcmp(op, entry_aten_conv_transpose2d_input.op) == 0) return &entry_aten_conv_transpose2d_input;

    static const Pt2ArgDefault args_aten_conv_transpose3d_input[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
        {"stride", PT2_D_INTS, "1,1,1"},
        {"padding", PT2_D_INTS, "0,0,0"},
        {"output_padding", PT2_D_INTS, "0,0,0"},
        {"groups", PT2_D_INT, "1"},
        {"dilation", PT2_D_INTS, "1,1,1"},
    };
    static const Pt2DefaultsEntry entry_aten_conv_transpose3d_input = {"aten::conv_transpose3d.input", args_aten_conv_transpose3d_input, 8};
    if (strcmp(op, entry_aten_conv_transpose3d_input.op) == 0) return &entry_aten_conv_transpose3d_input;

    static const Pt2ArgDefault args_aten_copy__default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"src", PT2_D_NO_DEFAULT, ""},
        {"non_blocking", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_copy__default = {"aten::copy_.default", args_aten_copy__default, 3};
    if (strcmp(op, entry_aten_copy__default.op) == 0) return &entry_aten_copy__default;

    static const Pt2ArgDefault args_aten_cos_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_cos_default = {"aten::cos.default", args_aten_cos_default, 1};
    if (strcmp(op, entry_aten_cos_default.op) == 0) return &entry_aten_cos_default;

    static const Pt2ArgDefault args_aten_cosh_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_cosh_default = {"aten::cosh.default", args_aten_cosh_default, 1};
    if (strcmp(op, entry_aten_cosh_default.op) == 0) return &entry_aten_cosh_default;

    static const Pt2ArgDefault args_aten_cumsum_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_cumsum_default = {"aten::cumsum.default", args_aten_cumsum_default, 3};
    if (strcmp(op, entry_aten_cumsum_default.op) == 0) return &entry_aten_cumsum_default;

    static const Pt2ArgDefault args_aten_diag_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"diagonal", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_diag_default = {"aten::diag.default", args_aten_diag_default, 2};
    if (strcmp(op, entry_aten_diag_default.op) == 0) return &entry_aten_diag_default;

    static const Pt2ArgDefault args_aten_div_Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_div_Tensor = {"aten::div.Tensor", args_aten_div_Tensor, 2};
    if (strcmp(op, entry_aten_div_Tensor.op) == 0) return &entry_aten_div_Tensor;

    static const Pt2ArgDefault args_aten_dropout_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"p", PT2_D_NO_DEFAULT, ""},
        {"train", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_dropout_default = {"aten::dropout.default", args_aten_dropout_default, 3};
    if (strcmp(op, entry_aten_dropout_default.op) == 0) return &entry_aten_dropout_default;

    static const Pt2ArgDefault args_aten_einsum_default[] = {
        {"equation", PT2_D_NO_DEFAULT, ""},
        {"tensors", PT2_D_NO_DEFAULT, ""},
        {"path", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_einsum_default = {"aten::einsum.default", args_aten_einsum_default, 3};
    if (strcmp(op, entry_aten_einsum_default.op) == 0) return &entry_aten_einsum_default;

    static const Pt2ArgDefault args_aten_elu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"alpha", PT2_D_INT, "1"},
        {"scale", PT2_D_INT, "1"},
        {"input_scale", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_elu_default = {"aten::elu.default", args_aten_elu_default, 4};
    if (strcmp(op, entry_aten_elu_default.op) == 0) return &entry_aten_elu_default;

    static const Pt2ArgDefault args_aten_embedding_default[] = {
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"indices", PT2_D_NO_DEFAULT, ""},
        {"padding_idx", PT2_D_INT, "-1"},
        {"scale_grad_by_freq", PT2_D_BOOL, "0"},
        {"sparse", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_embedding_default = {"aten::embedding.default", args_aten_embedding_default, 5};
    if (strcmp(op, entry_aten_embedding_default.op) == 0) return &entry_aten_embedding_default;

    static const Pt2ArgDefault args_aten_erf_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_erf_default = {"aten::erf.default", args_aten_erf_default, 1};
    if (strcmp(op, entry_aten_erf_default.op) == 0) return &entry_aten_erf_default;

    static const Pt2ArgDefault args_aten_exp_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_exp_default = {"aten::exp.default", args_aten_exp_default, 1};
    if (strcmp(op, entry_aten_exp_default.op) == 0) return &entry_aten_exp_default;

    static const Pt2ArgDefault args_aten_exp__default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_exp__default = {"aten::exp_.default", args_aten_exp__default, 1};
    if (strcmp(op, entry_aten_exp__default.op) == 0) return &entry_aten_exp__default;

    static const Pt2ArgDefault args_aten_expand_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"size", PT2_D_NO_DEFAULT, ""},
        {"implicit", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_expand_default = {"aten::expand.default", args_aten_expand_default, 3};
    if (strcmp(op, entry_aten_expand_default.op) == 0) return &entry_aten_expand_default;

    static const Pt2ArgDefault args_aten_expand_as_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_expand_as_default = {"aten::expand_as.default", args_aten_expand_as_default, 2};
    if (strcmp(op, entry_aten_expand_as_default.op) == 0) return &entry_aten_expand_as_default;

    static const Pt2ArgDefault args_aten_expm1_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_expm1_default = {"aten::expm1.default", args_aten_expm1_default, 1};
    if (strcmp(op, entry_aten_expm1_default.op) == 0) return &entry_aten_expm1_default;

    static const Pt2ArgDefault args_aten_feature_alpha_dropout_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"p", PT2_D_NO_DEFAULT, ""},
        {"train", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_feature_alpha_dropout_default = {"aten::feature_alpha_dropout.default", args_aten_feature_alpha_dropout_default, 3};
    if (strcmp(op, entry_aten_feature_alpha_dropout_default.op) == 0) return &entry_aten_feature_alpha_dropout_default;

    static const Pt2ArgDefault args_aten_feature_dropout_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"p", PT2_D_NO_DEFAULT, ""},
        {"train", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_feature_dropout_default = {"aten::feature_dropout.default", args_aten_feature_dropout_default, 3};
    if (strcmp(op, entry_aten_feature_dropout_default.op) == 0) return &entry_aten_feature_dropout_default;

    static const Pt2ArgDefault args_aten_flatten_using_ints[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"start_dim", PT2_D_INT, "0"},
        {"end_dim", PT2_D_INT, "-1"},
    };
    static const Pt2DefaultsEntry entry_aten_flatten_using_ints = {"aten::flatten.using_ints", args_aten_flatten_using_ints, 3};
    if (strcmp(op, entry_aten_flatten_using_ints.op) == 0) return &entry_aten_flatten_using_ints;

    static const Pt2ArgDefault args_aten_flip_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dims", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_flip_default = {"aten::flip.default", args_aten_flip_default, 2};
    if (strcmp(op, entry_aten_flip_default.op) == 0) return &entry_aten_flip_default;

    static const Pt2ArgDefault args_aten_floor_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_floor_default = {"aten::floor.default", args_aten_floor_default, 1};
    if (strcmp(op, entry_aten_floor_default.op) == 0) return &entry_aten_floor_default;

    static const Pt2ArgDefault args_aten_gelu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"approximate", PT2_D_STRING, "none"},
    };
    static const Pt2DefaultsEntry entry_aten_gelu_default = {"aten::gelu.default", args_aten_gelu_default, 2};
    if (strcmp(op, entry_aten_gelu_default.op) == 0) return &entry_aten_gelu_default;

    static const Pt2ArgDefault args_aten_glu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "-1"},
    };
    static const Pt2DefaultsEntry entry_aten_glu_default = {"aten::glu.default", args_aten_glu_default, 2};
    if (strcmp(op, entry_aten_glu_default.op) == 0) return &entry_aten_glu_default;

    static const Pt2ArgDefault args_aten_grid_sampler_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"grid", PT2_D_NO_DEFAULT, ""},
        {"interpolation_mode", PT2_D_NO_DEFAULT, ""},
        {"padding_mode", PT2_D_NO_DEFAULT, ""},
        {"align_corners", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_grid_sampler_default = {"aten::grid_sampler.default", args_aten_grid_sampler_default, 5};
    if (strcmp(op, entry_aten_grid_sampler_default.op) == 0) return &entry_aten_grid_sampler_default;

    static const Pt2ArgDefault args_aten_group_norm_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"num_groups", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NONE, ""},
        {"bias", PT2_D_NONE, ""},
        {"eps", PT2_D_FLOAT, "1e-05"},
        {"cudnn_enabled", PT2_D_BOOL, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_group_norm_default = {"aten::group_norm.default", args_aten_group_norm_default, 6};
    if (strcmp(op, entry_aten_group_norm_default.op) == 0) return &entry_aten_group_norm_default;

    static const Pt2ArgDefault args_aten_gru_input[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"hx", PT2_D_NO_DEFAULT, ""},
        {"params", PT2_D_NO_DEFAULT, ""},
        {"has_biases", PT2_D_NO_DEFAULT, ""},
        {"num_layers", PT2_D_NO_DEFAULT, ""},
        {"dropout", PT2_D_NO_DEFAULT, ""},
        {"train", PT2_D_NO_DEFAULT, ""},
        {"bidirectional", PT2_D_NO_DEFAULT, ""},
        {"batch_first", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_gru_input = {"aten::gru.input", args_aten_gru_input, 9};
    if (strcmp(op, entry_aten_gru_input.op) == 0) return &entry_aten_gru_input;

    static const Pt2ArgDefault args_aten_hamming_window_default[] = {
        {"window_length", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
        {"layout", PT2_D_NONE, ""},
        {"device", PT2_D_NONE, ""},
        {"pin_memory", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_hamming_window_default = {"aten::hamming_window.default", args_aten_hamming_window_default, 5};
    if (strcmp(op, entry_aten_hamming_window_default.op) == 0) return &entry_aten_hamming_window_default;

    static const Pt2ArgDefault args_aten_hann_window_default[] = {
        {"window_length", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
        {"layout", PT2_D_NONE, ""},
        {"device", PT2_D_NONE, ""},
        {"pin_memory", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_hann_window_default = {"aten::hann_window.default", args_aten_hann_window_default, 5};
    if (strcmp(op, entry_aten_hann_window_default.op) == 0) return &entry_aten_hann_window_default;

    static const Pt2ArgDefault args_aten_hardshrink_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"lambd", PT2_D_FLOAT, "0.5"},
    };
    static const Pt2DefaultsEntry entry_aten_hardshrink_default = {"aten::hardshrink.default", args_aten_hardshrink_default, 2};
    if (strcmp(op, entry_aten_hardshrink_default.op) == 0) return &entry_aten_hardshrink_default;

    static const Pt2ArgDefault args_aten_hardsigmoid_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_hardsigmoid_default = {"aten::hardsigmoid.default", args_aten_hardsigmoid_default, 1};
    if (strcmp(op, entry_aten_hardsigmoid_default.op) == 0) return &entry_aten_hardsigmoid_default;

    static const Pt2ArgDefault args_aten_hardswish_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_hardswish_default = {"aten::hardswish.default", args_aten_hardswish_default, 1};
    if (strcmp(op, entry_aten_hardswish_default.op) == 0) return &entry_aten_hardswish_default;

    static const Pt2ArgDefault args_aten_hardtanh_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"min_val", PT2_D_INT, "-1"},
        {"max_val", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_hardtanh_default = {"aten::hardtanh.default", args_aten_hardtanh_default, 3};
    if (strcmp(op, entry_aten_hardtanh_default.op) == 0) return &entry_aten_hardtanh_default;

    static const Pt2ArgDefault args_aten_im2col_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"dilation", PT2_D_NO_DEFAULT, ""},
        {"padding", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_im2col_default = {"aten::im2col.default", args_aten_im2col_default, 5};
    if (strcmp(op, entry_aten_im2col_default.op) == 0) return &entry_aten_im2col_default;

    static const Pt2ArgDefault args_aten_instance_norm_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NO_DEFAULT, ""},
        {"running_mean", PT2_D_NO_DEFAULT, ""},
        {"running_var", PT2_D_NO_DEFAULT, ""},
        {"use_input_stats", PT2_D_NO_DEFAULT, ""},
        {"momentum", PT2_D_NO_DEFAULT, ""},
        {"eps", PT2_D_NO_DEFAULT, ""},
        {"cudnn_enabled", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_instance_norm_default = {"aten::instance_norm.default", args_aten_instance_norm_default, 9};
    if (strcmp(op, entry_aten_instance_norm_default.op) == 0) return &entry_aten_instance_norm_default;

    static const Pt2ArgDefault args_aten_istft_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"n_fft", PT2_D_NO_DEFAULT, ""},
        {"hop_length", PT2_D_NONE, ""},
        {"win_length", PT2_D_NONE, ""},
        {"window", PT2_D_NONE, ""},
        {"center", PT2_D_BOOL, "1"},
        {"normalized", PT2_D_BOOL, "0"},
        {"onesided", PT2_D_NONE, ""},
        {"length", PT2_D_NONE, ""},
        {"return_complex", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_istft_default = {"aten::istft.default", args_aten_istft_default, 10};
    if (strcmp(op, entry_aten_istft_default.op) == 0) return &entry_aten_istft_default;

    static const Pt2ArgDefault args_aten_layer_norm_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"normalized_shape", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NONE, ""},
        {"bias", PT2_D_NONE, ""},
        {"eps", PT2_D_FLOAT, "1e-05"},
        {"cudnn_enable", PT2_D_BOOL, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_layer_norm_default = {"aten::layer_norm.default", args_aten_layer_norm_default, 6};
    if (strcmp(op, entry_aten_layer_norm_default.op) == 0) return &entry_aten_layer_norm_default;

    static const Pt2ArgDefault args_aten_leaky_relu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"negative_slope", PT2_D_FLOAT, "0.01"},
    };
    static const Pt2DefaultsEntry entry_aten_leaky_relu_default = {"aten::leaky_relu.default", args_aten_leaky_relu_default, 2};
    if (strcmp(op, entry_aten_leaky_relu_default.op) == 0) return &entry_aten_leaky_relu_default;

    static const Pt2ArgDefault args_aten_linalg_vector_norm_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"ord", PT2_D_INT, "2"},
        {"dim", PT2_D_NONE, ""},
        {"keepdim", PT2_D_BOOL, "0"},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_linalg_vector_norm_default = {"aten::linalg_vector_norm.default", args_aten_linalg_vector_norm_default, 5};
    if (strcmp(op, entry_aten_linalg_vector_norm_default.op) == 0) return &entry_aten_linalg_vector_norm_default;

    static const Pt2ArgDefault args_aten_linear_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
        {"bias", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_linear_default = {"aten::linear.default", args_aten_linear_default, 3};
    if (strcmp(op, entry_aten_linear_default.op) == 0) return &entry_aten_linear_default;

    static const Pt2ArgDefault args_aten_log_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_log_default = {"aten::log.default", args_aten_log_default, 1};
    if (strcmp(op, entry_aten_log_default.op) == 0) return &entry_aten_log_default;

    static const Pt2ArgDefault args_aten_log10_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_log10_default = {"aten::log10.default", args_aten_log10_default, 1};
    if (strcmp(op, entry_aten_log10_default.op) == 0) return &entry_aten_log10_default;

    static const Pt2ArgDefault args_aten_log1p_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_log1p_default = {"aten::log1p.default", args_aten_log1p_default, 1};
    if (strcmp(op, entry_aten_log1p_default.op) == 0) return &entry_aten_log1p_default;

    static const Pt2ArgDefault args_aten_log_sigmoid_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_log_sigmoid_default = {"aten::log_sigmoid.default", args_aten_log_sigmoid_default, 1};
    if (strcmp(op, entry_aten_log_sigmoid_default.op) == 0) return &entry_aten_log_sigmoid_default;

    static const Pt2ArgDefault args_aten_log_softmax_int[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_log_softmax_int = {"aten::log_softmax.int", args_aten_log_softmax_int, 3};
    if (strcmp(op, entry_aten_log_softmax_int.op) == 0) return &entry_aten_log_softmax_int;

    static const Pt2ArgDefault args_aten_logsumexp_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"keepdim", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_logsumexp_default = {"aten::logsumexp.default", args_aten_logsumexp_default, 3};
    if (strcmp(op, entry_aten_logsumexp_default.op) == 0) return &entry_aten_logsumexp_default;

    static const Pt2ArgDefault args_aten_lstm_input[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"hx", PT2_D_NO_DEFAULT, ""},
        {"params", PT2_D_NO_DEFAULT, ""},
        {"has_biases", PT2_D_NO_DEFAULT, ""},
        {"num_layers", PT2_D_NO_DEFAULT, ""},
        {"dropout", PT2_D_NO_DEFAULT, ""},
        {"train", PT2_D_NO_DEFAULT, ""},
        {"bidirectional", PT2_D_NO_DEFAULT, ""},
        {"batch_first", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_lstm_input = {"aten::lstm.input", args_aten_lstm_input, 9};
    if (strcmp(op, entry_aten_lstm_input.op) == 0) return &entry_aten_lstm_input;

    static const Pt2ArgDefault args_aten_matmul_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_matmul_default = {"aten::matmul.default", args_aten_matmul_default, 2};
    if (strcmp(op, entry_aten_matmul_default.op) == 0) return &entry_aten_matmul_default;

    static const Pt2ArgDefault args_aten_max_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_max_default = {"aten::max.default", args_aten_max_default, 1};
    if (strcmp(op, entry_aten_max_default.op) == 0) return &entry_aten_max_default;

    static const Pt2ArgDefault args_aten_max_dim[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"keepdim", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_max_dim = {"aten::max.dim", args_aten_max_dim, 3};
    if (strcmp(op, entry_aten_max_dim.op) == 0) return &entry_aten_max_dim;

    static const Pt2ArgDefault args_aten_max_other[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_max_other = {"aten::max.other", args_aten_max_other, 2};
    if (strcmp(op, entry_aten_max_other.op) == 0) return &entry_aten_max_other;

    static const Pt2ArgDefault args_aten_max_pool1d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0"},
        {"dilation", PT2_D_INTS, "1"},
        {"ceil_mode", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_max_pool1d_default = {"aten::max_pool1d.default", args_aten_max_pool1d_default, 6};
    if (strcmp(op, entry_aten_max_pool1d_default.op) == 0) return &entry_aten_max_pool1d_default;

    static const Pt2ArgDefault args_aten_max_pool1d_with_indices_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0"},
        {"dilation", PT2_D_INTS, "1"},
        {"ceil_mode", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_max_pool1d_with_indices_default = {"aten::max_pool1d_with_indices.default", args_aten_max_pool1d_with_indices_default, 6};
    if (strcmp(op, entry_aten_max_pool1d_with_indices_default.op) == 0) return &entry_aten_max_pool1d_with_indices_default;

    static const Pt2ArgDefault args_aten_max_pool2d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0,0"},
        {"dilation", PT2_D_INTS, "1,1"},
        {"ceil_mode", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_max_pool2d_default = {"aten::max_pool2d.default", args_aten_max_pool2d_default, 6};
    if (strcmp(op, entry_aten_max_pool2d_default.op) == 0) return &entry_aten_max_pool2d_default;

    static const Pt2ArgDefault args_aten_max_pool2d_with_indices_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0,0"},
        {"dilation", PT2_D_INTS, "1,1"},
        {"ceil_mode", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_max_pool2d_with_indices_default = {"aten::max_pool2d_with_indices.default", args_aten_max_pool2d_with_indices_default, 6};
    if (strcmp(op, entry_aten_max_pool2d_with_indices_default.op) == 0) return &entry_aten_max_pool2d_with_indices_default;

    static const Pt2ArgDefault args_aten_max_pool3d_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0,0,0"},
        {"dilation", PT2_D_INTS, "1,1,1"},
        {"ceil_mode", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_max_pool3d_default = {"aten::max_pool3d.default", args_aten_max_pool3d_default, 6};
    if (strcmp(op, entry_aten_max_pool3d_default.op) == 0) return &entry_aten_max_pool3d_default;

    static const Pt2ArgDefault args_aten_max_pool3d_with_indices_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"kernel_size", PT2_D_NO_DEFAULT, ""},
        {"stride", PT2_D_INTS, ""},
        {"padding", PT2_D_INTS, "0,0,0"},
        {"dilation", PT2_D_INTS, "1,1,1"},
        {"ceil_mode", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_max_pool3d_with_indices_default = {"aten::max_pool3d_with_indices.default", args_aten_max_pool3d_with_indices_default, 6};
    if (strcmp(op, entry_aten_max_pool3d_with_indices_default.op) == 0) return &entry_aten_max_pool3d_with_indices_default;

    static const Pt2ArgDefault args_aten_maximum_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_maximum_default = {"aten::maximum.default", args_aten_maximum_default, 2};
    if (strcmp(op, entry_aten_maximum_default.op) == 0) return &entry_aten_maximum_default;

    static const Pt2ArgDefault args_aten_mean_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_mean_default = {"aten::mean.default", args_aten_mean_default, 2};
    if (strcmp(op, entry_aten_mean_default.op) == 0) return &entry_aten_mean_default;

    static const Pt2ArgDefault args_aten_mean_dim[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"keepdim", PT2_D_BOOL, "0"},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_mean_dim = {"aten::mean.dim", args_aten_mean_dim, 4};
    if (strcmp(op, entry_aten_mean_dim.op) == 0) return &entry_aten_mean_dim;

    static const Pt2ArgDefault args_aten_min_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_min_default = {"aten::min.default", args_aten_min_default, 1};
    if (strcmp(op, entry_aten_min_default.op) == 0) return &entry_aten_min_default;

    static const Pt2ArgDefault args_aten_min_dim[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"keepdim", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_min_dim = {"aten::min.dim", args_aten_min_dim, 3};
    if (strcmp(op, entry_aten_min_dim.op) == 0) return &entry_aten_min_dim;

    static const Pt2ArgDefault args_aten_min_other[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_min_other = {"aten::min.other", args_aten_min_other, 2};
    if (strcmp(op, entry_aten_min_other.op) == 0) return &entry_aten_min_other;

    static const Pt2ArgDefault args_aten_minimum_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_minimum_default = {"aten::minimum.default", args_aten_minimum_default, 2};
    if (strcmp(op, entry_aten_minimum_default.op) == 0) return &entry_aten_minimum_default;

    static const Pt2ArgDefault args_aten_mish_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_mish_default = {"aten::mish.default", args_aten_mish_default, 1};
    if (strcmp(op, entry_aten_mish_default.op) == 0) return &entry_aten_mish_default;

    static const Pt2ArgDefault args_aten_mm_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"mat2", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_mm_default = {"aten::mm.default", args_aten_mm_default, 2};
    if (strcmp(op, entry_aten_mm_default.op) == 0) return &entry_aten_mm_default;

    static const Pt2ArgDefault args_aten_mul_Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_mul_Tensor = {"aten::mul.Tensor", args_aten_mul_Tensor, 2};
    if (strcmp(op, entry_aten_mul_Tensor.op) == 0) return &entry_aten_mul_Tensor;

    static const Pt2ArgDefault args_aten_neg_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_neg_default = {"aten::neg.default", args_aten_neg_default, 1};
    if (strcmp(op, entry_aten_neg_default.op) == 0) return &entry_aten_neg_default;

    static const Pt2ArgDefault args_aten_ones_like_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
        {"layout", PT2_D_NONE, ""},
        {"device", PT2_D_NONE, ""},
        {"pin_memory", PT2_D_NONE, ""},
        {"memory_format", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_ones_like_default = {"aten::ones_like.default", args_aten_ones_like_default, 6};
    if (strcmp(op, entry_aten_ones_like_default.op) == 0) return &entry_aten_ones_like_default;

    static const Pt2ArgDefault args_aten_pad_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"pad", PT2_D_NO_DEFAULT, ""},
        {"mode", PT2_D_STRING, "constant"},
        {"value", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_pad_default = {"aten::pad.default", args_aten_pad_default, 4};
    if (strcmp(op, entry_aten_pad_default.op) == 0) return &entry_aten_pad_default;

    static const Pt2ArgDefault args_aten_permute_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dims", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_permute_default = {"aten::permute.default", args_aten_permute_default, 2};
    if (strcmp(op, entry_aten_permute_default.op) == 0) return &entry_aten_permute_default;

    static const Pt2ArgDefault args_aten_pixel_shuffle_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"upscale_factor", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_pixel_shuffle_default = {"aten::pixel_shuffle.default", args_aten_pixel_shuffle_default, 2};
    if (strcmp(op, entry_aten_pixel_shuffle_default.op) == 0) return &entry_aten_pixel_shuffle_default;

    static const Pt2ArgDefault args_aten_pixel_unshuffle_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"downscale_factor", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_pixel_unshuffle_default = {"aten::pixel_unshuffle.default", args_aten_pixel_unshuffle_default, 2};
    if (strcmp(op, entry_aten_pixel_unshuffle_default.op) == 0) return &entry_aten_pixel_unshuffle_default;

    static const Pt2ArgDefault args_aten_pow_Tensor_Scalar[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"exponent", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_pow_Tensor_Scalar = {"aten::pow.Tensor_Scalar", args_aten_pow_Tensor_Scalar, 2};
    if (strcmp(op, entry_aten_pow_Tensor_Scalar.op) == 0) return &entry_aten_pow_Tensor_Scalar;

    static const Pt2ArgDefault args_aten_pow_Tensor_Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"exponent", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_pow_Tensor_Tensor = {"aten::pow.Tensor_Tensor", args_aten_pow_Tensor_Tensor, 2};
    if (strcmp(op, entry_aten_pow_Tensor_Tensor.op) == 0) return &entry_aten_pow_Tensor_Tensor;

    static const Pt2ArgDefault args_aten_prelu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_prelu_default = {"aten::prelu.default", args_aten_prelu_default, 2};
    if (strcmp(op, entry_aten_prelu_default.op) == 0) return &entry_aten_prelu_default;

    static const Pt2ArgDefault args_aten_prod_dim_int[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"keepdim", PT2_D_BOOL, "0"},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_prod_dim_int = {"aten::prod.dim_int", args_aten_prod_dim_int, 4};
    if (strcmp(op, entry_aten_prod_dim_int.op) == 0) return &entry_aten_prod_dim_int;

    static const Pt2ArgDefault args_aten_reciprocal_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_reciprocal_default = {"aten::reciprocal.default", args_aten_reciprocal_default, 1};
    if (strcmp(op, entry_aten_reciprocal_default.op) == 0) return &entry_aten_reciprocal_default;

    static const Pt2ArgDefault args_aten_relu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_relu_default = {"aten::relu.default", args_aten_relu_default, 1};
    if (strcmp(op, entry_aten_relu_default.op) == 0) return &entry_aten_relu_default;

    static const Pt2ArgDefault args_aten_relu6_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_relu6_default = {"aten::relu6.default", args_aten_relu6_default, 1};
    if (strcmp(op, entry_aten_relu6_default.op) == 0) return &entry_aten_relu6_default;

    static const Pt2ArgDefault args_aten_relu6__default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_relu6__default = {"aten::relu6_.default", args_aten_relu6__default, 1};
    if (strcmp(op, entry_aten_relu6__default.op) == 0) return &entry_aten_relu6__default;

    static const Pt2ArgDefault args_aten_relu__default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_relu__default = {"aten::relu_.default", args_aten_relu__default, 1};
    if (strcmp(op, entry_aten_relu__default.op) == 0) return &entry_aten_relu__default;

    static const Pt2ArgDefault args_aten_repeat_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"repeats", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_repeat_default = {"aten::repeat.default", args_aten_repeat_default, 2};
    if (strcmp(op, entry_aten_repeat_default.op) == 0) return &entry_aten_repeat_default;

    static const Pt2ArgDefault args_aten_reshape_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"shape", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_reshape_default = {"aten::reshape.default", args_aten_reshape_default, 2};
    if (strcmp(op, entry_aten_reshape_default.op) == 0) return &entry_aten_reshape_default;

    static const Pt2ArgDefault args_aten_reshape_as_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_reshape_as_default = {"aten::reshape_as.default", args_aten_reshape_as_default, 2};
    if (strcmp(op, entry_aten_reshape_as_default.op) == 0) return &entry_aten_reshape_as_default;

    static const Pt2ArgDefault args_aten_rms_norm_default[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"normalized_shape", PT2_D_NO_DEFAULT, ""},
        {"weight", PT2_D_NONE, ""},
        {"eps", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_rms_norm_default = {"aten::rms_norm.default", args_aten_rms_norm_default, 4};
    if (strcmp(op, entry_aten_rms_norm_default.op) == 0) return &entry_aten_rms_norm_default;

    static const Pt2ArgDefault args_aten_rnn_tanh_input[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"hx", PT2_D_NO_DEFAULT, ""},
        {"params", PT2_D_NO_DEFAULT, ""},
        {"has_biases", PT2_D_NO_DEFAULT, ""},
        {"num_layers", PT2_D_NO_DEFAULT, ""},
        {"dropout", PT2_D_NO_DEFAULT, ""},
        {"train", PT2_D_NO_DEFAULT, ""},
        {"bidirectional", PT2_D_NO_DEFAULT, ""},
        {"batch_first", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_rnn_tanh_input = {"aten::rnn_tanh.input", args_aten_rnn_tanh_input, 9};
    if (strcmp(op, entry_aten_rnn_tanh_input.op) == 0) return &entry_aten_rnn_tanh_input;

    static const Pt2ArgDefault args_aten_roll_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"shifts", PT2_D_NO_DEFAULT, ""},
        {"dims", PT2_D_INTS, ""},
    };
    static const Pt2DefaultsEntry entry_aten_roll_default = {"aten::roll.default", args_aten_roll_default, 3};
    if (strcmp(op, entry_aten_roll_default.op) == 0) return &entry_aten_roll_default;

    static const Pt2ArgDefault args_aten_round_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_round_default = {"aten::round.default", args_aten_round_default, 1};
    if (strcmp(op, entry_aten_round_default.op) == 0) return &entry_aten_round_default;

    static const Pt2ArgDefault args_aten_rsqrt_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_rsqrt_default = {"aten::rsqrt.default", args_aten_rsqrt_default, 1};
    if (strcmp(op, entry_aten_rsqrt_default.op) == 0) return &entry_aten_rsqrt_default;

    static const Pt2ArgDefault args_aten_scaled_dot_product_attention_default[] = {
        {"query", PT2_D_NO_DEFAULT, ""},
        {"key", PT2_D_NO_DEFAULT, ""},
        {"value", PT2_D_NO_DEFAULT, ""},
        {"attn_mask", PT2_D_NONE, ""},
        {"dropout_p", PT2_D_FLOAT, "0.0"},
        {"is_causal", PT2_D_BOOL, "0"},
        {"scale", PT2_D_NONE, ""},
        {"enable_gqa", PT2_D_BOOL, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_scaled_dot_product_attention_default = {"aten::scaled_dot_product_attention.default", args_aten_scaled_dot_product_attention_default, 8};
    if (strcmp(op, entry_aten_scaled_dot_product_attention_default.op) == 0) return &entry_aten_scaled_dot_product_attention_default;

    static const Pt2ArgDefault args_aten_select_int[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"index", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_select_int = {"aten::select.int", args_aten_select_int, 3};
    if (strcmp(op, entry_aten_select_int.op) == 0) return &entry_aten_select_int;

    static const Pt2ArgDefault args_aten_selu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_selu_default = {"aten::selu.default", args_aten_selu_default, 1};
    if (strcmp(op, entry_aten_selu_default.op) == 0) return &entry_aten_selu_default;

    static const Pt2ArgDefault args_aten_sigmoid_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_sigmoid_default = {"aten::sigmoid.default", args_aten_sigmoid_default, 1};
    if (strcmp(op, entry_aten_sigmoid_default.op) == 0) return &entry_aten_sigmoid_default;

    static const Pt2ArgDefault args_aten_sign_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_sign_default = {"aten::sign.default", args_aten_sign_default, 1};
    if (strcmp(op, entry_aten_sign_default.op) == 0) return &entry_aten_sign_default;

    static const Pt2ArgDefault args_aten_silu_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_silu_default = {"aten::silu.default", args_aten_silu_default, 1};
    if (strcmp(op, entry_aten_silu_default.op) == 0) return &entry_aten_silu_default;

    static const Pt2ArgDefault args_aten_sin_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_sin_default = {"aten::sin.default", args_aten_sin_default, 1};
    if (strcmp(op, entry_aten_sin_default.op) == 0) return &entry_aten_sin_default;

    static const Pt2ArgDefault args_aten_sinh_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_sinh_default = {"aten::sinh.default", args_aten_sinh_default, 1};
    if (strcmp(op, entry_aten_sinh_default.op) == 0) return &entry_aten_sinh_default;

    static const Pt2ArgDefault args_aten_slice_Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
        {"start", PT2_D_NONE, ""},
        {"end", PT2_D_NONE, ""},
        {"step", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_slice_Tensor = {"aten::slice.Tensor", args_aten_slice_Tensor, 5};
    if (strcmp(op, entry_aten_slice_Tensor.op) == 0) return &entry_aten_slice_Tensor;

    static const Pt2ArgDefault args_aten_slice_scatter_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"src", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
        {"start", PT2_D_NONE, ""},
        {"end", PT2_D_NONE, ""},
        {"step", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_slice_scatter_default = {"aten::slice_scatter.default", args_aten_slice_scatter_default, 6};
    if (strcmp(op, entry_aten_slice_scatter_default.op) == 0) return &entry_aten_slice_scatter_default;

    static const Pt2ArgDefault args_aten_softmax_int[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_softmax_int = {"aten::softmax.int", args_aten_softmax_int, 3};
    if (strcmp(op, entry_aten_softmax_int.op) == 0) return &entry_aten_softmax_int;

    static const Pt2ArgDefault args_aten_softplus_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"beta", PT2_D_INT, "1"},
        {"threshold", PT2_D_INT, "20"},
    };
    static const Pt2DefaultsEntry entry_aten_softplus_default = {"aten::softplus.default", args_aten_softplus_default, 3};
    if (strcmp(op, entry_aten_softplus_default.op) == 0) return &entry_aten_softplus_default;

    static const Pt2ArgDefault args_aten_softshrink_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"lambd", PT2_D_FLOAT, "0.5"},
    };
    static const Pt2DefaultsEntry entry_aten_softshrink_default = {"aten::softshrink.default", args_aten_softshrink_default, 2};
    if (strcmp(op, entry_aten_softshrink_default.op) == 0) return &entry_aten_softshrink_default;

    static const Pt2ArgDefault args_aten_split_Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"split_size", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_split_Tensor = {"aten::split.Tensor", args_aten_split_Tensor, 3};
    if (strcmp(op, entry_aten_split_Tensor.op) == 0) return &entry_aten_split_Tensor;

    static const Pt2ArgDefault args_aten_split_with_sizes_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"split_sizes", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_split_with_sizes_default = {"aten::split_with_sizes.default", args_aten_split_with_sizes_default, 3};
    if (strcmp(op, entry_aten_split_with_sizes_default.op) == 0) return &entry_aten_split_with_sizes_default;

    static const Pt2ArgDefault args_aten_sqrt_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_sqrt_default = {"aten::sqrt.default", args_aten_sqrt_default, 1};
    if (strcmp(op, entry_aten_sqrt_default.op) == 0) return &entry_aten_sqrt_default;

    static const Pt2ArgDefault args_aten_square_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_square_default = {"aten::square.default", args_aten_square_default, 1};
    if (strcmp(op, entry_aten_square_default.op) == 0) return &entry_aten_square_default;

    static const Pt2ArgDefault args_aten_squeeze_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_squeeze_default = {"aten::squeeze.default", args_aten_squeeze_default, 1};
    if (strcmp(op, entry_aten_squeeze_default.op) == 0) return &entry_aten_squeeze_default;

    static const Pt2ArgDefault args_aten_squeeze_dim[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_squeeze_dim = {"aten::squeeze.dim", args_aten_squeeze_dim, 2};
    if (strcmp(op, entry_aten_squeeze_dim.op) == 0) return &entry_aten_squeeze_dim;

    static const Pt2ArgDefault args_aten_stack_default[] = {
        {"tensors", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_stack_default = {"aten::stack.default", args_aten_stack_default, 2};
    if (strcmp(op, entry_aten_stack_default.op) == 0) return &entry_aten_stack_default;

    static const Pt2ArgDefault args_aten_stft_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"n_fft", PT2_D_NO_DEFAULT, ""},
        {"hop_length", PT2_D_NONE, ""},
        {"win_length", PT2_D_NONE, ""},
        {"window", PT2_D_NONE, ""},
        {"normalized", PT2_D_BOOL, "0"},
        {"onesided", PT2_D_NONE, ""},
        {"return_complex", PT2_D_NONE, ""},
        {"align_to_window", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_stft_default = {"aten::stft.default", args_aten_stft_default, 9};
    if (strcmp(op, entry_aten_stft_default.op) == 0) return &entry_aten_stft_default;

    static const Pt2ArgDefault args_aten_sub_Tensor[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"other", PT2_D_NO_DEFAULT, ""},
        {"alpha", PT2_D_INT, "1"},
    };
    static const Pt2DefaultsEntry entry_aten_sub_Tensor = {"aten::sub.Tensor", args_aten_sub_Tensor, 3};
    if (strcmp(op, entry_aten_sub_Tensor.op) == 0) return &entry_aten_sub_Tensor;

    static const Pt2ArgDefault args_aten_sum_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_sum_default = {"aten::sum.default", args_aten_sum_default, 2};
    if (strcmp(op, entry_aten_sum_default.op) == 0) return &entry_aten_sum_default;

    static const Pt2ArgDefault args_aten_sum_dim_IntList[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"keepdim", PT2_D_BOOL, "0"},
        {"dtype", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_sum_dim_IntList = {"aten::sum.dim_IntList", args_aten_sum_dim_IntList, 4};
    if (strcmp(op, entry_aten_sum_dim_IntList.op) == 0) return &entry_aten_sum_dim_IntList;

    static const Pt2ArgDefault args_aten_t_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_t_default = {"aten::t.default", args_aten_t_default, 1};
    if (strcmp(op, entry_aten_t_default.op) == 0) return &entry_aten_t_default;

    static const Pt2ArgDefault args_aten_tan_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_tan_default = {"aten::tan.default", args_aten_tan_default, 1};
    if (strcmp(op, entry_aten_tan_default.op) == 0) return &entry_aten_tan_default;

    static const Pt2ArgDefault args_aten_tanh_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_tanh_default = {"aten::tanh.default", args_aten_tanh_default, 1};
    if (strcmp(op, entry_aten_tanh_default.op) == 0) return &entry_aten_tanh_default;

    static const Pt2ArgDefault args_aten_tensor_split_indices[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"indices", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_tensor_split_indices = {"aten::tensor_split.indices", args_aten_tensor_split_indices, 3};
    if (strcmp(op, entry_aten_tensor_split_indices.op) == 0) return &entry_aten_tensor_split_indices;

    static const Pt2ArgDefault args_aten_tensor_split_sections[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"sections", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_tensor_split_sections = {"aten::tensor_split.sections", args_aten_tensor_split_sections, 3};
    if (strcmp(op, entry_aten_tensor_split_sections.op) == 0) return &entry_aten_tensor_split_sections;

    static const Pt2ArgDefault args_aten_transpose_int[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim0", PT2_D_NO_DEFAULT, ""},
        {"dim1", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_transpose_int = {"aten::transpose.int", args_aten_transpose_int, 3};
    if (strcmp(op, entry_aten_transpose_int.op) == 0) return &entry_aten_transpose_int;

    static const Pt2ArgDefault args_aten_trunc_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_trunc_default = {"aten::trunc.default", args_aten_trunc_default, 1};
    if (strcmp(op, entry_aten_trunc_default.op) == 0) return &entry_aten_trunc_default;

    static const Pt2ArgDefault args_aten_unbind_int[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_INT, "0"},
    };
    static const Pt2DefaultsEntry entry_aten_unbind_int = {"aten::unbind.int", args_aten_unbind_int, 2};
    if (strcmp(op, entry_aten_unbind_int.op) == 0) return &entry_aten_unbind_int;

    static const Pt2ArgDefault args_aten_unflatten_int[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
        {"sizes", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_unflatten_int = {"aten::unflatten.int", args_aten_unflatten_int, 3};
    if (strcmp(op, entry_aten_unflatten_int.op) == 0) return &entry_aten_unflatten_int;

    static const Pt2ArgDefault args_aten_unsqueeze_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"dim", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_unsqueeze_default = {"aten::unsqueeze.default", args_aten_unsqueeze_default, 2};
    if (strcmp(op, entry_aten_unsqueeze_default.op) == 0) return &entry_aten_unsqueeze_default;

    static const Pt2ArgDefault args_aten_upsample_bicubic2d_vec[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
        {"align_corners", PT2_D_NO_DEFAULT, ""},
        {"scale_factors", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_upsample_bicubic2d_vec = {"aten::upsample_bicubic2d.vec", args_aten_upsample_bicubic2d_vec, 4};
    if (strcmp(op, entry_aten_upsample_bicubic2d_vec.op) == 0) return &entry_aten_upsample_bicubic2d_vec;

    static const Pt2ArgDefault args_aten_upsample_bilinear2d_vec[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
        {"align_corners", PT2_D_NO_DEFAULT, ""},
        {"scale_factors", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_upsample_bilinear2d_vec = {"aten::upsample_bilinear2d.vec", args_aten_upsample_bilinear2d_vec, 4};
    if (strcmp(op, entry_aten_upsample_bilinear2d_vec.op) == 0) return &entry_aten_upsample_bilinear2d_vec;

    static const Pt2ArgDefault args_aten_upsample_linear1d_vec[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
        {"align_corners", PT2_D_NO_DEFAULT, ""},
        {"scale_factors", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_upsample_linear1d_vec = {"aten::upsample_linear1d.vec", args_aten_upsample_linear1d_vec, 4};
    if (strcmp(op, entry_aten_upsample_linear1d_vec.op) == 0) return &entry_aten_upsample_linear1d_vec;

    static const Pt2ArgDefault args_aten_upsample_nearest1d_vec[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
        {"scale_factors", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_upsample_nearest1d_vec = {"aten::upsample_nearest1d.vec", args_aten_upsample_nearest1d_vec, 3};
    if (strcmp(op, entry_aten_upsample_nearest1d_vec.op) == 0) return &entry_aten_upsample_nearest1d_vec;

    static const Pt2ArgDefault args_aten_upsample_nearest2d_vec[] = {
        {"input", PT2_D_NO_DEFAULT, ""},
        {"output_size", PT2_D_NO_DEFAULT, ""},
        {"scale_factors", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_upsample_nearest2d_vec = {"aten::upsample_nearest2d.vec", args_aten_upsample_nearest2d_vec, 3};
    if (strcmp(op, entry_aten_upsample_nearest2d_vec.op) == 0) return &entry_aten_upsample_nearest2d_vec;

    static const Pt2ArgDefault args_aten_view_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
        {"size", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_view_default = {"aten::view.default", args_aten_view_default, 2};
    if (strcmp(op, entry_aten_view_default.op) == 0) return &entry_aten_view_default;

    static const Pt2ArgDefault args_aten_view_as_complex_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_view_as_complex_default = {"aten::view_as_complex.default", args_aten_view_as_complex_default, 1};
    if (strcmp(op, entry_aten_view_as_complex_default.op) == 0) return &entry_aten_view_as_complex_default;

    static const Pt2ArgDefault args_aten_view_as_real_default[] = {
        {"self", PT2_D_NO_DEFAULT, ""},
    };
    static const Pt2DefaultsEntry entry_aten_view_as_real_default = {"aten::view_as_real.default", args_aten_view_as_real_default, 1};
    if (strcmp(op, entry_aten_view_as_real_default.op) == 0) return &entry_aten_view_as_real_default;

    static const Pt2ArgDefault args_aten_zeros_default[] = {
        {"size", PT2_D_NO_DEFAULT, ""},
        {"dtype", PT2_D_NONE, ""},
        {"layout", PT2_D_NONE, ""},
        {"device", PT2_D_NONE, ""},
        {"pin_memory", PT2_D_NONE, ""},
    };
    static const Pt2DefaultsEntry entry_aten_zeros_default = {"aten::zeros.default", args_aten_zeros_default, 5};
    if (strcmp(op, entry_aten_zeros_default.op) == 0) return &entry_aten_zeros_default;
    return 0;
}

} // namespace pnnx

#endif // PNNX_ATEN_DEFAULTS_TABLE_H
