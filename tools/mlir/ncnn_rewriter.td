// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_REWRITER_TD
#define NCNN_REWRITER_TD

include "tf_ops.td"
include "ncnn_ops.td"

def get_attr_f : NativeCodeCall<"$0.getValue<FloatAttr>(0)">;

def EqualOperands : Constraint<CPred<"$0 == $1">>;

def FuseInstanceNormPattern : Pat<
    (TF_ReshapeOp
        (TF_AddV2Op
            (TF_MulOp
                $reshaped__,
                (TF_MulOp:$rsqrt_var_eps_gamma
                    (TF_RsqrtOp
                        (TF_AddV2Op
                            (TF_MeanOp
                                (TF_SquaredDifferenceOp
                                    $reshaped_,
                                    (TF_MeanOp:$mean
                                        (TF_ReshapeOp:$reshaped $x, (TF_ConstOp ElementsAttr)),
                                        (TF_ConstOp:$reduce_axis ElementsAttr),
                                        ConstBoolAttrTrue // keep_dims
                                    )
                                ),
                                $reduce_axis_,
                                ConstBoolAttrTrue // keep_dims
                            ),
                            (TF_ConstOp ElementsAttr:$epsilon)
                        )
                    ),
                    $gamma
                )
            ),
            (TF_SubOp
                $beta,
                (TF_MulOp $rsqrt_var_eps_gamma_, $mean_)
            )
        ),
        (TF_ConstOp ElementsAttr)
    ),

    (NCNN_InstanceNormOp $x, $gamma, $beta, (get_attr_f $epsilon)),

    [
        (EqualOperands $reshaped, $reshaped_),
        (EqualOperands $reshaped, $reshaped__),
        (EqualOperands $reduce_axis, $reduce_axis_),
        (EqualOperands $rsqrt_var_eps_gamma, $rsqrt_var_eps_gamma_),
        (EqualOperands $mean, $mean_)
    ]
>;

#endif // NCNN_REWRITER_TD
