#ifndef TOOLS_ONNX_ONNX2NCNN_H
#define TOOLS_ONNX_ONNX2NCNN_H

#include "onnx.pb.h"

#include <stdio.h>
void convert(onnx::ModelProto model, FILE* pp, FILE* bp);

#endif
