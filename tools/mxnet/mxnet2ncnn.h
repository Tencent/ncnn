#ifndef TOOLS_MXNET_MXNET2NCNN_H
#define TOOLS_MXNET_MXNET2NCNN_H

#include <stdio.h>

void convert(FILE* mxnet_json, FILE* mxnet_param, FILE* pp, FILE* bp);

#endif
