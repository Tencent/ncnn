#ifndef _HELP_UTILS_H
#define _HELP_UTILS_H

#include <iostream>
#include <string>
#include <sstream>  
#include <fstream>  
#include <stdlib.h>  
#include <vector>
#include "mat.h"

namespace ncnn {
/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_in_f32(int layer_index, const char* layer_name, const Mat& bottom_blob);

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_out_f32(int layer_index, const char* layer_name, const Mat& top_blob);

/*
 * Extract the layer top blob S32 feature maps
 */
void extract_feature_out_s32(int layer_index, const char* layer_name, const Mat& top_blob);

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_in_s8(int layer_index, const char* layer_name, const Mat& bottom_blob);

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_out_s8(int layer_index, const char* layer_name, const Mat& top_blob);

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_out_s16(int layer_index, const char* layer_name, const Mat& top_blob);

/*
 * Extract the conv layer kernel weight value
 */
void extract_kernel_s8(int layer_index, const char* layer_name, const Mat& _kernel, const Mat& _bias, int num_input, int num_output, const int _kernel_size);

/*
 * Extract the dw-conv layer kernel weight value
 */
void extract_kernel_dw_s8(int layer_index, const char* layer_name, const Mat& _kernel, const Mat& _bias, int num_output, int group, const int _kernel_size);

/*
 * Extract the layer kernel weight value
 */
void extract_kernel_f32(int layer_index, const char* layer_name, const Mat& _kernel, const Mat& _bias, int num_input, int num_output, const int _kernel_size);

/*
 * Extract the blob feature map
 */
void extract_feature_blob_f32(const char* comment, const char* layer_name, const Mat& blob);

/*
 * Extract the blob feature map
 */
void extract_feature_blob_s8(const char* comment, const char* layer_name, const Mat& blob);

/*
 * Extract the blob feature map
 */
void extract_feature_blob_s16(const char* comment, const char* layer_name, const Mat& blob);
}

#endif //_HELP_UTILS_H