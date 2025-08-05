#ifndef NCNN_MODULE_H
#define NCNN_MODULE_H

#include "py/runtime.h"

extern mp_obj_t mp_ncnn_version(void);

/* option api */
extern mp_obj_t mp_ncnn_option_create(void);
extern mp_obj_t mp_ncnn_option_destroy(mp_obj_t ncnn_option_obj);
extern mp_obj_t mp_ncnn_option_get_num_threads(mp_obj_t ncnn_option_obj);
extern mp_obj_t mp_ncnn_option_set_num_threads(mp_obj_t ncnn_option_obj, mp_obj_t num_threads_obj);
extern mp_obj_t mp_ncnn_option_get_use_local_pool_allocator(mp_obj_t ncnn_option_obj);
extern mp_obj_t mp_ncnn_option_set_use_local_pool_allocator(mp_obj_t ncnn_option_obj, mp_obj_t use_local_pool_allocator_obj);
extern mp_obj_t mp_ncnn_option_set_blob_allocator(mp_obj_t ncnn_option_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_option_set_workspace_allocator(mp_obj_t ncnn_option_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_option_get_use_vulkan_compute(mp_obj_t ncnn_option_obj);
extern mp_obj_t mp_ncnn_option_set_use_vulkan_compute(mp_obj_t ncnn_option_obj, mp_obj_t use_vulkan_compute_obj);

/* mat api */
extern mp_obj_t mp_ncnn_mat_create(void);
extern mp_obj_t mp_ncnn_mat_create_1d(mp_obj_t w_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_create_2d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_create_3d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_4d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_1d(mp_obj_t w_obj, mp_obj_t data_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_create_external_2d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_3d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_4d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_1d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_2d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_3d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_4d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_1d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_2d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_3d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_4d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_destroy(mp_obj_t ncnn_mat_obj);

extern mp_obj_t mp_ncnn_mat_fill_float(mp_obj_t ncnn_mat_obj, mp_obj_t v_obj);

extern mp_obj_t mp_ncnn_mat_clone(mp_obj_t ncnn_mat_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_reshape_1d(mp_obj_t ncnn_mat_obj, mp_obj_t w_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_reshape_2d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_reshape_3d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_reshape_4d(size_t n_args, const mp_obj_t* args);

extern mp_obj_t mp_ncnn_mat_get_dims(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_w(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_h(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_d(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_c(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_elemsize(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_elempack(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_cstep(mp_obj_t ncnn_mat_obj);
extern mp_obj_t mp_ncnn_mat_get_data(mp_obj_t ncnn_mat_obj);

extern mp_obj_t mp_ncnn_mat_get_channel_data(mp_obj_t ncnn_mat_obj, mp_obj_t c_obj);

#endif