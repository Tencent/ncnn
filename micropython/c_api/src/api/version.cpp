#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

mp_obj_t mp_ncnn_version(void)
{
    const char* ver = ncnn_version();
    return mp_obj_new_str(ver, strlen(ver));
}
}
