#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

#if NCNN_PIXEL_DRAWING
mp_obj_t mp_ncnn_draw_rectangle_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c1(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_rectangle_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c2(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_rectangle_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c3(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_rectangle_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c4(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c1(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c2(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c3(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c4(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}

mp_obj_t mp_ncnn_draw_circle_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c1(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_circle_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c2(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_circle_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c3(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_circle_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c4(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}

mp_obj_t mp_ncnn_draw_line_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c1(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_line_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c2(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_line_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c3(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_line_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c4(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
#endif /* NCNN_PIXEL_DRAWING */
}