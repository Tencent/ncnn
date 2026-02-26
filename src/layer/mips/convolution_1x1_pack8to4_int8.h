// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void conv1x1s1_sgemm_pack8to4_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_pack8to4_int8_msa(bottom_im2col, top_blob, kernel, opt);
}

static void conv1x1s2_sgemm_pack8to4_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const int64_t* r0 = bottom_blob.channel(p);
        int64_t* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j < outw; j++)
            {
                outptr[0] = r0[0];

                r0 += 2;
                outptr += 1;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack8to4_int8_msa(bottom_blob_shrinked, top_blob, kernel, opt);
}
