// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

            int remain = size;

#if __AVX__ || __SSE__
			__m128 k_data = _mm_loadu_ps(kernel0);

			for (; remain > 0; remain--)
			{
				float r_array[4] = { *r0, *r1, *r2, *r3 };
				__m128 r_data = _mm_loadu_ps(r_array);
				__m128 sum = _mm_mul_ps(k_data, r_data);
				*outptr += sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];

				r0++;
				r1++;
				r2++;
				r3++;
				outptr++;
			}
#else
            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }
#endif

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            int size = outw * outh;

            int remain = size;

#if __AVX__ || __SSE__
#if __AVX__
			int circle_num = size / 8;
			__m256 k_data = _mm256_set1_ps(k0);
			int index = 0;
			for (; index < circle_num; index++)
			{
				int index_offset = index * 8;
				__m256 out_data = _mm256_loadu_ps(outptr + index_offset);
				__m256 r_data = _mm256_loadu_ps(r0 + index_offset);
				out_data = _mm256_add_ps(_mm256_mul_ps(r_data, k_data), out_data);
				_mm256_storeu_ps(outptr + index_offset, out_data);
			}

			for (index = 8 * index; index < size; index++)
			{
				outptr[index] += r0[index] * k0;
			}
#else
			int circle_num = size / 4;
			__m128 k_data = _mm_set1_ps(k0);
			int index = 0;
			for (; index < circle_num; index++)
			{
				int index_offset = index * 4;
				__m128 out_data = _mm_loadu_ps(outptr + index_offset);
				__m128 r_data = _mm_loadu_ps(r0 + index_offset);
				out_data = _mm_add_ps(_mm_mul_ps(r_data, k_data), out_data);
				_mm_storeu_ps(outptr + index_offset, out_data);
			}

			for (index = 4 * index; index < size; index++)
			{
				outptr[index] += r0[index] * k0;
			}
#endif
#else
            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }
#endif

        }
    }

}

static void conv1x1s2_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

#if __AVX__ || __SSE__
			float k_array[4] = { k0, k1, k2, k3 };
			__m128 k_data = _mm_loadu_ps(k_array);

			for (int i = 0; i < outh; i++)
			{
				int remain = outw;

				for (; remain > 0; remain--)
				{
					float r_array[4] = { *r0, *r1, *r2, *r3 };
					__m128 r_data = _mm_loadu_ps(r_array);
					__m128 sum = _mm_mul_ps(k_data, r_data);
					*outptr += sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];

					r0 += 2;
					r1 += 2;
					r2 += 2;
					r3 += 2;
					outptr++;
				}

				r0 += tailstep;
				r1 += tailstep;
				r2 += tailstep;
				r3 += tailstep;
			}
#else
            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;
                    float sum1 = *r1 * k1;
                    float sum2 = *r2 * k2;
                    float sum3 = *r3 * k3;

                    *outptr += sum + sum1 + sum2 + sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }
#endif

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;

                    *outptr += sum;

                    r0 += 2;
                    outptr++;
                }

                r0 += tailstep;
            }

        }
    }

}
