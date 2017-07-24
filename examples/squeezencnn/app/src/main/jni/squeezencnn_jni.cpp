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

#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "net.h"

#include "squeezenet_v1.1.id.h"

#include <sys/time.h>
#include <unistd.h>

static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
//     fprintf(stderr, "%.2fms   %s\n", elasped, comment);
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%.2fms   %s", elasped, comment);
}

static std::vector<unsigned char> squeezenet_param;
static std::vector<unsigned char> squeezenet_bin;
static std::vector<std::string> squeezenet_words;
static ncnn::Net squeezenet;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

// public native boolean Init(byte[] param, byte[] bin, byte[] words);
JNIEXPORT jboolean JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Init(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin, jbyteArray words)
{
    // init param
    {
        int len = env->GetArrayLength(param);
        squeezenet_param.resize(len);
        env->GetByteArrayRegion(param, 0, len, (jbyte*)squeezenet_param.data());
        int ret = squeezenet.load_param(squeezenet_param.data());
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_param %d %d", ret, len);
    }

    // init bin
    {
        int len = env->GetArrayLength(bin);
        squeezenet_bin.resize(len);
        env->GetByteArrayRegion(bin, 0, len, (jbyte*)squeezenet_bin.data());
        int ret = squeezenet.load_model(squeezenet_bin.data());
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model %d %d", ret, len);
    }

    // init words
    {
        int len = env->GetArrayLength(words);
        std::string words_buffer;
        words_buffer.resize(len);
        env->GetByteArrayRegion(words, 0, len, (jbyte*)words_buffer.data());
        squeezenet_words = split_string(words_buffer, "\n");
    }

    return JNI_TRUE;
}

// public native String Detect(Bitmap bitmap);
JNIEXPORT jstring JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap)
{
    bench_start();

    // ncnn from bitmap
    ncnn::Mat in;
    {
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        int width = info.width;
        int height = info.height;
        if (width != 227 || height != 227)
            return NULL;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        void* indata;
        AndroidBitmap_lockPixels(env, bitmap, &indata);

        in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2BGR, width, height);

        AndroidBitmap_unlockPixels(env, bitmap);
    }

    // squeezenet
    std::vector<float> cls_scores;
    {
        const float mean_vals[3] = {104.f, 117.f, 123.f};
        in.substract_mean_normalize(mean_vals, 0);

        ncnn::Extractor ex = squeezenet.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(4);

        ex.input(squeezenet_v1_1_param_id::BLOB_data, in);

        ncnn::Mat out;
        ex.extract(squeezenet_v1_1_param_id::BLOB_prob, out);

        cls_scores.resize(out.c);
        for (int j=0; j<out.c; j++)
        {
            const float* prob = out.data + out.cstep * j;
            cls_scores[j] = prob[0];
        }
    }

    // return top class
    int top_class = 0;
    float max_score = 0.f;
    for (size_t i=0; i<cls_scores.size(); i++)
    {
        float s = cls_scores[i];
//         __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%d %f", i, s);
        if (s > max_score)
        {
            top_class = i;
            max_score = s;
        }
    }

    const std::string& word = squeezenet_words[top_class];
    char tmp[32];
    sprintf(tmp, "%.3f", max_score);
    std::string result_str = std::string(word.c_str() + 10) + " = " + tmp;

    // +10 to skip leading n03179701
    jstring result = env->NewStringUTF(result_str.c_str());

    bench_end("detect");

    return result;
}

}
