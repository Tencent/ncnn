// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// convert piper checkpoints to ncnn models
//  1. checkout https://github.com/OHF-Voice/piper1-gpl (113931937cf235fc881afd1ca4be209bc6919bc7)
//  2. apply patch piper1-gpl.patch from https://github.com/nihui/ncnn-android-piper
//  3. setup piper with
//      python3 -m venv .venv
//      source .venv/bin/activate
//      python3 -m pip install -e .[train]
//  4. download piper checkpoint file (*.ckpt) from https://huggingface.co/datasets/rhasspy/piper-checkpoints
//  5. install pnnx via pip install -U pnnx
//  6. obtain export_ncnn.py script from https://github.com/nihui/ncnn-android-piper
//      python export_ncnn.py en.ckpt

// convert word list to simple phonemizer dict
//  1. prepare word list from https://github.com/Alexir/CMUdict
//  2. for each word, get phonemes via command "./espeak-ng -q -v en-us --ipa word"
//  3. obtain config.json file from https://huggingface.co/datasets/rhasspy/piper-checkpoints
//  4. replace phonemes with ids according to phoneme_id_map in config.json
//  5. write dict binary
//      word1 \0x00 ids1 \0xff word2 \0x00 ids2 \0xff .....

#include "layer.h"
#include "mat.h"
#include "net.h"

#include <ctype.h>
#include <stdio.h>
#include <map>
#include <vector>

class relative_embeddings_k_module : public ncnn::Layer
{
public:
    relative_embeddings_k_module()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        const int window_size = 4;

        const int wsize = bottom_blob.w;
        const int len = bottom_blob.h;
        const int num_heads = bottom_blob.c;

        top_blob.create(len, len, num_heads);

        top_blob.fill(0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const ncnn::Mat x0 = bottom_blob.channel(q);
            ncnn::Mat out0 = top_blob.channel(q);

            for (int i = 0; i < len; i++)
            {
                const float* xptr = x0.row(i) + std::max(0, window_size - i);
                float* outptr = out0.row(i) + std::max(i - window_size, 0);
                const int wsize2 = std::min(len, i - window_size + wsize) - std::max(i - window_size, 0);
                for (int j = 0; j < wsize2; j++)
                {
                    *outptr++ = *xptr++;
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(relative_embeddings_k_module)

class relative_embeddings_v_module : public ncnn::Layer
{
public:
    relative_embeddings_v_module()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        const int window_size = 4;

        const int wsize = window_size * 2 + 1;
        const int len = bottom_blob.h;
        const int num_heads = bottom_blob.c;

        top_blob.create(wsize, len, num_heads);

        top_blob.fill(0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const ncnn::Mat x0 = bottom_blob.channel(q);
            ncnn::Mat out0 = top_blob.channel(q);

            for (int i = 0; i < len; i++)
            {
                const float* xptr = x0.row(i) + std::max(i - window_size, 0);
                float* outptr = out0.row(i) + std::max(0, window_size - i);
                const int wsize2 = std::min(len, i - window_size + wsize) - std::max(i - window_size, 0);
                for (int j = 0; j < wsize2; j++)
                {
                    *outptr++ = *xptr++;
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(relative_embeddings_v_module)

class piecewise_rational_quadratic_transform_module : public ncnn::Layer
{
public:
    piecewise_rational_quadratic_transform_module()
    {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& h = bottom_blobs[0];
        const ncnn::Mat& x1 = bottom_blobs[1];
        ncnn::Mat& outputs = top_blobs[0];

        const int num_bins = 10;
        const int filter_channels = 192;
        const bool reverse = true;
        const float tail_bound = 5.0f;
        const float DEFAULT_MIN_BIN_WIDTH = 1e-3f;
        const float DEFAULT_MIN_BIN_HEIGHT = 1e-3f;
        const float DEFAULT_MIN_DERIVATIVE = 1e-3f;

        const int batch_size = x1.w;
        const int h_params_per_item = 2 * num_bins + (num_bins - 1); // 29

        outputs = x1.clone();

        float* out_ptr = outputs;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < batch_size; ++i)
        {
            const float current_x = ((const float*)x1)[i];

            const float* h_data = h.row(i);

            if (current_x < -tail_bound || current_x > tail_bound)
            {
                continue;
            }

            std::vector<float> unnormalized_widths(num_bins);
            std::vector<float> unnormalized_heights(num_bins);
            std::vector<float> unnormalized_derivatives(num_bins + 1);

            const float inv_sqrt_filter_channels = 1.0f / sqrtf(filter_channels);
            for (int j = 0; j < num_bins; ++j)
            {
                unnormalized_widths[j] = h_data[j] * inv_sqrt_filter_channels;
            }
            for (int j = 0; j < num_bins; ++j)
            {
                unnormalized_heights[j] = h_data[num_bins + j] * inv_sqrt_filter_channels;
            }
            for (int j = 0; j < num_bins - 1; ++j)
            {
                unnormalized_derivatives[j + 1] = h_data[2 * num_bins + j];
            }

            const float constant = logf(expf(1.f - DEFAULT_MIN_DERIVATIVE) - 1.f);
            unnormalized_derivatives[0] = constant;
            unnormalized_derivatives[num_bins] = constant;

            const float left = -tail_bound, right = tail_bound;
            const float bottom = -tail_bound, top = tail_bound;

            // Softmax + Affine
            std::vector<float> widths(num_bins);
            float w_max = -INFINITY;
            for (float val : unnormalized_widths) w_max = std::max(w_max, val);
            float w_sum = 0.f;
            for (int j = 0; j < num_bins; ++j)
            {
                widths[j] = expf(unnormalized_widths[j] - w_max);
                w_sum += widths[j];
            }
            for (int j = 0; j < num_bins; ++j)
            {
                widths[j] = DEFAULT_MIN_BIN_WIDTH + (1.f - DEFAULT_MIN_BIN_WIDTH * num_bins) * (widths[j] / w_sum);
            }

            // cumwidths
            std::vector<float> cumwidths(num_bins + 1);
            cumwidths[0] = left;
            float current_w_sum = 0.f;
            for (int j = 0; j < num_bins - 1; ++j)
            {
                current_w_sum += widths[j];
                cumwidths[j + 1] = left + (right - left) * current_w_sum;
            }
            cumwidths[num_bins] = right;

            // heights
            std::vector<float> heights(num_bins);
            float h_max = -INFINITY;
            for (float val : unnormalized_heights) h_max = std::max(h_max, val);
            float h_sum = 0.f;
            for (int j = 0; j < num_bins; ++j)
            {
                heights[j] = expf(unnormalized_heights[j] - h_max);
                h_sum += heights[j];
            }
            for (int j = 0; j < num_bins; ++j)
            {
                heights[j] = DEFAULT_MIN_BIN_HEIGHT + (1.f - DEFAULT_MIN_BIN_HEIGHT * num_bins) * (heights[j] / h_sum);
            }

            // cumheights
            std::vector<float> cumheights(num_bins + 1);
            cumheights[0] = bottom;
            float current_h_sum = 0.f;
            for (int j = 0; j < num_bins - 1; ++j)
            {
                current_h_sum += heights[j];
                cumheights[j + 1] = bottom + (top - bottom) * current_h_sum;
            }
            cumheights[num_bins] = top;

            // Softplus
            std::vector<float> derivatives(num_bins + 1);
            for (int j = 0; j < num_bins + 1; ++j)
            {
                float x = unnormalized_derivatives[j];
                derivatives[j] = DEFAULT_MIN_DERIVATIVE + (x > 0 ? x + logf(1.f + expf(-x)) : logf(1.f + expf(x)));
            }

            // bin_idx
            int bin_idx = 0;
            if (reverse)
            {
                auto it = std::upper_bound(cumheights.begin(), cumheights.end(), current_x);
                bin_idx = std::distance(cumheights.begin(), it) - 1;
            }
            else
            {
                auto it = std::upper_bound(cumwidths.begin(), cumwidths.end(), current_x);
                bin_idx = std::distance(cumwidths.begin(), it) - 1;
            }
            bin_idx = std::max(0, std::min(bin_idx, num_bins - 1));

            // collect coeffs
            const float input_cumwidths = cumwidths[bin_idx];
            const float input_bin_widths = cumwidths[bin_idx + 1] - cumwidths[bin_idx];
            const float input_cumheights = cumheights[bin_idx];
            const float input_heights = cumheights[bin_idx + 1] - cumheights[bin_idx];
            const float input_derivatives = derivatives[bin_idx];
            const float input_derivatives_plus_one = derivatives[bin_idx + 1];
            const float delta = input_heights / input_bin_widths;

            // apply transform
            if (reverse)
            {
                float a = (current_x - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * delta) + input_heights * (delta - input_derivatives);
                float b = input_heights * input_derivatives - (current_x - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * delta);
                float c = -delta * (current_x - input_cumheights);
                float discriminant = b * b - 4 * a * c;
                discriminant = std::max(0.f, discriminant);
                float root = (2 * c) / (-b - sqrtf(discriminant));
                out_ptr[i] = root * input_bin_widths + input_cumwidths;
            }
            else
            {
                float theta = (current_x - input_cumwidths) / input_bin_widths;
                float theta_one_minus_theta = theta * (1 - theta);
                float numerator = input_heights * (delta * theta * theta + input_derivatives * theta_one_minus_theta);
                float denominator = delta + ((input_derivatives + input_derivatives_plus_one - 2 * delta) * theta_one_minus_theta);
                out_ptr[i] = input_cumheights + numerator / denominator;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(piecewise_rational_quadratic_transform_module)

static bool is_word_eos(const char* word)
{
    const char c = word[0];
    return c == ',' || c == '.' || c == ';' || c == '?' || c == '!';
}

static void find_word_id(const std::map<unsigned int, std::vector<const char*> >& dict, const char* word, const unsigned char*& ids)
{
    ids = 0;

    unsigned char first_char = toupper(word[0]);
    if (dict.find(first_char) == dict.end())
        return;

    const std::vector<const char*>& wordlist = dict.at(first_char);
    for (size_t i = 0; i < wordlist.size(); i++)
    {
        if (strcasecmp(wordlist[i], word) == 0)
        {
            // hit
            ids = (const unsigned char*)(wordlist[i] + strlen(wordlist[i]) + 1);
            return;
        }
    }
}

static void simple_phonemize(const char* text, std::vector<int>& sequence_ids)
{
    // this is a very simple g2p function, it works for english only

    // load dict buffer
    std::vector<unsigned char> dictbinbuf;
    {
        FILE* fp = fopen("en-word_id.bin", "rb");
        if (!fp)
            return;

        fseek(fp, 0, SEEK_END);
        size_t len = ftell(fp);
        rewind(fp);

        dictbinbuf.resize(len);
        fread(dictbinbuf.data(), 1, len, fp);

        fclose(fp);
    }

    // build dict
    std::map<unsigned int, std::vector<const char*> > dict;
    {
        const unsigned char* p = dictbinbuf.data();
        const char* word = (const char*)p;
        for (size_t i = 0; i < dictbinbuf.size(); i++)
        {
            if (dictbinbuf[i] == 0xff)
            {
                unsigned int first_char = toupper(word[0]);
                dict[first_char].push_back(word);
                word = (const char*)(p + i + 1);
            }
        }
    }

    // phonemize mainpart
    {
        const int ID_PAD = 0;   // interleaved
        const int ID_BOS = 1;   // beginning of sentence
        const int ID_EOS = 2;   // end of sentence
        const int ID_SPACE = 3; // space

        bool last_char_is_control = false;
        bool sentence_begin = true;
        bool sentence_end = true;

        char word[256];

        const char* p = text;
        while (*p)
        {
            if (sentence_end && !last_char_is_control)
            {
                sequence_ids.push_back(ID_BOS);
                sequence_ids.push_back(ID_PAD);
                sentence_end = false;
            }

            if (sentence_begin || last_char_is_control)
            {
                // the very first word
            }
            else
            {
                // space id
                sequence_ids.push_back(ID_SPACE);
                sequence_ids.push_back(ID_PAD);
            }

            if (isalnum((unsigned char)*p))
            {
                char* pword = word;

                // alpha or number
                *pword++ = *p++;

                // consume word
                int wordlen = 1;
                while (isalnum((unsigned char)*p) && wordlen < 233)
                {
                    *pword++ = *p++;
                    wordlen++;
                }

                *pword = '\0';

                if (is_word_eos(word))
                {
                    if (!sentence_end)
                        sequence_ids.push_back(ID_EOS);
                    sentence_end = true;
                    last_char_is_control = false;
                    sentence_begin = false;
                    continue;
                }

                const unsigned char* ids = 0;
                find_word_id(dict, word, ids);
                if (ids)
                {
                    const unsigned char* pids = ids;
                    while (*pids != 0xff)
                    {
                        sequence_ids.push_back(*pids);
                        sequence_ids.push_back(ID_PAD);
                        pids++;
                    }
                }
                else
                {
                    // no such word, spell alphabet one by one
                    char tmp[2] = {'\0', '\0'};
                    for (size_t i = 0; i < strlen(word); i++)
                    {
                        tmp[0] = word[i];
                        find_word_id(dict, tmp, ids);
                        if (ids)
                        {
                            const unsigned char* pids = ids;
                            while (*pids != 0xff)
                            {
                                sequence_ids.push_back(*pids);
                                sequence_ids.push_back(ID_PAD);
                                pids++;
                            }
                            if (i + 1 != strlen(word))
                            {
                                sequence_ids.push_back(ID_SPACE);
                                sequence_ids.push_back(ID_PAD);
                            }
                        }
                        else
                        {
                            fprintf(stderr, "word char %c not recognized\n", word[i]);
                        }
                    }
                }

                last_char_is_control = false;
                sentence_begin = false;
                continue;
            }
            else
            {
                // skip control character
                p++;
                last_char_is_control = true;
            }
        }

        if (!sentence_end)
            sequence_ids.push_back(ID_EOS);
    }
}

static void path_attention(const ncnn::Mat& logw, const ncnn::Mat& m_p, const ncnn::Mat& logs_p, float noise_scale, float length_scale, ncnn::Mat& z_p)
{
    const int x_lengths = logw.w;

    // assert m_p.h == logs_p.h
    const int depth = m_p.h;

    std::vector<int> w_ceil(x_lengths);
    int y_lengths = 0;
    for (int i = 0; i < x_lengths; i++)
    {
        w_ceil[i] = (int)ceilf(expf(logw[i]) * length_scale);
        y_lengths += w_ceil[i];
    }

    z_p.create(y_lengths, depth);

    for (int i = 0; i < depth; i++)
    {
        const float* m_p_ptr = m_p.row(i);
        const float* logs_p_ptr = logs_p.row(i);
        float* ptr = z_p.row(i);

        for (int j = 0; j < x_lengths; j++)
        {
            const float m = m_p_ptr[j];
            const float nl = expf(logs_p_ptr[j]) * noise_scale;
            const int duration = w_ceil[j];

            for (int k = 0; k < duration; k++)
            {
                ptr[k] = m + (rand() / (float)RAND_MAX) * nl;
            }
            ptr += duration;
        }
    }
}

static int tts_piper(const char* text, int speaker_id, std::vector<short>& pcm)
{
    // zh models could be found at
    // https://github.com/nihui/ncnn-android-piper/tree/master/app/src/main/assets

    // hyper parameters from https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/en/en_US/libritts_r/medium/config.json
    const float noise_scale = 0.333f;
    const float length_scale = 1.f;
    const float noise_scale_w = 0.333f;

    // phonemize
    ncnn::Mat sequence;
    {
        std::vector<int> sequence_ids;
        simple_phonemize(text, sequence_ids);

        const int sequence_length = (int)sequence_ids.size();

        sequence.create(sequence_length);
        memcpy(sequence, sequence_ids.data(), sequence_length * sizeof(int));
    }

    // enc_p
    ncnn::Mat x;
    ncnn::Mat m_p;
    ncnn::Mat logs_p;
    {
        ncnn::Net enc_p;
        enc_p.opt.use_vulkan_compute = true;
        enc_p.register_custom_layer("piper.train.vits.attentions.relative_embeddings_k_module", relative_embeddings_k_module_layer_creator);
        enc_p.register_custom_layer("piper.train.vits.attentions.relative_embeddings_v_module", relative_embeddings_v_module_layer_creator);
        enc_p.load_param("en_enc_p.ncnn.param");
        enc_p.load_model("en_enc_p.ncnn.bin");

        ncnn::Extractor ex = enc_p.create_extractor();

        ex.input("in0", sequence);

        ex.extract("out0", x);
        ex.extract("out1", m_p);
        ex.extract("out2", logs_p);
    }

    // emb_g
    ncnn::Mat g;
    {
        ncnn::Net emb_g;
        emb_g.opt.use_vulkan_compute = true;
        emb_g.load_param("en_emb_g.ncnn.param");
        emb_g.load_model("en_emb_g.ncnn.bin");

        ncnn::Mat speaker_id_mat(1);
        {
            int* p = speaker_id_mat;
            p[0] = speaker_id;
        }

        ncnn::Extractor ex = emb_g.create_extractor();

        ex.input("in0", speaker_id_mat);

        ex.extract("out0", g);

        g = g.reshape(1, g.w);
    }

    // dp
    ncnn::Mat logw;
    {
        ncnn::Net dp;
        dp.opt.use_vulkan_compute = true;
        dp.register_custom_layer("piper.train.vits.modules.piecewise_rational_quadratic_transform_module", piecewise_rational_quadratic_transform_module_layer_creator);
        dp.load_param("en_dp.ncnn.param");
        dp.load_model("en_dp.ncnn.bin");

        ncnn::Mat noise(x.w, 2);
        for (int i = 0; i < noise.w * noise.h; i++)
        {
            noise[i] = rand() / (float)RAND_MAX * noise_scale_w;
        }

        ncnn::Extractor ex = dp.create_extractor();

        ex.input("in0", x);
        ex.input("in1", noise);
        ex.input("in2", g);

        ex.extract("out0", logw);
    }

    // path attention
    ncnn::Mat z_p;
    {
        path_attention(logw, m_p, logs_p, noise_scale, length_scale, z_p);
    }

    // flow
    ncnn::Mat z;
    {
        ncnn::Net flow;
        flow.opt.use_vulkan_compute = true;
        flow.load_param("en_flow.ncnn.param");
        flow.load_model("en_flow.ncnn.bin");

        ncnn::Extractor ex = flow.create_extractor();

        ex.input("in0", z_p);
        ex.input("in1", g);

        ex.extract("out0", z);
    }

    // dec
    ncnn::Mat o;
    {
        ncnn::Net dec;
        dec.opt.use_vulkan_compute = true;
        dec.load_param("en_dec.ncnn.param");
        dec.load_model("en_dec.ncnn.bin");

        ncnn::Extractor ex = dec.create_extractor();

        ex.input("in0", z);
        ex.input("in1", g);

        ex.extract("out0", o);
    }

    // normalize and clip
    {
        float volume = 1.f;
        float absmax = 0.f;
        for (int i = 0; i < o.w; i++)
        {
            absmax = std::max(absmax, fabs(o[i]));
        }
        if (absmax > 1e-8)
        {
            for (int i = 0; i < o.w; i++)
            {
                float v = o[i] / absmax * volume;
                v = std::min(std::max(v, -1.f), 1.f);
                o[i] = v;
            }
        }
    }

    // 16bit pcm
    {
        pcm.resize(o.w);
        for (int i = 0; i < o.w; i++)
        {
            pcm[i] = (short)(o[i] * 32767);
        }
    }

    return 0;
}

static void save_pcm_to_wav(const char* path, const short* pcm, int num_samples, int sample_rate)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return;

    // write wav header
    {
        int16_t num_channels = 1;
        int16_t bits_per_sample = 16;
        int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
        int16_t block_align = num_channels * bits_per_sample / 8;
        int32_t data_chunk_size = num_samples * num_channels * bits_per_sample / 8;
        int32_t chunk_size = 36 + data_chunk_size;

        // RIFF header
        fwrite("RIFF", 1, 4, f);
        fwrite(&chunk_size, 4, 1, f);
        fwrite("WAVE", 1, 4, f);

        // fmt subchunk
        fwrite("fmt ", 1, 4, f);
        int32_t subchunk1_size = 16;
        int16_t audio_format = 1; // PCM
        fwrite(&subchunk1_size, 4, 1, f);
        fwrite(&audio_format, 2, 1, f);
        fwrite(&num_channels, 2, 1, f);
        fwrite(&sample_rate, 4, 1, f);
        fwrite(&byte_rate, 4, 1, f);
        fwrite(&block_align, 2, 1, f);
        fwrite(&bits_per_sample, 2, 1, f);

        // data subchunk
        fwrite("data", 1, 4, f);
        fwrite(&data_chunk_size, 4, 1, f);
    }

    fwrite(pcm, sizeof(short), num_samples, f);
    fclose(f);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [sentences] [speaker id 0~903] [out path]\n", argv[0]);
        fprintf(stderr, "       %s \"Hello World\" 0 out.wav\n", argv[0]);
        fprintf(stderr, "       %s \"Happy New Year\" 123 out.wav\n", argv[0]);
        return 0;
    }

    const char* text = argv[1];
    const int speaker_id = atoi(argv[2]);
    const char* outpath = argv[3];

    std::vector<short> pcm;
    tts_piper(text, speaker_id, pcm);

    // "sample_rate": 22050
    save_pcm_to_wav(outpath, pcm.data(), pcm.size(), 22050);

    return 0;
}
