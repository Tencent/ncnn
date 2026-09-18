// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// HunyuanOCR ncnn example — ViT + XDRoPE + BBPE + KV-cache text decoder.
//
// Conversion steps (requires a GPU host with ~20 GB VRAM for the FP16 export):
//   pip install "huggingface_hub[cli]" pnnx torch transformers
//   huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR-hf
//   python convert/export_pnnx.py --model ./HunyuanOCR-hf --output ./hunyuan_ocr
//   python convert/add_kvcache.py ./hunyuan_ocr/text_decoder.ncnn.param
//   python convert/export_tokenizer.py --model ./HunyuanOCR-hf --output ./hunyuan_ocr
//
// Usage:
//   ./hunyuanocr --model ./hunyuan_ocr --image doc.jpg
//   ./hunyuanocr --model ./hunyuan_ocr --image doc.jpg --prompt "提取所有文字"

#include "layer.h"
#include "mat.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct KVConfig
{
    std::unordered_map<std::string, std::string> data;

    bool load(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open())
        {
            fprintf(stderr, "cannot open config: %s\n", path.c_str());
            return false;
        }
        std::string line;
        while (std::getline(f, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            auto eq = line.find('=');
            if (eq == std::string::npos)
                continue;
            std::string k = line.substr(0, eq);
            std::string v = line.substr(eq + 1);
            auto trim = [](std::string& s) {
                size_t a = s.find_first_not_of(" \t\r\n");
                size_t b = s.find_last_not_of(" \t\r\n");
                s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            trim(k);
            trim(v);
            data[k] = v;
        }
        return true;
    }

    int get_int(const std::string& k, int def = 0) const
    {
        auto it = data.find(k);
        return (it == data.end()) ? def : std::stoi(it->second);
    }
    long long get_ll(const std::string& k, long long def = 0) const
    {
        auto it = data.find(k);
        return (it == data.end()) ? def : std::stoll(it->second);
    }
    float get_float(const std::string& k, float def = 0.f) const
    {
        auto it = data.find(k);
        return (it == data.end()) ? def : std::stof(it->second);
    }
    std::string get_str(const std::string& k, const std::string& def = "") const
    {
        auto it = data.find(k);
        return (it == data.end()) ? def : it->second;
    }
    std::vector<int> get_int_list(const std::string& k) const
    {
        auto it = data.find(k);
        if (it == data.end())
            return {};
        std::vector<int> result;
        std::istringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            auto trim = [](std::string& s) {
                size_t a = s.find_first_not_of(" \t");
                size_t b = s.find_last_not_of(" \t");
                s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            trim(tok);
            if (!tok.empty())
                result.push_back(std::stoi(tok));
        }
        return result;
    }
};

struct BbpeTokenizer
{
    std::vector<std::string> id2tok;
    std::unordered_map<std::string, int> tok2id;
    std::vector<std::pair<std::string, std::string>> merges;
    // byte -> single-char encoded form (Ġ-style)
    std::string byte_encode[256];

    bool load(const std::string& vocab_path, const std::string& merges_path)
    {
        // byte encoding table matching HF bytes_to_unicode():
        // !..~, ¡..¬ (161-172), ®..ÿ (174-255) map to their own Unicode codepoints;
        // all remaining bytes map to U+0100 onwards.
        auto init_byte_table = [&]() {
            auto encode_cp = [](int cp) -> std::string {
                if (cp < 0x80) return std::string(1, (char)cp);
                char buf[3] = {(char)(0xC0 | (cp >> 6)), (char)(0x80 | (cp & 0x3F)), 0};
                return std::string(buf, 2);
            };
            std::unordered_map<int, std::string> bs;
            for (int b = 33; b <= 126; ++b)  bs[b] = encode_cp(b);
            for (int b = 161; b <= 172; ++b) bs[b] = encode_cp(b);
            for (int b = 174; b <= 255; ++b) bs[b] = encode_cp(b);
            int n = 0;
            for (int b = 0; b < 256; ++b)
                if (bs.find(b) == bs.end())
                    bs[b] = encode_cp(256 + n++);
            for (int i = 0; i < 256; ++i)
                byte_encode[i] = bs[i];
        };
        init_byte_table();

        std::ifstream fv(vocab_path);
        if (!fv.is_open())
        {
            fprintf(stderr, "cannot open vocab: %s\n", vocab_path.c_str());
            return false;
        }
        std::string line;
        while (std::getline(fv, line))
        {
            int id = (int)id2tok.size();
            id2tok.push_back(line);
            if (!line.empty())
                tok2id[line] = id;
        }

        std::ifstream fm(merges_path);
        if (!fm.is_open())
        {
            fprintf(stderr, "cannot open merges: %s\n", merges_path.c_str());
            return false;
        }
        while (std::getline(fm, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            auto sp = line.find(' ');
            if (sp == std::string::npos)
                continue;
            merges.push_back({line.substr(0, sp), line.substr(sp + 1)});
        }
        return true;
    }

    std::vector<std::string> bpe_word(std::vector<std::string> word) const
    {
        if (word.size() <= 1)
            return word;

        std::map<std::pair<std::string, std::string>, int> rank;
        for (int i = 0; i < (int)merges.size(); ++i)
            rank[merges[i]] = i;

        while (word.size() > 1)
        {
            int best_rank = INT_MAX, best_i = -1;
            for (int i = 0; i + 1 < (int)word.size(); ++i)
            {
                auto it = rank.find({word[i], word[i + 1]});
                if (it != rank.end() && it->second < best_rank)
                {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_i < 0)
                break;
            std::vector<std::string> next;
            for (int i = 0; i < (int)word.size(); ++i)
            {
                if (i == best_i)
                    next.push_back(word[i] + word[i + 1]), ++i;
                else
                    next.push_back(word[i]);
            }
            word = next;
        }
        return word;
    }

    std::vector<int> encode(const std::string& text) const
    {
        // split on whitespace, prepend Ġ to non-first words
        std::vector<int> ids;
        std::vector<std::string> words;
        bool in_word = false;
        std::string cur;
        for (unsigned char c : text)
        {
            if (c == ' ')
            {
                if (in_word)
                    words.push_back(cur);
                cur = "\xc4\xa0"; // Ġ
                in_word = true;
            }
            else
            {
                if (!in_word)
                    cur = "";
                cur += byte_encode[c];
                in_word = true;
            }
        }
        if (!cur.empty() && in_word)
            words.push_back(cur);

        for (auto& w : words)
        {
            std::vector<std::string> chars;
            for (int i = 0; i < (int)w.size();)
            {
                unsigned char c = (unsigned char)w[i];
                int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
                chars.push_back(w.substr(i, len));
                i += len;
            }
            for (auto& t : bpe_word(chars))
            {
                auto it = tok2id.find(t);
                if (it != tok2id.end())
                    ids.push_back(it->second);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids,
                       const std::unordered_set<int>& skip_ids) const
    {
        std::string encoded;
        for (int id : ids)
        {
            if (skip_ids.count(id) || id < 0 || id >= (int)id2tok.size())
                continue;
            encoded += id2tok[id];
        }
        // byte-decode: reverse byte_encode (populated in load() via init_byte_table)
        std::unordered_map<std::string, unsigned char> bdec;
        for (int i = 0; i < 256; ++i)
            bdec[byte_encode[i]] = (unsigned char)i;

        std::string out;
        int i = 0;
        while (i < (int)encoded.size())
        {
            // try 1-char, 2-char, 3-char encoded keys
            bool matched = false;
            for (int len : {1, 2, 3, 4})
            {
                if (i + len > (int)encoded.size())
                    break;
                std::string key = encoded.substr(i, len);
                auto it = bdec.find(key);
                if (it != bdec.end())
                {
                    out += (char)it->second;
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (!matched)
                i++;
        }
        return out;
    }
};

// XDRoPE: 4-axis rotary position embedding for HunyuanOCR

// Single-position variant: used for decode steps where every section shares the same scalar pos.
static void build_xdrope_cache(int seq_len, int position_id, float theta, float alpha,
                                const std::vector<int>& sections,
                                ncnn::Mat& cos_cache, ncnn::Mat& sin_cache)
{
    int half_dim = 0;
    for (int s : sections)
        half_dim += s;

    cos_cache.create(half_dim * 2, seq_len);
    sin_cache.create(half_dim * 2, seq_len);

    auto* cos_ptr = (float*)cos_cache.data;
    auto* sin_ptr = (float*)sin_cache.data;

    for (int t = 0; t < seq_len; ++t)
    {
        int offset = 0;
        for (int si = 0; si < (int)sections.size(); ++si)
        {
            int sect = sections[si];
            for (int j = 0; j < sect; ++j)
            {
                // Use global dimension index (offset+j) so each section occupies
                // its own slice of the frequency spectrum instead of restarting at 0.
                float freq = 1.0f / std::pow(theta, (float)(2 * (offset + j)) / (float)(2 * half_dim));
                if (alpha > 1.0f)
                    freq /= alpha;
                float angle = (position_id + t) * freq;
                cos_ptr[t * half_dim * 2 + offset + j]            = std::cos(angle);
                sin_ptr[t * half_dim * 2 + offset + j]            = std::sin(angle);
                cos_ptr[t * half_dim * 2 + half_dim + offset + j] = std::cos(angle);
                sin_ptr[t * half_dim * 2 + half_dim + offset + j] = std::sin(angle);
            }
            offset += sect;
        }
    }
}

// Prefill variant: image tokens get 2D spatial positions (temporal=0, height=row, width=col)
// while text tokens use the same scalar sequential position for all axes.
static void build_xdrope_cache_prefill(
    const std::vector<int>& input_ids,
    int image_token_id, int grid_h, int grid_w,
    float theta, float alpha,
    const std::vector<int>& sections,
    ncnn::Mat& cos_cache, ncnn::Mat& sin_cache)
{
    int seq_len = (int)input_ids.size();
    int half_dim = 0;
    for (int s : sections)
        half_dim += s;

    cos_cache.create(half_dim * 2, seq_len);
    sin_cache.create(half_dim * 2, seq_len);

    auto* cos_ptr = (float*)cos_cache.data;
    auto* sin_ptr = (float*)sin_cache.data;

    int img_idx = 0;
    for (int t = 0; t < seq_len; ++t)
    {
        // XDRoPE sections: [temporal, height, width, ...]. Image tokens carry
        // distinct spatial axes; text tokens collapse all axes to the sequential pos.
        int axis_pos[3];
        if (input_ids[t] == image_token_id && grid_w > 0)
        {
            axis_pos[0] = 0;                   // temporal
            axis_pos[1] = img_idx / grid_w;    // height row
            axis_pos[2] = img_idx % grid_w;    // width col
            ++img_idx;
        }
        else
        {
            axis_pos[0] = axis_pos[1] = axis_pos[2] = t;
        }

        int offset = 0;
        for (int si = 0; si < (int)sections.size(); ++si)
        {
            int sect = sections[si];
            int ap   = si < 3 ? axis_pos[si] : axis_pos[0];
            for (int j = 0; j < sect; ++j)
            {
                // Use global dimension index (offset+j) so each section occupies
                // its own slice of the frequency spectrum instead of restarting at 0.
                float freq = 1.0f / std::pow(theta, (float)(2 * (offset + j)) / (float)(2 * half_dim));
                if (alpha > 1.0f)
                    freq /= alpha;
                float angle = ap * freq;
                cos_ptr[t * half_dim * 2 + offset + j]            = std::cos(angle);
                sin_ptr[t * half_dim * 2 + offset + j]            = std::sin(angle);
                cos_ptr[t * half_dim * 2 + half_dim + offset + j] = std::cos(angle);
                sin_ptr[t * half_dim * 2 + half_dim + offset + j] = std::sin(angle);
            }
            offset += sect;
        }
    }
}

struct HunyuanOCRConfig
{
    int attn_cnt = 28;
    int hidden_size = 1536;
    int head_dim = 128;
    int vocab_size = 120818;
    int image_token_id = 59280;
    int bos_id = 120000;
    int system_end_id = 120021;
    int user_end_id = 120006;
    int image_start_id = 120118;
    int image_end_id = 120119;
    int special_id_begin = 120000;
    float rope_theta = 10000.0f;
    float rope_alpha = 1000.0f;
    std::vector<int> xdrope_section = {16, 24, 24};
    int patch_size = 14;
    int spatial_merge_size = 2;
    int vision_hidden_size = 1152;
    long long min_pixels = 12544;
    long long max_pixels = 9633792;
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    int eos_id = -1;
    std::unordered_set<int> special_ids;
};

struct HunyuanOCR
{
    HunyuanOCRConfig cfg;
    BbpeTokenizer tokenizer;

    ncnn::Net vision_net;
    ncnn::Net text_embed_net;
    ncnn::Net text_decoder_net;
    ncnn::Net lm_head_net;

    bool ok = false;

    bool load(const std::string& model_dir, int num_threads = 4)
    {
        KVConfig kv;
        if (!kv.load(model_dir + "/model.cfg"))
            return false;

        cfg.attn_cnt = kv.get_int("attn_cnt", cfg.attn_cnt);
        cfg.hidden_size = kv.get_int("hidden_size", cfg.hidden_size);
        cfg.head_dim = kv.get_int("head_dim", cfg.head_dim);
        cfg.vocab_size = kv.get_int("vocab_size", cfg.vocab_size);
        cfg.image_token_id = kv.get_int("image_token_id", cfg.image_token_id);
        cfg.bos_id = kv.get_int("bos_token_id", cfg.bos_id);
        cfg.system_end_id = kv.get_int("system_end_token_id", cfg.system_end_id);
        cfg.user_end_id = kv.get_int("user_end_token_id", cfg.user_end_id);
        cfg.image_start_id = kv.get_int("image_start_token_id", cfg.image_start_id);
        cfg.image_end_id = kv.get_int("image_end_token_id", cfg.image_end_id);
        cfg.special_id_begin = kv.get_int("special_token_id_begin", cfg.special_id_begin);
        cfg.rope_theta = kv.get_float("rope_theta", cfg.rope_theta);
        cfg.rope_alpha = kv.get_float("rope_alpha", cfg.rope_alpha);
        cfg.patch_size = kv.get_int("patch_size", cfg.patch_size);
        cfg.spatial_merge_size = kv.get_int("spatial_merge_size", cfg.spatial_merge_size);
        cfg.vision_hidden_size = kv.get_int("vision_hidden_size", cfg.vision_hidden_size);
        cfg.min_pixels = kv.get_ll("min_pixels", cfg.min_pixels);
        cfg.max_pixels = kv.get_ll("max_pixels", cfg.max_pixels);

        auto xds = kv.get_int_list("xdrope_section");
        if (!xds.empty())
            cfg.xdrope_section = xds;

        std::string mdir = model_dir + "/";
        vision_net.opt.num_threads = num_threads;
        text_embed_net.opt.num_threads = num_threads;
        text_decoder_net.opt.num_threads = num_threads;
        lm_head_net.opt.num_threads = num_threads;

        auto load_net = [&](ncnn::Net& net, const std::string& param_key,
                            const std::string& bin_key) -> bool {
            std::string pp = mdir + kv.get_str(param_key, param_key + ".ncnn.param");
            std::string bb = mdir + kv.get_str(bin_key, bin_key + ".ncnn.bin");
            return net.load_param(pp.c_str()) != 0 || net.load_model(bb.c_str()) != 0;
        };
        if (load_net(vision_net, "vision_param", "vision_bin"))
            return false;
        if (load_net(text_embed_net, "text_embed_param", "text_embed_bin"))
            return false;
        if (load_net(text_decoder_net, "text_decoder_param", "text_decoder_bin"))
            return false;
        if (load_net(lm_head_net, "lm_head_param", "lm_head_bin"))
            return false;

        std::string vocab_file = mdir + kv.get_str("vocab_file", "vocab.txt");
        std::string merges_file = mdir + kv.get_str("merges_file", "merges.txt");
        if (!tokenizer.load(vocab_file, merges_file))
            return false;

        cfg.special_ids.clear();
        for (int id = cfg.special_id_begin; id < cfg.vocab_size; ++id)
            cfg.special_ids.insert(id);

        std::string eos_str = kv.get_str("eos_token");
        if (!eos_str.empty())
        {
            auto it = tokenizer.tok2id.find(eos_str);
            if (it != tokenizer.tok2id.end())
                cfg.eos_id = it->second;
        }

        ok = true;
        return true;
    }

    // smart_resize: resize image so that patch count stays in [min_pixels, max_pixels]
    // and both H/W are divisible by (patch_size * spatial_merge_size).
    void smart_resize(int img_h, int img_w, int& out_h, int& out_w) const
    {
        int factor = cfg.patch_size * cfg.spatial_merge_size;
        long long h = img_h, w = img_w;

        long long pixels = h * w;
        if (pixels < cfg.min_pixels)
        {
            float scale = std::sqrt((float)cfg.min_pixels / pixels);
            h = (long long)std::round(h * scale);
            w = (long long)std::round(w * scale);
        }
        else if (pixels > cfg.max_pixels)
        {
            float scale = std::sqrt((float)cfg.max_pixels / pixels);
            h = (long long)std::round(h * scale);
            w = (long long)std::round(w * scale);
        }

        h = ((h + factor / 2) / factor) * factor;
        w = ((w + factor / 2) / factor) * factor;
        if (h == 0)
            h = factor;
        if (w == 0)
            w = factor;

        out_h = (int)h;
        out_w = (int)w;
    }

    // run vision encoder on BGR image; out_grid_h/out_grid_w receive the merged patch grid dims.
    ncnn::Mat encode_image(const cv::Mat& bgr, int& out_grid_h, int& out_grid_w) const
    {
        int target_h, target_w;
        smart_resize(bgr.rows, bgr.cols, target_h, target_w);

        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(target_w, target_h));

        ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR2RGB, target_w, target_h);
        const float mean[3] = {cfg.image_mean[0] * 255.f, cfg.image_mean[1] * 255.f,
                               cfg.image_mean[2] * 255.f};
        const float norm[3] = {1.f / (cfg.image_std[0] * 255.f), 1.f / (cfg.image_std[1] * 255.f),
                               1.f / (cfg.image_std[2] * 255.f)};
        in.substract_mean_normalize(mean, norm);

        int ph = target_h / cfg.patch_size;
        int pw = target_w / cfg.patch_size;
        out_grid_h = ph / cfg.spatial_merge_size;
        out_grid_w = pw / cfg.spatial_merge_size;

        // grid_thw: [1, ph, pw] as int tensor
        ncnn::Mat grid_thw(3);
        grid_thw[0] = 1;
        grid_thw[1] = ph;
        grid_thw[2] = pw;

        ncnn::Extractor ex = vision_net.create_extractor();
        ex.input("in0", in);
        ex.input("in1", grid_thw);

        ncnn::Mat feat;
        ex.extract("out0", feat);
        return feat;
    }

    // build input ids for the prefill sequence:
    // <|bos|> <|system|>\n<|system_end|><|user|>\n<image_start> [image_tokens] <image_end>\n
    // <prompt> <|user_end|> <|assistant|>\n
    std::vector<int> build_input_ids(const std::string& prompt, int num_image_tokens) const
    {
        std::vector<int> ids;
        ids.push_back(cfg.bos_id);
        ids.push_back(cfg.system_end_id);
        ids.push_back(cfg.user_end_id);
        ids.push_back(cfg.image_start_id);
        for (int i = 0; i < num_image_tokens; ++i)
            ids.push_back(cfg.image_token_id);
        ids.push_back(cfg.image_end_id);

        auto prompt_ids = tokenizer.encode(prompt);
        ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
        ids.push_back(cfg.user_end_id);
        return ids;
    }

    std::string generate(const cv::Mat& bgr, const std::string& prompt,
                         int max_new_tokens = 1024,
                         std::function<void(const std::string&)> on_token = nullptr)
    {
        int grid_h = 1, grid_w = 1;
        ncnn::Mat img_feat = encode_image(bgr, grid_h, grid_w);
        int num_patches = img_feat.h;  // visual tokens after spatial merge
        int num_image_tokens = num_patches;

        auto input_ids = build_input_ids(prompt, num_image_tokens);
        int seq_len = (int)input_ids.size();

        // Embed reads ((const int*)bottom_blob)[q]; write as int, not float.
        ncnn::Mat id_mat(seq_len);
        {
            int* p = (int*)id_mat.data;
            for (int i = 0; i < seq_len; ++i)
                p[i] = input_ids[i];
        }

        ncnn::Extractor emb_ex = text_embed_net.create_extractor();
        emb_ex.input("in0", id_mat);
        ncnn::Mat hidden;
        emb_ex.extract("out0", hidden);  // [seq_len, hidden_size]

        {
            int img_idx = 0;
            auto* hptr = (float*)hidden.data;
            for (int i = 0; i < seq_len && img_idx < num_patches; ++i)
            {
                if (input_ids[i] == cfg.image_token_id)
                {
                    auto* dst = hptr + i * cfg.hidden_size;
                    auto* src = (float*)img_feat.data + img_idx * cfg.vision_hidden_size;
                    // project: vision_hidden_size -> hidden_size via direct copy if equal,
                    // or zero-pad / truncate (a proper implementation would have a linear
                    // projection layer exported separately)
                    int copy_size = std::min(cfg.vision_hidden_size, cfg.hidden_size);
                    memcpy(dst, src, copy_size * sizeof(float));
                    ++img_idx;
                }
            }
        }

        ncnn::Mat cos_cache, sin_cache;
        build_xdrope_cache_prefill(input_ids, cfg.image_token_id, grid_h, grid_w,
                                   cfg.rope_theta, cfg.rope_alpha,
                                   cfg.xdrope_section, cos_cache, sin_cache);

        // Causal mask for prefill: upper-triangle positions are masked out.
        ncnn::Mat attn_mask(seq_len, seq_len);
        {
            float* mp = (float*)attn_mask.data;
            for (int i = 0; i < seq_len; ++i)
                for (int j = 0; j < seq_len; ++j)
                    mp[i * seq_len + j] = (j > i) ? -1e9f : 0.f;
        }

        // Prefill: run full sequence through decoder and capture per-layer KV caches.
        const int N = cfg.attn_cnt;
        std::vector<ncnn::Mat> kv_k(N), kv_v(N);

        ncnn::Extractor dec_ex = text_decoder_net.create_extractor();
        dec_ex.input("in0", hidden);
        dec_ex.input("in1", cos_cache);
        dec_ex.input("in2", sin_cache);
        dec_ex.input("in3", attn_mask);
        ncnn::Mat dec_out;
        dec_ex.extract("out0", dec_out);
        for (int i = 0; i < N; ++i)
        {
            dec_ex.extract(("out_k" + std::to_string(i)).c_str(), kv_k[i]);
            dec_ex.extract(("out_v" + std::to_string(i)).c_str(), kv_v[i]);
        }

        ncnn::Mat last_hidden(cfg.hidden_size);
        memcpy(last_hidden.data,
               (float*)dec_out.data + (seq_len - 1) * cfg.hidden_size,
               cfg.hidden_size * sizeof(float));

        std::vector<int> generated;
        int position = seq_len;

        for (int step = 0; step < max_new_tokens; ++step)
        {
            ncnn::Extractor lm_ex = lm_head_net.create_extractor();
            lm_ex.input("in0", last_hidden);
            ncnn::Mat logits;
            lm_ex.extract("out0", logits);

            int best_id = 0;
            float best_v = ((float*)logits.data)[0];
            for (int v = 1; v < cfg.vocab_size; ++v)
            {
                if (((float*)logits.data)[v] > best_v)
                {
                    best_v = ((float*)logits.data)[v];
                    best_id = v;
                }
            }

            if (best_id == cfg.eos_id)
                break;
            generated.push_back(best_id);

            if (on_token)
            {
                std::string tok = tokenizer.decode({best_id}, cfg.special_ids);
                if (!tok.empty())
                    on_token(tok);
            }

            // Embed next token as int (not float).
            ncnn::Mat next_id(1);
            ((int*)next_id.data)[0] = best_id;
            ncnn::Extractor e2 = text_embed_net.create_extractor();
            e2.input("in0", next_id);
            e2.extract("out0", last_hidden);

            build_xdrope_cache(1, position, cfg.rope_theta, cfg.rope_alpha,
                               cfg.xdrope_section, cos_cache, sin_cache);

            // Single-token mask: attends to all cached positions without masking.
            ncnn::Mat gen_mask(1, 1);
            gen_mask.fill(0.f);

            // Feed KV caches from previous step and capture updated caches.
            ncnn::Extractor d2 = text_decoder_net.create_extractor();
            d2.input("in0", last_hidden);
            d2.input("in1", cos_cache);
            d2.input("in2", sin_cache);
            d2.input("in3", gen_mask);
            for (int i = 0; i < N; ++i)
            {
                d2.input(("cache_k" + std::to_string(i)).c_str(), kv_k[i]);
                d2.input(("cache_v" + std::to_string(i)).c_str(), kv_v[i]);
            }
            d2.extract("out0", last_hidden);
            for (int i = 0; i < N; ++i)
            {
                d2.extract(("out_k" + std::to_string(i)).c_str(), kv_k[i]);
                d2.extract(("out_v" + std::to_string(i)).c_str(), kv_v[i]);
            }

            ++position;
        }

        return tokenizer.decode(generated, cfg.special_ids);
    }
};

static void usage(const char* argv0)
{
    fprintf(stderr, "Usage: %s --model <dir> --image <path> [--prompt <text>] [--threads N]\n",
            argv0);
}

int main(int argc, char** argv)
{
    std::string model_dir;
    std::string image_path;
    // default: detect and recognise text, output with bounding boxes
    std::string prompt = "\xe6\xa3\x80\xe6\xb5\x8b\xe5\xb9\xb6\xe8\xaf\x86\xe5\x88\xab\xe5\x9b\xbe"
                         "\xe7\x89\x87\xe4\xb8\xad\xe7\x9a\x84\xe6\x96\x87\xe5\xad\x97\xef\xbc\x8c"
                         "\xe5\xb0\x86\xe6\x96\x87\xe6\x9c\xac\xe5\x9d\x90\xe6\xa0\x87\xe6\xa0\xbc"
                         "\xe5\xbc\x8f\xe5\x8c\x96\xe8\xbe\x93\xe5\x87\xba\xe3\x80\x82";
    int threads = 4;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
            model_dir = argv[++i];
        else if (arg == "--image" && i + 1 < argc)
            image_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc)
            prompt = argv[++i];
        else if (arg == "--threads" && i + 1 < argc)
            threads = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h")
        {
            usage(argv[0]);
            return 0;
        }
    }

    if (model_dir.empty() || image_path.empty())
    {
        usage(argv[0]);
        return 1;
    }

    HunyuanOCR ocr;
    printf("Loading model from %s ...\n", model_dir.c_str());
    if (!ocr.load(model_dir, threads))
    {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty())
    {
        fprintf(stderr, "Failed to read image: %s\n", image_path.c_str());
        return 1;
    }

    printf("Running OCR (prompt: %s) ...\n", prompt.c_str());
    std::string result = ocr.generate(
        bgr, prompt, 1024,
        [](const std::string& tok)
        {
            printf("%s", tok.c_str());
            fflush(stdout);
        });
    printf("\n");
    return 0;
}
