// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// convert openai-whisper checkpoints to ncnn models

// convert vocab.json to simple vocab.txt

// whisper speech recognition implemented with ncnn library

#include "net.h"
#include "layer.h"
#include "layer_type.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <vector>

static std::vector<short> load_wav_samples(const char* wavpath)
{
    std::vector<short> samples;

    FILE* fp = fopen(wavpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "open %s failed\n", wavpath);
        return samples;
    }

    const int wav_header_bytes = 44;

    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);

    samples.resize((len - wav_header_bytes) / sizeof(short));

    rewind(fp);

    fseek(fp, wav_header_bytes, SEEK_SET);

    fread(samples.data(), 1, len - wav_header_bytes, fp);

    fclose(fp);

    return samples;
}

// tokenizer for handling text tokens
class Tokenizer
{
public:
    // predefined tokens in whisper
    enum
    {
        LangEn = 50259,
        LangYue = 50358,
        NoSpeech = 50363
    };

    std::vector<std::string> reverse_vocab;

    uint8_t byte_decoder[512]; // unicode code point to byte value

    // generate byte decoder for tokenization
    void generate_byte_decoder()
    {
        // initialize array to 0
        memset(byte_decoder, 0, 512 * sizeof(uint8_t));

        // define function to check if char is in "printable" range
        auto is_printable = [](int b) {
            return (b >= '!' && b <= '~')    // '!' to '~'
                || (b >= 161 && b <= 172)    // '¡' to '¬'
                || (b >= 174 && b <= 255);   // '®' to 'ÿ'
        };

        // handle "printable" characters
        // for these chars, key and value are the same
        for (int b = 0; b < 256; ++b)
        {
            if (is_printable(b))
            {
                byte_decoder[b] = static_cast<uint8_t>(b);
            }
        }

        // handle remaining characters
        // for these chars, key starts from 256 and increments
        int n = 0;
        for (int b = 0; b < 256; ++b)
        {
            if (!is_printable(b))
            {
                byte_decoder[256 + n] = static_cast<uint8_t>(b);
                n++;
            }
        }
    }

    // convert utf-8 string to code points
    std::vector<uint32_t> utf8_to_codepoints(const std::string& s) const
    {
        std::vector<uint32_t> codepoints;
        for (size_t i = 0; i < s.length();)
        {
            uint32_t cp = 0;
            int len = 0;
            unsigned char c = s[i];

            if (c < 0x80) // 1-byte
            {
                cp = c;
                len = 1;
            }
            else if ((c & 0xE0) == 0xC0) // 2-byte
            {
                cp = ((s[i] & 0x1F) << 6) | (s[i + 1] & 0x3F);
                len = 2;
            }
            else if ((c & 0xF0) == 0xE0) // 3-byte
            {
                cp = ((s[i] & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
                len = 3;
            }
            else if ((c & 0xF8) == 0xF0) // 4-byte
            {
                cp = ((s[i] & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
                len = 4;
            }
            else
            {
                // invalid utf-8 start byte, skip
                i++;
                continue;
            }
            codepoints.push_back(cp);
            i += len;
        }
        return codepoints;
    }

    bool load(const char* vocab_path)
    {
        // generate decoder when loading
        generate_byte_decoder();

        {
            FILE* fp = fopen(vocab_path, "rb");
            if (!fp)
            {
                fprintf(stderr, "open vocab.txt failed\n");
                return false;
            }

            char line[256];
            while (!feof(fp))
            {
                char* s = fgets(line, 255, fp);
                if (!s)
                    break;

                int vocab_len = strlen(line);
                if (vocab_len > 1)
                {
                    // drop the tail newline
                    vocab_len -= 1;
                }

                reverse_vocab.push_back(std::string(line, vocab_len));
            }

            fclose(fp);
        }

        return true;
    }

    // decode token ids to text
    std::string decode(const std::vector<int>& tokens) const
    {
        std::string outstring;
        bool in_timestamp = false;

        // step 1: concatenate token ids to a string with special unicode characters
        std::string text_buffer;
        for (int token_id : tokens)
        {
            if (token_id < 50257)
            {
                text_buffer += reverse_vocab[token_id];
                continue;
            }

            // handle timestamp tokens (50364-51864)
            // https://huggingface.co/openai/whisper-tiny/blob/main/tokenizer_config.json
            if (token_id >= 50364 && token_id <= 51864)
            {
                int timestamp = (token_id - 50364) * 2;

                char tmp[256];
                sprintf(tmp, " [%d.%02d] ", timestamp / 100, timestamp % 100);

                if (in_timestamp)
                {
                    // step 2: translate the special string back to original byte stream
                    std::vector<uint32_t> codepoints = utf8_to_codepoints(text_buffer);

                    std::vector<uint8_t> byte_sequence;
                    for (uint32_t cp : codepoints)
                    {
                        byte_sequence.push_back(byte_decoder[cp]);
                    }

                    std::string s(byte_sequence.begin(), byte_sequence.end());

                    text_buffer.clear();

                    outstring += s;
                    outstring += tmp;
                    outstring += "\n";

                    in_timestamp = false;
                }
                else
                {
                    outstring += tmp;
                    in_timestamp = true;
                }
            }

            // ignore functional/special tokens
            // <|endoftext|> and higher are special tokens
        }

        if (!text_buffer.empty())
        {
            // step 2: translate the special string back to original byte stream
            std::vector<uint32_t> codepoints = utf8_to_codepoints(text_buffer);

            std::vector<uint8_t> byte_sequence;
            for (uint32_t cp : codepoints)
            {
                byte_sequence.push_back(byte_decoder[cp]);
            }

            std::string s(byte_sequence.begin(), byte_sequence.end());

            outstring += s;
        }

        return outstring;
    }
};

// result class for beam search
class Result
{
public:
    std::vector<int> ids;
    float score;

    std::vector<ncnn::Mat> kvcache;
};

// main whisper implementation class
class Whisper
{
public:
    int load();

    int detect_lang(const std::vector<short>& samples, std::string& lang) const;
    int transcribe(const std::vector<short>& samples, std::string& text) const;

protected:
    int extract_fbank_feature(const std::vector<short>& samples, ncnn::Mat& input_features) const;
    int run_encoder(const ncnn::Mat& input_features, ncnn::Mat& encoder_states) const;
    int run_decoder_prefill(const std::vector<int>& tokens, const ncnn::Mat& encoder_states, ncnn::Mat& last_logits, std::vector<ncnn::Mat>& out_kvcache) const;
    int run_decoder_step(const std::vector<int>& tokens, const ncnn::Mat& encoder_states, ncnn::Mat& last_logits, const std::vector<ncnn::Mat>& kvcache, std::vector<ncnn::Mat>& out_kvcache) const;

protected:
    ncnn::Net fbank;

    ncnn::Net encoder;

    ncnn::Net embed_token;
    ncnn::Net embed_position;
    ncnn::Net decoder;

    ncnn::Net proj_out;

    Tokenizer tokenizer;

protected:
    std::vector<int> kv_cache_indexes;
    std::vector<int> out_kv_cache_indexes;
};

int Whisper::load()
{
    // configure options for networks
    fbank.opt.use_fp16_packed = false;
    fbank.opt.use_fp16_storage = false;
    fbank.opt.use_fp16_arithmetic = false;

    encoder.opt.use_fp16_packed = false;
    encoder.opt.use_fp16_storage = false;
    encoder.opt.use_fp16_arithmetic = false;

    decoder.opt.use_fp16_packed = false;
    decoder.opt.use_fp16_storage = false;
    decoder.opt.use_fp16_arithmetic = false;

    proj_out.opt.use_fp16_packed = false;
    proj_out.opt.use_fp16_storage = false;
    proj_out.opt.use_fp16_arithmetic = false;

    fbank.load_param("whisper_tiny_fbank.ncnn.param");
    fbank.load_model("whisper_tiny_fbank.ncnn.bin");

    encoder.load_param("whisper_tiny_encoder.ncnn.param");
    encoder.load_model("whisper_tiny_encoder.ncnn.bin");

    embed_token.load_param("whisper_tiny_embed_token.ncnn.param");
    embed_token.load_model("whisper_tiny_embed_token.ncnn.bin");

    embed_position.load_param("whisper_tiny_embed_position.ncnn.param");
    embed_position.load_model("whisper_tiny_embed_position.ncnn.bin");

    decoder.load_param("whisper_tiny_decoder.ncnn.param");
    decoder.load_model("whisper_tiny_decoder.ncnn.bin");

    proj_out.load_param("whisper_tiny_proj_out.ncnn.param");
    proj_out.load_model("whisper_tiny_proj_out.ncnn.bin");

    tokenizer.load("vocab.txt");

    // resolve kv cache blob indexes
    for (size_t i = 0; i < decoder.layers().size(); i++)
    {
        const ncnn::Layer* mha = decoder.layers()[i];
        if (mha->typeindex != ncnn::LayerType::MultiHeadAttention)
            continue;

        const size_t input_count = mha->bottoms.size();
        const size_t output_count = mha->tops.size();

        if (output_count == 3)
        {
            kv_cache_indexes.push_back(mha->bottoms[input_count - 2]);
            kv_cache_indexes.push_back(mha->bottoms[input_count - 1]);
            out_kv_cache_indexes.push_back(mha->tops[output_count - 2]);
            out_kv_cache_indexes.push_back(mha->tops[output_count - 1]);
        }
    }

    return 0;
}

// apply log_softmax to matrix in-place
static void log_softmax_inplace(ncnn::Mat& m)
{
    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Softmax");
    ncnn::ParamDict pd;
    pd.set(0, 0);
    op->load_param(pd);
    op->forward_inplace(m, opt);
    delete op;

    // apply log
    for (int i = 0; i < m.w; i++)
    {
        m[i] = logf(m[i]);
    }
}

int Whisper::detect_lang(const std::vector<short>& samples, std::string& lang) const
{
    std::vector<int> ids = {50258}; // SOT token

    ncnn::Mat input_features;
    extract_fbank_feature(samples, input_features);

    const int seqlen = input_features.w;

    ncnn::Mat encoder_states;
    run_encoder(input_features, encoder_states);

    ncnn::Mat logits;
    std::vector<ncnn::Mat> out_kvcache;
    run_decoder_prefill(ids, encoder_states, logits, out_kvcache);

    log_softmax_inplace(logits);

    // find the language token with highest probability
    std::vector<std::pair<float, int>> vec;
    for (int j = Tokenizer::LangEn; j <= Tokenizer::LangYue; j++)
    {
        vec.emplace_back(logits[j], j);
    }
    vec.emplace_back(logits[Tokenizer::NoSpeech], Tokenizer::NoSpeech);
    auto max_elem = std::max_element(vec.begin(), vec.end(), std::less<std::pair<float, int>>());
    int max_token_id = max_elem->second;
    auto max_token = tokenizer.reverse_vocab[max_token_id];

    if (max_token_id >= Tokenizer::LangEn && max_token_id <= Tokenizer::LangYue)
    {
        lang = max_token.substr(2, max_token.size() - 4);
    }
    else if (max_token_id == Tokenizer::NoSpeech)
    {
        lang = "no_speech";
    }
    else
    {
        lang = "unknown";
    }

    return 0;
}

int Whisper::transcribe(const std::vector<short>& samples, std::string& text) const
{
    // initialize with prompt tokens
    // https://huggingface.co/openai/whisper-tiny/blob/main/tokenizer_config.json
    std::vector<int> ids = {50258, 50260, 50359, 50363}; // SOT, zh, transcribe, notimestamps

    ncnn::Mat input_features;
    extract_fbank_feature(samples, input_features);

    const int seqlen = input_features.w;

    ncnn::Mat encoder_states;
    run_encoder(input_features, encoder_states);

    const int eot = 50257;
    const int beam_size = 5;
    const int max_candidates = 5;

    std::vector<Result> finished_beams;

    std::vector<Result> beams(1);
    beams[0].ids = ids;
    beams[0].score = 0.f;

    int step = 0;

    // beam search loop
    for (;;)
    {
        std::vector<Result> candidates;

        for (size_t i = 0; i < beams.size(); i++)
        {
            const Result& beam = beams[i];

            ncnn::Mat logits;
            std::vector<ncnn::Mat> out_kvcache;
            if (step == 0)
            {
                run_decoder_prefill(beam.ids, encoder_states, logits, out_kvcache);
            }
            else
            {
                run_decoder_step(beam.ids, encoder_states, logits, beam.kvcache, out_kvcache);
            }

            log_softmax_inplace(logits);

            // get topk candidates
            const int topk = 5;
            std::vector<std::pair<float, int> > vec(logits.w);
            for (int j = 0; j < logits.w; j++)
            {
                vec[j] = std::make_pair(logits[j], j);
            }
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<float, int> >());

            for (int j = 0; j < topk; j++)
            {
                int next_id = vec[j].second;
                float next_id_score = vec[j].first;

                Result candidate;
                candidate.ids = beam.ids;
                candidate.ids.push_back(next_id);
                candidate.score = beam.score + next_id_score;
                candidate.kvcache = out_kvcache;

                candidates.push_back(candidate);
            }
        }

        // sort candidates by score
        std::sort(candidates.begin(), candidates.end(), [](const Result& a, const Result& b) { return a.score > b.score; });

        beams.clear();
        for (size_t i = 0; i < candidates.size(); i++)
        {
            const Result& candidate = candidates[i];

            if (candidate.ids.back() == eot)
            {
                finished_beams.push_back(candidate);
            }
            else
            {
                beams.push_back(candidate);
            }
        }

        if (beams.size() > beam_size)
        {
            beams.resize(beam_size);
        }

        step++;

        if (beams.empty())
        {
            break;
        }

        if (finished_beams.size() >= max_candidates)
        {
            break;
        }
    }

    if (finished_beams.empty())
    {
        // no results
        return 0;
    }

    // find the best result based on average score
    int max_avg_score_index = 0;
    float max_avg_score = -FLT_MAX;
    for (size_t i = 0; i < finished_beams.size(); i++)
    {
        const Result& result = finished_beams[i];
        float avg_score = result.score / result.ids.size();
        if (avg_score > max_avg_score)
        {
            max_avg_score_index = (int)i;
            max_avg_score = avg_score;
        }
    }

    const Result& best_result = finished_beams[max_avg_score_index];

    text = tokenizer.decode(best_result.ids);

    return 0;
}

int Whisper::extract_fbank_feature(const std::vector<short>& samples, ncnn::Mat& input_features) const
{
    const int samples_size = (int)samples.size();

    // pad to 480000, normalize samples to -1~1
    ncnn::Mat waveform(480000);
    waveform.fill(0.f);
    {
        for (int i = 0; i < samples_size; i++)
        {
            waveform[i] = samples[i] / 32768.0f;
        }
    }

    ncnn::Extractor ex = fbank.create_extractor();

    ex.input("in0", waveform);

    ex.extract("out0", input_features);

    // drop the last frame
    {
        ncnn::Mat input_features_3k(input_features.w - 1, input_features.h);
        for (int i = 0; i < input_features.h; i++)
        {
            memcpy(input_features_3k.row(i), input_features.row(i), (input_features.w - 1) * sizeof(float));
        }
        input_features = input_features_3k;
    }

    return 0;
}

int Whisper::run_encoder(const ncnn::Mat& input_features, ncnn::Mat& encoder_states) const
{
    ncnn::Extractor ex = encoder.create_extractor();

    ex.input("in0", input_features);

    ex.extract("out0", encoder_states);

    return 0;
}

int Whisper::run_decoder_prefill(const std::vector<int>& tokens, const ncnn::Mat& encoder_states, ncnn::Mat& last_logits, std::vector<ncnn::Mat>& out_kvcache) const
{
    const int dst_seqlen = tokens.size();

    // token embedding
    ncnn::Mat token_embeds;
    {
        ncnn::Mat input_tokens(dst_seqlen);
        int* p = input_tokens;
        memcpy(p, tokens.data(), tokens.size() * sizeof(int));

        ncnn::Extractor ex = embed_token.create_extractor();
        ex.input("in0", input_tokens);
        ex.extract("out0", token_embeds);
    }

    // position embedding
    ncnn::Mat position_embeds;
    {
        ncnn::Mat input_positions(dst_seqlen);
        int* p = input_positions;
        for (int i = 0; i < dst_seqlen; i++)
        {
            p[i] = i;
        }

        ncnn::Extractor ex = embed_position.create_extractor();
        ex.input("in0", input_positions);
        ex.extract("out0", position_embeds);
    }

    // input embedding = token + position
    ncnn::Mat input_embeds;
    {
        input_embeds.create_like(token_embeds);
        for (int i = 0; i < input_embeds.total(); i++)
        {
            input_embeds[i] = token_embeds[i] + position_embeds[i];
        }
    }

    // create attention mask (causal mask)
    ncnn::Mat attention_mask(dst_seqlen, dst_seqlen);
    attention_mask.fill(0.f);
    for (int i = 0; i < dst_seqlen; i++)
    {
        for (int j = i + 1; j < dst_seqlen; j++)
        {
            attention_mask.row(i)[j] = -INFINITY;
        }
    }

    ncnn::Mat output_states;
    {
        ncnn::Extractor ex = decoder.create_extractor();
        ex.input("in0", input_embeds);
        ex.input("in1", encoder_states);
        ex.input("in2", attention_mask);

        out_kvcache.resize(out_kv_cache_indexes.size());
        for (size_t i = 0; i < out_kv_cache_indexes.size(); i++)
        {
            ex.extract(out_kv_cache_indexes[i], out_kvcache[i], 1);
        }

        ex.extract("out0", output_states);
    }

    // get last token's state for next token prediction
    ncnn::Mat last_state = output_states.row_range(dst_seqlen - 1, 1).clone();
    {
        ncnn::Extractor ex = proj_out.create_extractor();
        ex.input("in0", last_state);
        ex.extract("out0", last_logits);
    }

    last_logits = last_logits.reshape(last_logits.w);

    return 0;
}

int Whisper::run_decoder_step(const std::vector<int>& tokens, const ncnn::Mat& encoder_states, ncnn::Mat& last_logits, const std::vector<ncnn::Mat>& kvcache, std::vector<ncnn::Mat>& out_kvcache) const
{
    const int token_id = tokens.back();
    const int dst_seqlen = 1;

    // token embedding
    ncnn::Mat token_embeds;
    {
        ncnn::Mat input_tokens(dst_seqlen);
        ((int*)input_tokens)[0] = token_id;

        ncnn::Extractor ex = embed_token.create_extractor();
        ex.input("in0", input_tokens);
        ex.extract("out0", token_embeds);
    }

    // position embedding
    ncnn::Mat position_embeds;
    {
        ncnn::Mat input_positions(dst_seqlen);
        ((int*)input_positions)[0] = tokens.size() - 1;

        ncnn::Extractor ex = embed_position.create_extractor();
        ex.input("in0", input_positions);
        ex.extract("out0", position_embeds);
    }

    // input embedding = token + position
    ncnn::Mat input_embeds;
    {
        input_embeds.create_like(token_embeds);
        for (int i = 0; i < input_embeds.total(); i++)
        {
            input_embeds[i] = token_embeds[i] + position_embeds[i];
        }
    }

    // single token doesn't need attention mask
    ncnn::Mat attention_mask(dst_seqlen, dst_seqlen);
    attention_mask.fill(0.f);

    ncnn::Mat output_states;
    {
        ncnn::Extractor ex = decoder.create_extractor();
        ex.input("in0", input_embeds);
        ex.input("in1", encoder_states);
        ex.input("in2", attention_mask);

        // pass in kv cache from previous steps
        for (size_t i = 0; i < kv_cache_indexes.size(); i++)
        {
            ex.input(kv_cache_indexes[i], kvcache[i]);
        }

        // extract updated kv cache
        out_kvcache.resize(out_kv_cache_indexes.size());
        for (size_t i = 0; i < out_kv_cache_indexes.size(); i++)
        {
            ex.extract(out_kv_cache_indexes[i], out_kvcache[i], 1);
        }

        ex.extract("out0", output_states);
    }

    // get last token's state for prediction
    ncnn::Mat last_state = output_states.row_range(dst_seqlen - 1, 1).clone();
    {
        ncnn::Extractor ex = proj_out.create_extractor();
        ex.input("in0", last_state);
        ex.extract("out0", last_logits);
    }

    last_logits = last_logits.reshape(last_logits.w);

    return 0;
}

int main(int argc, char** argv)
{
    const char* wavpath = argc >= 2 ? argv[1] : "speech.wav";

    std::vector<short> samples;
    samples = load_wav_samples(wavpath);

    const int samples_size = (int)samples.size();
    if (samples_size == 0)
    {
        fprintf(stderr, "load wav failed\n");
        return -1;
    }

    Whisper whisper;
    whisper.load();

    // detect language first
    std::string lang;
    whisper.detect_lang(samples, lang);
    fprintf(stderr, "lang = %s\n", lang.c_str());

    // transcribe audio to text
    std::string text;
    whisper.transcribe(samples, text);
    fprintf(stderr, "text = %s\n", text.c_str());

    return 0;
}
