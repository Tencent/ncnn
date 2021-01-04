## current model load api
### Cons
#### long and awful code
#### two functions
#### deal float32 float16 quantized-u8
#### deal alignment size
```cpp
#if NCNN_STDIO
int Convolution::load_model(FILE* binfp)
{
    int nread;

    union
    {
        struct
        {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } flag_struct;

    nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "Convolution read flag_struct failed %d\n", nread);
        return -1;
    }

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    weight_data.create(weight_data_size);
    if (weight_data.empty())
        return -100;

    if (flag_struct.tag == 0x01306B47)
    {
        // half-precision weight data
        int align_weight_data_size = alignSize(weight_data_size * sizeof(unsigned short), 4);
        std::vector<unsigned short> float16_weights;
        float16_weights.resize(align_weight_data_size);
        nread = fread(float16_weights.data(), align_weight_data_size, 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Convolution read float16_weights failed %d\n", nread);
            return -1;
        }

        weight_data = Mat::from_float16(float16_weights.data(), weight_data_size);
        if (weight_data.empty())
            return -100;
    }
    else if (flag != 0)
    {
        // quantized weight data
        float quantization_value[256];
        nread = fread(quantization_value, 256 * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Convolution read quantization_value failed %d\n", nread);
            return -1;
        }

        int align_weight_data_size = alignSize(weight_data_size * sizeof(unsigned char), 4);
        std::vector<unsigned char> index_array;
        index_array.resize(align_weight_data_size);
        nread = fread(index_array.data(), align_weight_data_size, 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Convolution read index_array failed %d\n", nread);
            return -1;
        }

        float* weight_data_ptr = weight_data;
        for (int i = 0; i < weight_data_size; i++)
        {
            weight_data_ptr[i] = quantization_value[ index_array[i] ];
        }
    }
    else if (flag_struct.f0 == 0)
    {
        // raw weight data
        nread = fread(weight_data, weight_data_size * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Convolution read weight_data failed %d\n", nread);
            return -1;
        }
    }

    if (bias_term)
    {
        bias_data.create(num_output);
        if (bias_data.empty())
            return -100;
        nread = fread(bias_data, num_output * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Convolution read bias_data failed %d\n", nread);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STDIO

int Convolution::load_model(const unsigned char*& mem)
{
    union
    {
        struct
        {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } flag_struct;

    memcpy(&flag_struct, mem, sizeof(flag_struct));
    mem += sizeof(flag_struct);

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    if (flag_struct.tag == 0x01306B47)
    {
        // half-precision weight data
        weight_data = Mat::from_float16((unsigned short*)mem, weight_data_size);
        mem += alignSize(weight_data_size * sizeof(unsigned short), 4);
        if (weight_data.empty())
            return -100;
    }
    else if (flag != 0)
    {
        // quantized weight data
        const float* quantization_value = (const float*)mem;
        mem += 256 * sizeof(float);

        const unsigned char* index_array = (const unsigned char*)mem;
        mem += alignSize(weight_data_size * sizeof(unsigned char), 4);

        weight_data.create(weight_data_size);
        if (weight_data.empty())
            return -100;
        float* weight_data_ptr = weight_data;
        for (int i = 0; i < weight_data_size; i++)
        {
            weight_data_ptr[i] = quantization_value[ index_array[i] ];
        }
    }
    else if (flag_struct.f0 == 0)
    {
        // raw weight data
        weight_data = Mat(weight_data_size, (float*)mem);
        mem += weight_data_size * sizeof(float);
    }

    if (bias_term)
    {
        bias_data = Mat(num_output, (float*)mem);
        mem += num_output * sizeof(float);
    }

    return 0;
}
```

## new model load api proposed
### Pros
#### clean and simple api
#### element type detection
```cpp
int Convolution::load_model(const ModelBin& mb)
{
    // auto detect element type
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        // certain type specified
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}
```
