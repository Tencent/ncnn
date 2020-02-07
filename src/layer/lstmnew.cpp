#include "lstmnew.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(LstmNew)

LstmNew::LstmNew()
{
    one_blob_only = true;
    support_inplace = false;
}

int LstmNew::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);//262144

    return 0;
}

int LstmNew::load_model(const ModelBin& mb)
{
    int size = weight_data_size / num_output / 4;

    // raw weight data
    weight_i_data = mb.load(size, num_output * 4, 0);//256 1024
    if (weight_i_data.empty())
        return -100;

    weight_h_data = mb.load(num_output, num_output * 4, 0);//256 1024
    if (weight_h_data.empty())
        return -100;

    bias_c_data = mb.load(num_output*4, 0);//1024
    if (bias_c_data.empty())
        return -100;


    return 0;
}

int LstmNew::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
   

    size_t elemsize = bottom_blob.elemsize;

    int T = bottom_blob.c;
    int N=bottom_blob.h;
    int size = bottom_blob.w;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);//256
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    // internal cell state
    Mat cell(num_output, 4u, opt.workspace_allocator);//256
    if (cell.empty())
        return -100;
    // 4 x num_output
    Mat gates(num_output,4, 4u, opt.workspace_allocator);//256 4
    if (gates.empty())
        return -100;

 
    top_blob.create(num_output,N, T, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // unroll
    for (int t=0; t<T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
        const bool cont = t > 0;
        const float* x = bottom_blob.channel(t);
        for (int q=0; q<num_output; q++)
        {
            //float h_cont = cont ? hidden[q] : 0.f;

            const float* I_bias_c_data_ptr = (const float*)bias_c_data;
            const float* F_bias_c_data_ptr = (const float*)bias_c_data + num_output;
            const float* O_bias_c_data_ptr = (const float*)bias_c_data + 2 * num_output;
            const float* G_bias_c_data_ptr = (const float*)bias_c_data + 3 * num_output;

            float* gates_data_I = (float*)gates;
            float* gates_data_F = (float*)gates+ num_output;
            float* gates_data_O = (float*)gates+ 2 * num_output;
            float* gates_data_G = (float*)gates+ 3 * num_output;



            // gate I F O G
            const float* weight_h_data_I = (const float*)weight_h_data + weight_h_data.w * q;
            const float* weight_i_data_I = (const float*)weight_i_data + weight_i_data.w * q;
            const float* weight_h_data_F = (const float*)weight_h_data + weight_h_data.w * q + num_output * num_output;
            const float* weight_i_data_F = (const float*)weight_i_data + weight_i_data.w * q + num_output * size;
            const float* weight_h_data_O = (const float*)weight_h_data + weight_h_data.w * q + num_output * num_output * 2;
            const float* weight_i_data_O = (const float*)weight_i_data + weight_i_data.w * q + num_output * size * 2;
            const float* weight_h_data_G = (const float*)weight_h_data + weight_h_data.w * q + num_output * num_output * 3;
            const float* weight_i_data_G = (const float*)weight_i_data + weight_i_data.w * q + num_output * size * 3;

           
            float I = I_bias_c_data_ptr[q];
            float F = F_bias_c_data_ptr[q];
            float O = O_bias_c_data_ptr[q];
            float G = G_bias_c_data_ptr[q];


            for (int i=0; i<size; i++)
            {
                I += x[i] * weight_i_data_I[i];
                F += x[i] * weight_i_data_F[i];
                O += x[i] * weight_i_data_O[i];
                G += x[i] * weight_i_data_G[i];

                I += weight_h_data_I[i] * (!cont ? 0: hidden[i]);
                F += weight_h_data_F[i] * (!cont ? 0: hidden[i]);
                O += weight_h_data_O[i] * (!cont ? 0: hidden[i]);
                G += weight_h_data_G[i] * (!cont ? 0: hidden[i]);
            }

            // for (int i=0; i<num_output; ++i){
            //     I += weight_h_data_I[i] * (!cont ? 0: hidden[i]);
            //     F += weight_h_data_F[i] * (!cont ? 0: hidden[i]);
            //     O += weight_h_data_O[i] * (!cont ? 0: hidden[i]);
            //     G += weight_h_data_G[i] * (!cont ? 0: hidden[i]);

            // }
            gates_data_I[q] = I;
            gates_data_F[q] = F;
            gates_data_O[q] = O;
            gates_data_G[q] = G;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.channel(t);
        for (int q=0; q<num_output; q++)
        {
            float* gates_data_I = (float*)gates;
            float* gates_data_F = (float*)gates+ num_output;
            float* gates_data_O = (float*)gates+ 2 * num_output;
            float* gates_data_G = (float*)gates+ 3 * num_output;

            float I = gates_data_I[q];
            float F = gates_data_F[q];
            float O = gates_data_O[q];
            float G = gates_data_G[q];
	    
            I = 1.f / (1.f + exp(-I));
            F = cont ? 1.f / (1.f + exp(-F)) : 0.f;
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);

            //cell[q] is not initialized and so might be nan, and 0*nan evals to nan
            float cell2 = cont ? F * cell[q] + I * G  : I * G;
            float H = O * tanh(cell2);
            cell[q] = cell2;
            hidden[q] = H;
            output_data[q] = H;
        }

        // no cell output here
    }


    return 0;
}

} // namespace ncnn