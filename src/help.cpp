#include <sys/time.h>
#include <sys/stat.h> 
#include <stdio.h>
#include <string>
#include "net.h"
#include "help.h"

using namespace std;


int string_replase(string &s1, const string &s2, const string &s3)
{
	string::size_type pos = 0;
	string::size_type a = s2.size();
	string::size_type b = s3.size();
	while ((pos = s1.find(s2, pos)) != string::npos)
	{
		s1.replace(pos, a, s3);
		pos += b;
	}
	return 0;
}

namespace ncnn {
/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_in_f32(int layer_index, const char* layer_name, const Mat& bottom_blob)
{
    char file_path_input[128] = {'\0'};
    char file_dir[128] = {'\0'};
    FILE *pFile1 = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");
    
    sprintf(file_dir, "./output/");
    sprintf(file_path_input, "./output/L%.3d_%s_D_In.txt", layer_index, name.c_str());

    mkdir(file_dir, 0777);

    pFile1 = fopen(file_path_input,"w");
    if(pFile1 == NULL)
    {
        printf("open file error!\n");
    }

    int channel_num;

    channel_num = bottom_blob.c;
    
    //save botton feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile1, "Bottom blob channel %d:\n", k);

        //float *data = bottom_blob.data + bottom_blob.cstep*k;
        const float *data = bottom_blob.channel(k);
        for(int i = 0; i < bottom_blob.h; i++)
        {
            for(int j = 0; j < bottom_blob.w; j++)
            {
                fprintf(pFile1, "%s%8f ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile1, "\n");
            data += bottom_blob.w;
        }
        fprintf(pFile1, "\n");
    }      
    
    fclose(pFile1);   
    pFile1 = NULL;

}

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_in_s8(int layer_index, const char* layer_name, const Mat& bottom_blob)
{
    char file_path_input[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile1 = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");
    
    sprintf(file_dir, "./output/");
    sprintf(file_path_input, "./output/L%.3d_%s_D_In_S8.txt", layer_index, name.c_str());                                          
    mkdir(file_dir, 0777);

    pFile1 = fopen(file_path_input,"w");
    if(pFile1 == NULL)
    {
        printf("open file error!\n");
    }

    int channel_num;

    channel_num = bottom_blob.c;      
    
    //save botton feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile1, "Bottom blob channel %d:\n", k);

		const float* tmp = bottom_blob.channel(k);
		signed char *data = (signed char *)tmp;
        for(int i = 0; i < bottom_blob.h; i++)
        {
            for(int j = 0; j < bottom_blob.w; j++)
            {
                fprintf(pFile1, "%s%.4d ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile1, "\n");
            data += bottom_blob.w;
        }
        fprintf(pFile1, "\n");
    }      
    
    fclose(pFile1);   
    pFile1 = NULL;

}

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_out_s8(int layer_index, const char* layer_name, const Mat& top_blob)
{
    char file_path_input[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile1 = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");    
    
    sprintf(file_dir, "./output/");
    sprintf(file_path_input, "./output/L%.3d_%s_D_Out_S8.txt", layer_index, name.c_str());                                          
    mkdir(file_dir, 0777);

    pFile1 = fopen(file_path_input,"w");
    if(pFile1 == NULL)
    {
        printf("open file error!\n");
    }

    int channel_num;

    channel_num = top_blob.c;      
    
    //save botton feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile1, "Top blob channel %d:\n", k);

		const float* tmp = top_blob.channel(k);
		signed char *data = (signed char *)tmp;
        for(int i = 0; i < top_blob.h; i++)
        {
            for(int j = 0; j < top_blob.w; j++)
            {
                fprintf(pFile1, "%s%.4d ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile1, "\n");
            data += top_blob.w;
        }
        fprintf(pFile1, "\n");
    }      
    
    fclose(pFile1);   
    pFile1 = NULL;

}

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_out_s16(int layer_index, const char* layer_name, const Mat& top_blob)
{
    char file_path_input[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile1 = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");    
    
    sprintf(file_dir, "./output/");
    sprintf(file_path_input, "./output/L%.3d_%s_D_Out_S16.txt", layer_index, name.c_str());                                          
    mkdir(file_dir, 0777);

    pFile1 = fopen(file_path_input,"w");
    if(pFile1 == NULL)
    {
        printf("open file error!\n");
    }

    int channel_num;

    channel_num = top_blob.c;      
    
    //save botton feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile1, "Top blob channel %d:\n", k);

		const float* tmp = top_blob.channel(k);
		short *data = (short *)tmp;
        for(int i = 0; i < top_blob.h; i++)
        {
            for(int j = 0; j < top_blob.w; j++)
            {
                fprintf(pFile1, "%s%.6d ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile1, "\n");
            data += top_blob.w;
        }
        fprintf(pFile1, "\n");
    }      
    
    fclose(pFile1);   
    pFile1 = NULL;

}

/*
 * Extract the layer bottom blob and top blob 2 feature maps
 */
void extract_feature_out_f32(int layer_index, const char* layer_name, const Mat& top_blob)
{
    char file_path_output[128] = {'\0'};
    char file_dir[128] = {'\0'};
    //FILE *pFile1 = NULL;
    FILE *pFile2 = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");    
    
    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

    sprintf(file_path_output, "./output/L%.3d_%s_D_Out.txt", layer_index, name.c_str());

    pFile2 = fopen(file_path_output,"w");
    if(pFile2 == NULL)
    {
        printf("open file error!\n");
    }

	int channel_num;

    channel_num = top_blob.c;  
    
    //save top feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile2, "Top blob channel %d:\n", k);

        //float *data = top_blob.data + top_blob.cstep*k;
        const float *data = top_blob.channel(k);
        for(int i = 0; i < top_blob.h; i++)
        {
            for(int j = 0; j < top_blob.w; j++)
            {
                fprintf(pFile2, "%s%8f ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile2, "\n");
            data += top_blob.w;
        }
        fprintf(pFile2, "\n");
    }     

    //close file
    fclose(pFile2);   
    pFile2 = NULL;

}

/*
 * Extract the blob feature map
 */
void extract_feature_blob_f32(const char* comment, const char* layer_name, const Mat& blob)
{
    char file_path_output[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");    
    
    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

    sprintf(file_path_output, "./output/%s_%s_blob_data.txt", name.c_str(), comment);

    pFile = fopen(file_path_output,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }

	int channel_num = blob.c;
    
    //save top feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile, "blob channel %d:\n", k);

        //float *data = top_blob.data + top_blob.cstep*k;
        const float *data = blob.channel(k);
        for(int i = 0; i < blob.h; i++)
        {
            for(int j = 0; j < blob.w; j++)
            {
                fprintf(pFile, "%s%8f ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile, "\n");
            data += blob.w;
        }
        fprintf(pFile, "\n");
    }     

    //close file
    fclose(pFile);   
    pFile = NULL;
}

/*
 * Extract the blob feature map
 */
void extract_feature_blob_s8(const char* comment, const char* layer_name, const Mat& blob)
{
    char file_path_output[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");    
    
    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

    sprintf(file_path_output, "./output/%s_%s_blob_data.txt", name.c_str(), comment);

    pFile = fopen(file_path_output,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }

	int channel_num = blob.c;
    
    //save top feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile, "blob channel %d:\n", k);

        //float *data = top_blob.data + top_blob.cstep*k;
        const signed char *data = blob.channel(k);
        for(int i = 0; i < blob.h; i++)
        {
            for(int j = 0; j < blob.w; j++)
            {
                fprintf(pFile, "%s%.4d ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile, "\n");
            data += blob.w;
        }
        fprintf(pFile, "\n");
    }     

    //close file
    fclose(pFile);   
    pFile = NULL;
}

/*
 * Extract the blob feature map
 */
void extract_feature_blob_s16(const char* comment, const char* layer_name, const Mat& blob)
{
    char file_path_output[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");    
    
    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

    sprintf(file_path_output, "./output/%s_%s_blob_data.txt", name.c_str(), comment);

    pFile = fopen(file_path_output,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }

	int channel_num = blob.c;
    
    //save top feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile, "blob channel %d:\n", k);

        //float *data = top_blob.data + top_blob.cstep*k;
        const short *data = blob.channel(k);
        for(int i = 0; i < blob.h; i++)
        {
            for(int j = 0; j < blob.w; j++)
            {
                fprintf(pFile, "%s%.8d ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile, "\n");
            data += blob.w;
        }
        fprintf(pFile, "\n");
    }     

    //close file
    fclose(pFile);   
    pFile = NULL;
}

void extract_feature_out_s32(int layer_index, const char* layer_name, const Mat& top_blob)
{
    char file_path_output[128] = {'\0'};
    char file_dir[128] = {'\0'};
    //FILE *pFile1 = NULL;
    FILE *pFile2 = NULL;

    string name = layer_name;
    string_replase(name, "/", "-");    
    
    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

    sprintf(file_path_output, "./output/L%.3d_%s_D_Out_S32.txt", layer_index, name.c_str());

    pFile2 = fopen(file_path_output,"w");
    if(pFile2 == NULL)
    {
        printf("open file error!\n");
    }

	int channel_num;

    channel_num = top_blob.c;  
    
    //save top feature maps
    for(int k = 0; k < channel_num; k++)
    {
        fprintf(pFile2, "Top blob channel %d:\n", k);

        //float *data = top_blob.data + top_blob.cstep*k;
        const int* data = (const int*)top_blob.channel(k);
        for(int i = 0; i < top_blob.h; i++)
        {
            for(int j = 0; j < top_blob.w; j++)
            {
                fprintf(pFile2, "%s%.8d ", (data[j]<0)?"":" ", data[j]);
            }
            fprintf(pFile2, "\n");
            data += top_blob.w;
        }
        fprintf(pFile2, "\n");
    }     

    //close file
    fclose(pFile2);   
    pFile2 = NULL;

}

/*
 * Extract the layer kernel weight value
 */
void extract_kernel_f32(int layer_index, const char* layer_name, const Mat& _kernel, const Mat& _bias, int num_input, int num_output, const int _kernel_size)
{
    char file_path[128] = {'\0'};
    char file_dir[128] = {'\0'};
    FILE *pFile = NULL;
    
    string name = layer_name;
    string_replase(name, "/", "-");

    sprintf(file_dir, "./output/");
    sprintf(file_path, "./output/L%.3d_%s_K.txt", layer_index, name.c_str());
    mkdir(file_dir, 0777);

    pFile = fopen(file_path,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }
    
    const float *data = (const float*)_kernel.data;

    for(int k = 0; k < num_output; k++)
    {
        fprintf(pFile, "Kernel %d:\n", k);
        
        for(int n = 0; n < num_input; n++)
        {
            for(int i = 0; i < _kernel_size; i++)
            {
                for(int j = 0; j < _kernel_size; j++)
                {
                    fprintf(pFile, "%s%8f ", (data[j]<0)?"":" ", data[j]);
                }
                fprintf(pFile, "\n");
                data += _kernel_size;
            }
            fprintf(pFile, "\n");
        }   
        fprintf(pFile, "\n");
    }    

    if(_bias.w != 0)
    {
        fprintf(pFile, "Bias :\n");
        for(int k = 0; k < num_output; k++)
        {   
            const float *data_b = (const float*)_bias.data;
            fprintf(pFile, "%s%8f ", (data_b[k]<0)?"":" ", data_b[k]);
        }    
    } 
	else
	{
		fprintf(pFile, "Bias is 0:\n");
	}
    
    //close file
    fclose(pFile);   
    pFile = NULL;
}

void extract_kernel_dw_f32(int layer_index, const char* layer_name, const Mat& _kernel, const Mat& _bias, int num_output, int group, const int _kernel_size)
{
    char file_path[128] = {'\0'};
    char file_dir[128] = {'\0'};
    FILE *pFile = NULL;
    
    string name = layer_name;
    string_replase(name, "/", "-");

    sprintf(file_dir, "./output/");
    sprintf(file_path, "./output/L%.3d_%s_K.txt", layer_index, name.c_str());
    mkdir(file_dir, 0777);

    pFile = fopen(file_path,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }
    
    const float *data = (const float*)_kernel.data;

    if (group == num_output)
    {
        for(int k = 0; k < num_output; k++)
        {
            fprintf(pFile, "Kernel %d:\n", k);
            
            for(int i = 0; i < _kernel_size; i++)
            {
                for(int j = 0; j < _kernel_size; j++)
                {
                    fprintf(pFile, "%s%8f ", (data[j]<0)?"":" ", data[j]);
                }
                fprintf(pFile, "\n");
                data += _kernel_size;
            }

            fprintf(pFile, "\n");
        }    
    }
    else
    {
        printf("depthwiseconv output != group\n");
        //TODO
    }

    if(_bias.w != 0)
    {
        fprintf(pFile, "Bias :\n");
        for(int k = 0; k < num_output; k++)
        {   
            const float *data_b = (const float*)_bias.data;
            fprintf(pFile, "%s%8f ", (data_b[k]<0)?"":" ", data_b[k]);
        }    
    } 
	else
	{
		fprintf(pFile, "Bias is 0:\n");
	}
    
    //close file
    fclose(pFile);   
    pFile = NULL;
    data = NULL;
}

/*
 * Extract the layer kernel weight value
 */
void extract_kernel_s8(int layer_index, const char* layer_name, const Mat& _kernel, const Mat& _bias, int num_input, int num_output, const int _kernel_size)
{
    char file_path[128] = {'\0'};
    char file_dir[128] = {'\0'};
    FILE *pFile = NULL;
    
    string name = layer_name;
    string_replase(name, "/", "-");

    sprintf(file_dir, "./output/");
    sprintf(file_path, "./output/L%.3d_%s_K_In_S8.txt", layer_index, name.c_str());
    mkdir(file_dir, 0777);

    pFile = fopen(file_path,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }

    const float *kernel = _kernel;

    signed char *data = (signed char*)kernel;
    for(int k = 0; k < num_output; k++)
    {
        fprintf(pFile, "-----------Kernel Output %d:\n", k);
        
        for(int n = 0; n < num_input; n++)
        {
        	fprintf(pFile, "Kernel Input %d:\n", n);
            for(int i = 0; i < _kernel_size; i++)
            {
                for(int j = 0; j < _kernel_size; j++)
                {
                    fprintf(pFile, "%4d ", data[j]);
                }
                fprintf(pFile, "\n");
                data += _kernel_size;
            }
            fprintf(pFile, "\n");
        }   
        fprintf(pFile, "\n");
    }    


    if(_bias.w != 0)
    {
        fprintf(pFile, "Bias :\n");
        for(int k = 0; k < num_output; k++)
        {   
            const float *data_b = (const float *)_bias.data;
            fprintf(pFile, "%s%8f ", (data_b[k]<0)?"":" ", data_b[k]);
        }    
    } 
	else
	{
		fprintf(pFile, "Bias is 0:\n");
	}
    
    //close file
    fclose(pFile);   
    pFile = NULL;
}

void extract_kernel_dw_s8(int layer_index, const char* layer_name, const Mat& _kernel, const Mat& _bias, int num_output, int group, const int _kernel_size)
{
    char file_path[128] = {'\0'};
    char file_dir[128] = {'\0'};
    FILE *pFile = NULL;
    
    string name = layer_name;
    string_replase(name, "/", "-");

    sprintf(file_dir, "./output/");
    sprintf(file_path, "./output/L%.3d_%s_K_In_S8.txt", layer_index, name.c_str());
    mkdir(file_dir, 0777);

    pFile = fopen(file_path,"w");
    if(pFile == NULL)
    {
        printf("open file error!\n");
    }

    const float *kernel = _kernel;

    signed char *data = (signed char*)kernel;

    if(num_output == group)
    {
        for(int k = 0; k < num_output; k++)
        {
            fprintf(pFile, "-----------Kernel %d:\n", k);
            for(int i = 0; i < _kernel_size; i++)
            {
                for(int j = 0; j < _kernel_size; j++)
                {
                    fprintf(pFile, "%4d ", data[j]);
                }
                fprintf(pFile, "\n");
                data += _kernel_size;
            }

            fprintf(pFile, "\n");
        }    
    }
    else
    {
        printf("depthwiseconv output != group\n");
        //TODO
    }

    if(_bias.w != 0)
    {
        fprintf(pFile, "Bias :\n");
        for(int k = 0; k < num_output; k++)
        {   
            const float *data_b = (const float *)_bias.data;
            fprintf(pFile, "%s%8f  ", (data_b[k]<0)?"":" ", data_b[k]);
        }    
    } 
	else
	{
		fprintf(pFile, "Bias is 0:\n");
	}
    
    //close file
    fclose(pFile);   
    pFile = NULL;
    data = NULL;
}

}