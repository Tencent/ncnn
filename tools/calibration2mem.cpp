// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
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

#include <iostream>
#include <string>
#include <fstream>  
#include <sstream>  
#include <stdlib.h>  
#include <vector>
#include <cstring>

using namespace std;

typedef struct _quantizeParams
{
    string name;
    float  dataScale;
    float  weightScale;
}stQuantizeParams;

typedef struct _quantizeParamsBins
{
    int index;
    float  dataScale;
    float  weightScale;
}stQuantizeParamsBin;

typedef struct _layerName
{
    int layer_index;
    string layer_name;
}stLayerName;

//Convert string to num 
template <class Type>
Type stringToNum(const string& str)
{
    istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

int ConvertFileToBin(const char* param_path, const char* calibration_path, const char* calibrationbin_path)
{
    cout << "Convert Start..." << endl;

    ifstream in(calibration_path);
    string line;

    vector<string> objectLine;
    vector<string> objects;

    /*
    * read the file line to strings
    */
    if (in)
    {
        while (getline(in, line))
        {
            objectLine.push_back(line);
        }
    }
    else
    {
        cout << "ncnn calibration table txt, no such file" << endl;
        //return -1;
    }

    in.close();

    for (vector<string>::iterator iter = objectLine.begin(); iter != objectLine.end(); iter++)
    {
        //cout << (*iter) << endl;
        istringstream temp((*iter));

        string str1, str2;
        temp >> str1 >> str2;
        objects.push_back(str1);
        objects.push_back(str2);
    }
    /*
     * analyse ncnn.param file to find the relation between layer_name and layer_index
     */
    ifstream in1(param_path);
    string line1;

    vector<string> objectLine1;
    vector<string> objects1;
    vector<stLayerName> layerNames;

    /*
     * read the file line to strings
     */
    if (in1)
    {
        while (getline(in1, line1))
        {
            objectLine1.push_back(line1);
        }
    }
    else
    {
        cout << "ncnn.param, no such file" << endl;
        //return -1;
    }
    in1.close();

    int layer_index = 0;
    for (int i = 2; i < objectLine1.size(); i++)
    {
        istringstream temp(objectLine1[i]);
        string str1, str2;
        stLayerName layerNameTemp;

        temp >> str1 >> str2;
        
        layerNameTemp.layer_name = str2;
        layerNameTemp.layer_index = layer_index;
        layerNames.push_back(layerNameTemp);
        layer_index++;
    }

    /*
     * convert to memory
     */
    vector<stQuantizeParamsBin> ParamBins;
    vector<stQuantizeParams> CalbirationValues;
    for (int i = 0; i < layerNames.size(); i++)
    {
        string layer_name = layerNames[i].layer_name;
        bool flag = false;

        stQuantizeParams CalbirationValue;
        stQuantizeParamsBin ParamBinTemp;

        ParamBinTemp.index = layerNames[i].layer_index;
        ParamBinTemp.dataScale = 1.0f;
        ParamBinTemp.weightScale = 1.0f;

        CalbirationValue.name = layer_name;
        CalbirationValue.dataScale = 1.0f;
        CalbirationValue.weightScale = 1.0f;        


        for (vector<string>::iterator iter = objects.begin(); iter != objects.end(); iter++)
        {
            //data scale
            if(layer_name == *iter)
            {
                if (flag == false)
                {
                    float dataScale = stringToNum<double>(*(iter + 1));
                    ParamBinTemp.dataScale = dataScale;
                    CalbirationValue.dataScale = dataScale;
                    flag = true;
                }
            }

            //weight scale
            string param_name = layer_name + "_param_0";
            if(param_name == *iter)
            {
                float weightScale = stringToNum<double>(*(iter + 1));
                ParamBinTemp.weightScale = weightScale;
                CalbirationValue.weightScale = weightScale;
            }

        }

        ParamBins.push_back(ParamBinTemp);
        CalbirationValues.push_back(CalbirationValue);
    }

    int total_size = 0;

    total_size = ParamBins.size() * sizeof(stQuantizeParamsBin);
    fprintf(stderr, "Total size:%d, ParamNum:%ld, stParamBin size:%ld\n", total_size, ParamBins.size(), sizeof(stQuantizeParamsBin));
    /*
    * save to bin file
    */
    FILE* fp = fopen(calibrationbin_path, "wb");
    fwrite(&total_size, sizeof(total_size), 1, fp);
    for (int i = 0; i < ParamBins.size(); i++)
    {
        fwrite(&ParamBins[i], sizeof(stQuantizeParamsBin), 1, fp);
    }
    fclose(fp);


    //show the txt calibration table
    for (vector<stQuantizeParams>::iterator iter = CalbirationValues.begin(); iter != CalbirationValues.end(); iter++)
    {
        fprintf(stderr, "%-20s dataScale = %f\t weightScale = %f\n", iter->name.c_str(), iter->dataScale, iter->weightScale);
    }   


    fprintf(stderr, "Convert Calibration table from txt to binary file success...\n");

    return 0;
}

int LoadScaleParamFromBin(const char* calibrationbin_path)
{ 
    cout << "Read the binary file of calbiration table..." << endl;

    FILE* fp = fopen(calibrationbin_path, "rb");
    if (NULL == fp)
    {
        fprintf(stderr, "ncnn calbiration table binary, no such file\n");
        
        return -1;
    }

    int total_size = 0;
    int ret = 0;
    ret = fread(&total_size, sizeof(total_size), 1, fp);
    if (ret != 1)
    {
        fprintf(stderr, "read scalebin size error\n");

        return -1;
    }

    vector<stQuantizeParamsBin> ParamBins;
    stQuantizeParamsBin ParamBinTemp;
    int loop = total_size / sizeof(stQuantizeParamsBin);
    for (int i = 0; i < loop; i++)
    {
        ret = fread(&ParamBinTemp, sizeof(stQuantizeParamsBin), 1, fp);
        if (ret != 1)
        {
            fprintf(stderr, "read scalebin file error\n");

            return -1;
        }
        ParamBins.push_back(ParamBinTemp);
    }

    fclose(fp);

    //show the binary calibration table
    for (vector<stQuantizeParamsBin>::iterator iter = ParamBins.begin(); iter != ParamBins.end(); iter++)
    {
        printf("index = %-12d dataScale = %f\t weightScale = %f\n", iter->index, iter->dataScale, iter->weightScale);
    }   

    fprintf(stderr, "Read calibration table binary file done...\n");

    return 0;
}

int main(int argc, char** argv)
{
    cout << "--- Convert Calibration table txt to binary file " << endl;

    if(argc != 3)
    {
        fprintf(stderr, "Usage: %s [ncnn.param] [calibration.table]\n", argv[0]);
        return -1;      
    }

    const char* param_path = argv[1];
    const char* calibration_path = argv[2];

    const char* lastslash = strrchr(calibration_path, '/');
    const char* name = lastslash == NULL ? calibration_path : lastslash + 1;

    std::string calibrationbin_path = std::string(name) + ".bin";

    int res = 0;

    res = ConvertFileToBin(param_path, calibration_path, calibrationbin_path.c_str());
    if(res)
    {
        cout << "Convert failed, please check the input file is right..." << endl;
        return 0;
    }

    res = LoadScaleParamFromBin(calibrationbin_path.c_str());
    if(res)
    {
        cout << "Load Calibration binary file failed, please check the input file is right..." << endl;
        return 0;
    }    

    return 0;
}
