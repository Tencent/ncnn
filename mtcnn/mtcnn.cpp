#include <stdio.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"
using namespace std;
using namespace cv;

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};

struct orderScore
{
    float score;
    int oriOrder;
};
bool cmpScore(orderScore lsh, orderScore rsh){
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}
class mtcnn{
public:
    mtcnn();
    void detect(cv::Mat& image);
private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale);
    void nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname="Union");
    void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);

    ncnn::Net Pnet, Rnet, Onet;

    const float nms_threshold[3] = {0.5, 0.7, 0.7};
    const float threshold[3] = {0.8, 0.8, 0.8};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    std::vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
    int img_w, img_h;
};

mtcnn::mtcnn(){
    Pnet.load_param("det1.param");
    Pnet.load_model("det1.bin");
    Rnet.load_param("det2.param");
    Rnet.load_model("det2.bin");
    Onet.load_param("det3.param");
    Onet.load_model("det3.bin");
}

void mtcnn::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale){
    //for pooling 
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = score.data + score.cstep;
    float *plocal = location.data;
    Bbox bbox;
    orderScore order;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*row+1)/scale);
                bbox.y1 = round((stride*col+1)/scale);
                bbox.x2 = round((stride*row+1+cellsize)/scale);
                bbox.y2 = round((stride*col+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=*(plocal+channel*location.cstep);
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}
void mtcnn::nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for(int num=0;num<boundingBox_.size();num++){
            if(boundingBox_.at(num).exist){
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse 
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min")){
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}
void mtcnn::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<struct Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbh = (*it).x2 - (*it).x1 + 1;
            bbw = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[1]*bbh;
            y1 = (*it).y1 + (*it).regreCoord[0]*bbw;
            x2 = (*it).x2 + (*it).regreCoord[3]*bbh;
            y2 = (*it).y2 + (*it).regreCoord[2]*bbw;

            h = x2 - x1 + 1;
            w = y2 - y1 + 1;
          
            maxSide = (h>w)?h:w;
            x1 = x1 + h*0.5 - maxSide*0.5;
            y1 = y1 + w*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>height)(*it).x2 = height - 1;
            if((*it).y2>width)(*it).y2 = width - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}
void mtcnn::detect(cv::Mat &image){
    img_w = image.cols;
    img_h = image.rows;
    float minl = img_w<img_h?img_w:img_h;
    int MIN_DET_SIZE = 12;
    int minsize = 10;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    orderScore order;
    int count = 0;

    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, changedW, changedH);
        in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        std::vector<Bbox> boundingBox_;
        std::vector<orderScore> bboxScore_;
        generateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
        nms(boundingBox_, bboxScore_, nms_threshold[0]);

        for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        bboxScore_.clear();
        boundingBox_.clear();
    }
    printf("firstBbox_.size()=%d\n", firstBbox_.size());
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols);

    //second stage
    count = 0;
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat tempIm;
            tempIm.copyTo(image(temp));
            ncnn::Mat in = ncnn::Mat::from_pixels_resize(tempIm.data, ncnn::Mat::PIXEL_BGR, tempIm.cols, tempIm.rows, 24, 24);
            in.substract_mean_normalize(mean_vals, norm_vals);
            ncnn::Extractor ex = Rnet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox;
            ex.extract("prob1", score);
            ex.extract("conv5-2", bbox);
            if(*(score.data+score.cstep)>threshold[1]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(score.data+score.cstep);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    printf("secondBbox_.size()=%d\n", secondBbox_.size());
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols);

    //third stage 
    count = 0;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat tempIm;
            tempIm.copyTo(image(temp));
            ncnn::Mat in = ncnn::Mat::from_pixels_resize(tempIm.data, ncnn::Mat::PIXEL_BGR, tempIm.cols, tempIm.rows, 48, 48);
            in.substract_mean_normalize(mean_vals, norm_vals);
            ncnn::Extractor ex = Onet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox, keyPoint;
            ex.extract("prob1", score);
            ex.extract("conv6-2", bbox);
            ex.extract("conv6-3", keyPoint);
            if(*(score.data+score.cstep)>threshold[2]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(score.data+score.cstep);
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(keyPoint.data+num*keyPoint.cstep));
                }
                for(int num=0;num<5;num++){
                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(keyPoint.data+(num+5)*keyPoint.cstep));
                }

                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else
                (*it).exist=false;
            }
        }

    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++){
        if((*it).exist){
            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
            for(int num=0;num<5;num++)circle(image,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
        }
    }
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
}
int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    mtcnn mm;
    mm.detect(m);
    imshow("a",m);
    waitKey();
    return 0;
}