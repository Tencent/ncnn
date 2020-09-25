//test yuv420p to rgb and resize

#include "net.h"
#include <stdio.h>
#include <vector>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image/stb_image.h"
#define TJE_IMPLEMENTATION
#include "./stb_image/tiny_jpeg.h"
#include "./stb_image/timing.h"
//将此文件加入到cmakelist :ncnn_add_example(yuv2rgbresize)，编译可生成例子文件测试yuv420转rgb
//需要用的的4个文件stb_image.h,tiny_jpeg.h,timing.h,可取stb_image官网下载，以及1.yuv，或者加QQ2529834545获取,
void saveImage(const char *filename, int Width, int Height, int Channels, unsigned char *Output) {
	if (!tje_encode_to_file(filename, Width, Height, Channels, true, Output)) {
		fprintf(stderr, "save JPEG fail.\n");
		return;
	}
}

int testyuv2rgbresize(char *filename)
{
    int target_w=512;//1920;
	int target_h=288;//1080;
	FILE *f=fopen(filename,"rb");
	if(f==NULL){
		printf("open %s failed!!\n",filename);
		return -1;
	}
	printf("open yuv file ok\n");
	unsigned char *yuvdata=(unsigned char *)malloc(1920*1080*3);
	int size=fread(yuvdata,1,1920*1080*3/2,f);
	if(size!=1920*1080*3/2)
		return -2;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(yuvdata, ncnn::Mat::PIXEL_YUV420P2RGB, 1920, 1080, target_w, target_h);
	//ncnn::Mat in = ncnn::Mat::from_pixels_resize(yuvdata, ncnn::Mat::PIXEL_YUV420P2BGR, 1920, 1080, target_w, target_h);
	unsigned char *rgb=(unsigned char*)malloc(target_w*target_h*3);
	in.to_pixels(rgb,1);//1表示PIXEL_RGB，编译不过写的1
	saveImage("./1.jpg",target_w,target_h,3,rgb);
	fclose(f);
	free(yuvdata);
	free(rgb);
    return 0;
}

/*int testyuv2rgbresize2(char *filename)
{
    int target_w=512;
	int target_h=288;
	FILE *f=fopen(filename,"rb");
	if(f==NULL){
		printf("open %s failed!!\n",filename);
		return -1;
	}
	printf("open yuv file ok\n");
	unsigned char *yuvdata=(unsigned char *)malloc(1920*1080*3/2);
	unsigned char *yuv420sp=(unsigned char *)malloc(1920*1080*3/2);
	unsigned char *rgbdata=(unsigned char *)malloc(1920*1080*3);
	int size=fread(yuvdata,1,1920*1080*3/2,f);
	if(size!=1920*1080*3/2)
		return -2;
	ncnn::yuv420p2yuv420sp(yuvdata,yuvdata+1920*1080,yuvdata+1920*1080*5/4,1920,1080,yuv420sp);
	ncnn::yuv420sp2rgb(yuv420sp,1920,1080,rgbdata);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgbdata, ncnn::Mat::PIXEL_RGB, 1920, 1080, target_w, target_h);
	unsigned char *rgb=(unsigned char*)malloc(target_w*target_h*3);
	in.to_pixels(rgb,1);//1表示PIXEL_RGB，编译不过写的1
	saveImage("./1.jpg",target_w,target_h,3,rgb);
	fclose(f);
	free(yuvdata);
	free(rgb);
    return 0;
}*/

int main(int argc, char** argv){
	testyuv2rgbresize("./1.yuv");
 	return 0;
}


