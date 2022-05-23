// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#if __APPLE__
#include "mat.h"
#include "cpu.h"
#include "allocator.h"
#import <simd/simd.h>
#import <Accelerate/Accelerate.h>
#import <CoreGraphics/CoreGraphics.h>

namespace ncnn {
Mat Mat::from_apple_samplebuffer(CMSampleBufferRef samplebuffer) {
    CMFormatDescriptionRef des = CMSampleBufferGetFormatDescription(samplebuffer);
    if(!des) {
        return Mat();
    }
    if(CMFormatDescriptionGetMediaType(des)!=kCMMediaType_Video) {
        return Mat();
    }
    CVPixelBufferRef pixel = CMSampleBufferGetImageBuffer(samplebuffer);
    return Mat::from_apple_pixelbuffer(pixel);
}

Mat Mat::from_apple_pixelbuffer(CVPixelBufferRef pixelbuffer) {
    OSType format = CVPixelBufferGetPixelFormatType(pixelbuffer);
    if(format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
        size_t h = CVPixelBufferGetHeight(pixelbuffer);
        size_t w = CVPixelBufferGetWidth(pixelbuffer);
        vImage_Buffer a;
        if (vImageBuffer_Init(&a, h, w, 32, kvImageNoFlags)!=kvImageNoError) {
            return Mat();
        }
        vImage_YpCbCrPixelRange pixelRange = {16, 128, 265, 240, 235, 16, 240, 16};
        vImage_YpCbCrToARGB matrix;
        if (vImageConvert_YpCbCrToARGB_GenerateConversion(kvImage_YpCbCrToARGBMatrix_ITU_R_601_4,&pixelRange,&matrix,kvImage420Yp8_CbCr8,kvImageARGB8888,0)!=kvImageNoError) {
            return Mat();
        }
        Mat mat = Mat(w,h,(size_t)4,4);
        mat.dims = 2;
        if(CVPixelBufferLockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly) != 0) {
            mat.release();
            return Mat();
        }
        void* yData = CVPixelBufferGetBaseAddressOfPlane(pixelbuffer,0);
        void* uvData = CVPixelBufferGetBaseAddressOfPlane(pixelbuffer,1);
        vImage_Buffer yBuff = {yData,h,w,w};
        vImage_Buffer uvBuff = {uvData,h/2,w/2,w};
        if(vImageConvert_420Yp8_CbCr8ToARGB8888(&yBuff,&uvBuff,&a,&matrix,NULL,1,kvImageNoFlags)!=kvImageNoError) {
            mat.release();
            CVPixelBufferUnlockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);
            return Mat();
        }
        memcpy(mat.data,a.data,w*h*4);
        free(a.data);
        if(CVPixelBufferUnlockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly) != 0) {
            mat.release();
            CVPixelBufferUnlockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);
            return Mat();
        }
        return mat;
    } else if(format == kCVPixelFormatType_32ARGB) {
        size_t h = CVPixelBufferGetHeight(pixelbuffer);
        size_t w = CVPixelBufferGetWidth(pixelbuffer);
        Mat mat = Mat(w,h,(size_t)4,4);
        mat.dims = 2;
        if(CVPixelBufferLockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly) != 0) {
            return Mat();
        }
        void* pd = CVPixelBufferGetBaseAddress(pixelbuffer);
        memcpy(mat.data,pd,w*h*4);
        if(CVPixelBufferUnlockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly) != 0) {
            return Mat();
        }
        return mat;
    } else {
        return Mat();
    }
}
int Mat::to_apple_pixelbuffer(CVPixelBufferRef* pixelbuffer) {
    if(dims == 1) {
        return -1;
    } else if(dims == 2) {
        if(elempack==4) {
            if(elemsize==4) {
                void* p = NULL;
                posix_memalign(&p,64,w*h*elemsize);
                if(!p) {
                    posix_memalign(&p,64,w*h*elemsize);
                }
                memcpy(p,data,w*h*4);
                if( CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_32ARGB,p,w*4,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else if(elemsize==8) {
                void* p = NULL;
                posix_memalign(&p,64,w*h*elemsize);
                if(!p) {
                    posix_memalign(&p,64,w*h*elemsize);
                }
                memcpy(p,data,w*h*8);
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_64RGBAHalf,p,w*8,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else if(elemsize==16) {
                void* p = NULL;
                posix_memalign(&p,64,w*h*elemsize);
                if(!p) {
                    posix_memalign(&p,64,w*h*elemsize);
                }
                memcpy(p,data,w*h*16);
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_128RGBAFloat,p,w*16,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else {
                return -1;
            }
        } else if(elempack==1) {
            if(elemsize==1) {
                void* p = NULL;
                posix_memalign(&p,64,w*h*elemsize);
                if(!p) {
                    posix_memalign(&p,64,w*h*elemsize);
                }
                memcpy(p,data,w*h);
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_OneComponent8,p,w,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else if(elemsize==2) {
                void* p = NULL;
                posix_memalign(&p,64,w*h*elemsize);
                if(!p) {
                    posix_memalign(&p,64,w*h*elemsize);
                }
                memcpy(p,data,w*h*2);
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_OneComponent16Half,p,w*2,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else if(elemsize==4) {
                void* p = NULL;
                posix_memalign(&p,64,w*h*elemsize);
                if(!p) {
                    posix_memalign(&p,64,w*h*elemsize);
                }
                memcpy(p,data,w*h*4);
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_OneComponent32Float,p,w*4,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else {
                return -1;
            }
        } else {
            return -1;
        }
    } else if(dims==3) {
        if(elempack!=1) {
            return -1;
        }
        if(c!=1) {
            return -1;
        }
        if(d==3) {
            Mat m2 = depth_range(1,1);
            Mat m3 = depth_range(2,1);
            if(elemsize==1) {
                void * dpa = NULL;
                posix_memalign(&dpa,64,w*h*elemsize*4);
                if(!dpa) {
                    posix_memalign(&dpa,64,w*h*elemsize*4);
                }
                uint8_t* datas = (uint8_t*) dpa;
                int pth =get_cpu_count();
                uint8_t* dp =(uint8_t*)data;
                uint8_t* dp2 =(uint8_t*)m2.data;
                uint8_t* dp3 =(uint8_t*)m3.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=2) {
                        simd_uchar8 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],255,dp[i*w+j+1],dp2[i*w+j+1],dp3[i*w+j+1],255};
                        memcpy((datas+i*w+j),&p,8);
                    }
                }
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_32RGBA,datas,w*4,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else if(elemsize==4) {
                void * dpa = NULL;
                posix_memalign(&dpa,64,w*h*elemsize*4);
                if(!dpa) {
                    posix_memalign(&dpa,64,w*h*elemsize*4);
                }
                uint8_t* datas = (uint8_t*) dpa;
                int pth =get_cpu_count();
                ushort* dp =(ushort*)data;
                ushort* dp2 =(ushort*)m2.data;
                ushort* dp3 =(ushort*)m3.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=1) {
                        simd_ushort4 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],1};
                        memcpy(datas+(i*w+j)*4,&p,32);
                    }
                }
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_128RGBAFloat,datas,w*16,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else {
                return -1;
            }
        } else if(d==4) {
            Mat m2 = depth_range(1,1);
            Mat m3 = depth_range(2,1);
            Mat m4 = depth_range(3,1);
            if(elemsize==1) {
                void * dpa = NULL;
                posix_memalign(&dpa,64,w*h*elemsize*4);
                if(!dpa) {
                    posix_memalign(&dpa,64,w*h*elemsize*4);
                }
                uint8_t* datas = (uint8_t*) dpa;
                int pth =get_cpu_count();
                uint8_t* dp =(uint8_t*)data;
                uint8_t* dp2 =(uint8_t*)m2.data;
                uint8_t* dp3 =(uint8_t*)m3.data;
                uint8_t* dp4 =(uint8_t*)m4.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=2) {
                        simd_uchar8 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],dp4[i*w+j],dp[i*w+j+1],dp2[i*w+j+1],dp3[i*w+j+1],dp4[i*w+j+1]};
                        memcpy(datas+i*w+j,&p,8);
                    }
                }
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_32RGBA,datas,w*4,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else if(elemsize==4) {
                void * dpa = NULL;
                posix_memalign(&dpa,64,w*h*elemsize*4);
                if(!dpa) {
                    posix_memalign(&dpa,64,w*h*elemsize*4);
                }
                uint8_t* datas = (uint8_t*) dpa;
                int pth =get_cpu_count();
                ushort* dp =(ushort*)data;
                ushort* dp2 =(ushort*)m2.data;
                ushort* dp3 =(ushort*)m3.data;
                ushort* dp4 =(ushort*)m4.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=1) {
                        simd_ushort4 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],dp4[i*w+j]};
                        memcpy(datas+(i*w+j)*4,&p,32);
                    }
                }
                if(CVPixelBufferCreateWithBytes(NULL,w,h,kCVPixelFormatType_128RGBAFloat,datas,w*16,NULL,NULL,NULL,pixelbuffer)==kCVReturnSuccess) {
                    return 0;
                } else {
                    return -1;
                }
            } else {
                return -1;
            }
        } else {
            return -1;
        }
    } else {
        return -1;
    }
}
#if TARGET_OS_IOS
Mat Mat::from_apple_image(UIImage* image) {
    CGImageRef refImage = [image CGImage];
    CGSize size = image.size;

    int bitsPerComponent = 8;
    int bytePerPixel = 4;

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    int pixelCount = size.width * size.height;

    uint8_t *rgba = (uint8_t *)fastMalloc(pixelCount * bytePerPixel);
    if(!rgba)
        rgba = (uint8_t *)fastMalloc(pixelCount * bytePerPixel);

    CGContextRef context = CGBitmapContextCreate(rgba,
                           size.width,
                           size.height,
                           bitsPerComponent,
                           bytePerPixel * size.width,
                           colorSpace,
                           kCGImageAlphaNoneSkipLast);

    CGContextDrawImage(context, CGRectMake(0, 0, size.width, size.height), refImage);
    CGContextRelease(context);
    Mat mat = Mat(size.width,size.height,(size_t)4,4);
    mat.dims = 2;
    memcpy(mat.data,rgba,pixelCount * bytePerPixel);
    fastFree(rgba);
    return mat;
}
UIImage* Mat::to_apple_image() {
    int bytes_per_pix = 4;

    int bitsPerComponent = 8;

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    void*datas;

    if(dims == 1) {
        return nil;
    } else if(dims == 2) {
        if(elempack==4) {
            if(elemsize==4) {
                bytes_per_pix = 4;
                bitsPerComponent = 8;
            } else if(elemsize==8) {
                bytes_per_pix = 8;
                bitsPerComponent = 16;
            } else if(elemsize==16) {
                bytes_per_pix = 16;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else if(elempack==1) {
            if(elemsize==1) {
                bytes_per_pix = 1;
                bitsPerComponent = 8;
            } else if(elemsize==2) {
                bytes_per_pix = 2;
                bitsPerComponent = 16;
            } else if(elemsize==4) {
                bytes_per_pix = 4;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else {
            return nil;
        }
        datas = fastMalloc(w*h*bytes_per_pix);
        if(!datas)
            memcpy(datas,data,w*h*bytes_per_pix);
    } else if(dims==3) {
        if(elempack!=1) {
            return nil;
        }
        if(c!=1) {
            return nil;
        }
        if(d==3) {
            Mat m2 = depth_range(1,1);
            Mat m3 = depth_range(2,1);
            if(elemsize==1) {
                datas = fastMalloc(w*h*4);
                if(!datas)
                    datas = fastMalloc(w*h*4);
                int pth =get_cpu_count();
                uint8_t* dp =(uint8_t*)data;
                uint8_t* dp2 =(uint8_t*)m2.data;
                uint8_t* dp3 =(uint8_t*)m3.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=2) {
                        simd_uchar8 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],255,dp[i*w+j+1],dp2[i*w+j+1],dp3[i*w+j+1],255};
                        memcpy((uint8_t*)datas+i*w+j,&p,8);
                    }
                }
                bytes_per_pix = 4;
                bitsPerComponent = 8;
            } else if(elemsize==4) {
                datas = fastMalloc(w*h*16);
                if(!datas)
                    datas = fastMalloc(w*h*16);
                int pth =get_cpu_count();
                ushort* dp =(ushort*)data;
                ushort* dp2 =(ushort*)m2.data;
                ushort* dp3 =(ushort*)m3.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=1) {
                        simd_ushort4 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],1};
                        memcpy((uint8_t*)datas+(i*w+j)*4,&p,32);
                    }
                }
                bytes_per_pix = 16;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else if(d==4) {
            Mat m2 = depth_range(1,1);
            Mat m3 = depth_range(2,1);
            Mat m4 = depth_range(3,1);
            if(elemsize==1) {
                datas = fastMalloc(w*h*4);
                if(!datas)
                    datas = fastMalloc(w*h*4);
                int pth =get_cpu_count();
                uint8_t* dp =(uint8_t*)data;
                uint8_t* dp2 =(uint8_t*)m2.data;
                uint8_t* dp3 =(uint8_t*)m3.data;
                uint8_t* dp4 =(uint8_t*)m4.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=2) {
                        simd_uchar8 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],dp4[i*w+j],dp[i*w+j+1],dp2[i*w+j+1],dp3[i*w+j+1],dp4[i*w+j+1]};
                        memcpy((uint8_t*)datas+i*w+j,&p,8);
                    }
                }
                bytes_per_pix = 4;
                bitsPerComponent = 8;
            } else if(elemsize==4) {
                datas = fastMalloc(w*h*16);
                if(!datas)
                    datas = fastMalloc(w*h*16);
                int pth =get_cpu_count();
                ushort* dp =(ushort*)data;
                ushort* dp2 =(ushort*)m2.data;
                ushort* dp3 =(ushort*)m3.data;
                ushort* dp4 =(ushort*)m4.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=1) {
                        simd_ushort4 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],dp4[i*w+j]};
                        memcpy((uint8_t*)datas+(i*w+j)*4,&p,32);
                    }
                }
                bytes_per_pix = 16;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else {
            return nil;
        }
    } else {
        return nil;
    }
    CGContextRef newContext = CGBitmapContextCreate(datas,
                              w, h, bitsPerComponent,
                              w * bytes_per_pix,
                              colorSpace, kCGImageAlphaNoneSkipLast);

    CGImageRef frame = CGBitmapContextCreateImage(newContext);

    UIImage *image = [UIImage imageWithCGImage:frame];

    CGImageRelease(frame);

    CGContextRelease(newContext);

    CGColorSpaceRelease(colorSpace);

    fastFree(datas);

    return image;
}
#else
Mat Mat::from_apple_image(NSImage* image) {
    CGImageRef refImage = [image CGImageForProposedRect:nil context:[NSGraphicsContext currentContext] hints:nil];
    CGSize size = image.size;

    int bitsPerComponent = 8;
    int bytePerPixel = 4;

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    int pixelCount = size.width * size.height;

    uint8_t *rgba = (uint8_t *)fastMalloc(pixelCount * bytePerPixel);
    if(!rgba)
        rgba = (uint8_t *)fastMalloc(pixelCount * bytePerPixel);

    CGContextRef context = CGBitmapContextCreate(rgba,
                           size.width,
                           size.height,
                           bitsPerComponent,
                           bytePerPixel * size.width,
                           colorSpace,
                           kCGImageAlphaNoneSkipLast);

    CGContextDrawImage(context, CGRectMake(0, 0, size.width, size.height), refImage);
    CGContextRelease(context);
    Mat mat = Mat(size.width,size.height,(size_t)4,4);
    mat.dims = 2;
    memcpy(mat.data,rgba,pixelCount * bytePerPixel);
    fastFree(rgba);
    return mat;
}
NSImage* Mat::to_apple_image() {
    int bytes_per_pix = 4;

    int bitsPerComponent = 8;

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    void*datas;

    if(dims == 1) {
        return nil;
    } else if(dims == 2) {
        if(elempack==4) {
            if(elemsize==4) {
                bytes_per_pix = 4;
                bitsPerComponent = 8;
            } else if(elemsize==8) {
                bytes_per_pix = 8;
                bitsPerComponent = 16;
            } else if(elemsize==16) {
                bytes_per_pix = 16;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else if(elempack==1) {
            if(elemsize==1) {
                bytes_per_pix = 1;
                bitsPerComponent = 8;
            } else if(elemsize==2) {
                bytes_per_pix = 2;
                bitsPerComponent = 16;
            } else if(elemsize==4) {
                bytes_per_pix = 4;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else {
            return nil;
        }
        datas = fastMalloc(w*h*bytes_per_pix);
        if(!datas)
            datas = fastMalloc(w*h*bytes_per_pix);
        memcpy(datas,data,w*h*bytes_per_pix);
    } else if(dims==3) {
        if(elempack!=1) {
            return nil;
        }
        if(c!=1) {
            return nil;
        }
        if(d==3) {
            Mat m2 = depth_range(1,1);
            Mat m3 = depth_range(2,1);
            if(elemsize==1) {
                datas = fastMalloc(w*h*4);
                if(!datas) {
                    datas = fastMalloc(w*h*4);
                }
                int pth =get_cpu_count();
                uint8_t* dp =(uint8_t*)data;
                uint8_t* dp2 =(uint8_t*)m2.data;
                uint8_t* dp3 =(uint8_t*)m3.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=2) {
                        simd_uchar8 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],255,dp[i*w+j+1],dp2[i*w+j+1],dp3[i*w+j+1],255};
                        memcpy((uint8_t*)datas+i*w+j,&p,8);
                    }
                }
                bytes_per_pix = 4;
                bitsPerComponent = 8;
            } else if(elemsize==4) {
                datas = fastMalloc(w*h*16);
                if(!datas)
                    datas = fastMalloc(w*h*16);
                int pth =get_cpu_count();
                ushort* dp =(ushort*)data;
                ushort* dp2 =(ushort*)m2.data;
                ushort* dp3 =(ushort*)m3.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=1) {
                        simd_ushort4 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],1};
                        memcpy((uint8_t*)datas+(i*w+j)*4,&p,32);
                    }
                }
                bytes_per_pix = 16;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else if(d==4) {
            Mat m2 = depth_range(1,1);
            Mat m3 = depth_range(2,1);
            Mat m4 = depth_range(3,1);
            if(elemsize==1) {
                datas = fastMalloc(w*h*4);
                if(!datas)
                    datas = fastMalloc(w*h*4);
                int pth =get_cpu_count();
                uint8_t* dp =(uint8_t*)data;
                uint8_t* dp2 =(uint8_t*)m2.data;
                uint8_t* dp3 =(uint8_t*)m3.data;
                uint8_t* dp4 =(uint8_t*)m4.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=2) {
                        simd_uchar8 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],dp4[i*w+j],dp[i*w+j+1],dp2[i*w+j+1],dp3[i*w+j+1],dp4[i*w+j+1]};
                        memcpy((uint8_t*)datas+i*w+j,&p,8);
                    }
                }
                bytes_per_pix = 4;
                bitsPerComponent = 8;
            } else if(elemsize==4) {
                datas = fastMalloc(w*h*16);
                if(!datas)
                    datas = fastMalloc(w*h*16);
                int pth =get_cpu_count();
                ushort* dp =(ushort*)data;
                ushort* dp2 =(ushort*)m2.data;
                ushort* dp3 =(ushort*)m3.data;
                ushort* dp4 =(ushort*)m4.data;
                #pragma omp parallel for num_threads(pth)
                for(ushort i = 0; i<h; i++) {
                    for(ushort j = 0; j<w; j+=1) {
                        simd_ushort4 p = {dp[i*w+j],dp2[i*w+j],dp3[i*w+j],dp4[i*w+j]};
                        memcpy((uint8_t*)datas+(i*w+j)*4,&p,32);
                    }
                }
                bytes_per_pix = 16;
                bitsPerComponent = 32;
            } else {
                return nil;
            }
        } else {
            return nil;
        }
    } else {
        return nil;
    }
    CGContextRef newContext = CGBitmapContextCreate(datas,
                              w, h, bitsPerComponent,
                              w * bytes_per_pix,
                              colorSpace, kCGImageAlphaNoneSkipLast);

    CGImageRef frame = CGBitmapContextCreateImage(newContext);

    NSImage *image = [[NSImage alloc] initWithCGImage:frame size:NSMakeSize(w,h)];

    CGImageRelease(frame);

    CGContextRelease(newContext);

    CGColorSpaceRelease(colorSpace);

    fastFree(datas);

    return image;
}
#endif
}
#endif
