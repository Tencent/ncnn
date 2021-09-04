benchncnn can be used to test neural network inference performance

Only the network definition files (ncnn param) are required.

The large model binary files (ncnn bin) are not loaded but generated randomly for speed test.

More model networks may be added later.

---
Build
```shell
# assume you have already build ncnn library successfully
# uncomment the following line in <ncnn-root-dir>/CMakeLists.txt with your favorite editor

# add_subdirectory(benchmark)

cd <ncnn-root-dir>/<your-build-dir>
make -j4

# you can find benchncnn binary in <ncnn-root-dir>/<your-build-dir>/benchmark
```

Usage
```shell
# copy all param files to the current directory
./benchncnn [loop count] [num threads] [powersave] [gpu device] [cooling down]
```
run benchncnn on android device
```shell
# for running on android device, upload to /data/local/tmp/ folder
adb push benchncnn /data/local/tmp/
adb push <ncnn-root-dir>/benchmark/*.param /data/local/tmp/
adb shell

# executed in android adb shell
cd /data/local/tmp/
./benchncnn [loop count] [num threads] [powersave] [gpu device] [cooling down]
```

Parameter

|param|options|default|
|---|---|---|
|loop count|1~N|4|
|num threads|1~N|max_cpu_count|
|powersave|0=all cores, 1=little cores only, 2=big cores only|0|
|gpu device|-1=cpu-only, 0=gpu0, 1=gpu1 ...|-1|
|cooling down|0=disable, 1=enable|1|

---

Typical output (executed in android adb shell)


### AMD Ryzen Threadripper 3970X (Zen2 3.7 GHz ~ 4.5 GHz x 32)
```
i@s:~/qtang/ncnn/benchmark$ ../build-vulkan/benchmark/benchncnn 10 1 0 -1 0
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 0
          squeezenet  min =   11.73  max =   11.88  avg =   11.78
           mobilenet  min =   21.63  max =   21.73  avg =   21.68
        mobilenet_v2  min =   14.70  max =   14.95  avg =   14.82
        mobilenet_v3  min =   12.12  max =   12.17  avg =   12.15
          shufflenet  min =   14.08  max =   14.16  avg =   14.12
       shufflenet_v2  min =   25.99  max =   26.13  avg =   26.06
             mnasnet  min =   14.12  max =   14.17  avg =   14.14
     proxylessnasnet  min =   16.51  max =   16.71  avg =   16.61
     efficientnet_b0  min =   22.88  max =   22.97  avg =   22.93
        regnety_400m  min =   18.50  max =   18.61  avg =   18.56
           blazeface  min =    6.18  max =    6.27  avg =    6.21
           googlenet  min =   58.42  max =   58.60  avg =   58.49
            resnet18  min =   61.13  max =   61.84  avg =   61.40
             alexnet  min =   50.82  max =   50.98  avg =   50.92
               vgg16  min =  217.19  max =  218.40  avg =  217.87
            resnet50  min =  126.84  max =  137.46  avg =  128.21
      squeezenet_ssd  min =  114.24  max =  114.57  avg =  114.47
       mobilenet_ssd  min =   51.60  max =   51.89  avg =   51.77
      mobilenet_yolo  min =  125.09  max =  126.33  avg =  125.83
  mobilenetv2_yolov3  min =   57.51  max =   57.79  avg =   57.65
         yolov4-tiny  min =   85.65  max =   85.97  avg =   85.79
```

### NVIDIA Quadro RTX 8000 (TU102 SM x 72 + Tensor Core x 576)
```
i@s:~/qtang/ncnn/benchmark$ ../build-vulkan/benchmark/benchncnn 256 1 0 1 0
[0 Quadro RTX 8000]  queueC=2[8]  queueG=0[16]  queueT=1[2]
[0 Quadro RTX 8000]  bugsbn1=0  bugcopc=0  bugihfa=0
[0 Quadro RTX 8000]  fp16p=1  fp16s=1  fp16a=1  int8s=1  int8a=1
[0 Quadro RTX 8000]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
[1 Quadro RTX 8000]  queueC=2[8]  queueG=0[16]  queueT=1[2]
[1 Quadro RTX 8000]  bugsbn1=0  bugcopc=0  bugihfa=0
[1 Quadro RTX 8000]  fp16p=1  fp16s=1  fp16a=1  int8s=1  int8a=1
[1 Quadro RTX 8000]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 256
num_threads = 1
powersave = 0
gpu_device = 1
cooling_down = 0
          squeezenet  min =    0.84  max =    1.39  avg =    0.93
           mobilenet  min =    0.90  max =    2.30  avg =    0.91
        mobilenet_v2  min =    1.35  max =    9.59  avg =    1.46
        mobilenet_v3  min =    1.60  max =   77.94  avg =    2.12
          shufflenet  min =    0.86  max =    2.27  avg =    0.88
       shufflenet_v2  min =    1.25  max =    1.47  avg =    1.27
             mnasnet  min =    1.42  max =   20.77  avg =    1.72
     proxylessnasnet  min =    1.48  max =    1.67  avg =    1.49
     efficientnet_b0  min =    2.56  max =   12.86  avg =    2.77
        regnety_400m  min =    1.84  max =   14.98  avg =    2.42
           blazeface  min =    0.64  max =    0.90  avg =    0.65
           googlenet  min =    2.94  max =   76.82  avg =    3.45
            resnet18  min =    1.27  max =   10.56  avg =    1.56
             alexnet  min =    1.53  max =   71.76  avg =    1.96
               vgg16  min =    4.90  max =   78.12  avg =    5.80
            resnet50  min =    3.00  max =   12.51  avg =    3.07
      squeezenet_ssd  min =    5.60  max =   97.09  avg =    6.50
       mobilenet_ssd  min =    2.40  max =   93.64  avg =    3.30
      mobilenet_yolo  min =    2.96  max =   19.15  avg =    3.25
  mobilenetv2_yolov3  min =    4.52  max =   66.96  avg =    5.32
         yolov4-tiny  min =    9.32  max =   72.92  avg =   14.01

```

### NVIDIA RTX3090 (GA102 SM x 82 + Tensor Core 328)
```
(base) i@t:~/wls/ncnn/benchmark$ ../build/benchmark/benchncnn 32 1 0 0 0
[0 GeForce RTX 3090]  queueC=2[8]  queueG=0[16]  queueT=1[2]
[0 GeForce RTX 3090]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 GeForce RTX 3090]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 GeForce RTX 3090]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
[1 GeForce RTX 3090]  queueC=2[8]  queueG=0[16]  queueT=1[2]
[1 GeForce RTX 3090]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[1 GeForce RTX 3090]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[1 GeForce RTX 3090]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 32
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =    1.76  max =    2.74  avg =    1.80
     squeezenet_int8  min =   47.10  max =   47.75  avg =   47.21
           mobilenet  min =    4.77  max =    5.79  avg =    5.20
      mobilenet_int8  min =   64.19  max =   67.05  avg =   64.39
        mobilenet_v2  min =    2.44  max =   20.89  avg =    6.98
        mobilenet_v3  min =    2.75  max =    2.87  avg =    2.77
          shufflenet  min =    2.20  max =    2.62  avg =    2.46
       shufflenet_v2  min =    5.10  max =    7.43  avg =    5.75
             mnasnet  min =    3.47  max =    3.50  avg =    3.48
     proxylessnasnet  min =    2.59  max =    9.08  avg =    7.28
     efficientnet_b0  min =    3.87  max =    4.65  avg =    3.91
   efficientnetv2_b0  min =   29.48  max =   41.90  avg =   30.14
        regnety_400m  min =    2.89  max =    2.99  avg =    2.91
           blazeface  min =    1.55  max =    2.14  avg =    1.60
           googlenet  min =    4.33  max =   17.89  avg =    6.05
      googlenet_int8  min =  174.46  max =  178.19  avg =  174.74
            resnet18  min =    2.14  max =   11.04  avg =    5.33
       resnet18_int8  min =  193.37  max =  193.83  avg =  193.55
             alexnet  min =    2.37  max =   15.99  avg =    4.50
               vgg16  min =    4.55  max =   16.65  avg =    5.22
          vgg16_int8  min = 1538.76  max = 1544.81  avg = 1540.79
            resnet50  min =    4.13  max =   25.86  avg =    5.80
       resnet50_int8  min =  400.89  max =  401.72  avg =  401.29
      squeezenet_ssd  min =    6.95  max =    7.81  avg =    7.07
 squeezenet_ssd_int8  min =  158.51  max =  159.04  avg =  158.68
       mobilenet_ssd  min =    4.36  max =   18.98  avg =    9.40
  mobilenet_ssd_int8  min =  130.74  max =  130.92  avg =  130.83
      mobilenet_yolo  min =    3.96  max =   11.94  avg =    6.48
  mobilenetv2_yolov3  min =    6.07  max =    6.21  avg =    6.13
         yolov4-tiny  min =   13.01  max =   26.78  avg =   14.87
```

### AMD Ryzen Embedded V1605B (Zen 2.0 GHz ~ 3.6 GHz x 4 + Radeon Vega 8 1.1GHz 8CU)
```
C:\Users\i\Desktop\benchmark>benchncnn.exe 32 1 0 -1 0
loop_count = 32
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 0
          squeezenet  min =   22.13  max =   24.07  avg =   22.88
     squeezenet_int8  min =   58.54  max =   62.21  avg =   59.55
           mobilenet  min =   40.99  max =   43.67  avg =   41.70
      mobilenet_int8  min =   98.06  max =  111.37  avg =  101.15
        mobilenet_v2  min =   26.53  max =   28.96  avg =   27.81
        mobilenet_v3  min =   22.96  max =   25.25  avg =   23.30
          shufflenet  min =   20.17  max =   28.78  avg =   21.09
       shufflenet_v2  min =   19.06  max =   19.72  avg =   19.47
             mnasnet  min =   25.11  max =   39.53  avg =   27.54
     proxylessnasnet  min =   28.84  max =   35.16  avg =   30.03
     efficientnet_b0  min =   43.16  max =   46.03  avg =   43.65
   efficientnetv2_b0  min =   48.64  max =   52.07  avg =   49.62
        regnety_400m  min =   33.43  max =   35.87  avg =   33.97
           blazeface  min =    5.43  max =    6.04  avg =    5.56
           googlenet  min =   85.80  max =   90.93  avg =   87.65
      googlenet_int8  min =  214.37  max =  230.75  avg =  219.50
            resnet18  min =   76.58  max =   80.38  avg =   77.34
       resnet18_int8  min =  231.16  max =  255.22  avg =  236.65
             alexnet  min =   60.69  max =   64.06  avg =   61.34
               vgg16  min =  286.45  max =  307.04  avg =  290.86
          vgg16_int8  min = 1797.58  max = 2079.73  avg = 1844.78
            resnet50  min =  198.27  max =  215.03  avg =  201.37
       resnet50_int8  min =  493.52  max =  499.67  avg =  496.95
      squeezenet_ssd  min =  189.97  max =  198.53  avg =  192.10
 squeezenet_ssd_int8  min =  198.81  max =  214.55  avg =  203.59
       mobilenet_ssd  min =   87.56  max =   92.72  avg =   89.03
  mobilenet_ssd_int8  min =  196.97  max =  209.51  avg =  201.95
      mobilenet_yolo  min =  206.87  max =  218.48  avg =  210.84
  mobilenetv2_yolov3  min =  102.72  max =  108.18  avg =  104.62
         yolov4-tiny  min =  117.97  max =  134.73  avg =  121.26

C:\Users\i\Desktop\benchmark>benchncnn.exe 32 2 0 -1 0
loop_count = 32
num_threads = 2
powersave = 0
gpu_device = -1
cooling_down = 0
          squeezenet  min =   13.43  max =   14.35  avg =   13.62
     squeezenet_int8  min =   32.29  max =   50.76  avg =   33.56
           mobilenet  min =   23.42  max =   25.10  avg =   24.09
      mobilenet_int8  min =   51.99  max =   55.42  avg =   53.01
        mobilenet_v2  min =   15.45  max =   15.75  avg =   15.59
        mobilenet_v3  min =   14.32  max =   14.75  avg =   14.39
          shufflenet  min =   12.64  max =   12.83  avg =   12.69
       shufflenet_v2  min =   11.45  max =   12.44  avg =   11.60
             mnasnet  min =   14.43  max =   20.45  avg =   15.11
     proxylessnasnet  min =   16.18  max =   16.38  avg =   16.24
     efficientnet_b0  min =   25.25  max =   28.42  avg =   26.59
   efficientnetv2_b0  min =   27.57  max =   32.05  avg =   30.04
        regnety_400m  min =   22.74  max =   24.75  avg =   23.31
           blazeface  min =    3.44  max =    3.83  avg =    3.62
           googlenet  min =   49.39  max =   66.76  avg =   53.76
      googlenet_int8  min =  113.89  max =  136.75  avg =  119.29
            resnet18  min =   43.77  max =   67.24  avg =   46.14
       resnet18_int8  min =  121.44  max =  148.01  avg =  126.95
             alexnet  min =   34.46  max =   37.38  avg =   35.50
               vgg16  min =  177.16  max =  207.25  avg =  184.19
          vgg16_int8  min =  951.86  max = 1155.60  avg =  990.51
            resnet50  min =  112.28  max =  137.18  avg =  115.64
       resnet50_int8  min =  260.69  max =  272.26  avg =  265.89
      squeezenet_ssd  min =  108.07  max =  121.66  avg =  110.35
 squeezenet_ssd_int8  min =  109.01  max =  126.86  avg =  111.96
       mobilenet_ssd  min =   49.60  max =   52.62  avg =   50.46
  mobilenet_ssd_int8  min =  104.22  max =  111.07  avg =  106.33
      mobilenet_yolo  min =  117.42  max =  136.73  avg =  122.92
  mobilenetv2_yolov3  min =   61.66  max =   65.22  avg =   63.01
         yolov4-tiny  min =   72.64  max =   77.09  avg =   74.30

C:\Users\i\Desktop\benchmark>benchncnn.exe 32 4 0 -1 0
loop_count = 32
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 0
          squeezenet  min =    9.19  max =   14.82  avg =   11.15
     squeezenet_int8  min =   19.00  max =   40.30  avg =   24.80
           mobilenet  min =   18.02  max =   39.84  avg =   27.38
      mobilenet_int8  min =   28.04  max =   57.59  avg =   34.15
        mobilenet_v2  min =   10.26  max =   17.79  avg =   13.36
        mobilenet_v3  min =    8.87  max =   10.87  avg =    9.11
          shufflenet  min =    8.93  max =   11.96  avg =    9.34
       shufflenet_v2  min =    7.37  max =   13.10  avg =    8.72
             mnasnet  min =    9.24  max =   14.90  avg =   11.32
     proxylessnasnet  min =   10.21  max =   11.89  avg =   10.39
     efficientnet_b0  min =   16.22  max =   23.71  avg =   16.59
   efficientnetv2_b0  min =   17.44  max =   31.42  avg =   22.85
        regnety_400m  min =   18.32  max =   24.02  avg =   18.90
           blazeface  min =    2.22  max =    2.81  avg =    2.30
           googlenet  min =   31.52  max =   51.80  avg =   42.11
      googlenet_int8  min =   65.47  max =  114.41  avg =   75.98
            resnet18  min =   28.90  max =   64.62  avg =   37.58
       resnet18_int8  min =   71.29  max =  136.67  avg =  103.03
             alexnet  min =   23.67  max =   34.01  avg =   29.78
               vgg16  min =  142.18  max =  211.00  avg =  170.46
          vgg16_int8  min =  531.36  max =  871.25  avg =  625.60
            resnet50  min =   69.23  max =  108.67  avg =   73.68
       resnet50_int8  min =  149.18  max =  309.88  avg =  168.68
      squeezenet_ssd  min =   68.83  max =   81.70  avg =   71.01
 squeezenet_ssd_int8  min =   66.34  max =  118.16  avg =   74.34
       mobilenet_ssd  min =   29.96  max =   34.32  avg =   30.74
  mobilenet_ssd_int8  min =   56.87  max =   92.24  avg =   65.57
      mobilenet_yolo  min =   74.26  max =  113.91  avg =   81.28
  mobilenetv2_yolov3  min =   42.16  max =   63.49  avg =   45.34
         yolov4-tiny  min =   53.06  max =   69.84  avg =   55.81

C:\Users\i\Desktop\benchmark>benchncnn.exe 32 1 0 0 0
[0 AMD Radeon(TM) Vega 8 Graphics]  queueC=1[2]  queueG=0[1]  queueT=2[1]
[0 AMD Radeon(TM) Vega 8 Graphics]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 AMD Radeon(TM) Vega 8 Graphics]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 AMD Radeon(TM) Vega 8 Graphics]  subgroup=64  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 32
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =    6.78  max =    7.09  avg =    6.91
     squeezenet_int8  min =   58.93  max =   62.53  avg =   60.11
           mobilenet  min =    8.08  max =    8.39  avg =    8.25
      mobilenet_int8  min =   97.74  max =  116.77  avg =  100.17
        mobilenet_v2  min =    7.95  max =    8.27  avg =    8.14
        mobilenet_v3  min =    8.70  max =    9.70  avg =    9.02
          shufflenet  min =    6.36  max =    7.64  avg =    7.01
       shufflenet_v2  min =    7.04  max =    8.12  avg =    7.50
             mnasnet  min =    8.07  max =    9.08  avg =    8.38
     proxylessnasnet  min =    8.56  max =    9.66  avg =    8.81
     efficientnet_b0  min =   16.68  max =   18.00  avg =   17.30
   efficientnetv2_b0  min =  394.82  max =  404.88  avg =  401.05
        regnety_400m  min =   11.92  max =   12.17  avg =   12.03
           blazeface  min =    4.82  max =    6.50  avg =    5.42
           googlenet  min =   18.44  max =   19.66  avg =   19.18
      googlenet_int8  min =  213.41  max =  231.79  avg =  218.31
            resnet18  min =   14.27  max =   14.72  avg =   14.44
       resnet18_int8  min =  228.79  max =  249.65  avg =  236.06
             alexnet  min =   17.31  max =   18.31  avg =   17.69
               vgg16  min =  111.85  max =  123.35  avg =  112.98
          vgg16_int8  min = 1789.64  max = 1838.84  avg = 1826.05
            resnet50  min =   31.61  max =   32.86  avg =   32.12
       resnet50_int8  min =  483.57  max =  505.72  avg =  491.76
      squeezenet_ssd  min =   99.66  max =  105.68  avg =  104.57
 squeezenet_ssd_int8  min =  200.48  max =  208.71  avg =  203.02
       mobilenet_ssd  min =   33.45  max =   35.64  avg =   34.75
  mobilenet_ssd_int8  min =  195.14  max =  205.35  avg =  200.18
      mobilenet_yolo  min =   59.20  max =   61.06  avg =   60.47
  mobilenetv2_yolov3  min =   31.48  max =   33.25  avg =   32.84
         yolov4-tiny  min =   93.75  max =   97.45  avg =   96.00
 ```

### Qualcomm SM8150-AC Snapdragon 855+ (Kyro485 2.96 GHz + 2.42 GHz x 3 + 1.80 GHz x 4 + Adreno 640)
```
OnePlus7T:/data/local/tmp $ ./benchncnn 8 4 2 -1 1
loop_count = 8
num_threads = 4
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =    4.46  max =    4.58  avg =    4.51
     squeezenet_int8  min =    5.17  max =    5.29  avg =    5.21
           mobilenet  min =    5.94  max =    6.04  avg =    5.99
      mobilenet_int8  min =    5.08  max =    5.17  avg =    5.12
        mobilenet_v2  min =    5.08  max =    5.19  avg =    5.13
        mobilenet_v3  min =    4.93  max =    5.12  avg =    5.02
          shufflenet  min =    4.34  max =    4.50  avg =    4.40
       shufflenet_v2  min =    3.46  max =    3.57  avg =    3.52
             mnasnet  min =    4.62  max =    4.76  avg =    4.69
     proxylessnasnet  min =    5.48  max =    5.67  avg =    5.59
     efficientnet_b0  min =   10.01  max =   10.29  avg =   10.09
        regnety_400m  min =    8.26  max =    8.46  avg =    8.33
           blazeface  min =    1.92  max =    2.36  avg =    2.04
           googlenet  min =   17.38  max =   17.55  avg =   17.49
      googlenet_int8  min =   18.57  max =   18.69  avg =   18.63
            resnet18  min =   11.54  max =   11.80  avg =   11.63
       resnet18_int8  min =   18.13  max =   18.62  avg =   18.27
             alexnet  min =   16.53  max =   16.61  avg =   16.58
               vgg16  min =   65.81  max =   67.11  avg =   66.40
          vgg16_int8  min =  145.95  max =  148.61  avg =  147.95
            resnet50  min =   28.28  max =   28.36  avg =   28.32
       resnet50_int8  min =   34.74  max =   34.98  avg =   34.88
      squeezenet_ssd  min =   22.87  max =   23.42  avg =   23.05
 squeezenet_ssd_int8  min =   26.83  max =   26.94  avg =   26.89
       mobilenet_ssd  min =   16.71  max =   16.92  avg =   16.77
  mobilenet_ssd_int8  min =   14.55  max =   15.03  avg =   14.70
      mobilenet_yolo  min =   28.60  max =   29.43  avg =   28.84
  mobilenetv2_yolov3  min =   19.43  max =   19.84  avg =   19.61
         yolov4-tiny  min =   29.16  max =   29.57  avg =   29.38

OnePlus7T:/data/local/tmp $ ./benchncnn 8 1 2 -1 1
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =    9.69  max =    9.75  avg =    9.72
     squeezenet_int8  min =   10.24  max =   10.40  avg =   10.30
           mobilenet  min =   15.19  max =   15.29  avg =   15.23
      mobilenet_int8  min =   12.28  max =   12.37  avg =   12.31
        mobilenet_v2  min =   10.56  max =   10.66  avg =   10.61
        mobilenet_v3  min =    9.10  max =    9.25  avg =    9.14
          shufflenet  min =    6.95  max =    7.11  avg =    7.02
       shufflenet_v2  min =    6.66  max =    6.81  avg =    6.73
             mnasnet  min =    9.97  max =   10.13  avg =   10.03
     proxylessnasnet  min =   12.32  max =   12.47  avg =   12.37
     efficientnet_b0  min =   23.32  max =   23.45  avg =   23.37
        regnety_400m  min =   12.40  max =   12.50  avg =   12.44
           blazeface  min =    3.86  max =    4.23  avg =    3.96
           googlenet  min =   43.45  max =   43.59  avg =   43.52
      googlenet_int8  min =   42.83  max =   43.21  avg =   43.04
            resnet18  min =   23.57  max =   23.92  avg =   23.73
       resnet18_int8  min =   46.70  max =   47.05  avg =   46.91
             alexnet  min =   37.83  max =   37.95  avg =   37.89
               vgg16  min =  138.18  max =  138.84  avg =  138.53
          vgg16_int8  min =  368.28  max =  368.97  avg =  368.55
            resnet50  min =   68.52  max =   68.78  avg =   68.69
       resnet50_int8  min =   87.54  max =   88.07  avg =   87.92
      squeezenet_ssd  min =   36.45  max =   37.45  avg =   36.70
 squeezenet_ssd_int8  min =   47.19  max =   47.43  avg =   47.30
       mobilenet_ssd  min =   38.27  max =   38.39  avg =   38.36
  mobilenet_ssd_int8  min =   32.20  max =   33.01  avg =   32.49
      mobilenet_yolo  min =   69.79  max =   70.12  avg =   69.94
  mobilenetv2_yolov3  min =   38.65  max =   38.85  avg =   38.78
         yolov4-tiny  min =   52.62  max =   52.96  avg =   52.74

OnePlus7T:/data/local/tmp $ ./benchncnn 8 1 2 0 1
[0 Adreno (TM) 640]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 640]  buglssc=0  bugsbn1=0  buglbia=0  bugihfa=1
[0 Adreno (TM) 640]  fp16p=1  fp16s=0  fp16a=1  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = 0
cooling_down = 1
          squeezenet  min =    9.27  max =    9.56  avg =    9.43
           mobilenet  min =   13.04  max =   13.42  avg =   13.23
        mobilenet_v2  min =   10.92  max =   11.33  avg =   11.06
        mobilenet_v3  min =   12.28  max =   12.78  avg =   12.45
          shufflenet  min =    8.26  max =    8.47  avg =    8.38
       shufflenet_v2  min =    9.03  max =    9.28  avg =    9.14
             mnasnet  min =   11.40  max =   11.76  avg =   11.60
     proxylessnasnet  min =   12.40  max =   12.92  avg =   12.55
     efficientnet_b0  min =   23.04  max =   23.29  avg =   23.15
        regnety_400m  min =   15.85  max =   16.38  avg =   16.16
           blazeface  min =    2.80  max =    3.80  avg =    3.24
           googlenet  min =   29.84  max =   30.14  avg =   29.97
            resnet18  min =   25.12  max =   25.50  avg =   25.31
             alexnet  min =   30.62  max =   31.66  avg =   31.23
               vgg16  min =  159.00  max =  183.80  avg =  170.15
            resnet50  min =   59.69  max =   60.17  avg =   59.98
      squeezenet_ssd  min =   39.39  max =   40.21  avg =   39.97
       mobilenet_ssd  min =   27.95  max =   28.15  avg =   28.05
      mobilenet_yolo  min =   53.29  max =   54.21  avg =   53.98
  mobilenetv2_yolov3  min =   28.68  max =   28.92  avg =   28.79
```

### Qualcomm MSM6150 Snapdragon 675 (Kyro460 2.0GHz x 2 + Kyro460 1.7GHz x 6 + Adreno 612)
```
violet:/data/local/tmp/ncnn $ ./benchncnn 8 2 0
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
          squeezenet  min =   23.29  max =   24.65  avg =   23.95
     squeezenet_int8  min =   23.24  max =   61.55  avg =   31.20
           mobilenet  min =   31.60  max =   32.10  avg =   31.80
      mobilenet_int8  min =   30.35  max =   32.03  avg =   30.95
        mobilenet_v2  min =   25.92  max =   26.45  avg =   26.08
          shufflenet  min =   11.91  max =   12.11  avg =   12.00
             mnasnet  min =   21.38  max =   21.71  avg =   21.51
     proxylessnasnet  min =   25.53  max =   25.78  avg =   25.62
           googlenet  min =   93.62  max =  100.67  avg =   94.86
      googlenet_int8  min =   90.74  max =   91.06  avg =   90.87
            resnet18  min =   85.84  max =   87.37  avg =   86.50
       resnet18_int8  min =   77.88  max =   78.11  avg =   78.00
             alexnet  min =  196.33  max =  201.73  avg =  200.19
               vgg16  min =  560.71  max =  571.75  avg =  564.84
          vgg16_int8  min =  651.51  max =  652.68  avg =  652.12
            resnet50  min =  178.25  max =  179.86  avg =  178.77
       resnet50_int8  min =  181.07  max =  183.26  avg =  181.64
      squeezenet_ssd  min =   64.86  max =   68.39  avg =   66.05
 squeezenet_ssd_int8  min =   69.61  max =   70.37  avg =   69.93
       mobilenet_ssd  min =   65.92  max =   67.03  avg =   66.41
  mobilenet_ssd_int8  min =   61.54  max =   63.38  avg =   62.27
      mobilenet_yolo  min =  143.42  max =  146.69  avg =  144.33
    mobilenet_yolov3  min =  150.45  max =  152.30  avg =  151.36

violet:/data/local/tmp/ncnn $ ./benchncnn 8 1 0
loop_count = 8
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =   36.04  max =   37.25  avg =   36.48
     squeezenet_int8  min =   37.82  max =   79.20  avg =   43.13
           mobilenet  min =   54.29  max =   54.73  avg =   54.41
      mobilenet_int8  min =   58.90  max =   60.11  avg =   59.39
        mobilenet_v2  min =   38.64  max =   40.22  avg =   38.97
          shufflenet  min =   18.05  max =   18.39  avg =   18.19
             mnasnet  min =   34.65  max =   34.98  avg =   34.79
     proxylessnasnet  min =   42.61  max =   43.12  avg =   42.80
           googlenet  min =  164.74  max =  165.89  avg =  165.34
      googlenet_int8  min =  159.93  max =  160.38  avg =  160.12
            resnet18  min =  135.76  max =  137.93  avg =  136.98
       resnet18_int8  min =  140.22  max =  144.06  avg =  141.92
             alexnet  min =  391.01  max =  396.85  avg =  392.74
               vgg16  min = 1019.35  max = 1022.75  avg = 1021.26
          vgg16_int8  min = 1122.25  max = 1137.99  avg = 1124.78
            resnet50  min =  302.16  max =  304.22  avg =  303.05
       resnet50_int8  min =  318.35  max =  319.50  avg =  318.84
      squeezenet_ssd  min =   91.26  max =   94.86  avg =   92.39
 squeezenet_ssd_int8  min =  105.06  max =  106.17  avg =  105.56
       mobilenet_ssd  min =  105.01  max =  105.95  avg =  105.40
  mobilenet_ssd_int8  min =  119.93  max =  120.50  avg =  120.19
      mobilenet_yolo  min =  229.87  max =  230.76  avg =  230.21
    mobilenet_yolov3  min =  242.10  max =  242.91  avg =  242.47  
```

### Kirin 970 (Cortex-A73 2.4GHz x 4 + Cortex-A53 1.8GHz x 4)
```
HWEML:/data/local/tmp/ncnnbench $ ./benchncnn 8 4 2 -1 1
[0 Mali-G72]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-G72]  buglssc=0  bugsbn1=0  buglbia=0  bugihfa=1
[0 Mali-G72]  fp16p=1  fp16s=0  fp16a=1  int8s=0  int8a=0
loop_count = 8
num_threads = 4
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   24.38  max =   28.03  avg =   25.83
     squeezenet_int8  min =   21.79  max =   24.80  avg =   22.60
           mobilenet  min =   34.09  max =   36.88  avg =   35.93
      mobilenet_int8  min =   52.62  max =   61.70  avg =   55.38
        mobilenet_v2  min =   23.71  max =   25.70  avg =   24.49
        mobilenet_v3  min =   20.66  max =   25.68  avg =   23.07
          shufflenet  min =   17.89  max =   19.91  avg =   18.53
       shufflenet_v2  min =   13.73  max =   16.54  avg =   15.37
             mnasnet  min =   24.36  max =   27.14  avg =   25.58
     proxylessnasnet  min =   27.19  max =   29.70  avg =   28.59
     efficientnet_b0  min =   49.31  max =   50.26  avg =   49.70
        regnety_400m  min =   42.54  max =   51.22  avg =   46.71
           blazeface  min =    5.49  max =    7.67  avg =    6.27
           googlenet  min =   72.67  max =   81.22  avg =   75.92
      googlenet_int8  min =   67.60  max =   74.50  avg =   71.21
            resnet18  min =   69.32  max =   81.59  avg =   73.45
       resnet18_int8  min =   60.92  max =   68.11  avg =   64.18
             alexnet  min =   60.90  max =   79.28  avg =   66.72
               vgg16  min =  337.01  max =  378.89  avg =  352.37
          vgg16_int8  min =  465.88  max =  505.19  avg =  489.76
            resnet50  min =  207.75  max =  220.74  avg =  214.42
       resnet50_int8  min =  165.67  max =  183.80  avg =  171.27
      squeezenet_ssd  min =   72.77  max =   84.45  avg =   79.09
 squeezenet_ssd_int8  min =   75.37  max =   86.58  avg =   78.70
       mobilenet_ssd  min =   88.88  max =   96.43  avg =   92.02
  mobilenet_ssd_int8  min =   89.04  max =  101.35  avg =   92.23
      mobilenet_yolo  min =  189.73  max =  206.55  avg =  193.64
  mobilenetv2_yolov3  min =   99.08  max =  111.64  avg =  104.23

HWEML:/data/local/tmp/ncnnbench $ ./benchncnn 8 1 2 -1 1
[0 Mali-G72]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-G72]  buglssc=0  bugsbn1=0  buglbia=0  bugihfa=1
[0 Mali-G72]  fp16p=1  fp16s=0  fp16a=1  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   73.47  max =   81.39  avg =   76.06
     squeezenet_int8  min =   62.63  max =   73.66  avg =   66.52
           mobilenet  min =  103.85  max =  112.83  avg =  108.98
      mobilenet_int8  min =  152.27  max =  161.26  avg =  157.17
        mobilenet_v2  min =   70.53  max =   87.26  avg =   76.67
        mobilenet_v3  min =   59.87  max =   68.59  avg =   63.08
          shufflenet  min =   36.69  max =   41.45  avg =   39.24
       shufflenet_v2  min =   33.97  max =   37.84  avg =   35.03
             mnasnet  min =   69.24  max =   79.73  avg =   74.20
     proxylessnasnet  min =   78.63  max =   88.57  avg =   81.83
     efficientnet_b0  min =  147.45  max =  159.07  avg =  152.09
        regnety_400m  min =   90.83  max =   98.51  avg =   93.82
           blazeface  min =   10.05  max =   11.59  avg =   10.78
           googlenet  min =  240.26  max =  277.71  avg =  259.61
      googlenet_int8  min =  214.64  max =  233.56  avg =  225.01
            resnet18  min =  245.62  max =  268.49  avg =  260.37
       resnet18_int8  min =  184.85  max =  194.91  avg =  190.60
             alexnet  min =  202.52  max =  241.12  avg =  211.51
               vgg16  min = 1632.98  max = 1769.05  avg = 1710.89
          vgg16_int8  min = 1237.01  max = 1316.40  avg = 1273.44
            resnet50  min =  558.41  max =  601.59  avg =  581.26
       resnet50_int8  min =  425.26  max =  445.19  avg =  436.22
      squeezenet_ssd  min =  228.50  max =  255.89  avg =  244.63
 squeezenet_ssd_int8  min =  166.97  max =  193.77  avg =  180.22
       mobilenet_ssd  min =  226.54  max =  246.62  avg =  235.75
  mobilenet_ssd_int8  min =  231.35  max =  249.63  avg =  241.29
      mobilenet_yolo  min =  469.71  max =  508.79  avg =  497.50
  mobilenetv2_yolov3  min =  242.88  max =  265.30  avg =  254.68

HWEML:/data/local/tmp/ncnnbench $ ./benchncnn 4 1 2 0 1
[0 Mali-G72]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-G72]  buglssc=0  bugsbn1=0  buglbia=0  bugihfa=1
[0 Mali-G72]  fp16p=1  fp16s=0  fp16a=1  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = 0
cooling_down = 1
          squeezenet  min =   24.54  max =   25.75  avg =   25.16
           mobilenet  min =   22.03  max =   29.61  avg =   27.31
        mobilenet_v2  min =   20.15  max =   28.05  avg =   25.35
        mobilenet_v3  min =   34.26  max =   37.49  avg =   35.51
          shufflenet  min =   26.29  max =   27.68  avg =   26.86
       shufflenet_v2  min =   29.60  max =   32.08  avg =   31.27
             mnasnet  min =   25.85  max =   29.38  avg =   27.98
     proxylessnasnet  min =   23.64  max =   30.09  avg =   26.36
     efficientnet_b0  min =   52.55  max =   58.51  avg =   55.56
        regnety_400m  min =   37.81  max =   43.22  avg =   40.30
           blazeface  min =    9.14  max =   10.93  avg =   10.08
           googlenet  min =   60.19  max =   62.84  avg =   61.51
            resnet18  min =   50.42  max =   52.93  avg =   51.70
             alexnet  min =  195.34  max =  196.98  avg =  196.14
               vgg16  min =  725.88  max =  751.20  avg =  739.99
            resnet50  min =  124.47  max =  125.93  avg =  125.02
      squeezenet_ssd  min =   91.79  max =   97.04  avg =   93.56
       mobilenet_ssd  min =   51.81  max =   59.31  avg =   54.09
      mobilenet_yolo  min =  124.67  max =  127.62  avg =  126.53
  mobilenetv2_yolov3  min =   53.11  max =   54.81  avg =   54.11
```

### Qualcomm MSM8998 Snapdragon 835 (Kyro 2.45GHz x 4 + Kyro 1.9GHz x 4 + Adreno 540)
```
taimen:/data/local/tmp/ncnnbench $ ./benchncnn 8 4 2 -1 0
[0 Adreno (TM) 540]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 540]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 540]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 4
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   28.46  max =   30.89  avg =   29.77
     squeezenet_int8  min =   30.32  max =   32.92  avg =   31.68
           mobilenet  min =   36.65  max =   38.37  avg =   37.32
      mobilenet_int8  min =   62.91  max =   66.71  avg =   64.49
        mobilenet_v2  min =   27.85  max =   31.21  avg =   29.41
        mobilenet_v3  min =   23.83  max =   26.40  avg =   24.79
          shufflenet  min =   15.65  max =   16.88  avg =   16.27
       shufflenet_v2  min =   13.70  max =   14.49  avg =   14.08
             mnasnet  min =   25.04  max =   28.35  avg =   26.45
     proxylessnasnet  min =   27.49  max =   29.58  avg =   28.62
     efficientnet_b0  min =   48.43  max =   49.41  avg =   48.85
        regnety_400m  min =   42.48  max =   43.78  avg =   43.18
           blazeface  min =    4.39  max =    4.68  avg =    4.51
           googlenet  min =   75.98  max =   78.40  avg =   77.37
      googlenet_int8  min =   79.26  max =   83.20  avg =   80.55
            resnet18  min =   73.60  max =   76.97  avg =   75.63
       resnet18_int8  min =   62.93  max =   65.94  avg =   64.50
             alexnet  min =   64.18  max =   67.02  avg =   65.49
               vgg16  min =  389.39  max =  399.13  avg =  394.09
          vgg16_int8  min =  509.06  max =  524.41  avg =  514.76
            resnet50  min =  188.21  max =  194.58  avg =  191.98
       resnet50_int8  min =  182.84  max =  187.22  avg =  184.23
      squeezenet_ssd  min =   77.69  max =   81.17  avg =   79.24
 squeezenet_ssd_int8  min =   81.71  max =   84.12  avg =   82.90
       mobilenet_ssd  min =   78.35  max =   81.50  avg =   79.82
  mobilenet_ssd_int8  min =   96.84  max =  100.97  avg =   98.42
      mobilenet_yolo  min =  167.32  max =  170.71  avg =  168.87
  mobilenetv2_yolov3  min =   97.00  max =  102.11  avg =   99.01

taimen:/data/local/tmp/ncnnbench $ ./benchncnn 8 1 2 -1 1
[0 Adreno (TM) 540]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 540]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 540]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   67.25  max =   71.39  avg =   69.35
     squeezenet_int8  min =   62.12  max =   66.35  avg =   63.73
           mobilenet  min =  103.30  max =  110.39  avg =  107.13
      mobilenet_int8  min =  155.24  max =  161.42  avg =  157.82
        mobilenet_v2  min =   71.89  max =   74.73  avg =   73.48
        mobilenet_v3  min =   58.35  max =   63.43  avg =   60.68
          shufflenet  min =   35.96  max =   39.43  avg =   36.94
       shufflenet_v2  min =   35.53  max =   39.86  avg =   37.10
             mnasnet  min =   66.71  max =   74.00  avg =   68.65
     proxylessnasnet  min =   76.50  max =   82.20  avg =   78.57
     efficientnet_b0  min =  142.32  max =  152.17  avg =  146.14
        regnety_400m  min =   89.60  max =   98.27  avg =   92.62
           blazeface  min =   10.45  max =   12.81  avg =   11.07
           googlenet  min =  222.75  max =  233.61  avg =  228.38
      googlenet_int8  min =  206.70  max =  212.20  avg =  209.24
            resnet18  min =  210.86  max =  220.25  avg =  213.65
       resnet18_int8  min =  176.04  max =  183.58  avg =  178.71
             alexnet  min =  185.97  max =  195.91  avg =  191.40
               vgg16  min = 1176.82  max = 1200.64  avg = 1187.88
          vgg16_int8  min = 1086.52  max = 1105.00  avg = 1095.53
            resnet50  min =  517.48  max =  533.99  avg =  526.04
       resnet50_int8  min =  417.30  max =  435.81  avg =  422.36
      squeezenet_ssd  min =  164.88  max =  171.21  avg =  167.51
 squeezenet_ssd_int8  min =  164.78  max =  171.77  avg =  168.36
       mobilenet_ssd  min =  221.41  max =  229.13  avg =  226.18
  mobilenet_ssd_int8  min =  234.15  max =  245.91  avg =  239.01
      mobilenet_yolo  min =  471.34  max =  484.99  avg =  477.15
  mobilenetv2_yolov3  min =  249.14  max =  257.61  avg =  252.54

taimen:/data/local/tmp/ncnnbench $ ./benchncnn 8 1 2 0 1
[0 Adreno (TM) 540]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 540]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 540]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = 0
cooling_down = 1
          squeezenet  min =   18.74  max =   19.89  avg =   19.22
           mobilenet  min =   21.19  max =   25.61  avg =   22.94
        mobilenet_v2  min =   24.15  max =   34.68  avg =   30.12
        mobilenet_v3  min =   25.94  max =   33.15  avg =   30.09
          shufflenet  min =   25.05  max =   31.41  avg =   27.85
       shufflenet_v2  min =   28.82  max =   32.04  avg =   30.95
             mnasnet  min =   21.34  max =   27.69  avg =   24.17
     proxylessnasnet  min =   25.51  max =   30.03  avg =   28.01
     efficientnet_b0  min =   42.94  max =   47.44  avg =   45.28
        regnety_400m  min =   36.36  max =   55.73  avg =   41.82
           blazeface  min =   11.14  max =   13.11  avg =   12.20
           googlenet  min =   49.72  max =   56.92  avg =   51.79
            resnet18  min =   44.63  max =   47.37  avg =   45.86
             alexnet  min =   42.83  max =   46.34  avg =   44.63
               vgg16  min =  568.82  max =  586.75  avg =  578.60
            resnet50  min =  108.63  max =  115.76  avg =  110.38
      squeezenet_ssd  min =   85.22  max =  104.73  avg =   93.14
       mobilenet_ssd  min =   49.91  max =   56.86  avg =   52.33
      mobilenet_yolo  min =   98.76  max =  109.37  avg =  102.27
  mobilenetv2_yolov3  min =   57.49  max =   61.15  avg =   58.74
```

### Qualcomm SDM660 Snapdragon 660 (Kyro260 2.2GHz x 4 + Kyro260 1.84GHz x 4 + Adreno 512)
```
lavender:/data/local/tmp/ncnnbench $ ./benchncnn 8 8 0 -1 1
[0 Adreno (TM) 512]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 512]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 512]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 8
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   29.05  max =   44.86  avg =   33.26
     squeezenet_int8  min =   35.47  max =   37.10  avg =   36.09
           mobilenet  min =   31.59  max =   33.47  avg =   32.33
      mobilenet_int8  min =   77.50  max =   91.15  avg =   82.98
        mobilenet_v2  min =   33.63  max =   35.43  avg =   34.54
        mobilenet_v3  min =   29.97  max =   49.80  avg =   34.81
          shufflenet  min =   28.52  max =   30.09  avg =   29.09
       shufflenet_v2  min =   19.15  max =   21.15  avg =   19.99
             mnasnet  min =   29.91  max =   35.11  avg =   31.46
     proxylessnasnet  min =   33.28  max =  117.09  avg =   55.22
     efficientnet_b0  min =   52.29  max =   57.93  avg =   55.04
        regnety_400m  min =   96.05  max =  116.42  avg =  102.07
           blazeface  min =    7.98  max =   11.83  avg =    8.89
           googlenet  min =   76.88  max =  103.99  avg =   84.54
      googlenet_int8  min =   97.68  max =  118.56  avg =  104.92
            resnet18  min =   75.93  max =   89.31  avg =   80.00
       resnet18_int8  min =   73.27  max =   80.84  avg =   76.19
             alexnet  min =   90.94  max =  114.57  avg =   96.42
               vgg16  min =  381.30  max =  615.62  avg =  555.96
          vgg16_int8  min =  803.75  max = 1126.53  avg =  886.03
            resnet50  min =  257.38  max =  285.19  avg =  266.59
       resnet50_int8  min =  304.81  max =  338.01  avg =  314.84
      squeezenet_ssd  min =  117.59  max =  145.79  avg =  123.79
 squeezenet_ssd_int8  min =  132.80  max =  163.00  avg =  149.99
       mobilenet_ssd  min =  103.98  max =  126.90  avg =  113.10
  mobilenet_ssd_int8  min =  167.86  max =  188.46  avg =  180.56
      mobilenet_yolo  min =  201.75  max =  263.92  avg =  240.17
  mobilenetv2_yolov3  min =  143.76  max =  167.77  avg =  151.94
  
lavender:/data/local/tmp/ncnnbench $ ./benchncnn 4 1 2 -1 1
[0 Adreno (TM) 512]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 512]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 512]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   69.75  max =   71.33  avg =   70.38
     squeezenet_int8  min =   67.12  max =   68.07  avg =   67.59
           mobilenet  min =  107.65  max =  110.48  avg =  108.82
      mobilenet_int8  min =  163.13  max =  164.74  avg =  164.24
        mobilenet_v2  min =   75.50  max =   77.36  avg =   76.38
        mobilenet_v3  min =   59.05  max =   59.36  avg =   59.23
          shufflenet  min =   38.33  max =   38.74  avg =   38.57
       shufflenet_v2  min =   37.43  max =   38.97  avg =   38.32
             mnasnet  min =   69.29  max =   73.20  avg =   70.73
     proxylessnasnet  min =   80.81  max =   82.66  avg =   81.52
     efficientnet_b0  min =  151.20  max =  152.38  avg =  151.72
        regnety_400m  min =   93.53  max =   94.53  avg =   94.19
           blazeface  min =   12.15  max =   12.82  avg =   12.46
           googlenet  min =  239.63  max =  242.64  avg =  241.06
      googlenet_int8  min =  214.71  max =  216.53  avg =  215.79
            resnet18  min =  234.20  max =  238.74  avg =  236.90
       resnet18_int8  min =  181.57  max =  183.97  avg =  182.66
             alexnet  min =  205.94  max =  207.44  avg =  206.63
               vgg16  min = 1188.14  max = 1201.95  avg = 1196.93
          vgg16_int8  min = 1081.21  max = 1087.84  avg = 1085.17
            resnet50  min =  556.54  max =  566.68  avg =  561.21
       resnet50_int8  min =  433.19  max =  433.93  avg =  433.48
      squeezenet_ssd  min =  169.02  max =  170.54  avg =  169.73
 squeezenet_ssd_int8  min =  176.28  max =  177.90  avg =  176.87
       mobilenet_ssd  min =  228.15  max =  232.69  avg =  230.38
  mobilenet_ssd_int8  min =  236.97  max =  239.69  avg =  238.35
      mobilenet_yolo  min =  493.33  max =  506.34  avg =  499.79
  mobilenetv2_yolov3  min =  252.53  max =  261.58  avg =  256.30

lavender:/data/local/tmp/ncnnbench $ ./benchncnn 4 1 2 0 1
[0 Adreno (TM) 512]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 512]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 512]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = 0
cooling_down = 1
          squeezenet  min =   34.49  max =   34.65  avg =   34.55
           mobilenet  min =   54.45  max =   55.52  avg =   54.75
        mobilenet_v2  min =   39.32  max =   39.58  avg =   39.50
        mobilenet_v3  min =   36.13  max =   36.28  avg =   36.19
          shufflenet  min =   35.25  max =   35.42  avg =   35.31
       shufflenet_v2  min =   31.38  max =   31.70  avg =   31.53
             mnasnet  min =   40.95  max =   41.32  avg =   41.13
     proxylessnasnet  min =   43.81  max =   44.05  avg =   43.90
     efficientnet_b0  min =   68.34  max =   68.56  avg =   68.47
        regnety_400m  min =   53.89  max =   54.23  avg =   54.02
           blazeface  min =   19.82  max =   27.74  avg =   22.01
           googlenet  min =  119.46  max =  119.98  avg =  119.80
            resnet18  min =  115.56  max =  120.28  avg =  116.88
             alexnet  min =  102.06  max =  105.56  avg =  102.97
               vgg16  min = 1192.29  max = 1202.17  avg = 1197.03
            resnet50  min =  294.87  max =  298.79  avg =  296.05
      squeezenet_ssd  min =  167.85  max =  168.42  avg =  168.09
       mobilenet_ssd  min =  120.30  max =  120.37  avg =  120.34
      mobilenet_yolo  min =  256.60  max =  260.21  avg =  257.54
  mobilenetv2_yolov3  min =  121.48  max =  125.22  avg =  122.53
```

### Qualcomm MSM8996 Snapdragon 820 (Kyro 2.15GHz x 2 + Kyro 1.6GHz x 2)
```
root@msm8996:/data/local/tmp/ncnn # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   23.20  max =   24.06  avg =   23.63
       mobilenet  min =   35.89  max =   36.41  avg =   36.09
    mobilenet_v2  min =   27.04  max =   28.62  avg =   27.39
      shufflenet  min =   15.47  max =   16.45  avg =   16.00
       googlenet  min =   85.42  max =   86.15  avg =   85.81
        resnet18  min =   76.82  max =   79.63  avg =   78.50
         alexnet  min =  147.66  max =  156.92  avg =  152.95
           vgg16  min =  493.50  max =  515.03  avg =  507.34
  squeezenet-ssd  min =   56.31  max =   59.35  avg =   57.49
   mobilenet-ssd  min =   68.95  max =   74.24  avg =   71.39
  mobilenet-yolo  min =  142.52  max =  149.72  avg =  148.23

root@msm8996:/data/local/tmp/ncnn # ./benchncnn 8 1 2
loop_count = 8
num_threads = 1
powersave = 2
      squeezenet  min =   53.26  max =   53.37  avg =   53.31
       mobilenet  min =   96.37  max =   97.09  avg =   96.63
    mobilenet_v2  min =   63.00  max =   63.25  avg =   63.09
      shufflenet  min =   28.22  max =   28.88  avg =   28.48
       googlenet  min =  226.21  max =  228.31  avg =  227.22
        resnet18  min =  197.35  max =  198.55  avg =  197.84
         alexnet  min =  445.32  max =  449.62  avg =  446.65
           vgg16  min = 1416.39  max = 1450.95  avg = 1440.63
  squeezenet-ssd  min =  119.37  max =  119.77  avg =  119.56
   mobilenet-ssd  min =  183.04  max =  185.12  avg =  183.59
  mobilenet-yolo  min =  366.91  max =  369.87  avg =  368.40
```

### Qualcomm MSM8994 Snapdragon 810 (Cortex-A57 2.0GHz x 4 + Cortex-A53 1.55GHz x 4)
```
angler:/data/local/tmp $ ./benchncnn 8 8 0 -1 1
[0 Adreno (TM) 430]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 430]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 8
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   30.32  max =   31.57  avg =   30.98
     squeezenet_int8  min =   43.86  max =   45.85  avg =   44.63
           mobilenet  min =   36.41  max =   40.31  avg =   37.29
      mobilenet_int8  min =  100.97  max =  120.23  avg =  108.27
        mobilenet_v2  min =   35.45  max =   49.74  avg =   37.60
        mobilenet_v3  min =   31.73  max =   32.96  avg =   32.09
          shufflenet  min =   28.14  max =   44.45  avg =   30.67
       shufflenet_v2  min =   22.28  max =   29.52  avg =   23.65
             mnasnet  min =   31.64  max =   33.50  avg =   32.62
     proxylessnasnet  min =   36.67  max =   44.09  avg =   38.47
     efficientnet_b0  min =   59.78  max =   80.50  avg =   62.68
        regnety_400m  min =   78.18  max =  120.32  avg =   89.21
           blazeface  min =    8.15  max =   10.34  avg =    8.67
           googlenet  min =   93.20  max =   94.81  avg =   93.65
      googlenet_int8  min =  137.13  max =  157.18  avg =  149.36
            resnet18  min =   92.54  max =   99.54  avg =   95.33
       resnet18_int8  min =  118.58  max =  138.16  avg =  127.26
             alexnet  min =  104.60  max =  113.60  avg =  110.01
               vgg16  min =  572.58  max =  647.34  avg =  616.86
          vgg16_int8  min =  973.42  max = 1080.14  avg = 1025.06
            resnet50  min =  273.99  max =  299.86  avg =  286.42
       resnet50_int8  min =  324.39  max =  358.54  avg =  345.45
      squeezenet_ssd  min =  105.14  max =  131.91  avg =  112.98
 squeezenet_ssd_int8  min =  133.40  max =  159.71  avg =  147.66
       mobilenet_ssd  min =   94.06  max =  106.33  avg =  101.00
  mobilenet_ssd_int8  min =  134.43  max =  154.24  avg =  146.07
      mobilenet_yolo  min =  223.54  max =  281.09  avg =  246.72
  mobilenetv2_yolov3  min =  113.63  max =  132.06  avg =  126.55

angler:/data/local/tmp $ ./benchncnn 8 1 2 -1 1
[0 Adreno (TM) 430]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 430]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   73.43  max =   75.68  avg =   74.05
     squeezenet_int8  min =   89.35  max =   90.01  avg =   89.68
           mobilenet  min =  112.71  max =  114.10  avg =  113.15
      mobilenet_int8  min =  192.37  max =  193.31  avg =  192.89
        mobilenet_v2  min =   78.56  max =   78.90  avg =   78.75
        mobilenet_v3  min =   68.14  max =   68.75  avg =   68.33
          shufflenet  min =   45.11  max =   46.43  avg =   45.72
       shufflenet_v2  min =   39.51  max =   40.57  avg =   40.12
             mnasnet  min =   75.63  max =   76.00  avg =   75.79
     proxylessnasnet  min =   95.32  max =   95.71  avg =   95.49
     efficientnet_b0  min =  188.72  max =  193.88  avg =  192.17
        regnety_400m  min =   99.29  max =  100.73  avg =   99.98
           blazeface  min =   15.60  max =   15.88  avg =   15.72
           googlenet  min =  244.46  max =  245.62  avg =  245.10
      googlenet_int8  min =  295.42  max =  297.95  avg =  296.36
            resnet18  min =  218.37  max =  220.84  avg =  219.90
       resnet18_int8  min =  267.07  max =  268.35  avg =  267.63
             alexnet  min =  247.91  max =  248.35  avg =  248.08
               vgg16  min = 1113.08  max = 1146.56  avg = 1130.86
          vgg16_int8  min = 1629.60  max = 1683.80  avg = 1662.91
            resnet50  min =  544.25  max =  564.36  avg =  554.75
       resnet50_int8  min =  593.11  max =  595.40  avg =  594.21
      squeezenet_ssd  min =  167.40  max =  169.27  avg =  168.51
 squeezenet_ssd_int8  min =  229.23  max =  234.77  avg =  232.24
       mobilenet_ssd  min =  232.94  max =  235.56  avg =  234.51
  mobilenet_ssd_int8  min =  290.43  max =  292.55  avg =  291.50
      mobilenet_yolo  min =  523.04  max =  525.41  avg =  523.76
  mobilenetv2_yolov3  min =  269.57  max =  270.61  avg =  269.91

angler:/data/local/tmp $ ./benchncnn 4 1 2 0 1
[0 Adreno (TM) 430]  queueC=0[3]  queueG=0[3]  queueT=0[3]
[0 Adreno (TM) 430]  buglssc=0  bugsbn1=1  buglbia=0  bugihfa=0
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = 0
cooling_down = 1
          squeezenet  min =   39.49  max =   41.93  avg =   40.62
           mobilenet  min =   60.30  max =   61.81  avg =   60.88
        mobilenet_v2  min =   45.38  max =   47.10  avg =   45.88
        mobilenet_v3  min =   45.97  max =   47.39  avg =   46.69
          shufflenet  min =   29.12  max =   31.02  avg =   29.91
       shufflenet_v2  min =   47.58  max =   50.06  avg =   48.26
             mnasnet  min =   47.84  max =   49.17  avg =   48.26
     proxylessnasnet  min =   49.51  max =   51.03  avg =   49.97
     efficientnet_b0  min =  100.56  max =  105.60  avg =  102.45
        regnety_400m  min =   59.67  max =   61.24  avg =   60.56
           blazeface  min =   13.87  max =   13.98  avg =   13.93
           googlenet  min =  131.26  max =  136.33  avg =  133.40
            resnet18  min =  116.38  max =  117.92  avg =  116.93
             alexnet  min =   72.59  max =   73.94  avg =   73.29
               vgg16  min = 1090.07  max = 1101.71  avg = 1096.34
            resnet50  min =  299.76  max =  300.78  avg =  300.40
      squeezenet_ssd  min =  181.95  max =  182.83  avg =  182.39
       mobilenet_ssd  min =  148.44  max =  151.07  avg =  149.75
      mobilenet_yolo  min =  284.46  max =  285.74  avg =  285.39
  mobilenetv2_yolov3  min =  140.28  max =  148.62  avg =  144.83
```

### Qualcomm MSM8916 Snapdragon 410 (Cortex-A53 1.2GHz x 4)
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0 -1 1
no vulkan device
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   66.19  max =   72.82  avg =   68.21
     squeezenet_int8  min =  114.98  max =  130.46  avg =  118.16
           mobilenet  min =   89.62  max =   95.83  avg =   93.29
      mobilenet_int8  min =  241.53  max =  251.06  avg =  246.45
        mobilenet_v2  min =   78.90  max =   89.02  avg =   81.46
        mobilenet_v3  min =   63.31  max =   72.72  avg =   65.19
          shufflenet  min =   50.80  max =   59.33  avg =   53.10
       shufflenet_v2  min =   43.13  max =   50.89  avg =   44.53
             mnasnet  min =   72.71  max =   81.03  avg =   75.36
     proxylessnasnet  min =   78.31  max =   87.47  avg =   81.05
     efficientnet_b0  min =  133.79  max =  144.34  avg =  139.61
        regnety_400m  min =  112.34  max =  119.73  avg =  114.92
           blazeface  min =   17.01  max =   17.21  avg =   17.15
           googlenet  min =  187.44  max =  198.45  avg =  193.39
      googlenet_int8  min =  308.24  max =  414.77  avg =  333.01
            resnet18  min =  172.57  max =  185.75  avg =  178.58
       resnet18_int8  min =  259.60  max =  278.97  avg =  270.88
             alexnet  min =  186.46  max =  197.99  avg =  190.98
               vgg16  min =  807.01  max =  993.53  avg =  840.82
          vgg16_int8  min = 1552.74  max = 1616.45  avg = 1579.95
            resnet50  min =  416.01  max =  456.37  avg =  423.99
       resnet50_int8  min =  633.55  max =  665.31  avg =  650.97
      squeezenet_ssd  min =  189.73  max =  205.13  avg =  196.05
 squeezenet_ssd_int8  min =  303.99  max =  330.38  avg =  311.27
       mobilenet_ssd  min =  191.16  max =  201.49  avg =  195.73
  mobilenet_ssd_int8  min =  341.66  max =  360.41  avg =  352.74
      mobilenet_yolo  min =  404.64  max =  414.32  avg =  409.58
  mobilenetv2_yolov3  min =  255.36  max =  260.57  avg =  258.33

HM2014812:/data/local/tmp # ./benchncnn 4 1 0 -1 1
no vulkan device
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  157.84  max =  163.57  avg =  160.17
     squeezenet_int8  min =  235.90  max =  236.90  avg =  236.36
           mobilenet  min =  244.48  max =  245.33  avg =  244.93
      mobilenet_int8  min =  566.65  max =  585.54  avg =  574.41
        mobilenet_v2  min =  173.31  max =  184.20  avg =  179.05
        mobilenet_v3  min =  149.89  max =  151.90  avg =  150.65
          shufflenet  min =  103.08  max =  104.07  avg =  103.67
       shufflenet_v2  min =   88.62  max =   88.85  avg =   88.70
             mnasnet  min =  165.94  max =  166.74  avg =  166.42
     proxylessnasnet  min =  210.10  max =  215.64  avg =  212.17
     efficientnet_b0  min =  396.79  max =  409.39  avg =  401.78
        regnety_400m  min =  224.94  max =  226.49  avg =  225.46
           blazeface  min =   38.27  max =   39.03  avg =   38.67
           googlenet  min =  548.29  max =  556.97  avg =  551.88
      googlenet_int8  min =  763.95  max =  776.59  avg =  768.95
            resnet18  min =  496.89  max =  500.24  avg =  498.06
       resnet18_int8  min =  651.89  max =  655.10  avg =  653.40
             alexnet  min =  490.47  max =  492.03  avg =  491.29
               vgg16  min = 2203.58  max = 2236.58  avg = 2222.08
          vgg16_int8  min = 3753.17  max = 3761.56  avg = 3756.99
            resnet50  min = 1209.85  max = 1215.09  avg = 1212.06
       resnet50_int8  min = 1657.36  max = 1665.21  avg = 1660.50
      squeezenet_ssd  min =  366.46  max =  369.47  avg =  367.88
 squeezenet_ssd_int8  min =  601.46  max =  603.23  avg =  602.37
       mobilenet_ssd  min =  520.79  max =  523.17  avg =  521.59
  mobilenet_ssd_int8  min =  867.12  max =  876.73  avg =  872.79
      mobilenet_yolo  min = 1130.78  max = 1135.02  avg = 1132.42
  mobilenetv2_yolov3  min =  600.01  max =  602.18  avg =  600.88
```

### Raspberry Pi 3 Model B+ Broadcom BCM2837B0, Cortex-A53 (ARMv8) (1.4GHz x 4)
```
pi@raspberrypi:~ $ ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =  108.66  max =  109.24  avg =  108.96
       mobilenet  min =  151.78  max =  152.92  avg =  152.31
    mobilenet_v2  min =  193.14  max =  195.56  avg =  194.50
      shufflenet  min =   91.41  max =   92.19  avg =   91.75
       googlenet  min =  302.02  max =  304.08  avg =  303.24
        resnet18  min =  411.93  max =  423.14  avg =  416.54
         alexnet  min =  275.54  max =  276.50  avg =  276.13
           vgg16  min = 1845.36  max = 1925.95  avg = 1902.28
  squeezenet-ssd  min =  313.86  max =  317.35  avg =  315.28
   mobilenet-ssd  min =  262.91  max =  264.92  avg =  263.85
  mobilenet-yolo  min =  638.73  max =  641.27  avg =  639.87

```

### Raspberry Pi 4 Model B Broadcom BCM2711B0, Cortex-A72 (ARMv8) (1.5GHz x 4)
```
pi@raspberrypi:~ $ ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   57.47  max =   59.73  avg =   58.73
     squeezenet_int8  min =   77.41  max =   80.01  avg =   78.72
           mobilenet  min =   85.06  max =   86.67  avg =   86.01
      mobilenet_int8  min =  163.69  max =  185.67  avg =  168.48
        mobilenet_v2  min =   74.13  max =   76.76  avg =   75.84
        mobilenet_v3  min =   60.93  max =   61.46  avg =   61.25
          shufflenet  min =   37.62  max =   38.46  avg =   37.98
       shufflenet_v2  min =   33.00  max =   34.21  avg =   33.38
             mnasnet  min =   64.39  max =   65.01  avg =   64.64
     proxylessnasnet  min =   65.71  max =   66.71  avg =   66.06
           googlenet  min =  175.82  max =  176.69  avg =  176.24
      googlenet_int8  min =  187.11  max =  188.97  avg =  187.99
            resnet18  min =  233.36  max =  234.39  avg =  233.89
       resnet18_int8  min =  156.72  max =  173.10  avg =  159.56
             alexnet  min =  180.48  max =  197.66  avg =  183.05
               vgg16  min =  969.88  max = 1007.31  avg =  988.65
          vgg16_int8  min = 1206.02  max = 1258.90  avg = 1226.27
            resnet50  min =  480.30  max =  502.61  avg =  486.97
       resnet50_int8  min =  412.35  max =  465.48  avg =  421.58
      squeezenet_ssd  min =  183.15  max =  221.97  avg =  190.40
 squeezenet_ssd_int8  min =  233.73  max =  250.71  avg =  238.20
       mobilenet_ssd  min =  176.45  max =  197.79  avg =  180.52
  mobilenet_ssd_int8  min =  259.11  max =  272.16  avg =  261.29
      mobilenet_yolo  min =  423.25  max =  447.12  avg =  435.26
  mobilenetv2_yolov3  min =  241.08  max =  283.54  avg =  248.69
```

### Khadas VIM3, Amlogic A311D (Cortex-A73 2.2GHz x 4 + Cortex-A53 1.8GHz x 2)

```
khadas@Khadas:~/src/ncnn/build/benchmark$ ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   29.58  max =   30.03  avg =   29.80
     squeezenet_int8  min =   23.84  max =   24.06  avg =   23.98
           mobilenet  min =   41.79  max =   42.97  avg =   42.38
      mobilenet_int8  min =   20.87  max =   21.08  avg =   20.96
        mobilenet_v2  min =   38.08  max =   40.18  avg =   38.68
        mobilenet_v3  min =   28.25  max =   28.93  avg =   28.61
          shufflenet  min =   21.43  max =   21.97  avg =   21.80
       shufflenet_v2  min =   18.05  max =   18.50  avg =   18.29
             mnasnet  min =   33.17  max =   36.31  avg =   33.71
     proxylessnasnet  min =   35.14  max =   36.26  avg =   35.43
     efficientnet_b0  min =   54.37  max =   55.14  avg =   54.68
   efficientnetv2_b0  min =   60.19  max =   61.31  avg =   60.72
        regnety_400m  min =   44.23  max =   45.58  avg =   44.55
           blazeface  min =    6.25  max =    6.41  avg =    6.34
           googlenet  min =   80.06  max =   81.35  avg =   80.71
      googlenet_int8  min =   62.14  max =   63.17  avg =   62.56
            resnet18  min =   76.07  max =   77.96  avg =   76.69
       resnet18_int8  min =   47.49  max =   48.82  avg =   47.97
             alexnet  min =   67.96  max =   69.36  avg =   68.54
               vgg16  min =  409.22  max =  428.95  avg =  415.21
          vgg16_int8  min =  261.93  max =  268.01  avg =  263.84
            resnet50  min =  187.20  max =  190.96  avg =  188.85
       resnet50_int8  min =  122.45  max =  129.28  avg =  124.39
      squeezenet_ssd  min =   84.70  max =   86.36  avg =   85.47
 squeezenet_ssd_int8  min =   63.30  max =   65.30  avg =   63.75
       mobilenet_ssd  min =   87.79  max =   89.27  avg =   88.64
  mobilenet_ssd_int8  min =   45.61  max =   45.88  avg =   45.70
      mobilenet_yolo  min =  191.45  max =  199.32  avg =  193.56
  mobilenetv2_yolov3  min =  122.49  max =  126.64  avg =  124.34
         yolov4-tiny  min =  144.92  max =  153.81  avg =  146.79
           nanodet_m  min =   50.20  max =   57.32  avg =   51.66
          yolox-nano  min =   77.93  max =   81.16  avg =   78.94
     yolox-nano_int8  min =   70.78  max =   76.06  avg =   72.04
          yolox-tiny  min =  199.43  max =  202.16  avg =  200.72
     yolox-tiny_int8  min =  142.00  max =  143.77  avg =  143.04
             yolox_s  min =  666.32  max =  679.94  avg =  671.99
        yolox-s_int8  min =  458.40  max =  462.57  avg =  460.08
```

### Station P2, Rockchip RK3568 (Cortex-A55 2.0GHz x 4)

```
./benchncnn 4 4 0 -1 1
loop_count = 4
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   26.02  max =   27.15  avg =   26.74
     squeezenet_int8  min =   44.69  max =   45.70  avg =   45.24
           mobilenet  min =   32.63  max =   33.49  avg =   33.10
      mobilenet_int8  min =   44.23  max =   45.86  avg =   44.99
        mobilenet_v2  min =   31.59  max =   32.02  avg =   31.86
        mobilenet_v3  min =   25.71  max =   26.44  avg =   26.10
          shufflenet  min =   22.12  max =   23.17  avg =   22.52
       shufflenet_v2  min =   17.84  max =   18.21  avg =   17.96
             mnasnet  min =   28.26  max =   28.70  avg =   28.45
     proxylessnasnet  min =   31.96  max =   32.25  avg =   32.13
     efficientnet_b0  min =   53.17  max =   54.48  avg =   53.60
   efficientnetv2_b0  min =   70.08  max =   70.69  avg =   70.30
        regnety_400m  min =   40.80  max =   41.79  avg =   41.10
           blazeface  min =   10.79  max =   11.57  avg =   11.11
           googlenet  min =   83.66  max =   92.22  avg =   86.23
      googlenet_int8  min =  116.44  max =  118.34  avg =  117.08
            resnet18  min =   61.38  max =   62.52  avg =   61.94
       resnet18_int8  min =   95.58  max =   96.93  avg =   96.28
             alexnet  min =   69.90  max =   70.59  avg =   70.19
               vgg16  min =  334.24  max =  343.89  avg =  337.24
          vgg16_int8  min =  464.88  max =  474.71  avg =  468.29
            resnet50  min =  141.65  max =  146.23  avg =  143.78
       resnet50_int8  min =  230.36  max =  254.75  avg =  241.24
      squeezenet_ssd  min =   98.38  max =  104.60  avg =  100.50
 squeezenet_ssd_int8  min =  134.73  max =  137.88  avg =  136.12
       mobilenet_ssd  min =   77.48  max =   79.92  avg =   78.64
  mobilenet_ssd_int8  min =  101.44  max =  102.61  avg =  102.06
      mobilenet_yolo  min =  149.12  max =  150.14  avg =  149.76
  mobilenetv2_yolov3  min =  103.71  max =  107.81  avg =  105.69
         yolov4-tiny  min =  145.75  max =  149.35  avg =  147.09
           nanodet_m  min =   52.91  max =   54.06  avg =   53.53

./benchncnn 4 2 0 -1 1
loop_count = 4
num_threads = 2
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   33.78  max =   34.38  avg =   34.16
     squeezenet_int8  min =   61.66  max =   62.11  avg =   61.85
           mobilenet  min =   46.53  max =   46.74  avg =   46.62
      mobilenet_int8  min =   71.06  max =   71.76  avg =   71.38
        mobilenet_v2  min =   39.05  max =   39.38  avg =   39.19
        mobilenet_v3  min =   32.20  max =   32.47  avg =   32.29
          shufflenet  min =   27.13  max =   27.40  avg =   27.27
       shufflenet_v2  min =   23.38  max =   23.92  avg =   23.62
             mnasnet  min =   35.51  max =   35.73  avg =   35.62
     proxylessnasnet  min =   42.98  max =   43.16  avg =   43.06
     efficientnet_b0  min =   75.34  max =   75.79  avg =   75.61
   efficientnetv2_b0  min =  107.34  max =  107.83  avg =  107.60
        regnety_400m  min =   47.91  max =   48.20  avg =   48.02
           blazeface  min =   16.38  max =   16.63  avg =   16.49
           googlenet  min =  124.27  max =  125.24  avg =  124.65
      googlenet_int8  min =  177.78  max =  178.39  avg =  178.06
            resnet18  min =   82.02  max =   82.70  avg =   82.38
       resnet18_int8  min =  148.06  max =  149.03  avg =  148.39
             alexnet  min =  105.20  max =  105.91  avg =  105.54
               vgg16  min =  459.65  max =  464.94  avg =  462.02
          vgg16_int8  min =  737.54  max =  750.64  avg =  742.90
            resnet50  min =  204.44  max =  205.20  avg =  204.84
       resnet50_int8  min =  364.47  max =  366.04  avg =  365.53
      squeezenet_ssd  min =  124.42  max =  128.01  avg =  125.80
 squeezenet_ssd_int8  min =  179.29  max =  183.83  avg =  181.43
       mobilenet_ssd  min =  113.85  max =  115.50  avg =  114.41
  mobilenet_ssd_int8  min =  161.35  max =  162.38  avg =  161.71
      mobilenet_yolo  min =  214.95  max =  216.62  avg =  215.72
  mobilenetv2_yolov3  min =  134.23  max =  136.26  avg =  135.07
         yolov4-tiny  min =  194.72  max =  195.49  avg =  195.18
           nanodet_m  min =   67.67  max =   68.09  avg =   67.90

./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   54.31  max =   55.65  avg =   55.00
     squeezenet_int8  min =  103.96  max =  106.28  avg =  104.92
           mobilenet  min =   79.02  max =   79.46  avg =   79.25
      mobilenet_int8  min =  130.06  max =  130.61  avg =  130.36
        mobilenet_v2  min =   60.15  max =   60.66  avg =   60.31
        mobilenet_v3  min =   49.40  max =   49.57  avg =   49.49
          shufflenet  min =   39.39  max =   39.78  avg =   39.60
       shufflenet_v2  min =   35.48  max =   35.70  avg =   35.62
             mnasnet  min =   55.38  max =   56.10  avg =   55.71
     proxylessnasnet  min =   70.29  max =   70.48  avg =   70.35
     efficientnet_b0  min =  128.56  max =  129.96  avg =  129.26
   efficientnetv2_b0  min =  181.00  max =  181.56  avg =  181.24
        regnety_400m  min =   67.15  max =   69.62  avg =   67.95
           blazeface  min =   26.07  max =   26.58  avg =   26.33
           googlenet  min =  219.19  max =  221.32  avg =  220.01
      googlenet_int8  min =  317.62  max =  319.40  avg =  318.37
            resnet18  min =  135.33  max =  136.94  avg =  135.88
       resnet18_int8  min =  264.69  max =  265.51  avg =  265.16
             alexnet  min =  190.54  max =  193.50  avg =  191.88
               vgg16  min =  790.99  max =  809.24  avg =  795.85
          vgg16_int8  min = 1354.48  max = 1358.89  avg = 1357.40
            resnet50  min =  358.08  max =  362.96  avg =  360.29
       resnet50_int8  min =  667.92  max =  670.40  avg =  668.78
      squeezenet_ssd  min =  193.15  max =  194.02  avg =  193.49
 squeezenet_ssd_int8  min =  291.42  max =  294.70  avg =  293.16
       mobilenet_ssd  min =  189.54  max =  190.28  avg =  189.97
  mobilenet_ssd_int8  min =  289.94  max =  290.40  avg =  290.28
      mobilenet_yolo  min =  370.37  max =  384.69  avg =  375.11
  mobilenetv2_yolov3  min =  210.93  max =  211.70  avg =  211.40
         yolov4-tiny  min =  309.11  max =  310.74  avg =  309.89
           nanodet_m  min =  100.42  max =  112.25  avg =  103.66
```

### Rockchip RK3399 (Cortex-A72 1.8GHz x 2 + Cortex-A53 1.5GHz x 4)
```
rk3399_firefly_box:/data/local/tmp/ncnn/benchmark # ./benchncnn 8 2 2
loop_count = 8
num_threads = 2
powersave = 2
gpu_device = -1
          squeezenet  min =   52.53  max =   53.64  avg =   53.06
     squeezenet_int8  min =   53.37  max =   55.72  avg =   54.26
           mobilenet  min =   78.53  max =   81.46  avg =   79.53
      mobilenet_int8  min =   56.26  max =   62.04  avg =   58.40
        mobilenet_v2  min =   69.08  max =   69.97  avg =   69.44
          shufflenet  min =   31.57  max =   34.90  avg =   32.84
             mnasnet  min =   56.12  max =   57.29  avg =   56.54
     proxylessnasnet  min =   66.95  max =   67.46  avg =   67.13
           googlenet  min =  185.60  max =  203.72  avg =  191.80
      googlenet_int8  min =  167.17  max =  195.48  avg =  176.84
            resnet18  min =  192.91  max =  205.34  avg =  198.63
       resnet18_int8  min =  156.85  max =  173.24  avg =  162.57
             alexnet  min =  192.74  max =  209.14  avg =  197.55
               vgg16  min =  896.54  max =  947.90  avg =  924.92
          vgg16_int8  min =  974.32  max =  978.45  avg =  976.64
            resnet50  min =  436.12  max =  457.56  avg =  443.29
       resnet50_int8  min =  357.78  max =  389.60  avg =  369.63
      squeezenet_ssd  min =  144.73  max =  156.56  avg =  148.78
 squeezenet_ssd_int8  min =  173.36  max =  188.41  avg =  176.93
       mobilenet_ssd  min =  169.47  max =  195.27  avg =  174.54
  mobilenet_ssd_int8  min =  124.85  max =  140.70  avg =  129.52
      mobilenet_yolo  min =  387.88  max =  428.71  avg =  402.07
    mobilenet_yolov3  min =  409.21  max =  441.15  avg =  423.70

rk3399_firefly_box:/data/local/tmp/ncnn/benchmark # ./benchncnn 8 1 2
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
          squeezenet  min =   88.84  max =   91.30  avg =   90.01
     squeezenet_int8  min =   81.19  max =   83.46  avg =   81.69
           mobilenet  min =  134.79  max =  142.97  avg =  136.94
      mobilenet_int8  min =  105.89  max =  109.47  avg =  107.22
        mobilenet_v2  min =  106.92  max =  119.60  avg =  109.01
          shufflenet  min =   47.03  max =   48.43  avg =   47.69
             mnasnet  min =   90.78  max =   93.82  avg =   92.34
     proxylessnasnet  min =  109.38  max =  116.27  avg =  110.83
           googlenet  min =  325.96  max =  340.11  avg =  333.55
      googlenet_int8  min =  280.99  max =  286.43  avg =  283.21
            resnet18  min =  316.71  max =  328.74  avg =  321.68
       resnet18_int8  min =  253.65  max =  267.48  avg =  258.11
             alexnet  min =  310.41  max =  319.24  avg =  312.40
               vgg16  min = 1441.65  max = 1481.38  avg = 1468.75
          vgg16_int8  min = 1502.82  max = 1521.61  avg = 1512.19
            resnet50  min =  681.50  max =  692.14  avg =  686.59
       resnet50_int8  min =  558.08  max =  567.24  avg =  561.13
      squeezenet_ssd  min =  206.77  max =  216.37  avg =  210.85
 squeezenet_ssd_int8  min =  234.60  max =  245.13  avg =  241.38
       mobilenet_ssd  min =  271.13  max =  278.40  avg =  273.75
  mobilenet_ssd_int8  min =  216.88  max =  218.81  avg =  217.94
      mobilenet_yolo  min =  627.36  max =  636.86  avg =  632.40
    mobilenet_yolov3  min =  669.06  max =  682.47  avg =  676.11

rk3399_firefly_box:/data/local/tmp/ncnn/benchmark # ./benchncnn 8 4 1
loop_count = 8
num_threads = 4
powersave = 1
gpu_device = -1
          squeezenet  min =   58.57  max =   63.54  avg =   60.35
     squeezenet_int8  min =   62.79  max =   70.43  avg =   64.09
           mobilenet  min =   77.82  max =   95.34  avg =   80.56
      mobilenet_int8  min =   63.26  max =   78.81  avg =   67.81
        mobilenet_v2  min =   72.23  max =   84.33  avg =   74.97
          shufflenet  min =   41.25  max =   42.31  avg =   41.78
             mnasnet  min =   64.83  max =   82.47  avg =   67.73
     proxylessnasnet  min =   73.91  max =   85.34  avg =   76.67
           googlenet  min =  206.27  max =  280.66  avg =  227.77
      googlenet_int8  min =  192.79  max =  201.67  avg =  194.85
            resnet18  min =  203.68  max =  220.28  avg =  208.50
       resnet18_int8  min =  181.08  max =  193.67  avg =  183.65
             alexnet  min =  204.49  max =  208.71  avg =  206.48
               vgg16  min = 1031.40  max = 1059.07  avg = 1043.01
          vgg16_int8  min = 1173.33  max = 1192.29  avg = 1182.97
            resnet50  min =  410.29  max =  424.84  avg =  418.18
       resnet50_int8  min =  389.76  max =  398.02  avg =  392.88
      squeezenet_ssd  min =  169.58  max =  206.14  avg =  180.93
 squeezenet_ssd_int8  min =  199.68  max =  213.47  avg =  203.46
       mobilenet_ssd  min =  157.87  max =  173.44  avg =  162.57
  mobilenet_ssd_int8  min =  121.86  max =  133.69  avg =  125.92
      mobilenet_yolo  min =  349.75  max =  379.45  avg =  357.83
    mobilenet_yolov3  min =  363.76  max =  380.45  avg =  371.56

rk3399_firefly_box:/data/local/tmp/ncnn/benchmark # ./benchncnn 8 1 1
loop_count = 8
num_threads = 1
powersave = 1
gpu_device = -1
          squeezenet  min =  165.76  max =  171.54  avg =  167.61
     squeezenet_int8  min =  172.42  max =  183.19  avg =  174.43
           mobilenet  min =  245.50  max =  253.09  avg =  246.99
      mobilenet_int8  min =  221.14  max =  225.25  avg =  222.41
        mobilenet_v2  min =  190.55  max =  194.63  avg =  192.44
          shufflenet  min =   93.85  max =   98.10  avg =   95.70
             mnasnet  min =  174.12  max =  177.20  avg =  175.25
     proxylessnasnet  min =  213.46  max =  223.07  avg =  215.19
           googlenet  min =  667.97  max =  673.11  avg =  670.70
      googlenet_int8  min =  577.49  max =  579.45  avg =  578.19
            resnet18  min =  619.58  max =  626.98  avg =  622.85
       resnet18_int8  min =  527.11  max =  534.05  avg =  528.98
             alexnet  min =  762.35  max =  768.60  avg =  764.67
               vgg16  min = 3265.98  max = 3288.08  avg = 3279.45
          vgg16_int8  min = 3113.77  max = 3157.23  avg = 3134.39
            resnet50  min = 1321.07  max = 1341.97  avg = 1329.78
       resnet50_int8  min = 1187.20  max = 1195.61  avg = 1190.90
      squeezenet_ssd  min =  442.01  max =  457.50  avg =  450.00
 squeezenet_ssd_int8  min =  481.22  max =  501.44  avg =  488.83
       mobilenet_ssd  min =  497.80  max =  503.22  avg =  500.30
  mobilenet_ssd_int8  min =  447.33  max =  453.04  avg =  448.56
      mobilenet_yolo  min = 1115.70  max = 1121.13  avg = 1117.58
    mobilenet_yolov3  min = 1178.09  max = 1186.41  avg = 1181.39
```

### NanoPi R2S, Rockchip RK3328 (Cortex-A53 1.3GHz x 4) Armbian focal (21.05.1) aarch64
```
root@nanopi-r2s:~/ncnn/build/benchmark# ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   62.20  max =   62.81  avg =   62.49
     squeezenet_int8  min =   57.92  max =   71.46  avg =   59.76
           mobilenet  min =   82.88  max =   89.36  avg =   84.52
      mobilenet_int8  min =   57.16  max =   96.22  avg =   62.29
        mobilenet_v2  min =   73.68  max =   75.92  avg =   74.17
        mobilenet_v3  min =   59.57  max =   60.14  avg =   59.84
          shufflenet  min =   52.34  max =   52.70  avg =   52.53
       shufflenet_v2  min =   45.51  max =   45.92  avg =   45.73
             mnasnet  min =   67.75  max =   83.15  avg =   69.82
     proxylessnasnet  min =   81.70  max =   83.66  avg =   82.31
     efficientnet_b0  min =  121.10  max =  123.22  avg =  121.55
   efficientnetv2_b0  min =  138.93  max =  192.15  avg =  154.94
        regnety_400m  min =   99.62  max =  116.29  avg =  101.97
           blazeface  min =   18.80  max =   19.15  avg =   19.01
           googlenet  min =  176.36  max =  202.84  avg =  181.86
      googlenet_int8  min =  155.50  max =  190.50  avg =  161.20
            resnet18  min =  165.79  max =  201.57  avg =  172.56
       resnet18_int8  min =  122.24  max =  160.53  avg =  134.24
             alexnet  min =  227.07  max =  238.09  avg =  232.19
          vgg16_int8  min =  522.14  max =  551.75  avg =  531.68
            resnet50  min =  378.30  max =  440.21  avg =  388.56
       resnet50_int8  min =  315.76  max =  373.97  avg =  329.88
      squeezenet_ssd  min =  175.37  max =  200.86  avg =  179.01
 squeezenet_ssd_int8  min =  134.71  max =  147.57  avg =  136.57
       mobilenet_ssd  min =  174.43  max =  212.11  avg =  180.61
  mobilenet_ssd_int8  min =  119.41  max =  153.75  avg =  124.21
      mobilenet_yolo  min =  366.27  max =  422.67  avg =  383.65
  mobilenetv2_yolov3  min =  238.56  max =  281.97  avg =  247.56
         yolov4-tiny  min =  311.45  max =  333.32  avg =  316.79
           nanodet_m  min =  114.15  max =  122.39  avg =  115.44
           
root@nanopi-r2s:~/ncnn/build/benchmark# ./benchncnn 8 2 0
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   89.02  max =   90.52  avg =   89.35
     squeezenet_int8  min =   81.19  max =   81.90  avg =   81.42
           mobilenet  min =  131.47  max =  134.39  avg =  132.34
      mobilenet_int8  min =  102.20  max =  103.03  avg =  102.66
        mobilenet_v2  min =  102.40  max =  108.12  avg =  103.91
        mobilenet_v3  min =   89.17  max =   90.10  avg =   89.53
          shufflenet  min =   65.74  max =   68.86  avg =   66.50
       shufflenet_v2  min =   62.83  max =   64.41  avg =   63.25
             mnasnet  min =   98.01  max =   98.24  avg =   98.14
     proxylessnasnet  min =  121.10  max =  123.55  avg =  121.80
     efficientnet_b0  min =  187.79  max =  188.41  avg =  188.08
   efficientnetv2_b0  min =  211.96  max =  213.99  avg =  212.74
        regnety_400m  min =  124.98  max =  125.49  avg =  125.28
           blazeface  min =   24.91  max =   25.14  avg =   25.00
           googlenet  min =  278.47  max =  283.24  avg =  280.79
      googlenet_int8  min =  243.81  max =  247.82  avg =  245.30
            resnet18  min =  257.46  max =  259.29  avg =  258.29
       resnet18_int8  min =  187.18  max =  188.74  avg =  187.70
             alexnet  min =  384.52  max =  387.07  avg =  385.84
          vgg16_int8  min =  897.26  max =  901.68  avg =  899.19
            resnet50  min =  618.85  max =  623.92  avg =  620.85
       resnet50_int8  min =  512.33  max =  514.93  avg =  513.64
      squeezenet_ssd  min =  211.21  max =  218.71  avg =  213.02
 squeezenet_ssd_int8  min =  193.32  max =  193.97  avg =  193.70
       mobilenet_ssd  min =  271.11  max =  275.58  avg =  272.06
  mobilenet_ssd_int8  min =  208.80  max =  209.59  avg =  209.05
      mobilenet_yolo  min =  570.55  max =  575.98  avg =  572.73
  mobilenetv2_yolov3  min =  329.04  max =  353.84  avg =  340.42
         yolov4-tiny  min =  435.16  max =  463.68  avg =  457.69
           nanodet_m  min =  155.70  max =  159.13  avg =  156.50
```

### Rockchip RK3288 (Cortex-A17 1.8GHz x 4)
```
root@rk3288:/data/local/tmp/ncnn # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   51.43  max =   74.02  avg =   55.91
       mobilenet  min =  102.06  max =  125.67  avg =  106.02
    mobilenet_v2  min =   80.09  max =   99.23  avg =   85.40
      shufflenet  min =   34.91  max =   35.75  avg =   35.25
       googlenet  min =  181.72  max =  252.12  avg =  210.67
        resnet18  min =  198.86  max =  240.69  avg =  214.87
         alexnet  min =  154.68  max =  208.60  avg =  168.75
           vgg16  min = 1019.49  max = 1231.92  avg = 1129.09
  squeezenet-ssd  min =  133.38  max =  241.11  avg =  167.77
   mobilenet-ssd  min =  156.71  max =  216.70  avg =  175.31
  mobilenet-yolo  min =  396.78  max =  482.60  avg =  433.34
  
root@rk3288:/data/local/tmp/ncnn # ./benchncnn 8 1 0
loop_count = 8
num_threads = 1
powersave = 0
      squeezenet  min =  137.93  max =  140.76  avg =  138.71
       mobilenet  min =  244.01  max =  248.27  avg =  246.24
    mobilenet_v2  min =  177.94  max =  181.57  avg =  179.24
      shufflenet  min =   77.61  max =   78.30  avg =   77.94
       googlenet  min =  548.75  max =  559.40  avg =  553.00
        resnet18  min =  493.66  max =  510.55  avg =  500.37
         alexnet  min =  564.20  max =  604.87  avg =  581.30
           vgg16  min = 2425.03  max = 2447.25  avg = 2433.38
  squeezenet-ssd  min =  298.26  max =  304.67  avg =  302.00
   mobilenet-ssd  min =  465.65  max =  473.33  avg =  469.86
  mobilenet-yolo  min =  997.95  max = 1012.45  avg = 1002.32
```

### HiSilicon Hi3519V101 (Cortex-A17 1.2GHz x 1)
```
root@Hi3519:/ncnn-benchmark # taskset 2 ./benchncnn 8 1 0
loop_count = 8
num_threads = 1
powersave = 0
      squeezenet  min =  272.97  max =  275.84  avg =  274.85
 squeezenet-int8  min =  200.87  max =  202.47  avg =  201.74
       mobilenet  min =  480.90  max =  482.16  avg =  481.64
    mobilenet_v2  min =  350.01  max =  352.39  avg =  350.81
      shufflenet  min =  152.40  max =  153.17  avg =  152.80
       googlenet  min = 1096.65  max = 1101.35  avg = 1099.21
        resnet18  min =  983.92  max =  987.00  avg =  985.25
         alexnet  min = 1140.30  max = 1141.55  avg = 1140.92
  squeezenet-ssd  min =  574.62  max =  580.12  avg =  577.23
   mobilenet-ssd  min =  960.26  max =  969.13  avg =  965.93
  mobilenet-yolo  min = 1867.78  max = 1880.08  avg = 1873.89
```

### iPhone 5S (Apple A7 1.3GHz x 2)
```
iPhone:~ root# ./benchncnn 8 2 0 -1
[0 Apple A7 GPU]  queueC=0[8]  queueT=0[8]  memU=1  memDL=1  memHV=1
[0 Apple A7 GPU]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
          squeezenet  min =   49.21  max =   50.40  avg =   49.74
     squeezenet_int8  min =   54.73  max =   57.39  avg =   56.70
           mobilenet  min =   79.03  max =   80.00  avg =   79.44
      mobilenet_int8  min =  109.95  max =  112.69  avg =  111.38
        mobilenet_v2  min =   57.34  max =   57.88  avg =   57.47
        mobilenet_v3  min =   52.66  max =   53.73  avg =   53.12
          shufflenet  min =   32.78  max =   36.12  avg =   35.12
       shufflenet_v2  min =   31.25  max =   32.10  avg =   31.61
             mnasnet  min =   54.58  max =   56.12  avg =   55.44
     proxylessnasnet  min =   69.52  max =   72.42  avg =   70.40
           googlenet  min =  192.82  max =  194.20  avg =  193.35
      googlenet_int8  min =  235.43  max =  244.71  avg =  239.64
            resnet18  min =  164.33  max =  167.27  avg =  165.51
       resnet18_int8  min =  176.16  max =  179.73  avg =  178.60
             alexnet  min =  224.50  max =  228.21  avg =  226.51
               vgg16  min = 4262.28  max = 4400.29  avg = 4300.34
          vgg16_int8  min = 2835.84  max = 2955.22  avg = 2890.26
            resnet50  min =  542.66  max = 1344.49  avg =  737.05
       resnet50_int8  min =  426.08  max =  435.34  avg =  431.87
      squeezenet_ssd  min =  129.03  max =  131.44  avg =  129.99
 squeezenet_ssd_int8  min =  155.52  max =  161.42  avg =  158.51
       mobilenet_ssd  min =  168.18  max =  170.17  avg =  169.42
  mobilenet_ssd_int8  min =  205.78  max =  212.07  avg =  209.66
      mobilenet_yolo  min =  347.32  max =  363.15  avg =  355.72
  mobilenetv2_yolov3  min =  193.11  max =  196.64  avg =  194.31

iPhone:~ root# ./benchncnn 4 1 0 -1
[0 Apple A7 GPU]  queueC=0[8]  queueT=0[8]  memU=1  memDL=1  memHV=1
[0 Apple A7 GPU]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =   86.36  max =   86.81  avg =   86.57
     squeezenet_int8  min =   99.62  max =  100.07  avg =   99.83
           mobilenet  min =  143.11  max =  146.50  avg =  145.38
      mobilenet_int8  min =  202.25  max =  203.32  avg =  203.02
        mobilenet_v2  min =   97.56  max =   98.55  avg =   98.09
        mobilenet_v3  min =   87.45  max =   87.68  avg =   87.52
          shufflenet  min =   54.01  max =   54.13  avg =   54.08
       shufflenet_v2  min =   48.11  max =   48.65  avg =   48.36
             mnasnet  min =   95.02  max =   95.77  avg =   95.25
     proxylessnasnet  min =  123.91  max =  124.61  avg =  124.18
           googlenet  min =  344.23  max =  348.95  avg =  345.97
      googlenet_int8  min =  420.30  max =  420.99  avg =  420.65
            resnet18  min =  300.44  max =  301.36  avg =  300.99
       resnet18_int8  min =  308.60  max =  310.52  avg =  309.70
             alexnet  min =  423.92  max =  429.84  avg =  427.24
               vgg16  min = 4787.59  max = 5015.23  avg = 4900.43
          vgg16_int8  min = 3560.59  max = 3722.75  avg = 3639.88
            resnet50  min =  797.88  max = 1294.57  avg =  985.63
       resnet50_int8  min =  751.15  max =  760.25  avg =  757.89
      squeezenet_ssd  min =  193.75  max =  196.13  avg =  195.29
 squeezenet_ssd_int8  min =  243.78  max =  245.19  avg =  244.74
       mobilenet_ssd  min =  299.69  max =  307.22  avg =  305.12
  mobilenet_ssd_int8  min =  385.91  max =  389.82  avg =  388.48
      mobilenet_yolo  min =  657.00  max =  659.31  avg =  658.08
  mobilenetv2_yolov3  min =  335.59  max =  342.22  avg =  339.37

iPhone:~ root# ./benchncnn 4 1 0 0
[0 Apple A7 GPU]  queueC=0[8]  queueT=0[8]  memU=1  memDL=1  memHV=1
[0 Apple A7 GPU]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = 0
          squeezenet  min =  260.18  max =  262.55  avg =  261.09
           mobilenet  min =  288.73  max =  291.83  avg =  289.67
        mobilenet_v2  min =  265.72  max =  267.05  avg =  266.14
        mobilenet_v3  min =  255.86  max =  257.35  avg =  256.43
          shufflenet  min =  236.66  max =  239.49  avg =  237.98
       shufflenet_v2  min =  244.92  max =  247.75  avg =  246.22
             mnasnet  min =  254.75  max =  256.48  avg =  255.85
     proxylessnasnet  min =  281.42  max =  282.62  avg =  282.11
           googlenet  min =  745.36  max =  764.91  avg =  754.16
            resnet18  min =  721.26  max =  741.98  avg =  734.78
             alexnet  min =  521.43  max =  530.95  avg =  527.01
            resnet50  min = 1494.86  max = 1505.79  avg = 1501.49
      squeezenet_ssd  min = 1096.45  max = 1102.84  avg = 1098.55
       mobilenet_ssd  min =  639.50  max =  641.81  avg =  640.83
      mobilenet_yolo  min = 1445.16  max = 1450.94  avg = 1447.42
  mobilenetv2_yolov3  min = 1047.24  max = 1060.97  avg = 1052.86
```

### Freescale i.MX7 Dual (Cortex A7 1.0GHz x 2)
```
imx7d_pico:/data/local/tmp $ ./benchncnn 8 2 0 -1 1
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  219.06  max =  220.67  avg =  219.74
     squeezenet_int8  min =  160.49  max =  166.23  avg =  162.52
           mobilenet  min =  369.31  max =  374.73  avg =  373.00
      mobilenet_int8  min =  208.93  max =  215.28  avg =  211.55
        mobilenet_v2  min =  247.53  max =  253.69  avg =  250.61
        mobilenet_v3  min =  223.36  max =  229.34  avg =  225.98
          shufflenet  min =  142.62  max =  149.05  avg =  144.29
       shufflenet_v2  min =  137.73  max =  164.94  avg =  144.95
             mnasnet  min =  250.55  max =  256.91  avg =  252.96
     proxylessnasnet  min =  290.50  max =  306.90  avg =  294.47
     efficientnet_b0  min =  426.85  max =  432.62  avg =  431.36
        regnety_400m  min =  315.19  max =  321.19  avg =  316.99
           blazeface  min =   46.22  max =   50.55  avg =   46.87
           googlenet  min =  723.53  max =  729.07  avg =  726.19
      googlenet_int8  min =  496.12  max =  503.48  avg =  497.67
            resnet18  min =  653.64  max =  672.50  avg =  656.86
       resnet18_int8  min =  375.45  max =  384.86  avg =  380.34
             alexnet  min =  834.07  max =  855.77  avg =  839.92
      squeezenet_ssd  min =  519.21  max =  520.13  avg =  519.87
 squeezenet_ssd_int8  min =  400.46  max =  424.29  avg =  407.90
       mobilenet_ssd  min =  767.23  max =  773.95  avg =  770.57
  mobilenet_ssd_int8  min =  424.13  max =  430.35  avg =  428.67
      mobilenet_yolo  min = 1724.15  max = 1739.85  avg = 1729.92
  mobilenetv2_yolov3  min =  894.41  max =  906.72  avg =  900.18
         yolov4-tiny  min = 1175.09  max = 1210.57  avg = 1194.41

imx7d_pico:/data/local/tmp $ ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  411.10  max =  411.94  avg =  411.51
     squeezenet_int8  min =  284.90  max =  286.07  avg =  285.59
           mobilenet  min =  705.34  max =  706.15  avg =  705.68
      mobilenet_int8  min =  406.01  max =  413.78  avg =  410.08
        mobilenet_v2  min =  456.32  max =  457.13  avg =  456.83
        mobilenet_v3  min =  417.64  max =  418.78  avg =  418.24
          shufflenet  min =  251.05  max =  251.63  avg =  251.35
       shufflenet_v2  min =  238.09  max =  238.76  avg =  238.48
             mnasnet  min =  469.19  max =  469.61  avg =  469.33
     proxylessnasnet  min =  550.99  max =  551.84  avg =  551.41
     efficientnet_b0  min =  814.54  max =  815.54  avg =  815.06
        regnety_400m  min =  567.36  max =  567.45  avg =  567.42
           blazeface  min =   83.30  max =   83.43  avg =   83.37
           googlenet  min = 1367.22  max = 1367.76  avg = 1367.43
      googlenet_int8  min =  925.52  max =  926.48  avg =  925.96
            resnet18  min = 1231.85  max = 1235.23  avg = 1233.46
       resnet18_int8  min =  702.18  max =  704.35  avg =  703.06
             alexnet  min = 1593.15  max = 1604.42  avg = 1599.03
      squeezenet_ssd  min =  912.72  max =  913.16  avg =  912.92
 squeezenet_ssd_int8  min =  703.57  max =  705.24  avg =  704.26
       mobilenet_ssd  min = 1450.41  max = 1454.50  avg = 1452.28
  mobilenet_ssd_int8  min =  821.24  max =  822.69  avg =  822.15
      mobilenet_yolo  min = 3285.48  max = 3295.76  avg = 3289.43
  mobilenetv2_yolov3  min = 1671.48  max = 1682.96  avg = 1677.32
         yolov4-tiny  min = 2162.66  max = 2178.76  avg = 2171.22
```

### Loongson 2K1000 (GS264 1.0GHz x 2)
```
root@ls2k:~/ncnn/build/benchmark# ./benchncnn 4 2 0 -1 1
loop_count = 4
num_threads = 2
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  186.52  max =  188.70  avg =  187.19
           mobilenet  min =  278.43  max =  279.79  avg =  279.16
        mobilenet_v2  min =  223.91  max =  224.36  avg =  224.16
        mobilenet_v3  min =  180.59  max =  181.82  avg =  180.96
          shufflenet  min =  123.24  max =  123.64  avg =  123.48
       shufflenet_v2  min =  115.93  max =  117.35  avg =  116.59
             mnasnet  min =  206.54  max =  206.82  avg =  206.70
     proxylessnasnet  min =  229.80  max =  315.46  avg =  252.36
     efficientnet_b0  min =  339.09  max =  339.79  avg =  339.41
   efficientnetv2_b0  min =  384.24  max =  384.87  avg =  384.50
        regnety_400m  min =  271.61  max =  272.27  avg =  271.89
           blazeface  min =   36.06  max =   36.44  avg =   36.23
           googlenet  min =  655.98  max =  690.34  avg =  664.82
            resnet18  min =  497.96  max =  498.03  avg =  497.99
             alexnet  min =  509.80  max =  510.57  avg =  510.24
               vgg16  min = 2705.05  max = 3100.91  avg = 2876.61
            resnet50  min = 1258.32  max = 1297.43  avg = 1268.68
       mobilenet_ssd  min =  570.91  max =  572.03  avg =  571.44
      mobilenet_yolo  min = 1619.51  max = 1676.13  avg = 1636.07
  mobilenetv2_yolov3  min =  749.36  max =  797.85  avg =  761.68
         yolov4-tiny  min =  992.53  max = 1018.84  avg =  999.70
           nanodet_m  min =  301.72  max =  303.47  avg =  302.53

root@ls2k:~/ncnn/build/benchmark# ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  298.44  max =  300.93  avg =  299.44
           mobilenet  min =  473.52  max =  476.33  avg =  475.06
        mobilenet_v2  min =  343.32  max =  354.65  avg =  346.47
        mobilenet_v3  min =  284.11  max =  284.70  avg =  284.51
          shufflenet  min =  188.78  max =  189.04  avg =  188.88
       shufflenet_v2  min =  182.75  max =  183.07  avg =  182.92
             mnasnet  min =  335.42  max =  337.82  avg =  336.54
     proxylessnasnet  min =  384.64  max =  385.02  avg =  384.84
     efficientnet_b0  min =  572.26  max =  576.60  avg =  573.79
   efficientnetv2_b0  min =  646.99  max =  659.11  avg =  650.68
        regnety_400m  min =  426.79  max =  431.30  avg =  428.11
           blazeface  min =   57.62  max =   58.22  avg =   57.87
           googlenet  min = 1118.55  max = 1136.04  avg = 1123.23
            resnet18  min =  798.49  max =  801.61  avg =  800.10
             alexnet  min =  891.14  max =  903.55  avg =  895.12
               vgg16  min = 4412.31  max = 4480.41  avg = 4430.90
            resnet50  min = 2179.33  max = 2194.03  avg = 2184.99
       mobilenet_ssd  min =  974.85  max =  975.89  avg =  975.50
      mobilenet_yolo  min = 2541.57  max = 2560.65  avg = 2550.61
  mobilenetv2_yolov3  min = 1197.49  max = 1211.18  avg = 1201.43
         yolov4-tiny  min = 1535.79  max = 1695.41  avg = 1578.90
           nanodet_m  min =  450.95  max =  452.94  avg =  451.97
```

### Phytium FT-2000+/64 (FTC662 armv8 2.4GHz x 8)
```
[root@bogon benchmark]# ./benchncnn 10 1 0 -1 0
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 0
          squeezenet  min =   57.60  max =   59.78  avg =   58.51
     squeezenet_int8  min =   47.05  max =   47.89  avg =   47.40
           mobilenet  min =   91.08  max =   95.16  avg =   91.89
      mobilenet_int8  min =   60.27  max =   61.17  avg =   60.74
        mobilenet_v2  min =   63.38  max =   68.12  avg =   66.96
        mobilenet_v3  min =   53.34  max =   54.71  avg =   54.01
          shufflenet  min =   37.87  max =   41.78  avg =   39.37
       shufflenet_v2  min =   35.89  max =   37.30  avg =   36.40
             mnasnet  min =   59.57  max =   63.23  avg =   60.25
     proxylessnasnet  min =   71.24  max =   71.93  avg =   71.51
     efficientnet_b0  min =  134.34  max =  141.14  avg =  137.74
   efficientnetv2_b0  min =  143.82  max =  145.63  avg =  144.36
        regnety_400m  min =   76.96  max =   77.66  avg =   77.27
           blazeface  min =   11.57  max =   11.90  avg =   11.70
           googlenet  min =  188.10  max =  191.27  avg =  189.02
      googlenet_int8  min =  167.54  max =  169.63  avg =  168.38
            resnet18  min =  144.76  max =  163.39  avg =  154.95
       resnet18_int8  min =  124.14  max =  129.84  avg =  127.83
             alexnet  min =  198.22  max =  208.86  avg =  205.35
               vgg16  min =  848.10  max =  891.00  avg =  859.94
          vgg16_int8  min =  686.54  max =  742.77  avg =  704.74
            resnet50  min =  413.45  max =  428.84  avg =  417.81
       resnet50_int8  min =  306.32  max =  324.27  avg =  316.47
      squeezenet_ssd  min =  147.62  max =  149.58  avg =  148.48
 squeezenet_ssd_int8  min =  116.18  max =  134.86  avg =  126.93
       mobilenet_ssd  min =  188.49  max =  191.97  avg =  189.48
  mobilenet_ssd_int8  min =  120.28  max =  121.36  avg =  120.83
      mobilenet_yolo  min =  421.79  max =  425.68  avg =  423.51
  mobilenetv2_yolov3  min =  222.86  max =  225.58  avg =  224.01
         yolov4-tiny  min =  303.77  max =  310.70  avg =  307.45
           nanodet_m  min =   80.87  max =   82.11  avg =   81.35

[root@bogon benchmark]# ./benchncnn 10 8 0 -1 0
loop_count = 10
num_threads = 8
powersave = 0
gpu_device = -1
cooling_down = 0
          squeezenet  min =   14.53  max =   14.92  avg =   14.68
     squeezenet_int8  min =   11.67  max =   11.89  avg =   11.82
           mobilenet  min =   17.60  max =   20.05  avg =   18.34
      mobilenet_int8  min =    9.94  max =   10.22  avg =   10.08
        mobilenet_v2  min =   18.46  max =   19.18  avg =   18.81
        mobilenet_v3  min =   16.30  max =   16.71  avg =   16.45
          shufflenet  min =   14.65  max =   14.93  avg =   14.78
       shufflenet_v2  min =   11.23  max =   11.56  avg =   11.35
             mnasnet  min =   15.65  max =   16.08  avg =   15.92
     proxylessnasnet  min =   18.78  max =   21.72  avg =   19.68
     efficientnet_b0  min =   29.16  max =   29.62  avg =   29.37
   efficientnetv2_b0  min =   33.28  max =   35.48  avg =   34.23
        regnety_400m  min =   44.90  max =   47.36  avg =   46.32
           blazeface  min =    4.23  max =    4.43  avg =    4.30
           googlenet  min =   42.11  max =   42.98  avg =   42.38
      googlenet_int8  min =   33.24  max =   38.21  avg =   34.10
            resnet18  min =   33.27  max =   34.00  avg =   33.57
       resnet18_int8  min =   23.66  max =   24.78  avg =   24.24
             alexnet  min =   35.78  max =   37.68  avg =   36.46
               vgg16  min =  219.60  max =  235.79  avg =  222.11
          vgg16_int8  min =  128.64  max =  135.19  avg =  130.73
            resnet50  min =   84.15  max =   85.48  avg =   84.66
       resnet50_int8  min =   58.87  max =   61.98  avg =   59.85
      squeezenet_ssd  min =   47.60  max =   50.24  avg =   48.54
 squeezenet_ssd_int8  min =   36.42  max =   37.89  avg =   36.99
       mobilenet_ssd  min =   39.37  max =   42.63  avg =   41.06
  mobilenet_ssd_int8  min =   21.59  max =   22.05  avg =   21.83
      mobilenet_yolo  min =   83.16  max =   88.75  avg =   85.29
  mobilenetv2_yolov3  min =   58.13  max =   59.50  avg =   58.62
         yolov4-tiny  min =   74.18  max =   76.56  avg =   75.13
           nanodet_m  min =   25.16  max =   31.45  avg =   26.71
```

### nVIDIA RTX2060 of Notebook
```
C:\Users\ai\AppData\Local\Temp\benchmark>benchncnn.exe 64 1 0 0 0
[0 GeForce RTX 2060]  queueC=2[8]  queueG=0[16]  queueT=1[2]
[0 GeForce RTX 2060]  buglssc=0  bugihfa=0
[0 GeForce RTX 2060]  fp16p=1  fp16s=1  fp16a=1  int8s=1  int8a=1
loop_count = 64
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =    2.14  max =    2.93  avg =    2.26
           mobilenet  min =    2.08  max =    2.53  avg =    2.22
        mobilenet_v2  min =    2.81  max =    4.03  avg =    3.05
        mobilenet_v3  min =    2.90  max =    3.53  avg =    3.08
          shufflenet  min =    1.94  max =    4.27  avg =    2.55
       shufflenet_v2  min =    2.34  max =    2.97  avg =    2.49
             mnasnet  min =    2.11  max =    2.86  avg =    2.37
     proxylessnasnet  min =    2.27  max =    3.25  avg =    2.49
           googlenet  min =    4.34  max =    6.79  avg =    5.25
            resnet18  min =    2.60  max =    4.36  avg =    2.90
             alexnet  min =    2.79  max =    4.70  avg =    3.04
               vgg16  min =   11.40  max =   14.32  avg =   12.42
            resnet50  min =    5.26  max =    5.86  avg =    5.51
      squeezenet_ssd  min =    5.58  max =    7.94  avg =    6.56
       mobilenet_ssd  min =    3.47  max =    5.29  avg =    3.77
      mobilenet_yolo  min =    5.49  max =    6.19  avg =    5.70
  mobilenetv2_yolov3  min =    3.69  max =    5.14  avg =    3.91
```

### nVIDIA RTX2080 of Desktop
```
E:\projects\framework\ncnn\benchmark>benchncnn.exe 4096 1 0 0 0
[0 GeForce RTX 2080]  queueC=2[8]  queueG=0[16]  queueT=1[2]
[0 GeForce RTX 2080]  buglssc=0  bugihfa=0
[0 GeForce RTX 2080]  fp16p=1  fp16s=1  fp16a=1  int8s=1  int8a=1
loop_count = 4096
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =    1.39  max =   16.70  avg =    1.49
           mobilenet  min =    1.32  max =    2.55  avg =    1.42
        mobilenet_v2  min =    1.88  max =    5.02  avg =    2.00
        mobilenet_v3  min =    2.31  max =    3.58  avg =    2.45
          shufflenet  min =    1.45  max =    2.65  avg =    1.55
       shufflenet_v2  min =    1.90  max =    3.21  avg =    2.03
             mnasnet  min =    1.95  max =    3.17  avg =    2.09
     proxylessnasnet  min =    2.02  max =    2.95  avg =    2.16
           googlenet  min =    3.81  max =    5.91  avg =    4.05
            resnet18  min =    2.10  max =    3.28  avg =    2.24
             alexnet  min =    2.15  max =    3.35  avg =    2.30
               vgg16  min =    7.33  max =   11.12  avg =    7.80
            resnet50  min =    4.21  max =    6.70  avg =    4.49
      squeezenet_ssd  min =    4.58  max =    6.86  avg =    4.88
       mobilenet_ssd  min =    2.90  max =    4.52  avg =    3.09
      mobilenet_yolo  min =    4.15  max =    6.09  avg =    4.40
  mobilenetv2_yolov3  min =    3.04  max =    9.13  avg =    3.28
```

### NVIDIA Jetson AGX Xavier
```
$ ./benchncnn 8 4 2 -1 1
loop_count = 8
num_threads = 4
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =    8.21  max =    8.60  avg =    8.39
     squeezenet_int8  min =   15.07  max =   16.09  avg =   15.58
           mobilenet  min =   11.61  max =   12.14  avg =   11.87
      mobilenet_int8  min =   23.73  max =   24.24  avg =   23.96
        mobilenet_v2  min =    9.61  max =   10.02  avg =    9.82
        mobilenet_v3  min =    9.05  max =    9.90  avg =    9.32
          shufflenet  min =   10.20  max =   27.40  avg =   12.47
       shufflenet_v2  min =    7.88  max =    8.32  avg =    7.99
             mnasnet  min =    9.54  max =    9.86  avg =    9.67
     proxylessnasnet  min =   10.40  max =   10.75  avg =   10.53
     efficientnet_b0  min =   13.60  max =   21.72  avg =   15.14
   efficientnetv2_b0  min =   22.26  max =   23.89  avg =   23.18
        regnety_400m  min =   17.92  max =   23.25  avg =   19.07
           blazeface  min =    5.27  max =    5.49  avg =    5.37
           googlenet  min =   25.65  max =   28.43  avg =   26.88
      googlenet_int8  min =   43.53  max =   47.39  avg =   44.27
            resnet18  min =   15.40  max =   22.18  avg =   17.12
       resnet18_int8  min =   31.79  max =   33.27  avg =   32.28
             alexnet  min =   19.37  max =   26.23  avg =   22.10
               vgg16  min =   71.89  max =   77.37  avg =   73.72
          vgg16_int8  min =  142.28  max =  155.79  avg =  146.88
            resnet50  min =   48.77  max =   51.30  avg =   49.56
       resnet50_int8  min =   98.18  max =  101.89  avg =   99.62
      squeezenet_ssd  min =   28.66  max =   31.07  avg =   30.00
 squeezenet_ssd_int8  min =   41.00  max =   44.23  avg =   42.13
       mobilenet_ssd  min =   28.91  max =   31.07  avg =   29.82
  mobilenet_ssd_int8  min =   51.46  max =   55.86  avg =   52.38
      mobilenet_yolo  min =   49.70  max =   51.16  avg =   50.49
  mobilenetv2_yolov3  min =   32.42  max =   34.75  avg =   33.26
         yolov4-tiny  min =   40.33  max =   43.14  avg =   41.93
           nanodet_m  min =   15.71  max =   17.09  avg =   16.14


$ ./benchncnn 8 1 2 -1 1
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   22.23  max =   24.27  avg =   22.95
     squeezenet_int8  min =   46.30  max =   51.43  avg =   47.98
           mobilenet  min =   39.11  max =   41.02  avg =   39.68
      mobilenet_int8  min =   88.73  max =   92.65  avg =   90.71
        mobilenet_v2  min =   24.70  max =   24.82  avg =   24.77
        mobilenet_v3  min =   20.62  max =   22.85  avg =   21.36
          shufflenet  min =   14.86  max =   16.33  avg =   15.13
       shufflenet_v2  min =   15.26  max =   17.31  avg =   15.70
             mnasnet  min =   24.09  max =   26.18  avg =   24.49
     proxylessnasnet  min =   28.06  max =   30.31  avg =   28.79
     efficientnet_b0  min =   35.46  max =   38.73  avg =   36.76
   efficientnetv2_b0  min =   61.67  max =   65.31  avg =   63.37
        regnety_400m  min =   32.77  max =   35.13  avg =   33.74
           blazeface  min =   11.24  max =   14.33  avg =   11.81
           googlenet  min =   80.54  max =   86.01  avg =   83.25
      googlenet_int8  min =  144.73  max =  151.43  avg =  147.71
            resnet18  min =   49.64  max =   54.07  avg =   50.96
       resnet18_int8  min =  110.66  max =  114.77  avg =  112.45
             alexnet  min =   69.22  max =   79.85  avg =   73.69
               vgg16  min =  262.44  max =  271.50  avg =  266.18
          vgg16_int8  min =  544.21  max =  564.32  avg =  555.21
            resnet50  min =  172.05  max =  178.16  avg =  174.50
       resnet50_int8  min =  360.88  max =  370.64  avg =  364.51
      squeezenet_ssd  min =   68.69  max =   72.14  avg =   70.14
 squeezenet_ssd_int8  min =   99.38  max =  103.98  avg =  101.65
       mobilenet_ssd  min =   86.28  max =   95.09  avg =   89.34
  mobilenet_ssd_int8  min =  171.92  max =  178.76  avg =  175.84
      mobilenet_yolo  min =  171.12  max =  181.45  avg =  173.67
  mobilenetv2_yolov3  min =   91.30  max =   93.73  avg =   91.94
         yolov4-tiny  min =  106.43  max =  111.02  avg =  107.72
           nanodet_m  min =   35.99  max =   37.96  avg =   36.80


$ ./benchncnn 8 1 2 0 1
[0 NVIDIA Tegra Xavier (nvgpu)]  queueC=2[8]  queueG=0[16]  queueT=1[1]
[0 NVIDIA Tegra Xavier (nvgpu)]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 NVIDIA Tegra Xavier (nvgpu)]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 NVIDIA Tegra Xavier (nvgpu)]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = 0
cooling_down = 1
          squeezenet  min =    5.65  max =    6.25  avg =    5.85
     squeezenet_int8  min =   48.28  max =   52.99  avg =   50.05
           mobilenet  min =    5.82  max =    6.33  avg =    6.04
      mobilenet_int8  min =   89.35  max =   96.70  avg =   92.22
        mobilenet_v2  min =    7.17  max =    7.89  avg =    7.41
        mobilenet_v3  min =    8.32  max =    8.57  avg =    8.41
          shufflenet  min =    5.80  max =    6.13  avg =    5.98
       shufflenet_v2  min =    5.78  max =    7.07  avg =    6.69
             mnasnet  min =    6.43  max =    6.85  avg =    6.63
     proxylessnasnet  min =    6.65  max =    6.85  avg =    6.78
     efficientnet_b0  min =   11.81  max =   12.19  avg =   12.02
   efficientnetv2_b0  min =   19.43  max =   20.74  avg =   19.77
        regnety_400m  min =    7.71  max =    8.50  avg =    7.89
           blazeface  min =    2.90  max =    3.15  avg =    3.04
           googlenet  min =   10.88  max =   11.70  avg =   11.48
      googlenet_int8  min =  147.98  max =  153.42  avg =  150.24
            resnet18  min =    7.21  max =    7.46  avg =    7.30
       resnet18_int8  min =  112.84  max =  121.63  avg =  115.96
             alexnet  min =    7.91  max =    8.53  avg =    8.18
               vgg16  min =   32.77  max =   33.02  avg =   32.88
          vgg16_int8  min =  551.29  max =  568.78  avg =  556.85
            resnet50  min =   13.80  max =   14.03  avg =   13.90
       resnet50_int8  min =  360.47  max =  373.10  avg =  365.71
      squeezenet_ssd  min =   12.97  max =   13.57  avg =   13.26
 squeezenet_ssd_int8  min =  104.82  max =  107.61  avg =  106.37
       mobilenet_ssd  min =    8.30  max =    8.44  avg =    8.37
  mobilenet_ssd_int8  min =  174.79  max =  185.27  avg =  179.44
      mobilenet_yolo  min =   12.39  max =   12.68  avg =   12.51
  mobilenetv2_yolov3  min =   12.98  max =   13.48  avg =   13.13
         yolov4-tiny  min =   23.51  max =   24.84  avg =   23.96
           nanodet_m  min =    6.65  max =    6.80  avg =    6.72

        
$ ./benchncnn 8 1 0 0 1
[0 NVIDIA Tegra Xavier (nvgpu)]  queueC=2[8]  queueG=0[16]  queueT=1[1]
[0 NVIDIA Tegra Xavier (nvgpu)]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 NVIDIA Tegra Xavier (nvgpu)]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 NVIDIA Tegra Xavier (nvgpu)]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 8
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 1
          squeezenet  min =    4.63  max =    4.79  avg =    4.71
     squeezenet_int8  min =   46.50  max =   48.70  avg =   47.22
           mobilenet  min =    5.10  max =    5.20  avg =    5.15
      mobilenet_int8  min =   84.24  max =   88.19  avg =   85.59
        mobilenet_v2  min =    6.33  max =    6.51  avg =    6.41
        mobilenet_v3  min =    7.74  max =    7.91  avg =    7.85
          shufflenet  min =    5.28  max =    5.48  avg =    5.37
       shufflenet_v2  min =    6.13  max =    6.35  avg =    6.25
             mnasnet  min =    6.40  max =    6.58  avg =    6.49
     proxylessnasnet  min =    6.66  max =    7.16  avg =    6.82
     efficientnet_b0  min =   11.94  max =   12.12  avg =   12.05
   efficientnetv2_b0  min =   19.36  max =   20.60  avg =   19.69
        regnety_400m  min =    7.69  max =    8.40  avg =    7.91
           blazeface  min =    2.79  max =    3.17  avg =    2.99
           googlenet  min =   11.52  max =   12.57  avg =   11.90
      googlenet_int8  min =  144.39  max =  149.22  avg =  146.18
            resnet18  min =    7.30  max =    7.46  avg =    7.35
       resnet18_int8  min =  110.66  max =  115.50  avg =  112.27
             alexnet  min =    8.29  max =    8.40  avg =    8.33
               vgg16  min =   32.59  max =   33.11  avg =   32.83
          vgg16_int8  min =  545.74  max =  568.51  avg =  552.07
            resnet50  min =   13.71  max =   13.84  avg =   13.76
       resnet50_int8  min =  359.99  max =  369.29  avg =  365.05
      squeezenet_ssd  min =   13.09  max =   13.30  avg =   13.18
 squeezenet_ssd_int8  min =   99.73  max =  105.14  avg =  101.35
       mobilenet_ssd  min =    8.09  max =    8.50  avg =    8.25
  mobilenet_ssd_int8  min =  171.40  max =  176.93  avg =  174.56
      mobilenet_yolo  min =   12.17  max =   12.51  avg =   12.34
  mobilenetv2_yolov3  min =   12.99  max =   13.44  avg =   13.13
         yolov4-tiny  min =   23.50  max =   26.95  avg =   25.22
           nanodet_m  min =    6.41  max =    6.60  avg =    6.49
```

### MacBook Pro (13-inch, M1, 2020)
```
MacBook-Pro benchmark % ./benchncnn 10 1 0 -1 0
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 0
          squeezenet  min =    4.80  max =    5.05  avg =    4.86
     squeezenet_int8  min =    4.02  max =    4.13  avg =    4.04
           mobilenet  min =    9.09  max =    9.41  avg =    9.22
      mobilenet_int8  min =    4.65  max =    4.76  avg =    4.70
        mobilenet_v2  min =    5.64  max =    5.83  avg =    5.73
        mobilenet_v3  min =    4.64  max =    4.85  avg =    4.76
          shufflenet  min =    3.48  max =    3.63  avg =    3.56
       shufflenet_v2  min =    3.69  max =    3.81  avg =    3.73
             mnasnet  min =    5.67  max =    5.94  avg =    5.77
     proxylessnasnet  min =    7.03  max =    7.28  avg =    7.20
     efficientnet_b0  min =    9.13  max =    9.53  avg =    9.28
   efficientnetv2_b0  min =   17.37  max =   18.47  avg =   17.63
        regnety_400m  min =    7.64  max =    8.08  avg =    7.72
           blazeface  min =    1.80  max =    1.89  avg =    1.83
           googlenet  min =   25.71  max =   25.90  avg =   25.81
      googlenet_int8  min =   16.89  max =   17.10  avg =   16.97
            resnet18  min =   17.16  max =   17.28  avg =   17.20
       resnet18_int8  min =   15.55  max =   15.75  avg =   15.64
             alexnet  min =   30.60  max =   31.11  avg =   30.69
               vgg16  min =   73.41  max =   75.37  avg =   73.91
          vgg16_int8  min =  103.81  max =  105.15  avg =  104.19
            resnet50  min =   43.47  max =   44.24  avg =   43.68
       resnet50_int8  min =   30.37  max =   35.25  avg =   31.61
      squeezenet_ssd  min =   20.97  max =   21.21  avg =   21.12
 squeezenet_ssd_int8  min =   19.34  max =   19.54  avg =   19.42
       mobilenet_ssd  min =   22.18  max =   22.58  avg =   22.28
  mobilenet_ssd_int8  min =   13.27  max =   15.31  avg =   14.05
      mobilenet_yolo  min =   40.78  max =   41.04  avg =   40.89
  mobilenetv2_yolov3  min =   20.87  max =   21.92  avg =   21.02
         yolov4-tiny  min =   30.73  max =   32.37  avg =   31.29
           nanodet_m  min =    8.54  max =    8.86  avg =    8.65


MacBook-Pro benchmark % ./benchncnn 10 8 0 0 0
[0 Apple M1]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[0 Apple M1]  bugsbn1=0  bugbilz=151  bugcopc=0  bugihfa=0
[0 Apple M1]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 Apple M1]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 10
num_threads = 8
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =    1.86  max =    2.22  avg =    2.01
     squeezenet_int8  min =    2.38  max =    8.40  avg =    5.13
           mobilenet  min =    2.50  max =    2.91  avg =    2.64
      mobilenet_int8  min =    2.29  max =    5.26  avg =    3.54
        mobilenet_v2  min =    2.93  max =    3.12  avg =    2.98
        mobilenet_v3  min =    3.36  max =    3.61  avg =    3.48
          shufflenet  min =    1.99  max =    2.54  avg =    2.18
       shufflenet_v2  min =    2.35  max =    2.84  avg =    2.52
             mnasnet  min =    2.81  max =    3.33  avg =    2.92
     proxylessnasnet  min =    3.21  max =    3.62  avg =    3.36
     efficientnet_b0  min =    4.74  max =    5.73  avg =    5.07
   efficientnetv2_b0  min =   12.04  max =   13.04  avg =   12.61
        regnety_400m  min =    3.86  max =    4.04  avg =    3.98
           blazeface  min =    0.98  max =    1.11  avg =    1.03
           googlenet  min =    4.86  max =    5.38  avg =    5.02
      googlenet_int8  min =    9.43  max =   15.72  avg =   10.44
            resnet18  min =    3.92  max =    4.59  avg =    4.24
       resnet18_int8  min =    6.83  max =    7.57  avg =    7.35
             alexnet  min =    7.49  max =    7.87  avg =    7.65
               vgg16  min =   34.10  max =   35.29  avg =   34.60
          vgg16_int8  min =   40.09  max =   44.66  avg =   41.95
            resnet50  min =    7.22  max =    7.83  avg =    7.42
       resnet50_int8  min =   14.52  max =   20.56  avg =   15.78
      squeezenet_ssd  min =    8.52  max =   13.79  avg =    9.98
 squeezenet_ssd_int8  min =   12.38  max =   15.44  avg =   13.37
       mobilenet_ssd  min =    4.83  max =    6.00  avg =    5.31
  mobilenet_ssd_int8  min =    7.26  max =   13.12  avg =    9.01
      mobilenet_yolo  min =    7.22  max =    8.66  avg =    7.99
  mobilenetv2_yolov3  min =    7.46  max =    8.06  avg =    7.80
         yolov4-tiny  min =   12.17  max =   13.95  avg =   12.82
           nanodet_m  min =    3.54  max =    4.78  avg =    3.86
```
