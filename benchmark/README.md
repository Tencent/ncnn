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


Tips: Disable android UI server and set CPU and GPU to max frequency
```shell
# stopping android ui server, can be retarted later via adb shell start
adb root
adb shell stop

# executed in android adb shell
# set cpu performance mode
echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor

# set gpu performance mode (eg. RK3399)
echo "performance" > /sys/class/misc/mali0/device/devfreq/ff9a0000.gpu/governor
```

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
          squeezenet  min =    4.18  max =    4.29  avg =    4.22
     squeezenet_int8  min =    4.83  max =    4.97  avg =    4.90
           mobilenet  min =    5.73  max =    5.83  avg =    5.78
      mobilenet_int8  min =    4.89  max =    5.05  avg =    4.95
        mobilenet_v2  min =    4.93  max =    5.03  avg =    4.98
        mobilenet_v3  min =    4.41  max =    4.56  avg =    4.48
          shufflenet  min =    4.25  max =    4.41  avg =    4.33
       shufflenet_v2  min =    3.35  max =    3.46  avg =    3.40
             mnasnet  min =    4.52  max =    4.74  avg =    4.61
     proxylessnasnet  min =    5.31  max =    5.40  avg =    5.34
     efficientnet_b0  min =    8.98  max =    9.04  avg =    9.01
   efficientnetv2_b0  min =   14.26  max =   14.34  avg =   14.30
        regnety_400m  min =    8.09  max =    8.19  avg =    8.15
           blazeface  min =    1.87  max =    1.92  avg =    1.89
           googlenet  min =   16.77  max =   16.95  avg =   16.86
      googlenet_int8  min =   17.84  max =   17.94  avg =   17.89
            resnet18  min =   11.32  max =   11.88  avg =   11.45
       resnet18_int8  min =   17.81  max =   18.21  avg =   17.98
             alexnet  min =   15.96  max =   16.48  avg =   16.07
               vgg16  min =   66.39  max =   67.47  avg =   66.96
          vgg16_int8  min =  141.48  max =  144.30  avg =  143.01
            resnet50  min =   27.80  max =   28.02  avg =   27.91
       resnet50_int8  min =   33.63  max =   33.81  avg =   33.71
      squeezenet_ssd  min =   19.48  max =   19.96  avg =   19.62
 squeezenet_ssd_int8  min =   23.00  max =   23.54  avg =   23.20
       mobilenet_ssd  min =   15.15  max =   15.52  avg =   15.30
  mobilenet_ssd_int8  min =   13.41  max =   13.64  avg =   13.51
      mobilenet_yolo  min =   27.81  max =   28.63  avg =   28.10
  mobilenetv2_yolov3  min =   18.17  max =   18.75  avg =   18.31
         yolov4-tiny  min =   27.94  max =   28.19  avg =   28.05
           nanodet_m  min =    8.82  max =    8.87  avg =    8.84

OnePlus7T:/data/local/tmp $ ./benchncnn 8 1 2 -1 1
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =    9.42  max =    9.55  avg =    9.49
     squeezenet_int8  min =   10.06  max =   10.16  avg =   10.12
           mobilenet  min =   14.97  max =   15.05  avg =   15.01
      mobilenet_int8  min =   12.23  max =   12.30  avg =   12.26
        mobilenet_v2  min =   10.44  max =   10.50  avg =   10.46
        mobilenet_v3  min =    8.82  max =    9.02  avg =    8.88
          shufflenet  min =    6.98  max =    7.08  avg =    7.05
       shufflenet_v2  min =    6.73  max =    6.83  avg =    6.76
             mnasnet  min =    9.90  max =    9.98  avg =    9.93
     proxylessnasnet  min =   12.34  max =   12.50  avg =   12.41
     efficientnet_b0  min =   21.43  max =   21.46  avg =   21.45
   efficientnetv2_b0  min =   37.33  max =   37.48  avg =   37.42
        regnety_400m  min =   12.28  max =   12.37  avg =   12.31
           blazeface  min =    3.83  max =    3.95  avg =    3.88
           googlenet  min =   43.06  max =   43.11  avg =   43.08
      googlenet_int8  min =   42.54  max =   42.76  avg =   42.67
            resnet18  min =   23.32  max =   23.36  avg =   23.34
       resnet18_int8  min =   45.75  max =   46.09  avg =   45.95
             alexnet  min =   37.70  max =   38.98  avg =   38.09
               vgg16  min =  136.08  max =  137.16  avg =  136.68
          vgg16_int8  min =  360.40  max =  360.68  avg =  360.55
            resnet50  min =   67.69  max =   67.83  avg =   67.73
       resnet50_int8  min =   85.14  max =   85.30  avg =   85.21
      squeezenet_ssd  min =   35.14  max =   35.21  avg =   35.18
 squeezenet_ssd_int8  min =   45.78  max =   46.91  avg =   46.26
       mobilenet_ssd  min =   38.12  max =   38.19  avg =   38.17
  mobilenet_ssd_int8  min =   32.13  max =   32.28  avg =   32.23
      mobilenet_yolo  min =   69.44  max =   69.61  avg =   69.52
  mobilenetv2_yolov3  min =   38.15  max =   38.42  avg =   38.29
         yolov4-tiny  min =   51.27  max =   51.43  avg =   51.35
           nanodet_m  min =   16.92  max =   17.05  avg =   17.00

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

### Qualcomm MSM8996 Pro Snapdragon 821 (Kyro 2.35GHz x 2 + Kyro 2.19GHz x 2)
```
natrium:/data/local/tmp # ./benchncnn 8 4 0 -1 1
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   18.46  max =   19.12  avg =   18.78
     squeezenet_int8  min =   16.69  max =   17.22  avg =   16.95
           mobilenet  min =   27.33  max =   28.74  avg =   27.88
      mobilenet_int8  min =   20.14  max =   20.71  avg =   20.46
        mobilenet_v2  min =   21.94  max =   23.09  avg =   22.38
        mobilenet_v3  min =   18.81  max =   19.45  avg =   19.04
          shufflenet  min =   14.07  max =   14.75  avg =   14.29
       shufflenet_v2  min =   11.52  max =   11.92  avg =   11.71
             mnasnet  min =   20.41  max =   21.75  avg =   20.74
     proxylessnasnet  min =   22.99  max =   23.63  avg =   23.13
     efficientnet_b0  min =   34.74  max =   35.26  avg =   34.91
   efficientnetv2_b0  min =   41.16  max =   41.60  avg =   41.39
        regnety_400m  min =   44.27  max =   45.01  avg =   44.69
           blazeface  min =    4.25  max =    4.71  avg =    4.43
           googlenet  min =   54.88  max =   55.55  avg =   55.12
      googlenet_int8  min =   51.88  max =   52.72  avg =   52.25
            resnet18  min =   44.33  max =   45.44  avg =   44.88
       resnet18_int8  min =   51.24  max =   51.94  avg =   51.54
             alexnet  min =   38.62  max =   39.31  avg =   38.88
               vgg16  min =  242.53  max =  244.23  avg =  243.16
          vgg16_int8  min =  183.15  max =  204.96  avg =  192.16
            resnet50  min =  122.14  max =  124.29  avg =  122.94
       resnet50_int8  min =  116.61  max =  118.47  avg =  117.56
      squeezenet_ssd  min =   47.92  max =   49.01  avg =   48.45
 squeezenet_ssd_int8  min =   43.21  max =   44.45  avg =   43.76
       mobilenet_ssd  min =   56.92  max =   58.21  avg =   57.56
  mobilenet_ssd_int8  min =   42.26  max =   42.92  avg =   42.48
      mobilenet_yolo  min =  126.20  max =  128.50  avg =  127.10
  mobilenetv2_yolov3  min =   75.49  max =   76.50  avg =   76.01
         yolov4-tiny  min =   94.24  max =   95.75  avg =   94.83
           nanodet_m  min =   31.30  max =   31.93  avg =   31.62
    yolo-fastest-1.1  min =   16.89  max =   17.56  avg =   17.23
      yolo-fastestv2  min =   12.97  max =   13.50  avg =   13.15

natrium:/data/local/tmp # ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   46.27  max =   46.60  avg =   46.45
     squeezenet_int8  min =   41.33  max =   41.73  avg =   41.56
           mobilenet  min =   80.89  max =   81.16  avg =   81.00
      mobilenet_int8  min =   60.33  max =   62.29  avg =   61.33
        mobilenet_v2  min =   51.78  max =   52.02  avg =   51.88
        mobilenet_v3  min =   43.71  max =   44.17  avg =   43.91
          shufflenet  min =   24.96  max =   25.08  avg =   25.02
       shufflenet_v2  min =   24.09  max =   24.26  avg =   24.17
             mnasnet  min =   51.28  max =   51.42  avg =   51.35
     proxylessnasnet  min =   59.25  max =   59.66  avg =   59.48
     efficientnet_b0  min =   92.16  max =   92.34  avg =   92.22
   efficientnetv2_b0  min =  112.27  max =  113.63  avg =  113.17
        regnety_400m  min =   68.59  max =   68.85  avg =   68.75
           blazeface  min =    7.36  max =    7.83  avg =    7.59
           googlenet  min =  151.15  max =  151.53  avg =  151.37
      googlenet_int8  min =  152.01  max =  158.63  avg =  154.18
            resnet18  min =  121.49  max =  121.90  avg =  121.77
       resnet18_int8  min =  154.54  max =  166.73  avg =  161.30
             alexnet  min =   97.41  max =   97.74  avg =   97.62
               vgg16  min =  674.80  max =  675.86  avg =  675.38
          vgg16_int8  min =  593.42  max =  602.98  avg =  596.93
            resnet50  min =  360.44  max =  364.31  avg =  362.01
       resnet50_int8  min =  371.21  max =  386.24  avg =  381.53
      squeezenet_ssd  min =   97.72  max =   98.32  avg =   98.01
 squeezenet_ssd_int8  min =   98.33  max =   99.15  avg =   98.63
       mobilenet_ssd  min =  161.72  max =  161.89  avg =  161.79
  mobilenet_ssd_int8  min =  122.44  max =  123.38  avg =  123.00
      mobilenet_yolo  min =  367.34  max =  369.59  avg =  368.97
  mobilenetv2_yolov3  min =  190.09  max =  190.77  avg =  190.31
         yolov4-tiny  min =  241.59  max =  242.29  avg =  241.81
           nanodet_m  min =   63.03  max =   63.22  avg =   63.12
    yolo-fastest-1.1  min =   29.06  max =   29.22  avg =   29.12
      yolo-fastestv2  min =   22.72  max =   22.80  avg =   22.77
```

### Qualcomm MSM8994 Snapdragon 810 (Cortex-A57 2.0GHz x 4 + Cortex-A53 1.55GHz x 4)
```
angler:/data/local/tmp $ ./benchncnn 8 8 0 -1 1
loop_count = 8
num_threads = 8
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   25.83  max =   29.17  avg =   27.69
     squeezenet_int8  min =   24.18  max =   26.31  avg =   25.18
           mobilenet  min =   33.94  max =   35.29  avg =   34.44
      mobilenet_int8  min =   24.99  max =   26.12  avg =   25.46
        mobilenet_v2  min =   32.63  max =   34.44  avg =   33.56
        mobilenet_v3  min =   27.72  max =   30.14  avg =   29.35
          shufflenet  min =   23.23  max =   26.78  avg =   24.58
       shufflenet_v2  min =   21.04  max =   22.25  avg =   21.68
             mnasnet  min =   29.51  max =   31.26  avg =   30.27
     proxylessnasnet  min =   34.21  max =   37.55  avg =   35.20
     efficientnet_b0  min =   54.75  max =   60.45  avg =   56.38
   efficientnetv2_b0  min =   63.60  max =   67.51  avg =   64.81
        regnety_400m  min =   60.80  max =   72.33  avg =   68.27
           blazeface  min =    5.96  max =    7.22  avg =    6.41
           googlenet  min =   80.62  max =   94.46  avg =   86.50
      googlenet_int8  min =   69.05  max =   75.75  avg =   71.47
            resnet18  min =   63.90  max =   75.96  avg =   69.64
       resnet18_int8  min =   46.43  max =   62.23  avg =   53.22
             alexnet  min =   82.67  max =   90.25  avg =   87.03
               vgg16  min =  562.23  max =  636.26  avg =  594.82
          vgg16_int8  min =  303.42  max =  358.03  avg =  325.60
            resnet50  min =  233.47  max =  279.99  avg =  248.49
       resnet50_int8  min =  170.11  max =  198.27  avg =  183.35
      squeezenet_ssd  min =   86.97  max =  112.21  avg =   96.84
 squeezenet_ssd_int8  min =   66.09  max =   77.00  avg =   70.57
       mobilenet_ssd  min =   76.95  max =  101.74  avg =   87.73
  mobilenet_ssd_int8  min =   53.27  max =   60.50  avg =   57.46
      mobilenet_yolo  min =  206.42  max =  260.06  avg =  227.84
  mobilenetv2_yolov3  min =  129.32  max =  147.76  avg =  138.90
         yolov4-tiny  min =  184.85  max =  213.03  avg =  203.52
           nanodet_m  min =   47.66  max =   60.55  avg =   53.00

angler:/data/local/tmp $ ./benchncnn 8 1 2 -1 1
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   71.98  max =   73.95  avg =   73.05
     squeezenet_int8  min =   62.05  max =   62.16  avg =   62.10
           mobilenet  min =  113.19  max =  113.94  avg =  113.64
      mobilenet_int8  min =   79.60  max =   80.73  avg =   80.15
        mobilenet_v2  min =   79.50  max =   79.89  avg =   79.67
        mobilenet_v3  min =   68.32  max =   68.62  avg =   68.45
          shufflenet  min =   42.70  max =   43.01  avg =   42.84
       shufflenet_v2  min =   39.40  max =   39.88  avg =   39.58
             mnasnet  min =   76.70  max =   76.79  avg =   76.76
     proxylessnasnet  min =   95.45  max =   97.33  avg =   96.48
     efficientnet_b0  min =  168.99  max =  169.82  avg =  169.49
   efficientnetv2_b0  min =  190.62  max =  191.97  avg =  191.36
        regnety_400m  min =   98.06  max =   98.24  avg =   98.15
           blazeface  min =   12.70  max =   12.90  avg =   12.78
           googlenet  min =  238.46  max =  238.92  avg =  238.62
      googlenet_int8  min =  210.81  max =  211.14  avg =  211.00
            resnet18  min =  185.46  max =  186.29  avg =  185.93
       resnet18_int8  min =  155.11  max =  158.06  avg =  156.19
             alexnet  min =  201.09  max =  201.65  avg =  201.44
               vgg16  min = 1079.75  max = 1083.42  avg = 1081.78
          vgg16_int8  min =  815.97  max =  816.80  avg =  816.47
            resnet50  min =  502.17  max =  506.94  avg =  504.31
       resnet50_int8  min =  395.07  max =  396.10  avg =  395.61
      squeezenet_ssd  min =  163.81  max =  164.24  avg =  164.12
 squeezenet_ssd_int8  min =  149.15  max =  149.36  avg =  149.26
       mobilenet_ssd  min =  231.28  max =  231.62  avg =  231.37
  mobilenet_ssd_int8  min =  159.82  max =  160.00  avg =  159.89
      mobilenet_yolo  min =  519.76  max =  524.85  avg =  522.86
  mobilenetv2_yolov3  min =  273.42  max =  275.30  avg =  274.62
         yolov4-tiny  min =  346.24  max =  350.79  avg =  348.23
           nanodet_m  min =   97.26  max =   98.58  avg =   97.97

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
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   65.45  max =   73.59  avg =   68.10
     squeezenet_int8  min =   59.39  max =   65.54  avg =   61.14
           mobilenet  min =   86.69  max =   94.10  avg =   90.03
      mobilenet_int8  min =   62.22  max =   69.67  avg =   64.13
        mobilenet_v2  min =   77.98  max =   89.53  avg =   82.00
        mobilenet_v3  min =   62.17  max =   68.31  avg =   63.90
          shufflenet  min =   47.52  max =   53.76  avg =   49.92
       shufflenet_v2  min =   39.77  max =   46.08  avg =   40.66
             mnasnet  min =   69.27  max =   75.73  avg =   71.73
     proxylessnasnet  min =   78.72  max =   85.37  avg =   81.33
     efficientnet_b0  min =  126.62  max =  136.67  avg =  130.69
   efficientnetv2_b0  min =  143.24  max =  150.97  avg =  146.89
        regnety_400m  min =  108.79  max =  116.22  avg =  112.99
           blazeface  min =   14.85  max =   15.02  avg =   14.94
           googlenet  min =  180.91  max =  190.37  avg =  186.36
      googlenet_int8  min =  160.07  max =  170.86  avg =  165.05
            resnet18  min =  137.91  max =  155.37  avg =  144.99
       resnet18_int8  min =  104.34  max =  110.20  avg =  106.76
             alexnet  min =  105.30  max =  114.73  avg =  109.53
               vgg16  min =  829.16  max =  942.94  avg =  853.28
          vgg16_int8  min =  515.61  max =  547.32  avg =  526.50
            resnet50  min =  380.46  max =  443.90  avg =  393.71
       resnet50_int8  min =  318.06  max =  327.13  avg =  323.23
      squeezenet_ssd  min =  178.22  max =  189.02  avg =  184.51
 squeezenet_ssd_int8  min =  153.75  max =  163.44  avg =  158.05
       mobilenet_ssd  min =  189.45  max =  195.17  avg =  193.10
  mobilenet_ssd_int8  min =  132.59  max =  139.63  avg =  137.23
      mobilenet_yolo  min =  404.52  max =  414.20  avg =  409.97
  mobilenetv2_yolov3  min =  271.33  max =  279.98  avg =  275.08
         yolov4-tiny  min =  349.36  max =  372.54  avg =  357.98
           nanodet_m  min =  103.01  max =  111.71  avg =  105.82

HM2014812:/data/local/tmp # ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  147.48  max =  149.35  avg =  148.40
     squeezenet_int8  min =  143.20  max =  144.55  avg =  143.98
           mobilenet  min =  243.78  max =  244.33  avg =  244.08
      mobilenet_int8  min =  206.23  max =  207.13  avg =  206.55
        mobilenet_v2  min =  168.04  max =  170.37  avg =  169.06
        mobilenet_v3  min =  147.10  max =  147.91  avg =  147.55
          shufflenet  min =   88.47  max =   89.31  avg =   88.85
       shufflenet_v2  min =   84.47  max =   84.80  avg =   84.60
             mnasnet  min =  162.81  max =  163.93  avg =  163.22
     proxylessnasnet  min =  208.18  max =  209.15  avg =  208.61
     efficientnet_b0  min =  370.06  max =  371.14  avg =  370.64
   efficientnetv2_b0  min =  418.28  max =  429.68  avg =  423.01
        regnety_400m  min =  216.42  max =  217.19  avg =  216.71
           blazeface  min =   27.63  max =   28.67  avg =   28.00
           googlenet  min =  525.25  max =  528.83  avg =  526.23
      googlenet_int8  min =  469.78  max =  472.51  avg =  470.76
            resnet18  min =  396.46  max =  399.66  avg =  397.57
       resnet18_int8  min =  324.07  max =  326.64  avg =  325.34
             alexnet  min =  362.44  max =  363.02  avg =  362.68
               vgg16  min = 2174.86  max = 2252.92  avg = 2215.62
          vgg16_int8  min = 1726.07  max = 1732.69  avg = 1729.18
            resnet50  min = 1136.96  max = 1142.94  avg = 1139.91
       resnet50_int8  min =  977.73  max =  983.64  avg =  980.71
      squeezenet_ssd  min =  350.46  max =  353.35  avg =  351.37
 squeezenet_ssd_int8  min =  333.91  max =  336.59  avg =  334.77
       mobilenet_ssd  min =  513.18  max =  519.05  avg =  516.22
  mobilenet_ssd_int8  min =  424.37  max =  426.89  avg =  426.03
      mobilenet_yolo  min = 1143.20  max = 1145.04  avg = 1144.31
  mobilenetv2_yolov3  min =  617.45  max =  619.30  avg =  618.37
         yolov4-tiny  min =  839.32  max =  847.57  avg =  844.61
           nanodet_m  min =  208.41  max =  211.31  avg =  210.03
```

### Raspberry Pi 3 Model B+ Broadcom BCM2837B0, Cortex-A53 (ARMv8) (1.4GHz x 4)
```
pi@raspberrypi:~/ncnn/build/benchmark $ ./benchncnn 8 4 0 -1 1
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   93.48  max =   94.42  avg =   93.92
     squeezenet_int8  min =   75.56  max =   91.37  avg =   78.19
           mobilenet  min =  111.72  max =  116.75  avg =  113.22
      mobilenet_int8  min =   66.81  max =   67.65  avg =   67.27
        mobilenet_v2  min =  118.23  max =  122.53  avg =  120.10
        mobilenet_v3  min =   90.46  max =   92.37  avg =   91.43
          shufflenet  min =   63.71  max =   64.23  avg =   63.93
       shufflenet_v2  min =   55.14  max =   55.45  avg =   55.23
             mnasnet  min =   96.91  max =  100.11  avg =   98.45
     proxylessnasnet  min =  105.64  max =  109.58  avg =  107.21
     efficientnet_b0  min =  152.08  max =  153.21  avg =  152.51
   efficientnetv2_b0  min =  173.82  max =  174.44  avg =  174.09
        regnety_400m  min =  128.52  max =  130.30  avg =  129.22
           blazeface  min =   19.25  max =   20.16  avg =   19.86
           googlenet  min =  242.42  max =  248.31  avg =  244.42
      googlenet_int8  min =  189.70  max =  192.26  avg =  190.93
            resnet18  min =  279.99  max =  282.42  avg =  281.43
       resnet18_int8  min =  162.68  max =  167.57  avg =  165.72
             alexnet  min =  219.22  max =  227.34  avg =  222.01
            resnet50  min =  558.68  max =  567.64  avg =  562.70
       resnet50_int8  min =  391.51  max =  393.13  avg =  392.29
      squeezenet_ssd  min =  313.05  max =  316.22  avg =  314.47
 squeezenet_ssd_int8  min =  201.40  max =  202.80  avg =  201.96
       mobilenet_ssd  min =  242.57  max =  244.58  avg =  243.38
  mobilenet_ssd_int8  min =  138.13  max =  139.16  avg =  138.44
      mobilenet_yolo  min =  515.24  max =  527.39  avg =  519.71
  mobilenetv2_yolov3  min =  367.45  max =  379.95  avg =  374.68
         yolov4-tiny  min =  473.25  max =  481.18  avg =  475.87
           nanodet_m  min =  140.82  max =  142.07  avg =  141.42
    yolo-fastest-1.1  min =   84.35  max =   84.78  avg =   84.57
      yolo-fastestv2  min =   69.17  max =   69.71  avg =   69.42

pi@raspberrypi:~/ncnn/build/benchmark $ ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  152.36  max =  152.52  avg =  152.46
     squeezenet_int8  min =  138.38  max =  138.69  avg =  138.52
           mobilenet  min =  233.98  max =  238.35  avg =  235.30
      mobilenet_int8  min =  185.11  max =  185.42  avg =  185.30
        mobilenet_v2  min =  173.90  max =  175.93  avg =  175.17
        mobilenet_v3  min =  151.83  max =  153.28  avg =  152.52
          shufflenet  min =   91.71  max =   92.43  avg =   92.08
       shufflenet_v2  min =   97.29  max =   97.59  avg =   97.46
             mnasnet  min =  167.58  max =  168.03  avg =  167.86
     proxylessnasnet  min =  213.93  max =  216.89  avg =  215.50
     efficientnet_b0  min =  332.36  max =  332.48  avg =  332.42
   efficientnetv2_b0  min =  383.45  max =  384.49  avg =  383.92
        regnety_400m  min =  211.78  max =  212.81  avg =  212.09
           blazeface  min =   28.57  max =   29.41  avg =   28.94
           googlenet  min =  497.79  max =  499.53  avg =  498.56
      googlenet_int8  min =  429.97  max =  433.06  avg =  431.34
            resnet18  min =  423.38  max =  424.28  avg =  423.92
       resnet18_int8  min =  316.29  max =  317.27  avg =  316.74
             alexnet  min =  472.86  max =  473.48  avg =  473.15
            resnet50  min = 1100.52  max = 1103.82  avg = 1102.50
       resnet50_int8  min =  899.54  max =  902.28  avg =  901.05
      squeezenet_ssd  min =  404.44  max =  408.47  avg =  405.81
 squeezenet_ssd_int8  min =  324.34  max =  327.91  avg =  325.62
       mobilenet_ssd  min =  473.55  max =  474.66  avg =  474.11
  mobilenet_ssd_int8  min =  370.25  max =  371.03  avg =  370.56
      mobilenet_yolo  min = 1049.83  max = 1053.00  avg = 1051.51
  mobilenetv2_yolov3  min =  587.69  max =  588.59  avg =  588.22
         yolov4-tiny  min =  807.98  max =  809.10  avg =  808.64
           nanodet_m  min =  235.36  max =  236.96  avg =  236.19
    yolo-fastest-1.1  min =  105.76  max =  107.73  avg =  106.33
      yolo-fastestv2  min =   89.43  max =   89.72  avg =   89.54
```

### Raspberry Pi 4 Model B Broadcom BCM2711B0, Cortex-A72 (ARMv8) (1.5GHz x 4)
```
pi@raspberrypi:~ $ ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   58.38  max =   59.30  avg =   58.93
     squeezenet_int8  min =   48.98  max =   49.63  avg =   49.33
           mobilenet  min =   71.59  max =   72.33  avg =   72.08
      mobilenet_int8  min =   40.22  max =   40.35  avg =   40.30
        mobilenet_v2  min =   72.26  max =   73.16  avg =   72.62
        mobilenet_v3  min =   55.58  max =   56.64  avg =   56.34
          shufflenet  min =   37.93  max =   38.92  avg =   38.33
       shufflenet_v2  min =   29.54  max =   30.00  avg =   29.78
             mnasnet  min =   61.55  max =   62.15  avg =   61.82
     proxylessnasnet  min =   63.30  max =   63.68  avg =   63.45
     efficientnet_b0  min =   93.93  max =   95.05  avg =   94.39
   efficientnetv2_b0  min =  104.65  max =  105.15  avg =  104.85
        regnety_400m  min =   80.08  max =   81.99  avg =   81.09
           blazeface  min =   13.71  max =   14.04  avg =   13.82
           googlenet  min =  142.17  max =  143.88  avg =  143.09
      googlenet_int8  min =  117.55  max =  119.72  avg =  118.78
            resnet18  min =  175.44  max =  176.83  avg =  176.18
       resnet18_int8  min =   95.95  max =   99.11  avg =   97.99
             alexnet  min =  142.71  max =  144.85  avg =  143.52
               vgg16  min =  871.96  max =  875.45  avg =  873.71
          vgg16_int8  min =  455.05  max =  458.89  avg =  456.76
            resnet50  min =  334.35  max =  336.91  avg =  335.34
       resnet50_int8  min =  234.15  max =  238.99  avg =  236.38
      squeezenet_ssd  min =  179.60  max =  180.50  avg =  180.10
 squeezenet_ssd_int8  min =  130.65  max =  132.21  avg =  131.37
       mobilenet_ssd  min =  143.86  max =  145.48  avg =  144.75
  mobilenet_ssd_int8  min =   84.97  max =   85.71  avg =   85.31
      mobilenet_yolo  min =  321.30  max =  324.29  avg =  322.72
  mobilenetv2_yolov3  min =  217.92  max =  219.28  avg =  218.45
         yolov4-tiny  min =  280.18  max =  285.17  avg =  283.51
           nanodet_m  min =   80.26  max =   80.78  avg =   80.57
    yolo-fastest-1.1  min =   54.31  max =   55.96  avg =   55.11
      yolo-fastestv2  min =   44.74  max =   45.56  avg =   45.15

pi@raspberrypi:~ $ ./benchncnn 8 1 0
loop_count = 8
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   92.26  max =   92.88  avg =   92.60
     squeezenet_int8  min =   81.57  max =   82.20  avg =   81.90
           mobilenet  min =  145.36  max =  146.46  avg =  145.94
      mobilenet_int8  min =   99.54  max =   99.69  avg =   99.62
        mobilenet_v2  min =  109.98  max =  110.29  avg =  110.10
        mobilenet_v3  min =   88.16  max =   88.72  avg =   88.41
          shufflenet  min =   54.60  max =   55.03  avg =   54.76
       shufflenet_v2  min =   50.02  max =   50.66  avg =   50.30
             mnasnet  min =   99.74  max =  103.59  avg =  100.50
     proxylessnasnet  min =  117.14  max =  119.65  avg =  119.12
     efficientnet_b0  min =  194.20  max =  194.59  avg =  194.41
   efficientnetv2_b0  min =  221.52  max =  221.95  avg =  221.74
        regnety_400m  min =  135.36  max =  135.93  avg =  135.69
           blazeface  min =   17.29  max =   17.64  avg =   17.50
           googlenet  min =  282.88  max =  285.25  avg =  283.92
      googlenet_int8  min =  252.00  max =  252.58  avg =  252.23
            resnet18  min =  226.03  max =  226.82  avg =  226.49
       resnet18_int8  min =  188.88  max =  189.09  avg =  188.99
             alexnet  min =  213.34  max =  214.16  avg =  213.76
               vgg16  min = 1307.28  max = 1309.05  avg = 1307.79
          vgg16_int8  min = 1024.11  max = 1031.10  avg = 1026.32
            resnet50  min =  633.78  max =  638.23  avg =  636.02
       resnet50_int8  min =  501.96  max =  504.98  avg =  503.46
      squeezenet_ssd  min =  212.90  max =  215.44  avg =  214.85
 squeezenet_ssd_int8  min =  188.72  max =  190.73  avg =  189.38
       mobilenet_ssd  min =  294.98  max =  296.01  avg =  295.44
  mobilenet_ssd_int8  min =  200.44  max =  201.85  avg =  200.87
      mobilenet_yolo  min =  660.89  max =  662.27  avg =  661.82
  mobilenetv2_yolov3  min =  367.30  max =  368.69  avg =  368.05
         yolov4-tiny  min =  439.10  max =  441.09  avg =  440.07
           nanodet_m  min =  124.23  max =  124.88  avg =  124.42
    yolo-fastest-1.1  min =   68.99  max =   69.68  avg =   69.32
      yolo-fastestv2  min =   55.51  max =   56.02  avg =   55.87
```

### Raspberry Pi Zero 2 W Broadcom BCM2710A1, Cortex-A53 (ARMv8) (1.0GHz x 4)

```
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  119.52  max =  120.29  avg =  119.93
     squeezenet_int8  min =   96.32  max =   96.96  avg =   96.55
           mobilenet  min =  162.60  max =  165.49  avg =  163.19
      mobilenet_int8  min =   90.78  max =   91.39  avg =   91.03
        mobilenet_v2  min =  145.71  max =  148.83  avg =  147.39
        mobilenet_v3  min =  113.89  max =  151.95  avg =  119.04
          shufflenet  min =   72.72  max =   73.27  avg =   72.96
       shufflenet_v2  min =   63.64  max =   64.50  avg =   64.13
             mnasnet  min =  126.07  max =  126.93  avg =  126.53
     proxylessnasnet  min =  139.90  max =  140.84  avg =  140.35
     efficientnet_b0  min =  201.88  max =  202.55  avg =  202.14
   efficientnetv2_b0  min =  227.22  max =  228.84  avg =  228.09
        regnety_400m  min =  156.49  max =  157.47  avg =  156.96
           blazeface  min =   22.79  max =   23.28  avg =   23.10
           googlenet  min =  323.74  max =  324.90  avg =  324.45
      googlenet_int8  min =  250.86  max =  252.82  avg =  251.63
            resnet18  min =  351.37  max =  355.67  avg =  353.45
       resnet18_int8  min =  194.83  max =  196.68  avg =  195.51
             alexnet  min =  271.18  max =  273.53  avg =  272.18
            resnet50  min =  777.44  max =  797.47  avg =  782.63
       resnet50_int8  min =  496.78  max =  498.86  avg =  497.57
      squeezenet_ssd  min =  376.10  max =  382.41  avg =  379.13
 squeezenet_ssd_int8  min =  255.99  max =  257.57  avg =  256.78
       mobilenet_ssd  min =  338.64  max =  339.93  avg =  339.50
  mobilenet_ssd_int8  min =  190.24  max =  190.68  avg =  190.48
      mobilenet_yolo  min =  746.83  max =  748.14  avg =  747.53
  mobilenetv2_yolov3  min =  487.99  max =  491.18  avg =  489.37
         yolov4-tiny  min =  644.73  max =  652.24  avg =  646.64
           nanodet_m  min =  165.27  max =  167.12  avg =  166.27
    yolo-fastest-1.1  min =   98.74  max =  100.02  avg =   99.17
      yolo-fastestv2  min =   80.52  max =   81.86  avg =   81.29

loop_count = 8
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  240.53  max =  241.07  avg =  240.77
     squeezenet_int8  min =  212.63  max =  213.23  avg =  212.94
           mobilenet  min =  393.79  max =  394.04  avg =  393.94
      mobilenet_int8  min =  286.58  max =  286.95  avg =  286.75
        mobilenet_v2  min =  273.97  max =  274.51  avg =  274.23
        mobilenet_v3  min =  233.77  max =  234.59  avg =  234.20
          shufflenet  min =  133.05  max =  133.36  avg =  133.23
       shufflenet_v2  min =  128.86  max =  129.47  avg =  129.18
             mnasnet  min =  265.70  max =  266.17  avg =  265.93
     proxylessnasnet  min =  329.78  max =  330.54  avg =  330.13
     efficientnet_b0  min =  518.42  max =  519.38  avg =  519.00
   efficientnetv2_b0  min =  594.37  max =  595.17  avg =  594.74
        regnety_400m  min =  329.53  max =  330.44  avg =  329.87
           blazeface  min =   42.24  max =   45.56  avg =   43.96
           googlenet  min =  780.05  max =  780.63  avg =  780.39
      googlenet_int8  min =  663.83  max =  664.43  avg =  664.15
            resnet18  min =  653.62  max =  657.59  avg =  654.69
       resnet18_int8  min =  479.03  max =  479.72  avg =  479.40
             alexnet  min =  687.99  max =  690.34  avg =  689.15
            resnet50  min = 1800.97  max = 1806.11  avg = 1802.79
       resnet50_int8  min = 1311.68  max = 1314.56  avg = 1313.15
      squeezenet_ssd  min =  563.63  max =  565.57  avg =  564.44
 squeezenet_ssd_int8  min =  481.24  max =  483.97  avg =  482.20
       mobilenet_ssd  min =  799.21  max =  829.10  avg =  803.56
  mobilenet_ssd_int8  min =  568.11  max =  568.88  avg =  568.42
      mobilenet_yolo  min = 1815.60  max = 1816.44  avg = 1815.93
  mobilenetv2_yolov3  min =  951.34  max =  952.15  avg =  951.72
         yolov4-tiny  min = 1258.21  max = 1259.49  avg = 1258.66
           nanodet_m  min =  301.04  max =  304.09  avg =  301.70
    yolo-fastest-1.1  min =  155.04  max =  155.98  avg =  155.53
      yolo-fastestv2  min =  126.77  max =  127.40  avg =  127.05
```

### Banana Pi M2 Zero 2 AllWinner H2+, Cortex-A7 (ARMv7-A) (1.2GHz x 4)

```
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  230.97  max =  232.18  avg =  231.49
     squeezenet_int8  min =  171.12  max =  172.87  avg =  171.68
           mobilenet  min =  327.65  max =  340.92  avg =  329.88
      mobilenet_int8  min =  166.58  max =  169.55  avg =  167.47
        mobilenet_v2  min =  276.81  max =  278.67  avg =  277.55
        mobilenet_v3  min =  220.74  max =  225.14  avg =  222.08
          shufflenet  min =  147.97  max =  157.68  avg =  149.40
       shufflenet_v2  min =  146.56  max =  154.90  avg =  148.25
             mnasnet  min =  243.06  max =  244.47  avg =  243.80
     proxylessnasnet  min =  260.38  max =  261.47  avg =  260.66
     efficientnet_b0  min =  368.98  max =  371.03  avg =  369.96
   efficientnetv2_b0  min =  433.96  max =  459.25  avg =  437.52
        regnety_400m  min =  307.53  max =  312.29  avg =  308.68
           blazeface  min =   46.54  max =   47.35  avg =   46.98
           googlenet  min =  647.86  max =  669.20  avg =  651.19
      googlenet_int8  min =  439.90  max =  442.35  avg =  441.38
            resnet18  min =  642.53  max =  856.58  avg =  698.28
       resnet18_int8  min =  352.10  max =  354.51  avg =  353.44
             alexnet  min =  593.16  max =  624.20  avg =  598.66
            resnet50  min = 1556.12  max = 1782.22  avg = 1606.86
       resnet50_int8  min =  911.63  max =  999.42  avg =  924.37
      squeezenet_ssd  min =  653.85  max =  658.07  avg =  655.19
 squeezenet_ssd_int8  min =  456.26  max =  467.76  avg =  459.87
       mobilenet_ssd  min =  671.93  max =  682.64  avg =  674.88
  mobilenet_ssd_int8  min =  347.18  max =  349.07  avg =  347.81
      mobilenet_yolo  min = 1471.16  max = 1492.65  avg = 1479.30
  mobilenetv2_yolov3  min =  895.90  max =  906.60  avg =  899.74
         yolov4-tiny  min = 1178.53  max = 1205.79  avg = 1183.98
           nanodet_m  min =  358.89  max =  366.07  avg =  362.20
    yolo-fastest-1.1  min =  189.93  max =  192.18  avg =  190.91
      yolo-fastestv2  min =  158.60  max =  161.33  avg =  159.43

loop_count = 8
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  602.97  max =  604.97  avg =  603.46
     squeezenet_int8  min =  431.18  max =  432.42  avg =  431.77
           mobilenet  min =  971.52  max =  986.64  avg =  974.04
      mobilenet_int8  min =  556.74  max =  556.98  avg =  556.84
        mobilenet_v2  min =  682.85  max =  684.17  avg =  683.34
        mobilenet_v3  min =  585.10  max =  585.76  avg =  585.57
          shufflenet  min =  340.64  max =  342.63  avg =  341.26
       shufflenet_v2  min =  322.41  max =  324.13  avg =  323.35
             mnasnet  min =  644.30  max =  645.93  avg =  644.71
     proxylessnasnet  min =  732.50  max =  733.30  avg =  732.96
     efficientnet_b0  min = 1084.70  max = 1094.98  avg = 1086.52
   efficientnetv2_b0  min = 1282.27  max = 1283.67  avg = 1282.60
        regnety_400m  min =  764.60  max =  768.54  avg =  765.30
           blazeface  min =  100.48  max =  106.28  avg =  103.33
           googlenet  min = 1878.69  max = 1883.96  avg = 1880.76
      googlenet_int8  min = 1274.31  max = 1296.02  avg = 1279.59
            resnet18  min = 1837.91  max = 1843.95  avg = 1839.17
       resnet18_int8  min = 1011.98  max = 1014.43  avg = 1013.01
             alexnet  min = 1997.59  max = 2001.81  avg = 1999.42
            resnet50  min = 4844.31  max = 4857.05  avg = 4847.80
       resnet50_int8  min = 2792.59  max = 2810.08  avg = 2797.30
      squeezenet_ssd  min = 1438.96  max = 1443.31  avg = 1441.09
 squeezenet_ssd_int8  min = 1046.76  max = 1053.00  avg = 1049.22
       mobilenet_ssd  min = 2018.66  max = 2023.70  avg = 2019.67
  mobilenet_ssd_int8  min = 1129.16  max = 1130.62  avg = 1129.82
      mobilenet_yolo  min = 4724.90  max = 4728.57  avg = 4726.41
  mobilenetv2_yolov3  min = 2410.67  max = 2427.95  avg = 2413.89
         yolov4-tiny  min = 3177.27  max = 3185.52  avg = 3179.71
           nanodet_m  min =  761.38  max =  768.79  avg =  766.53
    yolo-fastest-1.1  min =  391.82  max =  393.32  avg =  392.39
      yolo-fastestv2  min =  316.93  max =  319.86  avg =  318.33
```

### Khadas VIM3, Amlogic A311D (Cortex-A73 2.2GHz x 4 + Cortex-A53 1.8GHz x 2)

```
vim3:/data/local/tmp # ./benchncnn 8 4 2 -1 1
loop_count = 8
num_threads = 4
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   30.98  max =   31.26  avg =   31.09
     squeezenet_int8  min =   24.70  max =   24.84  avg =   24.78
           mobilenet  min =   42.57  max =   43.37  avg =   42.96
      mobilenet_int8  min =   22.33  max =   22.52  avg =   22.44
        mobilenet_v2  min =   39.36  max =   39.77  avg =   39.56
        mobilenet_v3  min =   30.13  max =   30.45  avg =   30.28
          shufflenet  min =   21.62  max =   21.94  avg =   21.80
       shufflenet_v2  min =   18.83  max =   19.24  avg =   19.05
             mnasnet  min =   33.54  max =   34.08  avg =   33.80
     proxylessnasnet  min =   35.81  max =   36.05  avg =   35.95
     efficientnet_b0  min =   53.82  max =   54.44  avg =   54.21
   efficientnetv2_b0  min =   62.20  max =   62.60  avg =   62.43
        regnety_400m  min =   48.82  max =   49.27  avg =   49.05
           blazeface  min =    6.34  max =    6.51  avg =    6.43
           googlenet  min =   81.96  max =   82.53  avg =   82.23
      googlenet_int8  min =   64.42  max =   65.00  avg =   64.77
            resnet18  min =   77.00  max =   77.83  avg =   77.46
       resnet18_int8  min =   48.91  max =   49.14  avg =   49.05
             alexnet  min =   60.43  max =   60.93  avg =   60.69
               vgg16  min =  414.89  max =  423.00  avg =  418.75
          vgg16_int8  min =  245.58  max =  246.37  avg =  245.94
            resnet50  min =  185.53  max =  187.35  avg =  186.18
       resnet50_int8  min =  123.36  max =  124.75  avg =  124.17
      squeezenet_ssd  min =   85.87  max =   86.42  avg =   86.23
 squeezenet_ssd_int8  min =   64.90  max =   65.24  avg =   65.08
       mobilenet_ssd  min =   88.32  max =   90.02  avg =   89.10
  mobilenet_ssd_int8  min =   46.85  max =   47.18  avg =   46.98
      mobilenet_yolo  min =  192.33  max =  195.38  avg =  194.10
  mobilenetv2_yolov3  min =  127.33  max =  128.58  avg =  127.96
         yolov4-tiny  min =  150.44  max =  152.02  avg =  151.20
           nanodet_m  min =   54.22  max =   54.61  avg =   54.37
    yolo-fastest-1.1  min =   28.13  max =   28.76  avg =   28.40
      yolo-fastestv2  min =   22.10  max =   22.26  avg =   22.19

vim3:/data/local/tmp # ./benchncnn 4 1 2 -1 1
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   68.25  max =   68.85  avg =   68.67
     squeezenet_int8  min =   51.92  max =   52.08  avg =   52.01
           mobilenet  min =  112.69  max =  113.72  avg =  113.33
      mobilenet_int8  min =   66.43  max =   66.89  avg =   66.68
        mobilenet_v2  min =   81.36  max =   81.77  avg =   81.62
        mobilenet_v3  min =   62.33  max =   63.39  avg =   62.94
          shufflenet  min =   37.84  max =   38.03  avg =   37.93
       shufflenet_v2  min =   37.33  max =   38.08  avg =   37.68
             mnasnet  min =   73.83  max =   74.32  avg =   74.03
     proxylessnasnet  min =   85.19  max =   86.43  avg =   85.84
     efficientnet_b0  min =  138.68  max =  139.69  avg =  139.19
   efficientnetv2_b0  min =  167.53  max =  167.99  avg =  167.75
        regnety_400m  min =   94.78  max =   95.81  avg =   95.21
           blazeface  min =   11.22  max =   11.43  avg =   11.28
           googlenet  min =  229.35  max =  230.91  avg =  229.89
      googlenet_int8  min =  173.04  max =  173.48  avg =  173.24
            resnet18  min =  191.54  max =  193.78  avg =  192.49
       resnet18_int8  min =  132.97  max =  133.51  avg =  133.25
             alexnet  min =  140.31  max =  141.95  avg =  141.18
               vgg16  min = 1093.71  max = 1100.95  avg = 1097.64
          vgg16_int8  min =  734.44  max =  736.16  avg =  735.05
            resnet50  min =  530.38  max =  533.93  avg =  531.87
       resnet50_int8  min =  332.88  max =  334.22  avg =  333.71
      squeezenet_ssd  min =  159.08  max =  160.98  avg =  160.16
 squeezenet_ssd_int8  min =  126.97  max =  127.96  avg =  127.43
       mobilenet_ssd  min =  238.92  max =  241.14  avg =  239.70
  mobilenet_ssd_int8  min =  135.57  max =  136.02  avg =  135.78
      mobilenet_yolo  min =  539.59  max =  543.88  avg =  541.90
  mobilenetv2_yolov3  min =  281.32  max =  285.05  avg =  283.24
         yolov4-tiny  min =  381.99  max =  384.93  avg =  383.53
           nanodet_m  min =   98.32  max =   98.85  avg =   98.60
    yolo-fastest-1.1  min =   44.59  max =   44.95  avg =   44.80
      yolo-fastestv2  min =   36.88  max =   37.11  avg =   36.98
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

### Rock3A, Rockchip RK3568 (Cortex-A55 2.0GHz x 4) ubuntu 20.04

```
rock@rock3a:~/ncnn/build/benchmark$ ./benchncnn 8 4 0 -1 1
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   29.52  max =   30.30  avg =   29.76
     squeezenet_int8  min =   35.40  max =   36.19  avg =   35.88
           mobilenet  min =   34.47  max =   35.44  avg =   34.84
      mobilenet_int8  min =   34.19  max =   34.53  avg =   34.40
        mobilenet_v2  min =   35.75  max =   36.09  avg =   35.88
        mobilenet_v3  min =   28.12  max =   28.82  avg =   28.49
          shufflenet  min =   23.62  max =   24.08  avg =   23.84
       shufflenet_v2  min =   19.37  max =   19.64  avg =   19.52
             mnasnet  min =   30.84  max =   31.45  avg =   31.02
     proxylessnasnet  min =   35.73  max =   36.07  avg =   35.90
     efficientnet_b0  min =   48.16  max =   49.29  avg =   48.64
   efficientnetv2_b0  min =   66.62  max =   67.11  avg =   66.85
        regnety_400m  min =   41.11  max =   41.64  avg =   41.34
           blazeface  min =   12.38  max =   12.64  avg =   12.56
           googlenet  min =   86.73  max =   87.79  avg =   87.11
      googlenet_int8  min =  101.42  max =  103.87  avg =  102.55
            resnet18  min =   64.85  max =   65.84  avg =   65.23
       resnet18_int8  min =   93.55  max =   94.54  avg =   94.03
             alexnet  min =   70.89  max =   73.58  avg =   71.57
               vgg16  min =  356.13  max =  358.52  avg =  357.15
          vgg16_int8  min =  521.92  max =  524.13  avg =  523.11
            resnet50  min =  147.65  max =  150.33  avg =  148.52
       resnet50_int8  min =  191.94  max =  192.73  avg =  192.30
      squeezenet_ssd  min =  104.32  max =  105.75  avg =  105.00
 squeezenet_ssd_int8  min =  125.97  max =  127.53  avg =  126.70
       mobilenet_ssd  min =   82.29  max =   82.65  avg =   82.47
  mobilenet_ssd_int8  min =   79.26  max =   80.93  avg =   79.72
      mobilenet_yolo  min =  165.51  max =  165.86  avg =  165.72
  mobilenetv2_yolov3  min =  116.11  max =  116.83  avg =  116.43
         yolov4-tiny  min =  152.09  max =  153.39  avg =  152.60
           nanodet_m  min =   53.63  max =   54.14  avg =   53.92

rock@rock3a:~/ncnn/build/benchmark$ ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   62.47  max =   63.04  avg =   62.84
     squeezenet_int8  min =   67.23  max =   68.48  avg =   67.93
           mobilenet  min =   85.27  max =   85.69  avg =   85.49
      mobilenet_int8  min =   75.00  max =   75.48  avg =   75.26
        mobilenet_v2  min =   68.41  max =   69.09  avg =   68.76
        mobilenet_v3  min =   54.19  max =   54.52  avg =   54.34
          shufflenet  min =   45.90  max =   46.30  avg =   46.09
       shufflenet_v2  min =   39.64  max =   40.07  avg =   39.91
             mnasnet  min =   62.16  max =   62.41  avg =   62.30
     proxylessnasnet  min =   80.79  max =   81.41  avg =   81.12
     efficientnet_b0  min =  113.47  max =  113.68  avg =  113.57
   efficientnetv2_b0  min =  167.30  max =  167.58  avg =  167.44
        regnety_400m  min =   72.12  max =   72.24  avg =   72.17
           blazeface  min =   31.89  max =   32.04  avg =   31.95
           googlenet  min =  224.27  max =  224.86  avg =  224.55
      googlenet_int8  min =  240.02  max =  240.93  avg =  240.45
            resnet18  min =  150.25  max =  150.69  avg =  150.47
       resnet18_int8  min =  226.70  max =  228.19  avg =  227.56
             alexnet  min =  197.44  max =  199.16  avg =  198.17
               vgg16  min =  859.80  max =  860.79  avg =  860.35
          vgg16_int8  min = 1409.66  max = 1411.92  avg = 1411.07
            resnet50  min =  381.04  max =  382.73  avg =  381.86
       resnet50_int8  min =  441.78  max =  445.00  avg =  443.29
      squeezenet_ssd  min =  208.14  max =  208.67  avg =  208.41
 squeezenet_ssd_int8  min =  248.82  max =  250.80  avg =  249.89
       mobilenet_ssd  min =  200.95  max =  201.21  avg =  201.06
  mobilenet_ssd_int8  min =  173.81  max =  174.54  avg =  174.28
      mobilenet_yolo  min =  394.65  max =  395.00  avg =  394.78
  mobilenetv2_yolov3  min =  231.80  max =  232.27  avg =  232.08
         yolov4-tiny  min =  321.31  max =  322.43  avg =  321.79
           nanodet_m  min =  103.81  max =  104.61  avg =  104.25
```

### Rockchip RK3399 (Cortex-A72 1.8GHz x 2 + Cortex-A53 1.5GHz x 4)
```
nanopc-t4:/data/local/tmp # ./benchncnn 8 2 2 -1 1
loop_count = 8
num_threads = 2
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   43.73  max =   44.30  avg =   43.97
     squeezenet_int8  min =   37.92  max =   38.39  avg =   38.09
           mobilenet  min =   64.28  max =   66.66  avg =   65.14
      mobilenet_int8  min =   43.17  max =   43.73  avg =   43.38
        mobilenet_v2  min =   51.30  max =   52.18  avg =   51.75
        mobilenet_v3  min =   41.51  max =   43.25  avg =   42.10
          shufflenet  min =   27.43  max =   28.27  avg =   27.75
       shufflenet_v2  min =   24.96  max =   25.79  avg =   25.55
             mnasnet  min =   45.44  max =   46.95  avg =   46.16
     proxylessnasnet  min =   51.98  max =   53.52  avg =   52.48
     efficientnet_b0  min =   83.79  max =   84.68  avg =   84.27
   efficientnetv2_b0  min =   97.89  max =   99.27  avg =   98.55
        regnety_400m  min =   65.15  max =   65.89  avg =   65.41
           blazeface  min =    8.74  max =    8.89  avg =    8.80
           googlenet  min =  131.46  max =  140.16  avg =  133.24
      googlenet_int8  min =  115.72  max =  118.34  avg =  116.60
            resnet18  min =  111.77  max =  113.18  avg =  112.37
       resnet18_int8  min =   84.27  max =   84.90  avg =   84.49
             alexnet  min =  105.74  max =  109.87  avg =  107.15
               vgg16  min =  619.88  max =  634.59  avg =  629.15
          vgg16_int8  min =  447.14  max =  451.09  avg =  448.53
            resnet50  min =  291.51  max =  296.55  avg =  293.08
       resnet50_int8  min =  224.09  max =  227.03  avg =  225.02
      squeezenet_ssd  min =  109.72  max =  112.09  avg =  110.78
 squeezenet_ssd_int8  min =   93.41  max =   94.83  avg =   93.97
       mobilenet_ssd  min =  131.30  max =  132.82  avg =  131.94
  mobilenet_ssd_int8  min =   87.52  max =   88.89  avg =   88.35
      mobilenet_yolo  min =  288.02  max =  289.84  avg =  288.61
  mobilenetv2_yolov3  min =  168.45  max =  170.94  avg =  169.79
         yolov4-tiny  min =  217.45  max =  226.39  avg =  219.76
           nanodet_m  min =   65.74  max =   66.84  avg =   66.49
    yolo-fastest-1.1  min =   32.91  max =   33.74  avg =   33.37
      yolo-fastestv2  min =   28.90  max =   37.31  avg =   30.27

nanopc-t4:/data/local/tmp # ./benchncnn 8 1 2 -1 1
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
cooling_down = 1
          squeezenet  min =   71.35  max =   73.02  avg =   71.83
     squeezenet_int8  min =   60.39  max =   60.96  avg =   60.69
           mobilenet  min =  111.12  max =  113.02  avg =  111.99
      mobilenet_int8  min =   80.14  max =   81.59  avg =   81.00
        mobilenet_v2  min =   78.18  max =   80.89  avg =   79.18
        mobilenet_v3  min =   63.49  max =   64.26  avg =   63.90
          shufflenet  min =   38.90  max =   40.28  avg =   39.26
       shufflenet_v2  min =   37.72  max =   38.45  avg =   38.02
             mnasnet  min =   72.34  max =   73.59  avg =   72.87
     proxylessnasnet  min =   87.33  max =   89.70  avg =   88.45
     efficientnet_b0  min =  145.14  max =  146.77  avg =  145.93
   efficientnetv2_b0  min =  169.33  max =  171.16  avg =  170.16
        regnety_400m  min =   99.08  max =   99.80  avg =   99.47
           blazeface  min =   12.28  max =   12.69  avg =   12.48
           googlenet  min =  228.18  max =  229.36  avg =  228.64
      googlenet_int8  min =  201.62  max =  203.71  avg =  202.25
            resnet18  min =  175.71  max =  180.53  avg =  176.85
       resnet18_int8  min =  151.42  max =  152.45  avg =  151.83
             alexnet  min =  160.81  max =  186.24  avg =  165.30
               vgg16  min = 1044.34  max = 1080.88  avg = 1062.34
          vgg16_int8  min =  844.53  max =  851.71  avg =  848.65
            resnet50  min =  503.25  max =  505.20  avg =  504.18
       resnet50_int8  min =  397.71  max =  400.19  avg =  398.63
      squeezenet_ssd  min =  162.98  max =  165.97  avg =  164.34
 squeezenet_ssd_int8  min =  145.93  max =  148.59  avg =  146.94
       mobilenet_ssd  min =  226.54  max =  229.80  avg =  227.80
  mobilenet_ssd_int8  min =  159.97  max =  163.18  avg =  161.06
      mobilenet_yolo  min =  512.90  max =  517.47  avg =  515.06
  mobilenetv2_yolov3  min =  274.88  max =  280.24  avg =  276.36
         yolov4-tiny  min =  351.97  max =  358.70  avg =  355.60
           nanodet_m  min =   95.32  max =   97.83  avg =   96.28
    yolo-fastest-1.1  min =   43.47  max =   46.52  avg =   44.55
      yolo-fastestv2  min =   37.22  max =   37.63  avg =   37.45

nanopc-t4:/data/local/tmp # ./benchncnn 8 4 1 -1 1
loop_count = 8
num_threads = 4
powersave = 1
gpu_device = -1
cooling_down = 1
          squeezenet  min =   48.11  max =   48.51  avg =   48.24
     squeezenet_int8  min =   43.19  max =   44.17  avg =   43.40
           mobilenet  min =   65.47  max =   66.40  avg =   65.68
      mobilenet_int8  min =   49.15  max =   51.65  avg =   49.76
        mobilenet_v2  min =   53.60  max =   54.19  avg =   53.87
        mobilenet_v3  min =   52.83  max =   92.92  avg =   66.25
          shufflenet  min =   35.71  max =   36.03  avg =   35.83
       shufflenet_v2  min =   31.88  max =   32.38  avg =   32.16
             mnasnet  min =   51.59  max =   54.01  avg =   52.30
     proxylessnasnet  min =   60.11  max =   60.40  avg =   60.24
     efficientnet_b0  min =   98.22  max =   99.40  avg =   98.56
   efficientnetv2_b0  min =  114.19  max =  123.90  avg =  115.89
        regnety_400m  min =   85.89  max =   86.20  avg =   86.03
           blazeface  min =   11.23  max =   11.37  avg =   11.31
           googlenet  min =  142.25  max =  160.88  avg =  145.26
      googlenet_int8  min =  125.45  max =  128.50  avg =  125.96
            resnet18  min =  116.68  max =  118.26  avg =  117.00
       resnet18_int8  min =   88.43  max =   90.95  avg =   89.08
             alexnet  min =  150.91  max =  160.01  avg =  152.51
               vgg16  min =  674.91  max =  684.83  avg =  679.08
          vgg16_int8  min =  417.60  max =  422.52  avg =  419.60
            resnet50  min =  297.23  max =  299.37  avg =  298.03
       resnet50_int8  min =  243.99  max =  251.39  avg =  245.99
      squeezenet_ssd  min =  127.92  max =  128.53  avg =  128.17
 squeezenet_ssd_int8  min =  112.54  max =  114.63  avg =  113.19
       mobilenet_ssd  min =  136.43  max =  140.14  avg =  137.33
  mobilenet_ssd_int8  min =  102.14  max =  105.00  avg =  102.77
      mobilenet_yolo  min =  291.45  max =  294.04  avg =  292.63
  mobilenetv2_yolov3  min =  183.13  max =  187.00  avg =  184.05
         yolov4-tiny  min =  257.46  max =  268.76  avg =  260.49
           nanodet_m  min =   83.16  max =   91.03  avg =   84.77
    yolo-fastest-1.1  min =   43.53  max =   43.87  avg =   43.74
      yolo-fastestv2  min =   35.04  max =   35.54  avg =   35.17

nanopc-t4:/data/local/tmp # ./benchncnn 8 1 1 -1 1
loop_count = 8
num_threads = 1
powersave = 1
gpu_device = -1
cooling_down = 1
          squeezenet  min =  129.63  max =  130.58  avg =  129.85
     squeezenet_int8  min =  124.10  max =  126.34  avg =  124.81
           mobilenet  min =  207.92  max =  208.72  avg =  208.41
      mobilenet_int8  min =  175.55  max =  176.11  avg =  175.84
        mobilenet_v2  min =  143.02  max =  143.56  avg =  143.25
        mobilenet_v3  min =  133.11  max =  134.05  avg =  133.33
          shufflenet  min =   77.97  max =   78.54  avg =   78.19
       shufflenet_v2  min =   75.59  max =   76.05  avg =   75.82
             mnasnet  min =  139.86  max =  141.77  avg =  140.19
     proxylessnasnet  min =  178.57  max =  179.57  avg =  179.03
     efficientnet_b0  min =  316.10  max =  317.82  avg =  316.86
   efficientnetv2_b0  min =  359.26  max =  362.03  avg =  360.31
        regnety_400m  min =  182.64  max =  183.03  avg =  182.82
           blazeface  min =   25.81  max =   26.53  avg =   26.20
           googlenet  min =  448.45  max =  450.80  avg =  449.35
      googlenet_int8  min =  406.07  max =  410.65  avg =  408.04
            resnet18  min =  351.64  max =  362.12  avg =  354.19
       resnet18_int8  min =  298.10  max =  300.45  avg =  299.26
             alexnet  min =  586.92  max =  588.73  avg =  587.80
               vgg16  min = 2170.12  max = 2202.80  avg = 2183.32
          vgg16_int8  min = 1533.65  max = 1542.01  avg = 1537.33
            resnet50  min =  975.40  max =  977.79  avg =  976.61
       resnet50_int8  min =  851.59  max =  855.22  avg =  853.75
      squeezenet_ssd  min =  306.35  max =  307.54  avg =  306.96
 squeezenet_ssd_int8  min =  291.32  max =  292.87  avg =  292.18
       mobilenet_ssd  min =  423.70  max =  424.63  avg =  424.11
  mobilenet_ssd_int8  min =  358.62  max =  359.42  avg =  359.04
      mobilenet_yolo  min =  928.06  max =  929.25  avg =  928.55
  mobilenetv2_yolov3  min =  496.96  max =  499.29  avg =  497.73
         yolov4-tiny  min =  712.80  max =  714.15  avg =  713.55
           nanodet_m  min =  179.42  max =  180.60  avg =  179.75
    yolo-fastest-1.1  min =   88.06  max =   88.85  avg =   88.35
      yolo-fastestv2  min =   68.68  max =   69.83  avg =   69.08

nanopc-t4:/data/local/tmp # ./benchncnn 4 1 2 0 0
[0 Mali-T860]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-T860]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=1
[0 Mali-T860]  fp16-p/s/a=1/0/1  int8-p/s/a=1/0/0
[0 Mali-T860]  subgroup=0  basic=0  vote=0  ballot=0  shuffle=0
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = 0
cooling_down = 0
          squeezenet  min =   24.57  max =   24.71  avg =   24.64
           mobilenet  min =   35.86  max =   36.14  avg =   36.04
        mobilenet_v2  min =   30.18  max =   30.19  avg =   30.19
        mobilenet_v3  min =   30.88  max =   31.12  avg =   31.01
          shufflenet  min =   33.90  max =   33.98  avg =   33.93
       shufflenet_v2  min =   29.10  max =   29.14  avg =   29.12
             mnasnet  min =   30.49  max =   30.59  avg =   30.53
     proxylessnasnet  min =   33.56  max =   33.61  avg =   33.59
     efficientnet_b0  min =   51.15  max =   51.54  avg =   51.38
   efficientnetv2_b0  min =   86.26  max =   87.36  avg =   86.91
        regnety_400m  min =   38.44  max =   38.54  avg =   38.49
           blazeface  min =    9.66  max =    9.74  avg =    9.70
           googlenet  min =   80.62  max =   80.96  avg =   80.81
            resnet18  min =   74.07  max =   74.36  avg =   74.23
             alexnet  min =   76.84  max =   77.26  avg =   77.08
               vgg16  min =  300.71  max =  300.89  avg =  300.80
            resnet50  min =  175.96  max =  176.72  avg =  176.23
      squeezenet_ssd  min =   71.20  max =   71.38  avg =   71.32
       mobilenet_ssd  min =   76.99  max =   77.47  avg =   77.19
      mobilenet_yolo  min =  160.41  max =  160.84  avg =  160.62
  mobilenetv2_yolov3  min =   91.31  max =   91.37  avg =   91.35
         yolov4-tiny  min =  130.78  max =  131.54  avg =  131.16
           nanodet_m  min =   55.90  max =   56.03  avg =   55.96
    yolo-fastest-1.1  min =   25.50  max =   25.66  avg =   25.59
      yolo-fastestv2  min =   24.94  max =   25.07  avg =   25.01
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

### EAIDK 310, Rockchip RK3228H (Cortex-A53 1.3GHz x 4) fedora-28 aarch64
```
[openailab@MiWiFi-R1D-srv benchmark]$ ./benchncnn 8 4 0 -1 1
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   68.97  max =   71.42  avg =   69.65
     squeezenet_int8  min =   58.47  max =   59.58  avg =   58.77
           mobilenet  min =   90.87  max =  100.18  avg =   92.48
      mobilenet_int8  min =   59.46  max =   63.02  avg =   60.01
        mobilenet_v2  min =   82.92  max =  112.01  avg =   88.10
        mobilenet_v3  min =   66.65  max =   69.57  avg =   67.27
          shufflenet  min =   48.22  max =   48.49  avg =   48.34
       shufflenet_v2  min =   48.52  max =   52.88  avg =   49.17
             mnasnet  min =   75.63  max =   79.83  avg =   76.43
     proxylessnasnet  min =   84.73  max =   86.69  avg =   85.16
     efficientnet_b0  min =  125.69  max =  129.00  avg =  126.38
   efficientnetv2_b0  min =  144.44  max =  149.01  avg =  145.33
        regnety_400m  min =   99.69  max =  101.23  avg =  100.38
           blazeface  min =   15.84  max =   16.24  avg =   16.03
           googlenet  min =  194.64  max =  199.29  avg =  196.07
      googlenet_int8  min =  158.54  max =  165.64  avg =  160.25
            resnet18  min =  200.65  max =  221.60  avg =  204.30
       resnet18_int8  min =  122.69  max =  126.57  avg =  123.54
             alexnet  min =  175.54  max =  200.91  avg =  181.38
            resnet50  min =  428.75  max =  466.51  avg =  439.67
       resnet50_int8  min =  324.95  max =  347.47  avg =  329.74
      squeezenet_ssd  min =  199.86  max =  207.51  avg =  201.99
 squeezenet_ssd_int8  min =  150.35  max =  176.92  avg =  154.60
       mobilenet_ssd  min =  186.50  max =  189.92  avg =  188.09
  mobilenet_ssd_int8  min =  123.55  max =  127.17  avg =  124.63
      mobilenet_yolo  min =  393.83  max =  414.09  avg =  398.57
  mobilenetv2_yolov3  min =  263.49  max =  273.11  avg =  266.11
         yolov4-tiny  min =  342.33  max =  363.69  avg =  346.34
           nanodet_m  min =  119.66  max =  127.29  avg =  121.26
    yolo-fastest-1.1  min =   61.87  max =   90.26  avg =   65.77
      yolo-fastestv2  min =   48.48  max =   50.82  avg =   48.93

[openailab@MiWiFi-R1D-srv benchmark]$ ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  152.15  max =  152.67  avg =  152.43
     squeezenet_int8  min =  143.22  max =  144.24  avg =  143.61
           mobilenet  min =  237.77  max =  239.69  avg =  238.47
      mobilenet_int8  min =  199.91  max =  201.35  avg =  200.50
        mobilenet_v2  min =  169.67  max =  170.18  avg =  169.93
        mobilenet_v3  min =  150.06  max =  151.17  avg =  150.78
          shufflenet  min =   91.78  max =   92.38  avg =   92.06
       shufflenet_v2  min =  100.86  max =  101.75  avg =  101.50
             mnasnet  min =  165.10  max =  166.74  avg =  166.24
     proxylessnasnet  min =  218.42  max =  220.55  avg =  219.12
     efficientnet_b0  min =  348.00  max =  349.03  avg =  348.49
   efficientnetv2_b0  min =  404.06  max =  406.16  avg =  405.00
        regnety_400m  min =  209.48  max =  211.36  avg =  210.44
           blazeface  min =   31.31  max =   32.61  avg =   32.00
           googlenet  min =  510.38  max =  512.43  avg =  511.25
      googlenet_int8  min =  454.38  max =  456.19  avg =  455.02
            resnet18  min =  407.78  max =  409.45  avg =  408.34
       resnet18_int8  min =  357.01  max =  360.72  avg =  358.74
             alexnet  min =  504.12  max =  506.74  avg =  505.08
            resnet50  min = 1115.42  max = 1121.91  avg = 1118.67
       resnet50_int8  min =  973.38  max =  976.26  avg =  975.21
      squeezenet_ssd  min =  361.52  max =  363.69  avg =  362.38
 squeezenet_ssd_int8  min =  333.81  max =  337.16  avg =  335.24
       mobilenet_ssd  min =  477.43  max =  478.36  avg =  477.82
  mobilenet_ssd_int8  min =  409.33  max =  409.67  avg =  409.52
      mobilenet_yolo  min = 1048.79  max = 1057.72  avg = 1053.80
  mobilenetv2_yolov3  min =  567.04  max =  571.44  avg =  569.04
         yolov4-tiny  min =  788.40  max =  790.74  avg =  789.12
           nanodet_m  min =  253.68  max =  254.59  avg =  254.16
    yolo-fastest-1.1  min =  102.44  max =  103.11  avg =  102.67
      yolo-fastestv2  min =   82.19  max =   82.43  avg =   82.35
```

### NVIDIA Jetson Nano
```
[0 NVIDIA Tegra X1 (nvgpu)]  queueC=0[16]  queueG=0[16]  queueT=0[16]
[0 NVIDIA Tegra X1 (nvgpu)]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 NVIDIA Tegra X1 (nvgpu)]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 NVIDIA Tegra X1 (nvgpu)]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = 0
cooling_down = 1
          squeezenet  min =   12.15  max =   26.48  avg =   18.11
     squeezenet_int8  min =   27.60  max =   42.50  avg =   29.89
           mobilenet  min =   16.07  max =   16.10  avg =   16.09
      mobilenet_int8  min =   30.65  max =   32.15  avg =   31.07
        mobilenet_v2  min =   12.87  max =   13.15  avg =   12.99
        mobilenet_v3  min =   13.32  max =   16.65  avg =   14.57
          shufflenet  min =   14.21  max =   14.34  avg =   14.29
       shufflenet_v2  min =   13.03  max =   21.97  avg =   19.02
             mnasnet  min =   13.33  max =   13.64  avg =   13.49
     proxylessnasnet  min =   14.65  max =   14.91  avg =   14.76
     efficientnet_b0  min =   21.26  max =   21.41  avg =   21.35
   efficientnetv2_b0  min =   54.66  max =   60.81  avg =   57.16
        regnety_400m  min =   17.91  max =   18.08  avg =   18.01
           blazeface  min =    6.87  max =    7.03  avg =    6.94
           googlenet  min =   43.30  max =   43.54  avg =   43.43
      googlenet_int8  min =   80.07  max =   84.28  avg =   81.10
            resnet18  min =   43.89  max =   44.06  avg =   43.98
       resnet18_int8  min =   60.70  max =   63.43  avg =   61.60
             alexnet  min =   74.21  max =   75.20  avg =   74.45
               vgg16  min =  310.39  max =  310.65  avg =  310.52
          vgg16_int8  min =  293.15  max =  297.28  avg =  294.93
            resnet50  min =   93.03  max =   93.22  avg =   93.12
       resnet50_int8  min =  158.54  max =  161.25  avg =  159.56
      squeezenet_ssd  min =   55.88  max =   57.43  avg =   56.46
 squeezenet_ssd_int8  min =   72.42  max =   73.25  avg =   72.73
       mobilenet_ssd  min =   35.38  max =   37.57  avg =   36.63
  mobilenet_ssd_int8  min =   62.92  max =   64.97  avg =   63.63
      mobilenet_yolo  min =   76.56  max =   80.44  avg =   78.05
  mobilenetv2_yolov3  min =   46.35  max =   48.14  avg =   47.26
         yolov4-tiny  min =   95.38  max =   97.55  avg =   96.45
           nanodet_m  min =   22.82  max =   26.01  avg =   24.48
    yolo-fastest-1.1  min =   20.23  max =   25.51  avg =   21.52
      yolo-fastestv2  min =   20.67  max =   20.82  avg =   20.75

[0 NVIDIA Tegra X1 (nvgpu)]  queueC=0[16]  queueG=0[16]  queueT=0[16]
[0 NVIDIA Tegra X1 (nvgpu)]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 NVIDIA Tegra X1 (nvgpu)]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 NVIDIA Tegra X1 (nvgpu)]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 8
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 1
          squeezenet  min =   12.00  max =   15.41  avg =   13.55
     squeezenet_int8  min =   78.76  max =   79.14  avg =   78.91
           mobilenet  min =   16.03  max =   16.25  avg =   16.15
      mobilenet_int8  min =  107.58  max =  107.68  avg =  107.61
        mobilenet_v2  min =   12.84  max =   13.13  avg =   12.99
        mobilenet_v3  min =   13.29  max =   16.64  avg =   14.38
          shufflenet  min =   14.23  max =   14.54  avg =   14.34
       shufflenet_v2  min =   12.94  max =   13.21  avg =   13.02
             mnasnet  min =   13.42  max =   13.66  avg =   13.53
     proxylessnasnet  min =   14.64  max =   14.94  avg =   14.76
     efficientnet_b0  min =   21.28  max =   21.51  avg =   21.36
   efficientnetv2_b0  min =   74.32  max =   78.50  avg =   77.79
        regnety_400m  min =   17.94  max =   18.26  avg =   18.07
           blazeface  min =    6.83  max =    6.94  avg =    6.89
           googlenet  min =   43.45  max =   43.63  avg =   43.52
      googlenet_int8  min =  255.68  max =  256.33  avg =  255.92
            resnet18  min =   43.96  max =   44.06  avg =   44.01
       resnet18_int8  min =  192.01  max =  192.64  avg =  192.33
             alexnet  min =   74.04  max =   74.23  avg =   74.14
               vgg16  min =  310.32  max =  310.64  avg =  310.44
          vgg16_int8  min = 1003.05  max = 1004.27  avg = 1003.66
            resnet50  min =   93.05  max =   93.34  avg =   93.21
       resnet50_int8  min =  516.27  max =  517.12  avg =  516.69
      squeezenet_ssd  min =   56.67  max =   56.86  avg =   56.73
 squeezenet_ssd_int8  min =  182.96  max =  184.26  avg =  183.71
       mobilenet_ssd  min =   35.61  max =   35.70  avg =   35.65
  mobilenet_ssd_int8  min =  217.02  max =  217.50  avg =  217.23
      mobilenet_yolo  min =   78.10  max =   78.36  avg =   78.20
  mobilenetv2_yolov3  min =   49.86  max =   57.83  avg =   53.18
         yolov4-tiny  min =   96.76  max =   96.86  avg =   96.82
           nanodet_m  min =   25.26  max =   25.36  avg =   25.31
    yolo-fastest-1.1  min =   21.55  max =   24.22  avg =   23.78
      yolo-fastestv2  min =   20.80  max =   21.01  avg =   20.90

loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   30.03  max =   31.41  avg =   30.59
     squeezenet_int8  min =   27.32  max =   27.76  avg =   27.50
           mobilenet  min =   41.74  max =   42.57  avg =   42.05
      mobilenet_int8  min =   30.48  max =   31.57  avg =   30.85
        mobilenet_v2  min =   33.49  max =   34.18  avg =   33.83
        mobilenet_v3  min =   30.59  max =   30.96  avg =   30.79
          shufflenet  min =   21.07  max =   31.68  avg =   22.53
       shufflenet_v2  min =   19.55  max =   20.01  avg =   19.71
             mnasnet  min =   31.70  max =   32.26  avg =   31.93
     proxylessnasnet  min =   36.90  max =   38.55  avg =   37.27
     efficientnet_b0  min =   68.42  max =   77.60  avg =   70.60
   efficientnetv2_b0  min =   73.72  max =   81.05  avg =   75.31
        regnety_400m  min =   56.67  max =   66.82  avg =   58.24
           blazeface  min =    6.55  max =    6.96  avg =    6.74
           googlenet  min =   92.74  max =   94.22  avg =   93.12
      googlenet_int8  min =   80.86  max =   87.28  avg =   82.41
            resnet18  min =   83.10  max =   84.30  avg =   83.44
       resnet18_int8  min =   59.40  max =   65.86  avg =   60.70
             alexnet  min =   89.21  max =   92.45  avg =   89.98
               vgg16  min =  445.72  max =  451.09  avg =  447.39
          vgg16_int8  min =  292.81  max =  295.55  avg =  294.34
            resnet50  min =  203.42  max =  204.45  avg =  204.08
       resnet50_int8  min =  157.87  max =  160.30  avg =  158.67
      squeezenet_ssd  min =   85.60  max =   87.24  avg =   86.18
 squeezenet_ssd_int8  min =   73.10  max =   85.64  avg =   74.94
       mobilenet_ssd  min =   86.75  max =   96.51  avg =   88.49
  mobilenet_ssd_int8  min =   63.40  max =   71.57  avg =   64.97
      mobilenet_yolo  min =  193.84  max =  195.24  avg =  194.62
  mobilenetv2_yolov3  min =  115.80  max =  117.27  avg =  116.27
         yolov4-tiny  min =  156.30  max =  158.26  avg =  156.81
           nanodet_m  min =   46.64  max =   47.97  avg =   47.12
    yolo-fastest-1.1  min =   25.78  max =   27.86  avg =   26.29
      yolo-fastestv2  min =   20.54  max =   30.73  avg =   22.18

loop_count = 8
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   85.91  max =   86.86  avg =   86.14
     squeezenet_int8  min =   77.57  max =   78.10  avg =   77.69
           mobilenet  min =  137.43  max =  138.03  avg =  137.63
      mobilenet_int8  min =  108.06  max =  108.21  avg =  108.13
        mobilenet_v2  min =   93.81  max =   94.70  avg =   93.99
        mobilenet_v3  min =   81.77  max =   82.49  avg =   81.99
          shufflenet  min =   47.84  max =   48.46  avg =   48.17
       shufflenet_v2  min =   47.93  max =   48.23  avg =   48.09
             mnasnet  min =   91.73  max =   92.55  avg =   91.98
     proxylessnasnet  min =  115.41  max =  115.75  avg =  115.56
     efficientnet_b0  min =  225.64  max =  226.21  avg =  225.94
   efficientnetv2_b0  min =  239.71  max =  240.20  avg =  239.89
        regnety_400m  min =  118.46  max =  118.84  avg =  118.61
           blazeface  min =   15.58  max =   17.14  avg =   16.21
           googlenet  min =  286.85  max =  287.51  avg =  287.11
      googlenet_int8  min =  256.44  max =  256.74  avg =  256.53
            resnet18  min =  221.27  max =  221.93  avg =  221.60
       resnet18_int8  min =  189.95  max =  191.34  avg =  190.74
             alexnet  min =  284.30  max =  285.40  avg =  284.87
               vgg16  min = 1241.51  max = 1244.53  avg = 1242.90
          vgg16_int8  min = 1003.92  max = 1004.47  avg = 1004.29
            resnet50  min =  624.43  max =  625.34  avg =  624.84
       resnet50_int8  min =  516.64  max =  517.26  avg =  516.99
      squeezenet_ssd  min =  190.21  max =  191.35  avg =  190.71
 squeezenet_ssd_int8  min =  182.97  max =  184.19  avg =  183.38
       mobilenet_ssd  min =  275.60  max =  276.17  avg =  275.90
  mobilenet_ssd_int8  min =  216.67  max =  217.58  avg =  216.94
      mobilenet_yolo  min =  616.16  max =  617.45  avg =  616.71
  mobilenetv2_yolov3  min =  324.88  max =  325.73  avg =  325.19
         yolov4-tiny  min =  421.01  max =  423.52  avg =  422.14
           nanodet_m  min =  117.39  max =  117.75  avg =  117.54
    yolo-fastest-1.1  min =   54.55  max =   55.61  avg =   54.87
      yolo-fastestv2  min =   44.40  max =   44.78  avg =   44.57
```

### Rockchip RK3288-CG.W (Cortex-A17 1.8GHz x 4)
```
WW_Tinker_Board:/data/local/tmp # ./benchncnn 8 4 0 -1 1
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   56.61  max =   56.80  avg =   56.69
     squeezenet_int8  min =   40.63  max =   41.05  avg =   40.89
           mobilenet  min =   83.91  max =   84.59  avg =   84.23
      mobilenet_int8  min =   36.15  max =   36.44  avg =   36.25
        mobilenet_v2  min =   71.12  max =   71.73  avg =   71.54
        mobilenet_v3  min =   56.08  max =   56.56  avg =   56.28
          shufflenet  min =   37.39  max =   37.75  avg =   37.55
       shufflenet_v2  min =   35.19  max =   35.52  avg =   35.34
             mnasnet  min =   62.08  max =   62.36  avg =   62.24
     proxylessnasnet  min =   66.98  max =   67.38  avg =   67.16
     efficientnet_b0  min =  109.95  max =  110.71  avg =  110.15
   efficientnetv2_b0  min =  122.56  max =  123.31  avg =  122.94
        regnety_400m  min =   88.84  max =   89.19  avg =   88.99
           blazeface  min =   11.79  max =   11.92  avg =   11.85
           googlenet  min =  162.56  max =  165.39  avg =  163.19
      googlenet_int8  min =  110.35  max =  110.91  avg =  110.60
            resnet18  min =  172.39  max =  173.99  avg =  173.24
       resnet18_int8  min =   84.00  max =   84.40  avg =   84.19
             alexnet  min =  156.71  max =  158.23  avg =  157.59
               vgg16  min =  956.95  max =  964.32  avg =  960.60
          vgg16_int8  min =  388.10  max =  389.52  avg =  388.68
            resnet50  min =  403.05  max =  404.80  avg =  404.01
       resnet50_int8  min =  205.12  max =  207.42  avg =  206.19
      squeezenet_ssd  min =  163.61  max =  165.79  avg =  164.93
 squeezenet_ssd_int8  min =  125.88  max =  126.35  avg =  126.12
       mobilenet_ssd  min =  175.97  max =  176.86  avg =  176.39
  mobilenet_ssd_int8  min =   76.90  max =   77.74  avg =   77.35
      mobilenet_yolo  min =  385.59  max =  387.19  avg =  386.60
  mobilenetv2_yolov3  min =  234.88  max =  236.22  avg =  235.66
         yolov4-tiny  min =  307.44  max =  310.64  avg =  308.54
           nanodet_m  min =   92.54  max =   93.15  avg =   92.82
    yolo-fastest-1.1  min =   46.69  max =   47.02  avg =   46.83
      yolo-fastestv2  min =   38.37  max =   38.68  avg =   38.54

WW_Tinker_Board:/data/local/tmp # ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  138.27  max =  138.57  avg =  138.41
     squeezenet_int8  min =   85.97  max =   86.23  avg =   86.05
           mobilenet  min =  234.90  max =  235.08  avg =  235.00
      mobilenet_int8  min =   99.92  max =  100.45  avg =  100.12
        mobilenet_v2  min =  157.76  max =  157.99  avg =  157.86
        mobilenet_v3  min =  130.05  max =  130.23  avg =  130.17
          shufflenet  min =   74.48  max =   74.62  avg =   74.55
       shufflenet_v2  min =   74.05  max =   74.25  avg =   74.13
             mnasnet  min =  150.74  max =  151.03  avg =  150.87
     proxylessnasnet  min =  171.09  max =  171.23  avg =  171.16
     efficientnet_b0  min =  306.85  max =  307.02  avg =  306.97
   efficientnetv2_b0  min =  347.40  max =  347.87  avg =  347.64
        regnety_400m  min =  190.26  max =  190.33  avg =  190.29
           blazeface  min =   25.25  max =   25.68  avg =   25.47
           googlenet  min =  432.09  max =  432.48  avg =  432.32
      googlenet_int8  min =  275.55  max =  276.07  avg =  275.88
            resnet18  min =  355.11  max =  358.56  avg =  356.90
       resnet18_int8  min =  205.80  max =  206.68  avg =  206.26
             alexnet  min =  330.09  max =  330.29  avg =  330.15
               vgg16  min = 2122.95  max = 2124.45  avg = 2123.68
          vgg16_int8  min = 1048.53  max = 1049.29  avg = 1048.86
            resnet50  min = 1047.27  max = 1048.33  avg = 1047.63
       resnet50_int8  min =  517.75  max =  519.28  avg =  518.81
      squeezenet_ssd  min =  304.69  max =  305.75  avg =  305.16
 squeezenet_ssd_int8  min =  219.16  max =  219.94  avg =  219.45
       mobilenet_ssd  min =  483.73  max =  484.12  avg =  484.01
  mobilenet_ssd_int8  min =  208.89  max =  209.19  avg =  209.09
      mobilenet_yolo  min = 1092.75  max = 1093.70  avg = 1093.13
  mobilenetv2_yolov3  min =  560.66  max =  560.92  avg =  560.77
         yolov4-tiny  min =  704.69  max =  705.38  avg =  705.12
           nanodet_m  min =  187.13  max =  187.57  avg =  187.39
    yolo-fastest-1.1  min =   83.05  max =   83.11  avg =   83.08
      yolo-fastestv2  min =   72.19  max =   72.23  avg =   72.21

WW_Tinker_Board:/data/local/tmp # ./benchncnn 4 1 0 0 0
[0 Mali-T760]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-T760]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=1
[0 Mali-T760]  fp16-p/s/a=1/0/1  int8-p/s/a=1/0/0
[0 Mali-T760]  subgroup=0  basic=0  vote=0  ballot=0  shuffle=0
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =   41.78  max =   41.82  avg =   41.79
           mobilenet  min =   62.67  max =   62.80  avg =   62.74
        mobilenet_v2  min =   51.08  max =   51.26  avg =   51.17
        mobilenet_v3  min =   51.43  max =   51.70  avg =   51.51
          shufflenet  min =   56.83  max =   56.94  avg =   56.87
       shufflenet_v2  min =   48.46  max =   48.63  avg =   48.53
             mnasnet  min =   52.31  max =   52.63  avg =   52.42
     proxylessnasnet  min =   57.33  max =   57.46  avg =   57.41
     efficientnet_b0  min =   87.52  max =   87.80  avg =   87.62
   efficientnetv2_b0  min =  123.83  max =  124.67  avg =  124.34
        regnety_400m  min =   65.52  max =   65.81  avg =   65.64
           blazeface  min =   14.56  max =   14.73  avg =   14.62
           googlenet  min =  138.52  max =  139.39  avg =  138.89
            resnet18  min =  124.45  max =  124.81  avg =  124.58
             alexnet  min =  130.46  max =  130.68  avg =  130.54
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
          squeezenet  min =  220.10  max =  226.46  avg =  222.89
     squeezenet_int8  min =  159.26  max =  165.25  avg =  161.71
           mobilenet  min =  366.92  max =  373.78  avg =  371.55
      mobilenet_int8  min =  223.14  max =  229.66  avg =  225.66
        mobilenet_v2  min =  252.32  max =  259.41  avg =  255.54
        mobilenet_v3  min =  214.05  max =  222.24  avg =  217.53
          shufflenet  min =  137.02  max =  144.79  avg =  138.85
       shufflenet_v2  min =  134.89  max =  140.75  avg =  137.18
             mnasnet  min =  250.64  max =  256.75  avg =  253.33
     proxylessnasnet  min =  285.35  max =  291.43  avg =  288.37
     efficientnet_b0  min =  430.47  max =  436.63  avg =  434.75
        regnety_400m  min =  317.69  max =  325.77  avg =  321.24
           blazeface  min =   42.93  max =   43.30  avg =   43.14
           googlenet  min =  721.84  max =  728.40  avg =  724.23
      googlenet_int8  min =  504.07  max =  511.06  avg =  507.39
            resnet18  min =  645.61  max =  653.08  avg =  648.51
       resnet18_int8  min =  370.84  max =  514.38  avg =  392.80
             alexnet  min =  783.64  max =  794.83  avg =  786.95
      squeezenet_ssd  min =  508.71  max =  513.70  avg =  511.29
 squeezenet_ssd_int8  min =  402.85  max =  409.32  avg =  406.45
       mobilenet_ssd  min =  763.70  max =  771.52  avg =  767.61
  mobilenet_ssd_int8  min =  457.99  max =  460.85  avg =  459.76
      mobilenet_yolo  min = 1730.90  max = 1746.52  avg = 1741.26
  mobilenetv2_yolov3  min =  884.00  max =  892.97  avg =  889.38
         yolov4-tiny  min = 1181.20  max = 1218.20  avg = 1202.28
           nanodet_m  min =  331.53  max =  339.89  avg =  334.62

imx7d_pico:/data/local/tmp $ ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  408.39  max =  410.27  avg =  408.95
     squeezenet_int8  min =  290.25  max =  290.95  avg =  290.61
           mobilenet  min =  707.10  max =  711.64  avg =  708.47
      mobilenet_int8  min =  434.95  max =  436.16  avg =  435.66
        mobilenet_v2  min =  466.52  max =  467.41  avg =  466.96
        mobilenet_v3  min =  407.03  max =  408.29  avg =  407.56
          shufflenet  min =  240.65  max =  241.07  avg =  240.85
       shufflenet_v2  min =  229.27  max =  235.66  avg =  231.51
             mnasnet  min =  471.21  max =  471.48  avg =  471.35
     proxylessnasnet  min =  544.74  max =  547.62  avg =  546.20
     efficientnet_b0  min =  824.09  max =  824.44  avg =  824.20
        regnety_400m  min =  570.20  max =  571.73  avg =  570.82
           blazeface  min =   76.46  max =   77.05  avg =   76.81
           googlenet  min = 1368.82  max = 1369.99  avg = 1369.33
      googlenet_int8  min =  945.51  max =  946.61  avg =  945.91
            resnet18  min = 1237.79  max = 1257.12  avg = 1246.80
       resnet18_int8  min =  705.09  max =  706.72  avg =  705.63
             alexnet  min = 1516.35  max = 1522.82  avg = 1519.52
      squeezenet_ssd  min =  906.97  max =  908.48  avg =  907.68
 squeezenet_ssd_int8  min =  727.15  max =  728.16  avg =  727.77
       mobilenet_ssd  min = 1475.19  max = 1478.52  avg = 1476.81
  mobilenet_ssd_int8  min =  883.88  max =  890.68  avg =  885.90
      mobilenet_yolo  min = 3408.43  max = 3418.63  avg = 3412.52
  mobilenetv2_yolov3  min = 1685.18  max = 1695.89  avg = 1689.23
         yolov4-tiny  min = 2168.24  max = 2183.24  avg = 2175.93
           nanodet_m  min =  561.56  max =  562.05  avg =  561.72
```

### Z7-Lite 7020 XC7Z020CLG400-2 (Cortex-A9 766MHz x 2)
```
root@petalinux_hdmi:~# LD_LIBRARY_PATH=. ./benchncnn 8 2 0 -1 1
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  389.18  max =  390.13  avg =  389.60
     squeezenet_int8  min =  254.33  max =  255.24  avg =  254.85
           mobilenet  min =  623.71  max =  625.01  avg =  624.46
      mobilenet_int8  min =  240.40  max =  241.03  avg =  240.87
        mobilenet_v2  min =  450.00  max =  450.89  avg =  450.40
        mobilenet_v3  min =  362.99  max =  363.66  avg =  363.28
          shufflenet  min =  212.20  max =  213.28  avg =  212.84
       shufflenet_v2  min =  210.26  max =  212.64  avg =  211.53
             mnasnet  min =  408.67  max =  409.64  avg =  409.17
     proxylessnasnet  min =  449.86  max =  450.94  avg =  450.45
     efficientnet_b0  min =  737.40  max =  739.58  avg =  738.32
   efficientnetv2_b0  min =  848.58  max =  849.74  avg =  849.24
        regnety_400m  min =  501.32  max =  503.02  avg =  501.87
           blazeface  min =   70.89  max =   72.22  avg =   71.61
      squeezenet_ssd  min =  978.55  max =  979.86  avg =  979.22
 squeezenet_ssd_int8  min =  691.90  max =  694.18  avg =  692.73
       mobilenet_ssd  min = 1353.12  max = 1354.13  avg = 1353.53
  mobilenet_ssd_int8  min =  496.26  max =  497.29  avg =  496.61
           nanodet_m  min =  542.04  max =  546.29  avg =  544.73
    yolo-fastest-1.1  min =  282.75  max =  286.11  avg =  284.24
      yolo-fastestv2  min =  230.91  max =  232.74  avg =  231.56

root@petalinux_hdmi:~# LD_LIBRARY_PATH=. ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  637.19  max =  639.33  avg =  637.82
     squeezenet_int8  min =  390.31  max =  391.63  avg =  390.94
           mobilenet  min = 1085.54  max = 1085.96  avg = 1085.71
      mobilenet_int8  min =  437.28  max =  437.65  avg =  437.44
        mobilenet_v2  min =  716.03  max =  716.75  avg =  716.35
        mobilenet_v3  min =  587.83  max =  588.55  avg =  588.21
          shufflenet  min =  331.28  max =  331.97  avg =  331.63
       shufflenet_v2  min =  331.03  max =  333.19  avg =  331.76
             mnasnet  min =  682.68  max =  683.11  avg =  682.82
     proxylessnasnet  min =  763.89  max =  764.80  avg =  764.35
     efficientnet_b0  min = 1288.61  max = 1289.10  avg = 1288.81
   efficientnetv2_b0  min = 1499.12  max = 1500.11  avg = 1499.65
        regnety_400m  min =  852.03  max =  853.16  avg =  852.68
           blazeface  min =  109.40  max =  111.51  avg =  110.41
      squeezenet_ssd  min = 1493.25  max = 1497.00  avg = 1494.87
 squeezenet_ssd_int8  min = 1016.77  max = 1019.31  avg = 1017.99
       mobilenet_ssd  min = 2379.20  max = 2379.83  avg = 2379.64
  mobilenet_ssd_int8  min =  881.70  max =  881.89  avg =  881.83
           nanodet_m  min =  831.13  max =  832.58  avg =  831.87
    yolo-fastest-1.1  min =  466.80  max =  469.90  avg =  468.79
      yolo-fastestv2  min =  352.07  max =  355.20  avg =  353.36
```

### Loongson 2K1000 (GS264 1.0GHz x 2)
```
root@ls2k:~/ncnn/build/benchmark# ./benchncnn 10 2 0 -1 1
loop_count = 10
num_threads = 2
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  184.33  max =  184.94  avg =  184.65
     squeezenet_int8  min =  201.42  max =  201.99  avg =  201.72
           mobilenet  min =  277.17  max =  278.04  avg =  277.66
      mobilenet_int8  min =  234.61  max =  235.17  avg =  234.81
        mobilenet_v2  min =  223.10  max =  274.92  avg =  228.71
        mobilenet_v3  min =  185.79  max =  201.76  avg =  187.60
          shufflenet  min =  129.78  max =  131.09  avg =  130.28
       shufflenet_v2  min =  115.86  max =  116.77  avg =  116.42
             mnasnet  min =  213.92  max =  214.72  avg =  214.26
     proxylessnasnet  min =  240.05  max =  242.02  avg =  240.86
     efficientnet_b0  min =  347.52  max =  348.53  avg =  348.13
   efficientnetv2_b0  min =  382.78  max =  479.58  avg =  398.18
        regnety_400m  min =  270.00  max =  312.84  avg =  274.66
           blazeface  min =   37.60  max =   38.02  avg =   37.79
           googlenet  min =  659.55  max =  693.17  avg =  666.17
      googlenet_int8  min =  678.26  max =  718.39  avg =  682.79
            resnet18  min =  499.75  max =  766.88  avg =  532.49
       resnet18_int8  min =  500.38  max =  533.97  avg =  504.56
             alexnet  min =  508.49  max =  542.94  avg =  516.13
               vgg16  min = 2654.06  max = 3082.44  avg = 2762.51
          vgg16_int8  min = 2628.96  max = 2665.35  avg = 2647.12
            resnet50  min = 1256.97  max = 1417.45  avg = 1283.04
       resnet50_int8  min = 1232.55  max = 1276.94  avg = 1244.59
      squeezenet_ssd  min =  538.83  max =  588.03  avg =  553.44
 squeezenet_ssd_int8  min =  501.67  max =  532.61  avg =  505.72
       mobilenet_ssd  min =  571.14  max =  600.93  avg =  578.22
  mobilenet_ssd_int8  min =  478.67  max =  515.39  avg =  483.06
      mobilenet_yolo  min = 1644.48  max = 1729.17  avg = 1669.18
  mobilenetv2_yolov3  min =  752.22  max =  792.40  avg =  760.10
         yolov4-tiny  min =  994.48  max = 1096.10  avg = 1016.49
           nanodet_m  min =  299.12  max =  343.99  avg =  303.98
    yolo-fastest-1.1  min =  141.56  max =  142.93  avg =  142.04
      yolo-fastestv2  min =  125.66  max =  168.88  avg =  130.28

root@ls2k:~/ncnn/build/benchmark# ./benchncnn 4 1 0 -1 1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  295.48  max =  296.42  avg =  295.98
     squeezenet_int8  min =  334.05  max =  336.31  avg =  335.35
           mobilenet  min =  476.33  max =  479.00  avg =  477.41
      mobilenet_int8  min =  446.03  max =  448.21  avg =  446.73
        mobilenet_v2  min =  343.26  max =  343.97  avg =  343.69
        mobilenet_v3  min =  296.84  max =  297.31  avg =  297.11
          shufflenet  min =  202.31  max =  203.96  avg =  202.79
       shufflenet_v2  min =  181.69  max =  182.42  avg =  182.08
             mnasnet  min =  353.73  max =  354.12  avg =  353.99
     proxylessnasnet  min =  404.49  max =  405.00  avg =  404.75
     efficientnet_b0  min =  592.54  max =  593.81  avg =  593.14
   efficientnetv2_b0  min =  649.91  max =  651.49  avg =  650.54
        regnety_400m  min =  425.96  max =  426.33  avg =  426.12
           blazeface  min =   59.74  max =   60.19  avg =   59.90
           googlenet  min = 1120.13  max = 1217.54  avg = 1146.27
      googlenet_int8  min = 1205.17  max = 1213.43  avg = 1208.13
            resnet18  min =  803.07  max =  997.37  avg =  856.09
       resnet18_int8  min =  911.74  max =  916.16  avg =  913.31
             alexnet  min =  883.47  max =  903.08  avg =  889.06
               vgg16  min = 4425.52  max = 4587.36  avg = 4467.61
          vgg16_int8  min = 4896.90  max = 4993.15  avg = 4924.44
            resnet50  min = 2163.22  max = 2169.90  avg = 2167.49
       resnet50_int8  min = 2202.87  max = 2218.00  avg = 2210.51
      squeezenet_ssd  min =  831.06  max =  926.94  avg =  856.24
 squeezenet_ssd_int8  min =  800.52  max =  803.28  avg =  801.72
       mobilenet_ssd  min =  979.74  max =  980.82  avg =  980.22
  mobilenet_ssd_int8  min =  893.79  max =  895.41  avg =  894.51
      mobilenet_yolo  min = 2578.17  max = 2586.30  avg = 2582.55
  mobilenetv2_yolov3  min = 1190.77  max = 1207.67  avg = 1196.06
         yolov4-tiny  min = 1558.29  max = 1570.18  avg = 1561.52
           nanodet_m  min =  442.90  max =  444.27  avg =  443.72
    yolo-fastest-1.1  min =  203.60  max =  208.43  avg =  205.20
      yolo-fastestv2  min =  184.61  max =  185.05  avg =  184.75
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

### Intel Atom x5-Z8350
```
nihui@nihui-ROCK-Pi-X:~/ncnn/build/benchmark$ ./benchncnn 20 4 0 -1 1
loop_count = 20
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   50.22  max =   50.53  avg =   50.32
     squeezenet_int8  min =   77.92  max =   78.37  avg =   78.07
           mobilenet  min =   80.12  max =   81.53  avg =   80.35
      mobilenet_int8  min =  120.54  max =  124.10  avg =  120.84
        mobilenet_v2  min =   56.62  max =   60.12  avg =   58.37
        mobilenet_v3  min =   50.19  max =   50.41  avg =   50.27
          shufflenet  min =   37.96  max =   38.28  avg =   38.10
       shufflenet_v2  min =   35.28  max =   35.59  avg =   35.45
             mnasnet  min =   54.91  max =   55.10  avg =   55.01
     proxylessnasnet  min =   62.25  max =   62.59  avg =   62.40
     efficientnet_b0  min =  101.92  max =  105.73  avg =  102.27
   efficientnetv2_b0  min =  115.48  max =  117.25  avg =  115.89
        regnety_400m  min =   79.66  max =   81.70  avg =   79.95
           blazeface  min =   10.43  max =   10.60  avg =   10.49
           googlenet  min =  170.41  max =  173.44  avg =  170.68
      googlenet_int8  min =  253.06  max =  257.34  avg =  253.57
            resnet18  min =  127.19  max =  130.69  avg =  127.65
       resnet18_int8  min =  200.54  max =  204.25  avg =  200.88
             alexnet  min =  104.89  max =  110.89  avg =  105.56
               vgg16  min =  653.78  max =  661.34  avg =  655.44
          vgg16_int8  min =  974.72  max = 1006.48  avg =  978.76
            resnet50  min =  367.63  max =  371.74  avg =  368.27
       resnet50_int8  min =  574.94  max =  584.08  avg =  576.18
      squeezenet_ssd  min =  115.35  max =  116.47  avg =  115.62
 squeezenet_ssd_int8  min =  169.95  max =  170.75  avg =  170.26
       mobilenet_ssd  min =  167.00  max =  172.02  avg =  168.95
  mobilenet_ssd_int8  min =  244.91  max =  248.30  avg =  245.27
      mobilenet_yolo  min =  382.80  max =  393.23  avg =  385.79
  mobilenetv2_yolov3  min =  208.23  max =  211.54  avg =  209.64
         yolov4-tiny  min =  251.10  max =  263.77  avg =  256.37
           nanodet_m  min =   84.48  max =   84.95  avg =   84.70
    yolo-fastest-1.1  min =   44.11  max =   45.15  avg =   44.26
      yolo-fastestv2  min =   37.95  max =   38.52  avg =   38.34

nihui@nihui-ROCK-Pi-X:~/ncnn/build/benchmark$ ./benchncnn 10 1 0 -1 1
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =  130.52  max =  131.08  avg =  130.64
     squeezenet_int8  min =  231.03  max =  231.38  avg =  231.19
           mobilenet  min =  231.40  max =  231.74  avg =  231.61
      mobilenet_int8  min =  409.74  max =  410.02  avg =  409.85
        mobilenet_v2  min =  150.23  max =  150.72  avg =  150.47
        mobilenet_v3  min =  119.08  max =  119.34  avg =  119.20
          shufflenet  min =   72.62  max =   72.81  avg =   72.73
       shufflenet_v2  min =   73.63  max =   73.71  avg =   73.68
             mnasnet  min =  140.87  max =  141.09  avg =  140.98
     proxylessnasnet  min =  166.39  max =  166.75  avg =  166.54
     efficientnet_b0  min =  280.55  max =  281.30  avg =  280.77
   efficientnetv2_b0  min =  321.05  max =  321.24  avg =  321.16
        regnety_400m  min =  183.78  max =  184.64  avg =  183.91
           blazeface  min =   18.94  max =   19.08  avg =   19.01
           googlenet  min =  453.56  max =  454.71  avg =  454.15
      googlenet_int8  min =  791.40  max =  791.93  avg =  791.61
            resnet18  min =  365.87  max =  366.40  avg =  366.15
       resnet18_int8  min =  652.86  max =  653.39  avg =  653.09
             alexnet  min =  289.15  max =  290.25  avg =  289.65
               vgg16  min = 1887.16  max = 1887.73  avg = 1887.41
          vgg16_int8  min = 3211.44  max = 3213.39  avg = 3212.55
            resnet50  min = 1060.37  max = 1061.40  avg = 1060.80
       resnet50_int8  min = 1869.41  max = 1870.59  avg = 1870.17
      squeezenet_ssd  min =  277.23  max =  277.83  avg =  277.50
 squeezenet_ssd_int8  min =  455.54  max =  458.06  avg =  456.28
       mobilenet_ssd  min =  478.03  max =  478.83  avg =  478.32
  mobilenet_ssd_int8  min =  822.61  max =  822.96  avg =  822.79
      mobilenet_yolo  min = 1136.89  max = 1138.51  avg = 1137.74
  mobilenetv2_yolov3  min =  551.81  max =  552.53  avg =  552.14
         yolov4-tiny  min =  685.49  max =  686.15  avg =  685.79
           nanodet_m  min =  181.21  max =  181.52  avg =  181.32
    yolo-fastest-1.1  min =   82.21  max =   82.68  avg =   82.30
      yolo-fastestv2  min =   67.62  max =   68.36  avg =   68.10

root@nihui-ROCK-Pi-X:/home/nihui/osd/ncnn/build/benchmark# ./benchncnn 10 1 0 0 0
[0 Intel(R) HD Graphics (CHV)]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[0 Intel(R) HD Graphics (CHV)]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 Intel(R) HD Graphics (CHV)]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 Intel(R) HD Graphics (CHV)]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =   29.14  max =   29.76  avg =   29.45
           mobilenet  min =   36.19  max =   37.03  avg =   36.52
        mobilenet_v2  min =   30.39  max =   31.62  avg =   30.76
        mobilenet_v3  min =   31.60  max =   32.25  avg =   31.92
          shufflenet  min =   22.47  max =   23.19  avg =   22.70
       shufflenet_v2  min =   22.30  max =   24.16  avg =   23.12
             mnasnet  min =   29.40  max =   30.23  avg =   29.84
     proxylessnasnet  min =   31.00  max =   31.91  avg =   31.41
     efficientnet_b0  min =   58.03  max =   58.74  avg =   58.42
   efficientnetv2_b0  min =  131.17  max =  191.61  avg =  161.37
        regnety_400m  min =   40.30  max =   42.27  avg =   41.04
           blazeface  min =   15.06  max =   15.96  avg =   15.48
           googlenet  min =   85.37  max =   86.49  avg =   85.84
            resnet18  min =   93.87  max =   95.00  avg =   94.53
             alexnet  min =  110.96  max =  120.83  avg =  115.14
               vgg16  min =  798.75  max =  812.60  avg =  804.93
            resnet50  min =  213.12  max =  214.81  avg =  213.79
      squeezenet_ssd  min =  124.48  max =  125.18  avg =  124.87
       mobilenet_ssd  min =   84.04  max =   84.70  avg =   84.49
      mobilenet_yolo  min =  186.52  max =  189.61  avg =  188.53
  mobilenetv2_yolov3  min =  102.07  max =  102.97  avg =  102.39
         yolov4-tiny  min =  212.49  max =  214.75  avg =  213.77
           nanodet_m  min =   42.97  max =   45.58  avg =   44.05
    yolo-fastest-1.1  min =   27.14  max =   32.53  avg =   28.76
      yolo-fastestv2  min =   20.73  max =   25.90  avg =   22.97
```

### Intel Celeron N5105
```
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   18.06  max =   18.21  avg =   18.12
     squeezenet_int8  min =   24.55  max =   25.16  avg =   24.69
           mobilenet  min =   32.22  max =   32.70  avg =   32.40
      mobilenet_int8  min =   40.52  max =   40.59  avg =   40.54
        mobilenet_v2  min =   22.54  max =   22.71  avg =   22.65
        mobilenet_v3  min =   17.86  max =   19.02  avg =   18.09
          shufflenet  min =   11.23  max =   11.30  avg =   11.28
       shufflenet_v2  min =   11.04  max =   11.19  avg =   11.13
             mnasnet  min =   19.93  max =   20.09  avg =   20.01
     proxylessnasnet  min =   21.91  max =   22.00  avg =   21.95
     efficientnet_b0  min =   33.29  max =   33.66  avg =   33.50
   efficientnetv2_b0  min =   40.16  max =   40.63  avg =   40.34
        regnety_400m  min =   27.38  max =   27.59  avg =   27.50
           blazeface  min =    3.01  max =    3.11  avg =    3.04
           googlenet  min =   64.78  max =   65.16  avg =   65.01
      googlenet_int8  min =   80.11  max =   80.79  avg =   80.46
            resnet18  min =   53.91  max =   54.28  avg =   54.07
       resnet18_int8  min =   63.95  max =   64.20  avg =   64.06
             alexnet  min =   51.84  max =   52.17  avg =   52.00
               vgg16  min =  322.01  max =  324.34  avg =  322.72
          vgg16_int8  min =  323.83  max =  324.17  avg =  324.02
            resnet50  min =  152.66  max =  153.33  avg =  153.03
       resnet50_int8  min =  193.40  max =  194.55  avg =  194.03
      squeezenet_ssd  min =   44.07  max =   44.51  avg =   44.37
 squeezenet_ssd_int8  min =   51.08  max =   52.26  avg =   51.60
       mobilenet_ssd  min =   67.73  max =   68.21  avg =   67.98
  mobilenet_ssd_int8  min =   82.41  max =   82.70  avg =   82.55
      mobilenet_yolo  min =  157.38  max =  159.44  avg =  158.23
  mobilenetv2_yolov3  min =   83.35  max =   83.68  avg =   83.55
         yolov4-tiny  min =  107.25  max =  107.72  avg =  107.50
           nanodet_m  min =   26.93  max =   27.24  avg =   27.09
    yolo-fastest-1.1  min =   12.47  max =   12.71  avg =   12.61
      yolo-fastestv2  min =   10.65  max =   10.95  avg =   10.81

loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   54.43  max =   54.48  avg =   54.46
     squeezenet_int8  min =   79.32  max =   79.64  avg =   79.43
           mobilenet  min =  105.92  max =  106.12  avg =  106.03
      mobilenet_int8  min =  152.24  max =  152.28  avg =  152.26
        mobilenet_v2  min =   62.44  max =   62.83  avg =   62.57
        mobilenet_v3  min =   49.47  max =   49.55  avg =   49.50
          shufflenet  min =   27.32  max =   27.37  avg =   27.34
       shufflenet_v2  min =   29.85  max =   30.00  avg =   29.93
             mnasnet  min =   59.83  max =   60.09  avg =   59.98
     proxylessnasnet  min =   66.66  max =   66.84  avg =   66.76
     efficientnet_b0  min =  104.00  max =  104.19  avg =  104.08
   efficientnetv2_b0  min =  128.05  max =  128.39  avg =  128.21
        regnety_400m  min =   77.95  max =   78.03  avg =   78.00
           blazeface  min =    6.66  max =    6.77  avg =    6.70
           googlenet  min =  195.32  max =  195.75  avg =  195.52
      googlenet_int8  min =  275.81  max =  276.25  avg =  275.98
            resnet18  min =  160.94  max =  161.17  avg =  161.03
       resnet18_int8  min =  223.88  max =  224.12  avg =  224.03
             alexnet  min =  120.96  max =  121.16  avg =  121.05
               vgg16  min =  852.50  max =  853.66  avg =  853.04
          vgg16_int8  min = 1081.07  max = 1083.31  avg = 1082.18
            resnet50  min =  497.54  max =  497.85  avg =  497.67
       resnet50_int8  min =  681.79  max =  682.60  avg =  682.29
      squeezenet_ssd  min =  101.81  max =  102.49  avg =  102.13
 squeezenet_ssd_int8  min =  147.77  max =  148.52  avg =  148.04
       mobilenet_ssd  min =  215.63  max =  216.07  avg =  215.91
  mobilenet_ssd_int8  min =  305.65  max =  305.97  avg =  305.78
      mobilenet_yolo  min =  494.99  max =  495.41  avg =  495.16
  mobilenetv2_yolov3  min =  233.51  max =  234.26  avg =  233.84
         yolov4-tiny  min =  287.26  max =  287.89  avg =  287.50
           nanodet_m  min =   70.48  max =   70.73  avg =   70.61
    yolo-fastest-1.1  min =   27.32  max =   27.36  avg =   27.34
      yolo-fastestv2  min =   23.51  max =   23.85  avg =   23.76

[0 Intel(R) UHD Graphics (JSL)]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[0 Intel(R) UHD Graphics (JSL)]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 Intel(R) UHD Graphics (JSL)]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 Intel(R) UHD Graphics (JSL)]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =   14.71  max =   15.37  avg =   14.90
           mobilenet  min =   15.38  max =   16.34  avg =   16.07
        mobilenet_v2  min =   13.58  max =   14.52  avg =   14.23
        mobilenet_v3  min =   14.95  max =   15.81  avg =   15.20
          shufflenet  min =   11.93  max =   12.73  avg =   12.31
       shufflenet_v2  min =   14.47  max =   14.74  avg =   14.60
             mnasnet  min =   15.32  max =   17.13  avg =   15.95
     proxylessnasnet  min =   15.34  max =   16.25  avg =   15.66
     efficientnet_b0  min =   26.02  max =   26.19  avg =   26.11
   efficientnetv2_b0  min =   75.92  max =   76.18  avg =   76.07
        regnety_400m  min =   17.79  max =   18.00  avg =   17.91
           blazeface  min =    5.03  max =    5.96  avg =    5.65
           googlenet  min =   35.20  max =   35.40  avg =   35.32
            resnet18  min =   35.49  max =   35.61  avg =   35.56
             alexnet  min =   40.93  max =   41.25  avg =   41.11
               vgg16  min =  220.66  max =  222.18  avg =  221.42
            resnet50  min =   78.10  max =   78.48  avg =   78.28
      squeezenet_ssd  min =   46.90  max =   47.46  avg =   47.26
       mobilenet_ssd  min =   33.33  max =   33.54  avg =   33.44
      mobilenet_yolo  min =   67.54  max =   67.77  avg =   67.64
  mobilenetv2_yolov3  min =   38.98  max =   39.69  avg =   39.37
         yolov4-tiny  min =   68.01  max =   69.74  avg =   68.86
           nanodet_m  min =   17.41  max =   18.13  avg =   17.78
    yolo-fastest-1.1  min =   13.91  max =   14.18  avg =   14.03
      yolo-fastestv2  min =   15.94  max =   16.02  avg =   15.97
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

### Ingenic T40XP Xburst2 Core X2 1.4Ghz (without MSA)
```
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = 0
cooling_down = 0
          squeezenet  min =  921.23  max =  944.03  avg =  930.71
     squeezenet_int8  min = 3280.89  max = 3404.83  avg = 3359.68
           mobilenet  min = 1277.61  max = 1298.51  avg = 1284.38
      mobilenet_int8  min = 4342.67  max = 4350.21  avg = 4345.85
        mobilenet_v2  min =  780.92  max =  783.93  avg =  782.79
        mobilenet_v3  min =  650.59  max =  655.08  avg =  652.06
          shufflenet  min =  352.75  max =  353.69  avg =  353.24
       shufflenet_v2  min =  362.82  max =  364.08  avg =  363.38
             mnasnet  min =  790.45  max =  791.89  avg =  790.99
     proxylessnasnet  min =  868.71  max =  870.47  avg =  869.17
     efficientnet_b0  min = 1491.44  max = 1492.36  avg = 1491.95
   efficientnetv2_b0  min = 2135.04  max = 2148.02  avg = 2139.99
        regnety_400m  min = 1000.53  max = 1005.29  avg = 1001.81
           blazeface  min =  102.72  max =  104.18  avg =  103.51
           googlenet  min = 3652.89  max = 3705.40  avg = 3675.43
      googlenet_int8  min = 8067.30  max = 8070.22  avg = 8069.21
```
### MacBook Pro (15-inch, 2019) - 2.6GHz six cores Intel Core i7 && Radeon Pro 555X 4GB && Intel UHD Graphics 630 1536MB
```

➜  benchmark git:(master) ✗ ./benchncnn 10 1 0 -1
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =   14.68  max =   17.06  avg =   15.55
     squeezenet_int8  min =   51.64  max =   57.85  avg =   54.01
           mobilenet  min =   20.74  max =   25.38  avg =   22.77
      mobilenet_int8  min =   66.84  max =   91.01  avg =   75.69
        mobilenet_v2  min =   14.04  max =   20.06  avg =   16.36
        mobilenet_v3  min =   11.89  max =   16.22  avg =   13.58
          shufflenet  min =   13.74  max =   17.10  avg =   15.02
       shufflenet_v2  min =   12.73  max =   14.36  avg =   13.53
             mnasnet  min =   11.05  max =   17.79  avg =   13.82
     proxylessnasnet  min =   12.60  max =   27.38  avg =   17.55
     efficientnet_b0  min =   23.73  max =   26.82  avg =   25.45
   efficientnetv2_b0  min =   27.03  max =   33.89  avg =   30.78
        regnety_400m  min =   13.81  max =   21.50  avg =   15.40
           blazeface  min =    3.72  max =    4.98  avg =    4.43
           googlenet  min =   65.88  max =   76.62  avg =   69.40
      googlenet_int8  min =  192.07  max =  227.85  avg =  203.81
            resnet18  min =   79.45  max =   90.41  avg =   85.32
       resnet18_int8  min =  201.71  max =  222.31  avg =  207.39
             alexnet  min =   70.67  max =   80.13  avg =   74.43
               vgg16  min =  233.74  max =  261.62  avg =  250.99
          vgg16_int8  min = 1722.78  max = 1997.14  avg = 1772.71
            resnet50  min =  130.39  max =  135.31  avg =  133.27
       resnet50_int8  min =  439.69  max =  483.78  avg =  461.33
      squeezenet_ssd  min =  108.54  max =  122.15  avg =  115.02
 squeezenet_ssd_int8  min =  175.58  max =  185.09  avg =  181.33
       mobilenet_ssd  min =   51.89  max =   59.32  avg =   54.30
  mobilenet_ssd_int8  min =  140.15  max =  192.10  avg =  164.47
      mobilenet_yolo  min =  117.37  max =  131.89  avg =  126.34
  mobilenetv2_yolov3  min =   57.57  max =   72.29  avg =   64.92
         yolov4-tiny  min =  114.45  max =  123.15  avg =  116.91
           nanodet_m  min =   25.65  max =   33.27  avg =   28.75

➜  benchmark git:(master) ✗ ./benchncnn 10 1 0 0
[0 AMD Radeon Pro 555X]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[0 AMD Radeon Pro 555X]  bugsbn1=0  bugbilz=196  bugcopc=0  bugihfa=0
[0 AMD Radeon Pro 555X]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 AMD Radeon Pro 555X]  subgroup=64  basic=0  vote=0  ballot=0  shuffle=0
[1 Intel(R) UHD Graphics 630]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[1 Intel(R) UHD Graphics 630]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[1 Intel(R) UHD Graphics 630]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[1 Intel(R) UHD Graphics 630]  subgroup=32  basic=0  vote=0  ballot=0  shuffle=0
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = 0
cooling_down = 1
          squeezenet  min =    6.66  max =    7.30  avg =    6.91
     squeezenet_int8  min =   49.97  max =   60.92  avg =   53.86
           mobilenet  min =    6.99  max =    7.48  avg =    7.17
      mobilenet_int8  min =   70.46  max =   83.20  avg =   79.33
        mobilenet_v2  min =    9.56  max =   10.87  avg =   10.34
        mobilenet_v3  min =   11.48  max =   12.20  avg =   11.94
          shufflenet  min =    4.52  max =    5.25  avg =    4.96
       shufflenet_v2  min =    7.29  max =    9.65  avg =    7.99
             mnasnet  min =    9.82  max =   11.88  avg =   10.62
     proxylessnasnet  min =    7.85  max =    8.41  avg =    8.07
     efficientnet_b0  min =   17.34  max =   17.85  avg =   17.56
   efficientnetv2_b0  min =   21.95  max =   24.10  avg =   23.15
        regnety_400m  min =   13.54  max =   14.83  avg =   14.11
           blazeface  min =    3.26  max =    6.59  avg =    5.50
           googlenet  min =   17.62  max =   19.47  avg =   18.27
      googlenet_int8  min =  198.88  max =  247.97  avg =  223.31
            resnet18  min =   11.10  max =   12.01  avg =   11.59
       resnet18_int8  min =  225.56  max =  259.39  avg =  238.97
             alexnet  min =   17.66  max =   19.19  avg =   18.24
               vgg16  min =   53.20  max =   54.88  avg =   53.73
          vgg16_int8  min = 1747.52  max = 2130.08  avg = 1880.42
            resnet50  min =   27.38  max =   28.84  avg =   28.34
       resnet50_int8  min =  461.86  max =  579.83  avg =  528.15
      squeezenet_ssd  min =   19.99  max =   20.98  avg =   20.50
 squeezenet_ssd_int8  min =  185.20  max =  209.66  avg =  196.81
       mobilenet_ssd  min =   12.81  max =   14.21  avg =   13.48
  mobilenet_ssd_int8  min =  139.29  max =  168.38  avg =  148.20
      mobilenet_yolo  min =   19.50  max =   20.51  avg =   19.97
  mobilenetv2_yolov3  min =   15.95  max =   19.28  avg =   16.85
         yolov4-tiny  min =   21.43  max =   23.42  avg =   22.28
           nanodet_m  min =    7.95  max =    9.23  avg =    8.48

➜  benchmark git:(master) ✗ ./benchncnn 10 1 0 1
[0 AMD Radeon Pro 555X]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[0 AMD Radeon Pro 555X]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 AMD Radeon Pro 555X]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 AMD Radeon Pro 555X]  subgroup=64  basic=0  vote=0  ballot=0  shuffle=0
[1 Intel(R) UHD Graphics 630]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[1 Intel(R) UHD Graphics 630]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[1 Intel(R) UHD Graphics 630]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[1 Intel(R) UHD Graphics 630]  subgroup=32  basic=0  vote=0  ballot=0  shuffle=0
loop_count = 10
num_threads = 1
powersave = 0
gpu_device = 1
cooling_down = 1
          squeezenet  min =   11.06  max =   13.22  avg =   12.09
     squeezenet_int8  min =   54.87  max =   64.55  avg =   59.84
           mobilenet  min =   13.65  max =   16.70  avg =   14.81
      mobilenet_int8  min =   72.36  max =   93.58  avg =   86.40
        mobilenet_v2  min =   11.88  max =   15.90  avg =   13.47
        mobilenet_v3  min =   12.68  max =   16.16  avg =   14.56
          shufflenet  min =   13.87  max =   16.68  avg =   14.93
       shufflenet_v2  min =   11.73  max =   13.65  avg =   12.87
             mnasnet  min =   12.71  max =   15.56  avg =   14.22
     proxylessnasnet  min =   14.03  max =   17.28  avg =   15.37
     efficientnet_b0  min =   17.50  max =   21.46  avg =   19.30
   efficientnetv2_b0  min =   35.47  max =   38.58  avg =   36.89
        regnety_400m  min =   16.00  max =   19.45  avg =   17.48
           blazeface  min =    6.08  max =    7.18  avg =    6.39
           googlenet  min =   23.35  max =   29.68  avg =   25.77
      googlenet_int8  min =  198.49  max =  254.38  avg =  222.77
            resnet18  min =   21.85  max =   28.10  avg =   24.70
       resnet18_int8  min =  211.21  max =  279.55  avg =  222.64
             alexnet  min =   24.45  max =   30.47  avg =   26.87
               vgg16  min =  115.20  max =  117.76  avg =  116.48
          vgg16_int8  min = 1715.92  max = 1960.02  avg = 1800.21
            resnet50  min =   45.65  max =   46.25  avg =   46.05
       resnet50_int8  min =  448.13  max =  555.53  avg =  485.47
      squeezenet_ssd  min =   28.43  max =   33.26  avg =   29.85
 squeezenet_ssd_int8  min =  180.91  max =  202.51  avg =  190.84
       mobilenet_ssd  min =   21.03  max =   26.93  avg =   23.48
  mobilenet_ssd_int8  min =  154.41  max =  184.64  avg =  165.04
      mobilenet_yolo  min =   37.04  max =   38.64  avg =   37.52
  mobilenetv2_yolov3  min =   24.98  max =   30.03  avg =   27.70
         yolov4-tiny  min =   39.29  max =   50.25  avg =   44.18
           nanodet_m  min =   15.97  max =   20.27  avg =   17.93
```
