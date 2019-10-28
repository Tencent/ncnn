benchncnn can be used to test neural network inference performance

Only the network definition files (ncnn param) are required.

The large model binary files (ncnn bin) are not loaded but generated randomly for speed test.

More model networks may be added later.

---
Build
```
# assume you have already build ncnn library successfully
# uncomment the following line in <ncnn-root-dir>/CMakeLists.txt with your favorite editor

# add_subdirectory(benchmark)

$ cd <ncnn-root-dir>/<your-build-dir>
$ make -j4

# you can find benchncnn binary in <ncnn-root-dir>/<your-build-dir>/benchmark
```

Usage
```
# copy all param files to the current directory
$ ./benchncnn [loop count] [num threads] [powersave] [gpu device]
```
run benchncnn on android device
```
# for running on android device, upload to /data/local/tmp/ folder
$ adb push benchncnn /data/local/tmp/
$ adb push <ncnn-root-dir>/benchmark/*.param /data/local/tmp/
$ adb shell

# executed in android adb shell
$ cd /data/local/tmp/
$ ./benchncnn [loop count] [num threads] [powersave] [gpu device]
```

Parameter

|param|options|default|
|---|---|---|
|loop count|1~N|4|
|num threads|1~N|max_cpu_count|
|powersave|0=all cores, 1=little cores only, 2=big cores only|0|
|gpu device|-1=cpu-only, 0=gpu0, 1=gpu1 ...|-1|

---

Typical output (executed in android adb shell)

Qualcomm MSM6150 Snapdragon 675 (Kyro460 2.0GHz x 2 + Kyro460 1.7GHz x 6 + Adreno 612)
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

Kirin 970 (Cortex-A73 2.4GHz x 4 + Cortex-A53 1.8GHz x 4)
```
HWBKL:/data/local/tmp/ncnn $ ./benchncnn 8 4 2                                 
loop_count = 8
num_threads = 4
powersave = 2
gpu_device = -1
          squeezenet  min =   22.55  max =   27.76  avg =   25.71
     squeezenet-int8  min =   18.46  max =   24.04  avg =   19.83
           mobilenet  min =   32.52  max =   39.48  avg =   34.29
      mobilenet-int8  min =   21.65  max =   27.64  avg =   22.62
        mobilenet_v2  min =   29.93  max =   32.77  avg =   31.87
          shufflenet  min =   15.40  max =   19.51  avg =   17.56
             mnasnet  min =   25.10  max =   29.34  avg =   27.56
     proxylessnasnet  min =   33.08  max =   35.05  avg =   33.63
           googlenet  min =   81.98  max =   95.30  avg =   89.31
      googlenet-int8  min =   71.39  max =   76.15  avg =   73.74
            resnet18  min =   78.78  max =   87.98  avg =   86.15
       resnet18-int8  min =   66.45  max =   79.07  avg =   70.57
             alexnet  min =  139.34  max =  139.66  avg =  139.48
               vgg16  min =  427.03  max =  430.85  avg =  428.96
            resnet50  min =  343.06  max =  353.42  avg =  346.09
       resnet50-int8  min =  146.54  max =  150.83  avg =  148.85
      squeezenet-ssd  min =   57.13  max =   57.87  avg =   57.58
 squeezenet-ssd-int8  min =   56.35  max =   58.03  avg =   57.10
       mobilenet-ssd  min =   69.72  max =   75.62  avg =   72.84
  mobilenet-ssd-int8  min =   43.79  max =   49.95  avg =   44.73
      mobilenet-yolo  min =  179.57  max =  187.39  avg =  184.98
    mobilenet-yolov3  min =  164.52  max =  182.49  avg =  174.72
```

Qualcomm MSM8998 Snapdragon 835 (Kyro 2.45GHz x 4 + Kyro 1.9GHz x 4 + Adreno 540)
```
sagit:/data/local/tmp $ ./benchncnn 8 4 0
[0 Adreno (TM) 540]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 540]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
          squeezenet  min =   25.76  max =   26.92  avg =   26.12
     squeezenet_int8  min =   20.95  max =   21.23  avg =   21.07
           mobilenet  min =   38.37  max =   38.77  avg =   38.61
      mobilenet_int8  min =   30.31  max =   30.93  avg =   30.57
        mobilenet_v2  min =   30.23  max =   30.92  avg =   30.67
          shufflenet  min =   14.69  max =   14.89  avg =   14.78
             mnasnet  min =   26.89  max =   27.12  avg =   26.96
     proxylessnasnet  min =   30.80  max =   30.97  avg =   30.86
           googlenet  min =   90.19  max =   91.43  avg =   90.60
      googlenet_int8  min =   73.63  max =   74.12  avg =   73.92
            resnet18  min =   84.19  max =   86.84  avg =   85.56
       resnet18_int8  min =   61.74  max =   62.47  avg =   61.91
             alexnet  min =  142.65  max =  144.35  avg =  143.35
               vgg16  min =  467.25  max =  479.00  avg =  471.77
          vgg16_int8  min =  464.94  max =  468.86  avg =  466.73
            resnet50  min =  202.83  max =  204.22  avg =  203.36
       resnet50_int8  min =  165.61  max =  166.11  avg =  165.78
      squeezenet_ssd  min =   73.29  max =   75.00  avg =   73.99
 squeezenet_ssd_int8  min =   65.03  max =   66.28  avg =   65.50
       mobilenet_ssd  min =   88.01  max =   88.66  avg =   88.25
  mobilenet_ssd_int8  min =   69.95  max =   70.76  avg =   70.14
      mobilenet_yolo  min =  191.62  max =  237.58  avg =  212.80
    mobilenet_yolov3  min =  241.35  max =  243.13  avg =  242.27

sagit:/data/local/tmp $ ./benchncnn 8 1 0
[0 Adreno (TM) 540]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 540]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =   68.57  max =   69.67  avg =   68.88
     squeezenet_int8  min =   52.99  max =   53.82  avg =   53.31
           mobilenet  min =  116.61  max =  118.33  avg =  117.64
      mobilenet_int8  min =   96.25  max =   98.42  avg =   96.87
        mobilenet_v2  min =   78.55  max =   79.71  avg =   78.94
          shufflenet  min =   33.62  max =   34.23  avg =   34.01
             mnasnet  min =   74.20  max =   75.23  avg =   74.89
     proxylessnasnet  min =   87.76  max =   89.33  avg =   88.63
           googlenet  min =  278.71  max =  281.95  avg =  280.19
      googlenet_int8  min =  205.23  max =  206.50  avg =  205.75
            resnet18  min =  228.86  max =  231.37  avg =  230.13
       resnet18_int8  min =  162.87  max =  165.73  avg =  163.89
             alexnet  min =  359.06  max =  359.96  avg =  359.67
               vgg16  min = 1359.55  max = 1368.28  avg = 1364.26
          vgg16_int8  min =  987.93  max =  996.37  avg =  991.80
            resnet50  min =  552.06  max =  556.15  avg =  553.67
       resnet50_int8  min =  412.79  max =  415.59  avg =  414.15
      squeezenet_ssd  min =  158.16  max =  159.39  avg =  158.77
 squeezenet_ssd_int8  min =  132.39  max =  134.26  avg =  133.42
       mobilenet_ssd  min =  233.77  max =  242.49  avg =  238.20
  mobilenet_ssd_int8  min =  192.66  max =  200.20  avg =  197.47
      mobilenet_yolo  min =  522.35  max =  537.15  avg =  529.32
    mobilenet_yolov3  min =  535.72  max =  549.35  avg =  541.81

sagit:/data/local/tmp $ ./benchncnn 8 1 0 0
[0 Adreno (TM) 540]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 540]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 0
gpu_device = 0
          squeezenet  min =   35.06  max =   45.54  avg =   36.91
           mobilenet  min =   50.06  max =   51.50  avg =   51.07
        mobilenet_v2  min =   38.21  max =   41.10  avg =   39.14
          shufflenet  min =   34.92  max =   35.73  avg =   35.30
             mnasnet  min =   38.82  max =   39.16  avg =   39.02
     proxylessnasnet  min =   42.60  max =   43.93  avg =   43.22
           googlenet  min =  136.68  max =  139.14  avg =  138.05
            resnet18  min =  142.47  max =  143.61  avg =  142.96
             alexnet  min =  297.56  max =  303.92  avg =  300.53
               vgg16  min =  980.64  max =  998.57  avg =  988.27
            resnet50  min =  312.66  max =  315.18  avg =  314.44
      squeezenet_ssd  min =  189.98  max =  194.55  avg =  192.53
       mobilenet_ssd  min =  125.63  max =  126.95  avg =  126.17
      mobilenet_yolo  min =  260.15  max =  264.34  avg =  262.51
    mobilenet_yolov3  min =  249.49  max =  250.87  avg =  249.94
```

Qualcomm MSM8996 Snapdragon 820 (Kyro 2.15GHz x 2 + Kyro 1.6GHz x 2)
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

Qualcomm MSM8994 Snapdragon 810 (Cortex-A57 2.0GHz x 4 + Cortex-A53 1.55GHz x 4)
```
angler:/data/local/tmp $ ./benchncnn 8 8 0 -1
[0 Adreno (TM) 430]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 8
powersave = 0
gpu_device = -1
          squeezenet  min =   29.78  max =   31.54  avg =   30.44
     squeezenet_int8  min =   32.65  max =   33.24  avg =   32.92
           mobilenet  min =   35.30  max =   36.09  avg =   35.64
      mobilenet_int8  min =   42.39  max =   56.78  avg =   45.79
        mobilenet_v2  min =   34.43  max =   37.82  avg =   35.91
        mobilenet_v3  min =   29.32  max =   31.81  avg =   30.06
          shufflenet  min =   26.97  max =   39.34  avg =   28.97
       shufflenet_v2  min =   20.52  max =   21.59  avg =   21.04
             mnasnet  min =   31.04  max =   32.05  avg =   31.40
     proxylessnasnet  min =   34.84  max =   37.39  avg =   35.85
           googlenet  min =   92.10  max =  100.70  avg =   95.79
      googlenet_int8  min =  114.91  max =  128.51  avg =  122.20
            resnet18  min =   86.65  max =   98.13  avg =   91.50
       resnet18_int8  min =   98.86  max =  107.98  avg =  104.97
             alexnet  min =   59.56  max =   62.33  avg =   60.44
               vgg16  min =  478.32  max =  536.57  avg =  506.26
          vgg16_int8  min =  798.23  max =  982.39  avg =  864.55
            resnet50  min =  250.06  max =  324.21  avg =  294.96
       resnet50_int8  min =  266.27  max =  316.08  avg =  293.68
      squeezenet_ssd  min =  102.29  max =  118.00  avg =  108.56
 squeezenet_ssd_int8  min =  112.11  max =  128.01  avg =  120.85
       mobilenet_ssd  min =   77.02  max =   99.10  avg =   90.02
  mobilenet_ssd_int8  min =   85.34  max =   99.40  avg =   92.08
      mobilenet_yolo  min =  189.59  max =  250.30  avg =  215.45
  mobilenetv2_yolov3  min =  112.90  max =  134.83  avg =  122.34

angler:/data/local/tmp $ ./benchncnn 8 1 2 -1
[0 Adreno (TM) 430]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
          squeezenet  min =   71.93  max =   74.06  avg =   72.52
     squeezenet_int8  min =   80.48  max =   84.26  avg =   82.42
           mobilenet  min =  110.80  max =  112.19  avg =  111.59
      mobilenet_int8  min =  132.81  max =  137.21  avg =  135.70
        mobilenet_v2  min =   76.54  max =   77.15  avg =   76.84
        mobilenet_v3  min =   63.34  max =   65.32  avg =   64.36
          shufflenet  min =   42.66  max =   43.02  avg =   42.81
       shufflenet_v2  min =   37.79  max =   38.20  avg =   37.91
             mnasnet  min =   72.45  max =   73.98  avg =   73.20
     proxylessnasnet  min =   93.86  max =   95.09  avg =   94.45
           googlenet  min =  244.22  max =  247.57  avg =  245.56
      googlenet_int8  min =  299.63  max =  307.30  avg =  303.43
            resnet18  min =  215.69  max =  218.01  avg =  216.48
       resnet18_int8  min =  257.21  max =  260.61  avg =  259.00
             alexnet  min =  182.54  max =  191.83  avg =  185.48
               vgg16  min =  951.71  max =  957.57  avg =  954.09
          vgg16_int8  min = 1430.97  max = 1449.88  avg = 1441.17
            resnet50  min =  530.57  max =  535.83  avg =  534.17
       resnet50_int8  min =  574.92  max =  587.43  avg =  583.75
      squeezenet_ssd  min =  165.72  max =  167.60  avg =  166.36
 squeezenet_ssd_int8  min =  211.29  max =  216.03  avg =  214.33
       mobilenet_ssd  min =  230.97  max =  237.00  avg =  233.38
  mobilenet_ssd_int8  min =  276.50  max =  279.99  avg =  277.89
      mobilenet_yolo  min =  511.57  max =  513.35  avg =  512.41
  mobilenetv2_yolov3  min =  264.79  max =  266.26  avg =  265.64

angler:/data/local/tmp $ ./benchncnn 4 1 2 0
[0 Adreno (TM) 430]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = 0
          squeezenet  min =   63.62  max =   65.43  avg =   64.40
           mobilenet  min =  102.23  max =  102.41  avg =  102.31
        mobilenet_v2  min =   66.78  max =   67.72  avg =   67.33
        mobilenet_v3  min =   59.54  max =   61.45  avg =   60.78
          shufflenet  min =   40.87  max =   41.02  avg =   40.92
       shufflenet_v2  min =   63.76  max =   65.94  avg =   64.91
             mnasnet  min =   67.72  max =   69.09  avg =   68.43
     proxylessnasnet  min =   72.76  max =   74.54  avg =   73.35
           googlenet  min =  222.64  max =  225.33  avg =  224.27
            resnet18  min =  221.03  max =  221.44  avg =  221.32
             alexnet  min =  272.73  max =  289.68  avg =  281.79
               vgg16  min = 1485.61  max = 1500.48  avg = 1493.50
            resnet50  min =  543.39  max =  544.91  avg =  544.12
      squeezenet_ssd  min =  255.55  max =  261.69  avg =  258.16
       mobilenet_ssd  min =  223.17  max =  223.74  avg =  223.41
      mobilenet_yolo  min =  472.74  max =  474.75  avg =  473.81
  mobilenetv2_yolov3  min =  232.31  max =  233.25  avg =  232.73
```

Qualcomm MSM8916 Snapdragon 410 (Cortex-A53 1.2GHz x 4)
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0 -1
no vulkan device
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
          squeezenet  min =   69.84  max =   75.30  avg =   71.96
     squeezenet_int8  min =   81.75  max =   89.49  avg =   87.37
           mobilenet  min =   86.45  max =   92.54  avg =   88.71
      mobilenet_int8  min =  126.33  max =  135.35  avg =  129.80
        mobilenet_v2  min =   77.29  max =   82.98  avg =   79.48
        mobilenet_v3  min =   62.67  max =   69.64  avg =   65.90
          shufflenet  min =   49.72  max =   55.24  avg =   52.12
       shufflenet_v2  min =   41.56  max =   47.44  avg =   43.22
             mnasnet  min =   70.85  max =  101.80  avg =   77.14
     proxylessnasnet  min =   78.41  max =  118.13  avg =   86.30
           googlenet  min =  185.25  max =  193.25  avg =  190.61
      googlenet_int8  min =  249.37  max =  265.69  avg =  261.41
            resnet18  min =  161.23  max =  171.26  avg =  165.87
       resnet18_int8  min =  228.60  max =  235.20  avg =  231.43
             alexnet  min =  131.93  max =  138.04  avg =  134.94
               vgg16  min =  761.87  max =  782.58  avg =  771.16
          vgg16_int8  min = 1428.25  max = 1466.57  avg = 1444.17
            resnet50  min =  396.75  max =  413.57  avg =  400.44
       resnet50_int8  min =  515.11  max =  528.93  avg =  522.78
      squeezenet_ssd  min =  171.97  max =  181.60  avg =  176.61
 squeezenet_ssd_int8  min =  234.82  max =  251.06  avg =  242.53
       mobilenet_ssd  min =  189.11  max =  197.64  avg =  192.07
  mobilenet_ssd_int8  min =  237.46  max =  259.70  avg =  247.50
      mobilenet_yolo  min =  393.03  max =  398.29  avg =  396.03
  mobilenetv2_yolov3  min =  243.06  max =  251.99  avg =  248.98

HM2014812:/data/local/tmp # ./benchncnn 4 1 0 -1
no vulkan device
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =  154.41  max =  154.63  avg =  154.49
     squeezenet_int8  min =  198.03  max =  198.47  avg =  198.33
           mobilenet  min =  245.66  max =  246.02  avg =  245.89
      mobilenet_int8  min =  362.26  max =  367.15  avg =  364.51
        mobilenet_v2  min =  171.04  max =  172.61  avg =  172.03
        mobilenet_v3  min =  147.45  max =  148.01  avg =  147.80
          shufflenet  min =   99.12  max =   99.63  avg =   99.32
       shufflenet_v2  min =   87.12  max =   88.34  avg =   87.58
             mnasnet  min =  163.69  max =  164.31  avg =  163.96
     proxylessnasnet  min =  206.99  max =  207.19  avg =  207.09
           googlenet  min =  549.57  max =  552.38  avg =  550.31
      googlenet_int8  min =  713.62  max =  722.81  avg =  716.56
            resnet18  min =  496.47  max =  497.21  avg =  496.76
       resnet18_int8  min =  614.42  max =  618.19  avg =  616.69
             alexnet  min =  485.29  max =  485.48  avg =  485.41
               vgg16  min = 2249.25  max = 2258.91  avg = 2254.00
          vgg16_int8  min = 3633.70  max = 3640.76  avg = 3637.08
            resnet50  min = 1207.35  max = 1215.22  avg = 1211.77
       resnet50_int8  min = 1435.69  max = 1442.76  avg = 1439.70
      squeezenet_ssd  min =  351.84  max =  354.93  avg =  353.40
 squeezenet_ssd_int8  min =  522.87  max =  523.83  avg =  523.46
       mobilenet_ssd  min =  519.71  max =  525.70  avg =  523.20
  mobilenet_ssd_int8  min =  709.94  max =  713.54  avg =  711.16
      mobilenet_yolo  min = 1134.40  max = 1136.18  avg = 1135.55
  mobilenetv2_yolov3  min =  597.02  max =  603.85  avg =  600.56
```
Raspberry Pi 3 Model B+ Broadcom BCM2837B0, Cortex-A53 (ARMv8) (1.4GHz x 4 )
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

Rockchip RK3399 (Cortex-A72 1.8GHz x 2 + Cortex-A53 1.5GHz x 4)
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

Rockchip RK3288 (Cortex-A17 1.8GHz x 4)
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

HiSilicon Hi3519V101 (Cortex-A17 1.2GHz x 1)
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

iPhone 5S (Apple A7 1.3GHz x 2)
```
iPhone:~ root# ./benchncnn 8 2 0 -1
[0 Apple A7 GPU]  queueC=0[8]  queueT=0[8]  memU=1  memDL=1  memHV=1
[0 Apple A7 GPU]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
          squeezenet  min =   48.99  max =   49.78  avg =   49.45
     squeezenet_int8  min =   56.13  max =   57.42  avg =   56.81
           mobilenet  min =   84.16  max =   89.81  avg =   88.47
      mobilenet_int8  min =  109.50  max =  112.03  avg =  111.34
        mobilenet_v2  min =   71.56  max =   72.43  avg =   71.96
        mobilenet_v3  min =   65.05  max =   66.91  avg =   66.41
          shufflenet  min =   35.70  max =   36.21  avg =   35.97
       shufflenet_v2  min =   34.11  max =   35.09  avg =   34.43
             mnasnet  min =   75.07  max =   81.22  avg =   76.27
     proxylessnasnet  min =   98.09  max =  100.32  avg =   99.70
           googlenet  min =  193.53  max =  194.29  avg =  193.86
      googlenet_int8  min =  238.45  max =  241.87  avg =  240.04
            resnet18  min =  166.78  max =  169.38  avg =  167.62
       resnet18_int8  min =  174.25  max =  179.40  avg =  178.03
             alexnet  min =  223.84  max =  227.87  avg =  226.64
               vgg16  min = 4274.73  max = 4384.21  avg = 4321.12
          vgg16_int8  min = 2851.18  max = 2931.06  avg = 2886.38
            resnet50  min =  584.61  max = 1198.90  avg =  843.45
       resnet50_int8  min =  428.77  max =  433.07  avg =  431.64
      squeezenet_ssd  min =  124.95  max =  129.02  avg =  127.74
 squeezenet_ssd_int8  min =  155.81  max =  161.03  avg =  159.09
       mobilenet_ssd  min =  175.63  max =  181.72  avg =  177.58
  mobilenet_ssd_int8  min =  210.13  max =  215.42  avg =  211.35
      mobilenet_yolo  min =  416.58  max =  418.47  avg =  417.54
  mobilenetv2_yolov3  min =  191.08  max =  193.94  avg =  192.50

iPhone:~ root# ./benchncnn 4 1 0 -1
[0 Apple A7 GPU]  queueC=0[8]  queueT=0[8]  memU=1  memDL=1  memHV=1
[0 Apple A7 GPU]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =   86.93  max =   87.19  avg =   87.02
     squeezenet_int8  min =  100.08  max =  100.29  avg =  100.23
           mobilenet  min =  170.40  max =  171.36  avg =  170.88
      mobilenet_int8  min =  203.44  max =  205.93  avg =  204.54
        mobilenet_v2  min =  121.45  max =  122.10  avg =  121.78
        mobilenet_v3  min =   88.54  max =   89.30  avg =   88.86
          shufflenet  min =   54.38  max =   54.93  avg =   54.70
       shufflenet_v2  min =   53.28  max =   53.68  avg =   53.54
             mnasnet  min =  119.53  max =  119.96  avg =  119.77
     proxylessnasnet  min =  158.69  max =  159.54  avg =  159.13
           googlenet  min =  355.21  max =  356.21  avg =  355.57
      googlenet_int8  min =  419.25  max =  425.11  avg =  423.48
            resnet18  min =  304.66  max =  308.39  avg =  306.32
       resnet18_int8  min =  307.38  max =  310.87  avg =  309.92
             alexnet  min =  431.57  max =  432.70  avg =  432.16
               vgg16  min = 4864.05  max = 4990.12  avg = 4916.39
          vgg16_int8  min = 3634.18  max = 3774.51  avg = 3704.15
            resnet50  min = 1813.64  max = 1932.45  avg = 1883.24
       resnet50_int8  min =  759.16  max =  764.36  avg =  762.69
      squeezenet_ssd  min =  204.48  max =  205.46  avg =  204.97
 squeezenet_ssd_int8  min =  246.21  max =  248.67  avg =  247.90
       mobilenet_ssd  min =  309.56  max =  310.80  avg =  310.45
  mobilenet_ssd_int8  min =  394.91  max =  395.71  avg =  395.26
      mobilenet_yolo  min =  775.40  max =  775.98  avg =  775.60
  mobilenetv2_yolov3  min =  349.42  max =  349.89  avg =  349.63

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

Freescale i.MX7 Dual (Cortex A7 1.0GHz x 2)
```
imx7d_pico:/data/local/tmp $ ./benchncnn 8 2 0 -1
no vulkan device
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
          squeezenet  min =  224.67  max =  231.60  avg =  227.47
     squeezenet_int8  min =  185.46  max =  193.12  avg =  188.00
           mobilenet  min =  368.35  max =  377.54  avg =  373.52
      mobilenet_int8  min =  295.91  max =  311.14  avg =  300.22
        mobilenet_v2  min =  245.15  max =  258.91  avg =  251.11
        mobilenet_v3  min =  204.87  max =  214.75  avg =  208.43
          shufflenet  min =  136.80  max =  146.50  avg =  139.83
       shufflenet_v2  min =  134.49  max =  140.67  avg =  136.52
             mnasnet  min =  239.42  max =  259.82  avg =  244.24
     proxylessnasnet  min =  274.69  max =  314.04  avg =  283.27
           googlenet  min =  767.22  max =  777.88  avg =  772.83
      googlenet_int8  min =  649.88  max =  661.52  avg =  655.04
            resnet18  min =  801.59  max =  810.73  avg =  806.35
       resnet18_int8  min =  559.68  max =  565.63  avg =  562.97
             alexnet  min =  537.42  max =  546.84  avg =  543.54
          vgg16_int8  min = 4020.71  max = 4225.25  avg = 4125.73
       resnet50_int8  min = 1280.18  max = 1299.61  avg = 1289.15
      squeezenet_ssd  min =  507.03  max =  525.56  avg =  511.64
 squeezenet_ssd_int8  min =  516.40  max =  522.67  avg =  519.79
       mobilenet_ssd  min =  763.48  max =  853.31  avg =  782.99
  mobilenet_ssd_int8  min =  561.88  max =  571.76  avg =  565.32
      mobilenet_yolo  min = 1707.59  max = 1737.15  avg = 1724.02
  mobilenetv2_yolov3  min =  868.44  max =  881.45  avg =  873.60

imx7d_pico:/data/local/tmp $ ./benchncnn 4 1 0 -1
no vulkan device
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =  415.15  max =  420.91  avg =  416.71
     squeezenet_int8  min =  318.99  max =  327.53  avg =  321.54
           mobilenet  min =  700.08  max =  702.53  avg =  701.22
      mobilenet_int8  min =  552.53  max =  553.29  avg =  553.05
        mobilenet_v2  min =  451.33  max =  451.80  avg =  451.58
        mobilenet_v3  min =  381.48  max =  381.71  avg =  381.61
          shufflenet  min =  243.32  max =  247.22  avg =  245.32
       shufflenet_v2  min =  232.13  max =  238.69  avg =  235.41
             mnasnet  min =  452.86  max =  457.33  avg =  455.57
     proxylessnasnet  min =  522.94  max =  523.48  avg =  523.11
           googlenet  min = 1421.14  max = 1422.31  avg = 1421.91
      googlenet_int8  min = 1180.16  max = 1181.66  avg = 1181.07
            resnet18  min = 1515.22  max = 1519.08  avg = 1517.02
       resnet18_int8  min =  982.26  max =  983.99  avg =  983.23
             alexnet  min = 1043.95  max = 1044.17  avg = 1044.06
          vgg16_int8  min = 6408.50  max = 6452.92  avg = 6431.34
       resnet50_int8  min = 2327.96  max = 2335.53  avg = 2330.79
      squeezenet_ssd  min =  903.02  max =  903.65  avg =  903.42
 squeezenet_ssd_int8  min =  838.03  max =  841.02  avg =  839.47
       mobilenet_ssd  min = 1458.43  max = 1459.58  avg = 1459.14
  mobilenet_ssd_int8  min = 1038.04  max = 1038.91  avg = 1038.46
      mobilenet_yolo  min = 3284.78  max = 3292.92  avg = 3287.02
  mobilenetv2_yolov3  min = 1648.99  max = 1652.01  avg = 1650.45
```
