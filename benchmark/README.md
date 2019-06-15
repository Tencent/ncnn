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
          squeezenet  min =   35.20  max =   37.31  avg =   36.16
     squeezenet_int8  min =   33.28  max =   34.16  avg =   33.69
           mobilenet  min =   40.05  max =   41.64  avg =   40.77
      mobilenet_int8  min =   44.21  max =   59.67  avg =   47.32
        mobilenet_v2  min =   40.54  max =   44.47  avg =   41.67
          shufflenet  min =   26.27  max =   27.69  avg =   26.95
             mnasnet  min =   33.82  max =   35.53  avg =   34.56
     proxylessnasnet  min =   40.87  max =   41.85  avg =   41.48
           googlenet  min =  117.12  max =  124.40  avg =  119.08
      googlenet_int8  min =  115.56  max =  127.86  avg =  118.47
            resnet18  min =  115.12  max =  133.91  avg =  119.21
       resnet18_int8  min =  103.82  max =  120.64  avg =  110.19
             alexnet  min =  102.87  max =  113.87  avg =  106.37
               vgg16  min =  631.35  max =  803.15  avg =  704.54
          vgg16_int8  min =  733.03  max =  926.28  avg =  833.06
            resnet50  min =  239.58  max =  307.39  avg =  275.57
       resnet50_int8  min =  241.82  max =  299.77  avg =  271.43
      squeezenet_ssd  min =  105.07  max =  127.09  avg =  112.49
 squeezenet_ssd_int8  min =  111.01  max =  123.29  avg =  116.56
       mobilenet_ssd  min =   87.14  max =  103.73  avg =   90.35
  mobilenet_ssd_int8  min =   84.85  max =  100.21  avg =   89.86
      mobilenet_yolo  min =  193.35  max =  259.92  avg =  232.43
    mobilenet_yolov3  min =  201.78  max =  268.21  avg =  247.84

angler:/data/local/tmp $ ./benchncnn 8 1 2 -1
[0 Adreno (TM) 430]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
          squeezenet  min =   89.16  max =   90.35  avg =   89.45
     squeezenet_int8  min =   80.78  max =   83.93  avg =   82.89
           mobilenet  min =  129.52  max =  130.83  avg =  130.37
      mobilenet_int8  min =  135.67  max =  137.39  avg =  136.46
        mobilenet_v2  min =   92.56  max =   94.22  avg =   93.33
          shufflenet  min =   47.40  max =   47.71  avg =   47.53
             mnasnet  min =   85.46  max =   86.49  avg =   86.01
     proxylessnasnet  min =  105.07  max =  108.15  avg =  106.76
           googlenet  min =  346.85  max =  352.11  avg =  348.53
      googlenet_int8  min =  305.50  max =  308.97  avg =  308.10
            resnet18  min =  283.16  max =  288.63  avg =  284.99
       resnet18_int8  min =  269.03  max =  271.15  avg =  270.11
             alexnet  min =  308.02  max =  331.66  avg =  316.61
               vgg16  min = 1404.13  max = 1420.82  avg = 1411.80
          vgg16_int8  min = 1434.01  max = 1449.60  avg = 1443.90
            resnet50  min =  649.41  max =  657.73  avg =  655.96
       resnet50_int8  min =  617.58  max =  625.31  avg =  621.32
      squeezenet_ssd  min =  197.78  max =  200.01  avg =  198.99
 squeezenet_ssd_int8  min =  211.59  max =  217.95  avg =  215.20
       mobilenet_ssd  min =  263.36  max =  271.00  avg =  268.68
  mobilenet_ssd_int8  min =  274.52  max =  278.78  avg =  276.79
      mobilenet_yolo  min =  590.42  max =  596.09  avg =  593.38
    mobilenet_yolov3  min =  613.12  max =  632.20  avg =  625.98

angler:/data/local/tmp $ ./benchncnn 4 1 2 0
[0 Adreno (TM) 430]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 2
gpu_device = 0
          squeezenet  min =   63.34  max =   64.84  avg =   63.97
           mobilenet  min =  102.15  max =  102.58  avg =  102.31
        mobilenet_v2  min =   66.96  max =   68.38  avg =   67.53
          shufflenet  min =   41.24  max =   42.66  avg =   41.83
             mnasnet  min =   67.92  max =   68.70  avg =   68.15
     proxylessnasnet  min =   72.68  max =   74.70  avg =   73.68
           googlenet  min =  224.78  max =  225.32  avg =  225.09
            resnet18  min =  221.38  max =  221.93  avg =  221.71
             alexnet  min =  279.22  max =  288.89  avg =  282.13
               vgg16  min = 1511.11  max = 1520.28  avg = 1516.09
            resnet50  min =  543.91  max =  544.93  avg =  544.37
      squeezenet_ssd  min =  256.75  max =  263.39  avg =  260.09
       mobilenet_ssd  min =  223.12  max =  223.86  avg =  223.55
      mobilenet_yolo  min =  471.34  max =  474.97  avg =  473.00
    mobilenet_yolov3  min =  472.65  max =  476.39  avg =  474.20
```

Qualcomm MSM8916 Snapdragon 410 (Cortex-A53 1.2GHz x 4)
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0 -1
no vulkan device
loop_count = 8
num_threads = 4
powersave = 0
gpu_device = -1
          squeezenet  min =   74.66  max =   80.12  avg =   77.15
     squeezenet_int8  min =   81.34  max =   87.29  avg =   85.59
           mobilenet  min =  103.08  max =  108.38  avg =  104.80
      mobilenet_int8  min =  127.09  max =  128.51  avg =  127.75
        mobilenet_v2  min =   98.50  max =  105.00  avg =  100.97
          shufflenet  min =   51.58  max =   56.81  avg =   53.71
             mnasnet  min =   85.41  max =   89.80  avg =   87.53
     proxylessnasnet  min =   96.80  max =  103.95  avg =  101.19
           googlenet  min =  243.21  max =  247.12  avg =  244.48
      googlenet_int8  min =  254.19  max =  270.68  avg =  260.82
            resnet18  min =  217.00  max =  220.79  avg =  218.74
       resnet18_int8  min =  240.32  max =  276.83  avg =  251.17
             alexnet  min =  237.36  max =  248.34  avg =  242.39
               vgg16  min = 1335.34  max = 1402.95  avg = 1379.22
          vgg16_int8  min = 1429.89  max = 1462.62  avg = 1446.97
            resnet50  min =  477.94  max =  556.38  avg =  504.19
       resnet50_int8  min =  588.24  max =  643.33  avg =  602.23
      squeezenet_ssd  min =  193.36  max =  201.49  avg =  197.23
 squeezenet_ssd_int8  min =  249.88  max =  276.11  avg =  259.98
       mobilenet_ssd  min =  214.84  max =  220.13  avg =  217.64
  mobilenet_ssd_int8  min =  239.38  max =  264.35  avg =  246.79
      mobilenet_yolo  min =  472.71  max =  501.46  avg =  480.24
    mobilenet_yolov3  min =  481.33  max =  492.76  avg =  488.16

HM2014812:/data/local/tmp # ./benchncnn 4 1 0 -1
no vulkan device
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =  197.65  max =  202.06  avg =  198.80
     squeezenet_int8  min =  202.48  max =  202.95  avg =  202.65
           mobilenet  min =  295.04  max =  296.91  avg =  296.10
      mobilenet_int8  min =  361.93  max =  363.94  avg =  362.91
        mobilenet_v2  min =  226.53  max =  228.70  avg =  227.61
          shufflenet  min =  110.78  max =  113.33  avg =  112.25
             mnasnet  min =  208.05  max =  209.45  avg =  208.66
     proxylessnasnet  min =  258.52  max =  259.84  avg =  259.13
           googlenet  min =  794.38  max =  802.06  avg =  798.95
      googlenet_int8  min =  719.79  max =  728.64  avg =  724.49
            resnet18  min =  658.66  max =  667.04  avg =  662.81
       resnet18_int8  min =  633.08  max =  636.60  avg =  634.24
             alexnet  min =  920.75  max =  923.41  avg =  922.01
               vgg16  min = 3721.14  max = 3762.16  avg = 3739.58
          vgg16_int8  min = 3625.32  max = 3634.06  avg = 3628.52
            resnet50  min = 1467.75  max = 1478.21  avg = 1471.91
       resnet50_int8  min = 1501.08  max = 1506.64  avg = 1503.80
      squeezenet_ssd  min =  461.66  max =  465.73  avg =  464.46
 squeezenet_ssd_int8  min =  527.91  max =  532.13  avg =  529.92
       mobilenet_ssd  min =  596.02  max =  597.51  avg =  596.64
  mobilenet_ssd_int8  min =  701.93  max =  705.44  avg =  703.57
      mobilenet_yolo  min = 1340.58  max = 1345.12  avg = 1342.62
    mobilenet_yolov3  min = 1410.31  max = 1419.52  avg = 1415.60
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
rk3399_firefly_box:/data/local/tmp/ncnn # ./benchncnn 8 6 0 
loop_count = 8
num_threads = 2
powersave = 2
gpu_device = -1
          squeezenet  min =   51.60  max =   53.03  avg =   52.08
     squeezenet-int8  min =   62.11  max =   64.91  avg =   63.27
           mobilenet  min =   77.18  max =   79.11  avg =   77.75
      mobilenet-int8  min =   55.02  max =   59.89  avg =   57.44
        mobilenet_v2  min =   84.48  max =   85.52  avg =   85.13
          shufflenet  min =   36.04  max =   38.82  avg =   37.05
             mnasnet  min =   61.31  max =   62.40  avg =   61.77
     proxylessnasnet  min =   72.75  max =   73.53  avg =   73.12
           googlenet  min =  184.77  max =  191.39  avg =  187.33
      googlenet-int8  min =  186.17  max =  192.47  avg =  187.60
            resnet18  min =  204.00  max =  208.83  avg =  206.40
       resnet18-int8  min =  199.17  max =  208.99  avg =  201.45
             alexnet  min =  165.20  max =  176.01  avg =  168.54
               vgg16  min =  853.25  max =  894.94  avg =  875.34
            resnet50  min =  646.85  max =  665.59  avg =  654.79
       resnet50-int8  min =  399.45  max =  413.88  avg =  403.38
      squeezenet-ssd  min =  138.18  max =  150.14  avg =  143.34
 squeezenet-ssd-int8  min =  194.38  max =  202.95  avg =  196.60
       mobilenet-ssd  min =  156.85  max =  162.62  avg =  158.98
  mobilenet-ssd-int8  min =  119.37  max =  127.61  avg =  123.43
      mobilenet-yolo  min =  412.08  max =  445.90  avg =  423.03
    mobilenet-yolov3  min =  370.11  max =  390.11  avg =  376.91
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
iPhone:~ root# ./benchncnn 8 2 0
[0 Apple A7 GPU]  queueC=0[8]  queueT=0[8]  memU=1  memDL=1  memHV=1
[0 Apple A7 GPU]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
          squeezenet  min =   68.21  max =   72.00  avg =   70.36
     squeezenet_int8  min =   56.31  max =   58.27  avg =   57.04
           mobilenet  min =   85.74  max =   86.52  avg =   86.03
      mobilenet_int8  min =  111.06  max =  114.07  avg =  113.09
        mobilenet_v2  min =   68.72  max =   69.84  avg =   69.36
          shufflenet  min =   35.26  max =   36.54  avg =   35.77
             mnasnet  min =   68.63  max =   70.57  avg =   69.51
     proxylessnasnet  min =   92.44  max =   93.78  avg =   93.41
           googlenet  min =  280.98  max =  290.75  avg =  286.56
      googlenet_int8  min =  238.81  max =  270.71  avg =  246.85
            resnet18  min =  251.99  max =  260.40  avg =  255.23
       resnet18_int8  min =  179.41  max =  208.97  avg =  187.22
             alexnet  min =  329.07  max =  337.75  avg =  333.24
               vgg16  min = 4547.25  max = 4706.56  avg = 4647.60
          vgg16_int8  min = 3516.66  max = 3598.62  avg = 3546.62
            resnet50  min = 2657.13  max = 2710.55  avg = 2689.35
       resnet50_int8  min =  442.35  max =  596.75  avg =  464.38
      squeezenet_ssd  min =  180.00  max =  198.60  avg =  185.11
 squeezenet_ssd_int8  min =  155.91  max =  159.64  avg =  158.08
       mobilenet_ssd  min =  171.14  max =  172.65  avg =  172.05
  mobilenet_ssd_int8  min =  207.76  max =  211.34  avg =  209.93
      mobilenet_yolo  min =  379.55  max =  389.24  avg =  384.13
    mobilenet_yolov3  min =  410.48  max =  416.43  avg =  414.26

iPhone:~ root# ./benchncnn 4 1 0 0
[0 Apple A7 GPU]  queueC=0[8]  queueT=0[8]  memU=1  memDL=1  memHV=1
[0 Apple A7 GPU]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = 0
          squeezenet  min =  257.60  max =  260.76  avg =  259.57
           mobilenet  min =  288.68  max =  328.62  avg =  299.17
        mobilenet_v2  min =  263.82  max =  265.67  avg =  264.85
          shufflenet  min =  237.64  max =  238.88  avg =  238.13
             mnasnet  min =  255.72  max =  258.46  avg =  256.67
     proxylessnasnet  min =  280.92  max =  281.34  avg =  281.07
           googlenet  min =  749.29  max =  763.25  avg =  756.65
            resnet18  min =  731.45  max =  744.19  avg =  738.51
             alexnet  min =  522.82  max =  543.89  avg =  531.66
               vgg16  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
            resnet50  min = 1479.13  max = 1495.76  avg = 1486.67
      squeezenet_ssd  min = 1094.71  max = 1115.38  avg = 1100.96
       mobilenet_ssd  min =  638.81  max =  644.79  avg =  642.82
      mobilenet_yolo  min = 1365.58  max = 1374.82  avg = 1371.34
    mobilenet_yolov3  min = 1319.51  max = 1332.27  avg = 1325.04
```

Freescale i.MX7 Dual (Cortex A7 1.0GHz x 2)
```
imx7d_pico:/data/local/tmp # ./benchncnn 8 2 0 -1
no vulkan device
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
          squeezenet  min =  257.01  max =  264.40  avg =  259.36
     squeezenet_int8  min =  184.81  max =  193.88  avg =  188.20
           mobilenet  min =  412.64  max =  420.48  avg =  417.13
      mobilenet_int8  min =  294.15  max =  303.00  avg =  300.12
        mobilenet_v2  min =  319.58  max =  330.20  avg =  325.12
          shufflenet  min =  154.84  max =  172.62  avg =  159.53
             mnasnet  min =  281.86  max =  294.58  avg =  288.04
     proxylessnasnet  min =  341.37  max =  356.94  avg =  345.23
           googlenet  min =  963.52  max =  976.01  avg =  969.66
      googlenet_int8  min =  636.54  max =  649.23  avg =  643.85
            resnet18  min =  936.32  max = 1019.67  avg =  951.71
       resnet18_int8  min =  569.95  max =  576.27  avg =  573.34
             alexnet  min = 1238.50  max = 1345.29  avg = 1256.16
               vgg16  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
          vgg16_int8  min = 3984.30  max = 4080.91  avg = 4035.83
            resnet50  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
       resnet50_int8  min = 1342.88  max = 1431.57  avg = 1367.53
      squeezenet_ssd  min =  603.41  max =  616.67  avg =  606.44
 squeezenet_ssd_int8  min =  519.09  max =  528.18  avg =  523.99
       mobilenet_ssd  min =  826.06  max =  841.06  avg =  832.99
  mobilenet_ssd_int8  min =  549.39  max =  555.53  avg =  551.53
      mobilenet_yolo  min = 1905.68  max = 2090.86  avg = 1939.94
    mobilenet_yolov3  min = 1996.95  max = 2008.81  avg = 2001.69

imx7d_pico:/data/local/tmp # ./benchncnn 4 1 0 -1
no vulkan device
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =  477.31  max =  479.17  avg =  478.35
     squeezenet_int8  min =  339.14  max =  342.23  avg =  340.94
           mobilenet  min =  788.13  max =  790.99  avg =  789.61
      mobilenet_int8  min =  557.75  max =  560.75  avg =  559.57
        mobilenet_v2  min =  588.62  max =  598.17  avg =  594.10
          shufflenet  min =  265.33  max =  268.28  avg =  266.20
             mnasnet  min =  526.77  max =  541.61  avg =  533.76
     proxylessnasnet  min =  627.94  max =  629.29  avg =  628.61
           googlenet  min = 1824.24  max = 1867.65  avg = 1841.26
      googlenet_int8  min = 1167.88  max = 1169.38  avg = 1168.41
            resnet18  min = 1793.57  max = 1830.01  avg = 1803.27
       resnet18_int8  min = 1005.40  max = 1005.92  avg = 1005.68
             alexnet  min = 2446.74  max = 2452.12  avg = 2449.54
               vgg16  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
          vgg16_int8  min = 6743.75  max = 6838.15  avg = 6787.64
            resnet50  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
       resnet50_int8  min = 2438.62  max = 2459.22  avg = 2448.26
      squeezenet_ssd  min = 1079.45  max = 1082.11  avg = 1080.79
 squeezenet_ssd_int8  min =  871.46  max =  903.80  avg =  881.71
       mobilenet_ssd  min = 1573.47  max = 1576.09  avg = 1574.97
  mobilenet_ssd_int8  min = 1025.44  max = 1026.73  avg = 1026.31
      mobilenet_yolo  min = 3647.39  max = 3685.94  avg = 3670.43
    mobilenet_yolov3  min = 3833.90  max = 3838.05  avg = 3835.67
```
