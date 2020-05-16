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
$ ./benchncnn [loop count] [num threads] [powersave] [gpu device] [cooling down]
```
run benchncnn on android device
```
# for running on android device, upload to /data/local/tmp/ folder
$ adb push benchncnn /data/local/tmp/
$ adb push <ncnn-root-dir>/benchmark/*.param /data/local/tmp/
$ adb shell

# executed in android adb shell
$ cd /data/local/tmp/
$ ./benchncnn [loop count] [num threads] [powersave] [gpu device] [cooling down]
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
          squeezenet  min =   30.04  max =   31.50  avg =   30.81
     squeezenet_int8  min =   34.17  max =   39.91  avg =   34.98
           mobilenet  min =   35.53  max =   36.84  avg =   36.27
      mobilenet_int8  min =   43.11  max =   46.56  avg =   44.07
        mobilenet_v2  min =   35.87  max =   37.17  avg =   36.52
        mobilenet_v3  min =   32.02  max =   38.50  avg =   33.74
          shufflenet  min =   27.51  max =   35.04  avg =   29.05
       shufflenet_v2  min =   22.20  max =   32.48  avg =   23.93
             mnasnet  min =   31.56  max =   37.65  avg =   33.24
     proxylessnasnet  min =   35.75  max =   37.67  avg =   36.44
           googlenet  min =   91.70  max =  106.84  avg =   96.50
      googlenet_int8  min =  116.48  max =  133.12  avg =  125.38
            resnet18  min =   86.73  max =   96.19  avg =   91.24
       resnet18_int8  min =  103.79  max =  112.57  avg =  108.28
             alexnet  min =  103.67  max =  111.48  avg =  107.44
               vgg16  min =  593.11  max =  695.22  avg =  635.23
          vgg16_int8  min =  941.80  max = 1130.39  avg = 1020.08
            resnet50  min =  261.01  max =  364.68  avg =  306.03
       resnet50_int8  min =  275.98  max =  342.96  avg =  310.11
      squeezenet_ssd  min =   98.26  max =  127.12  avg =  111.21
 squeezenet_ssd_int8  min =  106.40  max =  131.56  avg =  115.15
       mobilenet_ssd  min =   88.25  max =  111.04  avg =  100.95
  mobilenet_ssd_int8  min =   89.84  max =  101.60  avg =   95.36
      mobilenet_yolo  min =  196.11  max =  299.58  avg =  233.38
  mobilenetv2_yolov3  min =  124.33  max =  141.38  avg =  131.51

angler:/data/local/tmp $ ./benchncnn 8 1 2 -1
[0 Adreno (TM) 430]  queueC=0[3]  queueT=0[3]  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 430]  fp16p=1  fp16s=0  fp16a=0  int8s=0  int8a=0
loop_count = 8
num_threads = 1
powersave = 2
gpu_device = -1
          squeezenet  min =   73.26  max =   73.37  avg =   73.31
     squeezenet_int8  min =   81.18  max =   84.50  avg =   82.63
           mobilenet  min =  112.50  max =  112.99  avg =  112.78
      mobilenet_int8  min =  137.53  max =  138.11  avg =  137.79
        mobilenet_v2  min =   78.43  max =   79.34  avg =   78.76
        mobilenet_v3  min =   67.52  max =   68.00  avg =   67.68
          shufflenet  min =   44.35  max =   45.14  avg =   44.74
       shufflenet_v2  min =   39.14  max =   39.82  avg =   39.63
             mnasnet  min =   75.36  max =   75.49  avg =   75.41
     proxylessnasnet  min =   95.58  max =   95.77  avg =   95.67
           googlenet  min =  248.28  max =  249.88  avg =  248.74
      googlenet_int8  min =  304.50  max =  307.04  avg =  306.00
            resnet18  min =  217.78  max =  219.64  avg =  218.92
       resnet18_int8  min =  251.31  max =  262.02  avg =  256.88
             alexnet  min =  270.77  max =  286.14  avg =  278.56
               vgg16  min = 1084.56  max = 1097.28  avg = 1089.98
          vgg16_int8  min = 1552.85  max = 1555.78  avg = 1554.47
            resnet50  min =  529.62  max =  534.36  avg =  531.10
       resnet50_int8  min =  585.55  max =  589.21  avg =  587.50
      squeezenet_ssd  min =  168.71  max =  170.17  avg =  169.45
 squeezenet_ssd_int8  min =  214.35  max =  217.70  avg =  216.67
       mobilenet_ssd  min =  227.44  max =  233.00  avg =  230.49
  mobilenet_ssd_int8  min =  274.47  max =  275.08  avg =  274.87
      mobilenet_yolo  min =  513.80  max =  515.88  avg =  514.96
  mobilenetv2_yolov3  min =  265.95  max =  268.74  avg =  267.71

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
          squeezenet  min =   66.28  max =   71.76  avg =   68.42
     squeezenet_int8  min =   80.27  max =   86.95  avg =   84.41
           mobilenet  min =   87.53  max =   93.44  avg =   91.11
      mobilenet_int8  min =  121.72  max =  134.20  avg =  128.89
        mobilenet_v2  min =   78.41  max =   84.16  avg =   80.65
        mobilenet_v3  min =   63.52  max =   68.81  avg =   65.06
          shufflenet  min =   50.21  max =   55.31  avg =   51.05
       shufflenet_v2  min =   41.84  max =   47.19  avg =   43.29
             mnasnet  min =   69.69  max =   75.25  avg =   71.82
     proxylessnasnet  min =   78.53  max =   83.97  avg =   80.69
           googlenet  min =  186.86  max =  194.13  avg =  191.75
      googlenet_int8  min =  254.33  max =  282.34  avg =  268.64
            resnet18  min =  162.89  max =  176.29  avg =  168.70
       resnet18_int8  min =  221.94  max =  233.40  avg =  228.47
             alexnet  min =  136.68  max =  147.43  avg =  141.04
               vgg16  min =  820.71  max = 1179.12  avg =  935.40
          vgg16_int8  min = 1489.99  max = 1728.10  avg = 1557.09
            resnet50  min =  417.40  max =  422.28  avg =  419.54
       resnet50_int8  min =  526.23  max =  556.69  avg =  540.22
      squeezenet_ssd  min =  176.31  max =  187.60  avg =  182.31
 squeezenet_ssd_int8  min =  238.51  max =  249.05  avg =  243.61
       mobilenet_ssd  min =  188.66  max =  197.45  avg =  193.79
  mobilenet_ssd_int8  min =  247.35  max =  269.80  avg =  253.90
      mobilenet_yolo  min =  395.71  max =  407.36  avg =  401.46
  mobilenetv2_yolov3  min =  250.01  max =  261.79  avg =  256.15

HM2014812:/data/local/tmp # ./benchncnn 4 1 0 -1
no vulkan device
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =  152.92  max =  155.51  avg =  153.70
     squeezenet_int8  min =  201.83  max =  205.87  avg =  203.62
           mobilenet  min =  242.75  max =  243.80  avg =  243.08
      mobilenet_int8  min =  360.94  max =  363.18  avg =  361.70
        mobilenet_v2  min =  169.94  max =  171.54  avg =  170.60
        mobilenet_v3  min =  148.18  max =  149.00  avg =  148.47
          shufflenet  min =   99.96  max =  100.62  avg =  100.32
       shufflenet_v2  min =   86.17  max =   87.17  avg =   86.68
             mnasnet  min =  163.09  max =  163.60  avg =  163.30
     proxylessnasnet  min =  208.37  max =  208.83  avg =  208.63
           googlenet  min =  550.35  max =  558.00  avg =  552.59
      googlenet_int8  min =  716.89  max =  729.11  avg =  723.14
            resnet18  min =  499.56  max =  500.78  avg =  499.96
       resnet18_int8  min =  614.61  max =  621.81  avg =  617.39
             alexnet  min =  485.58  max =  486.32  avg =  486.06
               vgg16  min = 2218.44  max = 2267.49  avg = 2239.35
          vgg16_int8  min = 3655.84  max = 3663.73  avg = 3659.39
            resnet50  min = 1220.40  max = 1227.92  avg = 1223.96
       resnet50_int8  min = 1449.73  max = 1452.61  avg = 1451.31
      squeezenet_ssd  min =  358.87  max =  361.47  avg =  360.33
 squeezenet_ssd_int8  min =  535.02  max =  538.82  avg =  536.98
       mobilenet_ssd  min =  523.74  max =  528.69  avg =  525.57
  mobilenet_ssd_int8  min =  713.93  max =  716.41  avg =  714.92
      mobilenet_yolo  min = 1130.12  max = 1135.57  avg = 1132.88
  mobilenetv2_yolov3  min =  603.32  max =  606.46  avg =  604.85
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
Raspberry Pi 4 Model B Broadcom BCM2711B0, Cortex-A72 (ARMv8) (1.5GHz x 4 )
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

Freescale i.MX7 Dual (Cortex A7 1.0GHz x 2)
```
imx7d_pico:/data/local/tmp $ ./benchncnn 8 2 0 -1
no vulkan device
loop_count = 8
num_threads = 2
powersave = 0
gpu_device = -1
          squeezenet  min =  227.22  max =  240.96  avg =  233.13
     squeezenet_int8  min =  184.70  max =  194.19  avg =  189.49
           mobilenet  min =  371.00  max =  379.42  avg =  376.17
      mobilenet_int8  min =  296.94  max =  307.36  avg =  303.22
        mobilenet_v2  min =  250.47  max =  261.67  avg =  255.94
        mobilenet_v3  min =  219.98  max =  229.24  avg =  223.64
          shufflenet  min =  141.67  max =  151.36  avg =  144.48
       shufflenet_v2  min =  139.62  max =  163.04  avg =  144.63
             mnasnet  min =  247.99  max =  260.33  avg =  252.61
     proxylessnasnet  min =  281.56  max =  297.47  avg =  289.67
           googlenet  min =  769.66  max =  791.01  avg =  779.46
      googlenet_int8  min =  656.84  max =  670.71  avg =  662.36
            resnet18  min =  806.68  max =  827.56  avg =  819.04
       resnet18_int8  min =  567.16  max =  575.90  avg =  570.64
             alexnet  min =  840.61  max =  908.73  avg =  855.15
      squeezenet_ssd  min =  519.63  max =  535.01  avg =  525.83
 squeezenet_ssd_int8  min =  515.51  max =  526.91  avg =  520.61
       mobilenet_ssd  min =  773.45  max =  784.11  avg =  779.29
  mobilenet_ssd_int8  min =  563.08  max =  570.83  avg =  565.40
      mobilenet_yolo  min = 1747.08  max = 1770.13  avg = 1758.12
  mobilenetv2_yolov3  min =  871.39  max =  884.43  avg =  877.60

imx7d_pico:/data/local/tmp $ ./benchncnn 4 1 0 -1
no vulkan device
loop_count = 4
num_threads = 1
powersave = 0
gpu_device = -1
          squeezenet  min =  431.89  max =  434.21  avg =  433.57
     squeezenet_int8  min =  342.41  max =  344.21  avg =  343.06
           mobilenet  min =  726.55  max =  727.96  avg =  727.51
      mobilenet_int8  min =  566.52  max =  567.63  avg =  566.91
        mobilenet_v2  min =  479.92  max =  482.83  avg =  481.06
        mobilenet_v3  min =  424.95  max =  427.36  avg =  425.91
          shufflenet  min =  246.83  max =  248.29  avg =  247.54
       shufflenet_v2  min =  244.47  max =  246.13  avg =  245.18
             mnasnet  min =  475.35  max =  475.83  avg =  475.66
     proxylessnasnet  min =  547.79  max =  564.61  avg =  552.22
           googlenet  min = 1452.86  max = 1457.48  avg = 1454.76
      googlenet_int8  min = 1192.39  max = 1214.75  avg = 1201.63
            resnet18  min = 1522.25  max = 1659.13  avg = 1563.40
       resnet18_int8  min =  992.90  max = 1001.80  avg =  995.79
             alexnet  min = 1620.82  max = 1626.94  avg = 1623.96
      squeezenet_ssd  min =  919.51  max =  922.87  avg =  921.18
 squeezenet_ssd_int8  min =  854.49  max =  879.17  avg =  864.31
       mobilenet_ssd  min = 1475.04  max = 1488.65  avg = 1478.95
  mobilenet_ssd_int8  min = 1040.01  max = 1041.69  avg = 1040.91
      mobilenet_yolo  min = 3413.03  max = 3423.75  avg = 3418.63
  mobilenetv2_yolov3  min = 1640.18  max = 1661.04  avg = 1652.19
```

nVIDIA RTX2060 of Notebook
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

nVIDIA RTX2080 of Desktop
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
