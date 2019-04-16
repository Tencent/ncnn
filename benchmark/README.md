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

```shell
chiron:/data/local/tmp/ncnn $ ./benchncnn 8 4 2
loop_count = 8                                                   
num_threads = 4                                                  
powersave = 2                                                    
gpu_device = -1                                            
          squeezenet  min =   38.51  max =   39.25  avg =   38.94
     squeezenet-int8  min =   30.54  max =   31.11  avg =   30.80
           mobilenet  min =   67.14  max =   68.90  avg =   68.00
      mobilenet-int8  min =   27.85  max =   28.17  avg =   28.01
        mobilenet_v2  min =   52.23  max =  251.20  avg =   77.76
          shufflenet  min =   37.54  max =   38.52  avg =   37.99
             mnasnet  min =   51.02  max =   51.70  avg =   51.37
     proxylessnasnet  min =   29.59  max =   30.37  avg =   29.90
           googlenet  min =   84.33  max =   84.84  avg =   84.56
      googlenet-int8  min =  116.08  max =  116.58  avg =  116.37
            resnet18  min =  136.64  max =  337.54  avg =  168.20
       resnet18-int8  min =  113.72  max =  114.97  avg =  114.25
             alexnet  min =  205.68  max =  358.51  avg =  234.93
               vgg16  min =  709.03  max =  887.07  avg =  763.75
            resnet50  min =  609.71  max =  840.24  avg =  688.00
       resnet50-int8  min =  280.26  max =  479.41  avg =  306.38
      squeezenet-ssd  min =  119.71  max =  122.11  avg =  120.90
 squeezenet-ssd-int8  min =  103.92  max =  303.06  avg =  130.58
       mobilenet-ssd  min =  141.68  max =  342.35  avg =  167.57
  mobilenet-ssd-int8  min =  110.75  max =  114.24  avg =  111.66
      mobilenet-yolo  min =  324.39  max =  523.89  avg =  375.21
    mobilenet-yolov3  min =  167.07  max =  170.77  avg =  168.55

chiron:/data/local/tmp/ncnn $ ./benchncnn 8 1 2
loop_count = 8                                                                                                                
num_threads = 1
powersave = 2
gpu_device = -1
          squeezenet  min =   65.12  max =   66.80  avg =   66.11
     squeezenet-int8  min =   51.58  max =   52.14  avg =   51.75
           mobilenet  min =  112.89  max =  114.90  avg =  113.68
      mobilenet-int8  min =   92.25  max =   94.40  avg =   93.34
        mobilenet_v2  min =   82.48  max =   83.63  avg =   82.94
          shufflenet  min =   36.13  max =   36.97  avg =   36.52
             mnasnet  min =   74.46  max =   75.34  avg =   74.87
     proxylessnasnet  min =   88.85  max =   89.50  avg =   89.12
           googlenet  min =  271.41  max =  274.74  avg =  273.31
      googlenet-int8  min =  201.53  max =  203.06  avg =  202.23
            resnet18  min =  237.97  max =  240.65  avg =  239.17
       resnet18-int8  min =  159.75  max =  161.64  avg =  160.36
             alexnet  min =  365.06  max =  366.67  avg =  365.91
               vgg16  min = 1314.94  max = 1318.95  avg = 1316.50
            resnet50  min = 1106.64  max = 1115.68  avg = 1110.61
       resnet50-int8  min =  400.05  max =  405.72  avg =  402.34
      squeezenet-ssd  min =  150.17  max =  151.11  avg =  150.61
 squeezenet-ssd-int8  min =  124.49  max =  125.98  avg =  125.31
       mobilenet-ssd  min =  228.97  max =  231.75  avg =  230.97
  mobilenet-ssd-int8  min =  183.39  max =  185.90  avg =  184.85
      mobilenet-yolo  min =  529.68  max =  540.27  avg =  534.55
    mobilenet-yolov3  min =  522.93  max =  531.25  avg =  525.13
            
chiron:/data/local/tmp/ncnn $ ./benchncnn 8 4 1
loop_count = 8                                                                                                                 
num_threads = 4
powersave = 1
gpu_device = -1
          squeezenet  min =   40.29  max =   40.66  avg =   40.46
     squeezenet-int8  min =   40.61  max =   50.62  avg =   42.08
           mobilenet  min =   50.78  max =   51.42  avg =   51.00
      mobilenet-int8  min =   62.09  max =   62.55  avg =   62.32
        mobilenet_v2  min =   52.83  max =   53.13  avg =   52.93
          shufflenet  min =   29.74  max =   29.97  avg =   29.84
             mnasnet  min =   43.11  max =   43.62  avg =   43.37
     proxylessnasnet  min =   50.32  max =   50.73  avg =   50.47
           googlenet  min =  140.51  max =  141.20  avg =  140.90
      googlenet-int8  min =  140.47  max =  140.91  avg =  140.67
            resnet18  min =  141.60  max =  144.12  avg =  142.42
       resnet18-int8  min =  130.68  max =  131.50  avg =  130.95
             alexnet  min =  151.18  max =  153.49  avg =  152.28
               vgg16  min =  674.35  max =  696.00  avg =  682.37
            resnet50  min =  575.88  max =  584.70  avg =  580.19
       resnet50-int8  min =  297.85  max =  298.65  avg =  298.18
      squeezenet-ssd  min =  106.52  max =  108.75  avg =  107.35
 squeezenet-ssd-int8  min =  119.45  max =  121.27  avg =  120.03
       mobilenet-ssd  min =  109.81  max =  114.22  avg =  110.72
  mobilenet-ssd-int8  min =  118.19  max =  119.55  avg =  118.65
      mobilenet-yolo  min =  266.90  max =  268.14  avg =  267.51
    mobilenet-yolov3  min =  245.19  max =  247.02  avg =  245.71

chiron:/data/local/tmp/ncnn $ ./benchncnn 8 1 1
loop_count = 8
num_threads = 1
powersave = 1
gpu_device = -1
          squeezenet  min =  132.23  max =  133.57  avg =  132.55
     squeezenet-int8  min =  125.51  max =  131.38  avg =  128.10
           mobilenet  min =  187.55  max =  190.19  avg =  188.53
      mobilenet-int8  min =  233.87  max =  242.32  avg =  238.71
        mobilenet_v2  min =  167.12  max =  168.25  avg =  167.58
          shufflenet  min =   83.73  max =   84.38  avg =   84.04
             mnasnet  min =  142.07  max =  143.37  avg =  142.78
     proxylessnasnet  min =  173.35  max =  177.28  avg =  174.93
           googlenet  min =  525.40  max =  534.24  avg =  530.01
      googlenet-int8  min =  470.90  max =  479.06  avg =  473.84
            resnet18  min =  502.06  max =  507.12  avg =  505.31
       resnet18-int8  min =  425.31  max =  439.68  avg =  434.78
             alexnet  min =  601.74  max =  606.15  avg =  603.52
               vgg16  min = 2565.99  max = 2589.02  avg = 2573.64
            resnet50  min = 2237.39  max = 2281.13  avg = 2261.25
       resnet50-int8  min = 1000.02  max = 1011.14  avg = 1006.33
      squeezenet-ssd  min =  323.94  max =  327.63  avg =  325.21
 squeezenet-ssd-int8  min =  337.32  max =  341.70  avg =  339.42
       mobilenet-ssd  min =  398.51  max =  401.32  avg =  399.86
  mobilenet-ssd-int8  min =  451.12  max =  462.66  avg =  454.15
      mobilenet-yolo  min =  922.58  max =  924.53  avg =  923.63
    mobilenet-yolov3  min =  897.05  max =  908.41  avg =  901.31
    
chiron:/data/local/tmp/ncnn $ ./benchncnn 8 1 1 0                
[0 Adreno (TM) 540]  queueC=0  queueT=0  memU=2  memDL=2  memHV=2
[0 Adreno (TM) 540]  fp16s=0  fp16a=0  int8s=0  int8a=0          
loop_count = 8                                                   
num_threads = 1                                                  
powersave = 1           
gpu_device = 0                                                   
          squeezenet  min =   99.87  max =  100.37  avg =  100.18
           mobilenet  min =  152.85  max =  154.95  avg =  153.70
        mobilenet_v2  min =  105.75  max =  106.83  avg =  106.37
          shufflenet  min =   51.94  max =   53.05  avg =   52.46
             mnasnet  min =   93.11  max =   94.35  avg =   93.76
     proxylessnasnet  min =   94.90  max =   99.87  avg =   98.62
           googlenet  min =  400.21  max =  403.04  avg =  401.75
            resnet18  min =  481.49  max =  487.42  avg =  485.33
             alexnet  min =  610.22  max =  616.31  avg =  613.54
###############################################################################
# below models have problems with vkQueueSubmit and  vkWaitForFences failed ###
###############################################################################
vkQueueSubmit failed -3 
vkWaitForFences failed 2
               vgg16  min =    1.16  max =    1.51  avg =    1.37
            resnet50  min =    5.75  max =    8.85  avg =    6.84
      squeezenet-ssd  min =   18.65  max =   19.15  avg =   18.88
       mobilenet-ssd  min =    7.05  max =    8.74  avg =    7.64
      mobilenet-yolo  min =   11.29  max =   12.74  avg =   11.64
    mobilenet-yolov3  min =    7.47  max =    9.95  avg =    8.25
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
angler:/data/local/tmp $ ./benchncnn 8 8 0
loop_count = 8
num_threads = 8
powersave = 0
      squeezenet  min =   35.57  max =   36.56  avg =   36.13
       mobilenet  min =   44.80  max =   56.80  avg =   47.91
    mobilenet_v2  min =   46.80  max =   64.64  avg =   50.34
      shufflenet  min =   28.24  max =   30.27  avg =   29.36
       googlenet  min =  118.82  max =  132.80  avg =  123.74
        resnet18  min =  119.55  max =  141.99  avg =  126.78
         alexnet  min =  104.52  max =  125.98  avg =  110.17
           vgg16  min =  815.12  max =  930.98  avg =  878.57
  squeezenet-ssd  min =  111.05  max =  130.23  avg =  119.43
   mobilenet-ssd  min =   88.88  max =  108.96  avg =   98.38
  mobilenet-yolo  min =  220.57  max =  263.42  avg =  241.03
```

Qualcomm MSM8916 Snapdragon 410 (Cortex-A53 1.2GHz x 4)
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   79.70  max =   85.42  avg =   82.22
       mobilenet  min =  119.87  max =  125.63  avg =  123.46
    mobilenet_v2  min =  125.65  max =  131.16  avg =  128.20
      shufflenet  min =   60.95  max =   66.03  avg =   63.03
       googlenet  min =  237.47  max =  256.79  avg =  245.65
        resnet18  min =  239.73  max =  250.41  avg =  245.87
         alexnet  min =  248.66  max =  279.08  avg =  267.41
           vgg16  min = 1429.50  max = 1510.46  avg = 1465.25
  squeezenet-ssd  min =  203.33  max =  213.85  avg =  209.81
   mobilenet-ssd  min =  215.26  max =  224.23  avg =  219.73
  mobilenet-yolo  min =  506.41  max =  520.50  avg =  513.30
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
loop_count = 8
num_threads = 2
powersave = 0
      squeezenet  min =   70.94  max =   72.40  avg =   71.75
       mobilenet  min =   89.24  max =   92.21  avg =   90.60
    mobilenet_v2  min =   71.70  max =   74.43  avg =   73.68
      shufflenet  min =   35.48  max =   41.40  avg =   38.94
       googlenet  min =  282.76  max =  295.00  avg =  289.64
        resnet18  min =  251.99  max =  260.40  avg =  255.23
         alexnet  min =  329.07  max =  337.75  avg =  333.24
           vgg16  min = 4547.25  max = 4706.56  avg = 4647.60
  squeezenet-ssd  min =  171.23  max =  180.49  avg =  175.54
   mobilenet-ssd  min =  174.56  max =  192.69  avg =  179.60
  mobilenet-yolo  min =  357.90  max =  363.93  avg =  360.97
```

Freescale i.MX7 Dual (Cortex A7 1.0GHz x 2)
```
imx7d_pico:/data/local/tmp # ./benchncnn 8 2 0
loop_count = 8
num_threads = 2
powersave = 0
      squeezenet  min =  269.26  max =  278.84  avg =  273.10
       mobilenet  min =  442.79  max =  445.82  avg =  444.46
    mobilenet_v2  min =  362.19  max =  364.58  avg =  363.33
      shufflenet  min =  171.30  max =  190.63  avg =  177.52
       googlenet  min =  975.95  max =  986.11  avg =  980.51
        resnet18  min = 1016.60  max = 1035.50  avg = 1021.75
         alexnet  min = 1240.54  max = 1254.86  avg = 1247.18
           vgg16  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
  squeezenet-ssd  min =  614.93  max =  623.15  avg =  619.56
   mobilenet-ssd  min =  842.83  max =  884.64  avg =  855.40
  mobilenet-yolo  min = 1772.24  max = 1924.37  avg = 1805.75
```
