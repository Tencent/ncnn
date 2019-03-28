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
