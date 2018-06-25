benchncnn can be used to test neural network inference performance

Only the network definition files (ncnn param) are required.

The large model binary files (ncnn bin) are not loaded but generated randomly for speed test.

More model networks may be added later.

---

Usage
```
# copy all param files to the current directory
./benchncnn [loop count] [num threads] [powersave]
```

Typical output (executed in android adb shell)

Qualcomm MSM8996 Snapdragon 820 (Kyro 2.15GHz x 2 + Kyro 1.6GHz x 2)
```
root@msm8996:/data/local/tmp/ncnn # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   34.99  max =   35.49  avg =   35.21
       mobilenet  min =   55.14  max =   66.88  avg =   56.87
    mobilenet_v2  min =   42.42  max =   43.28  avg =   42.94
      shufflenet  min =   24.02  max =   25.37  avg =   24.78
       googlenet  min =  116.84  max =  134.88  avg =  123.13
        resnet18  min =  113.23  max =  119.52  avg =  114.85
         alexnet  min =  186.85  max =  207.04  avg =  193.54
           vgg16  min =  616.64  max =  635.01  avg =  627.31
  squeezenet-ssd  min =   83.07  max =   97.50  avg =   86.36
   mobilenet-ssd  min =  105.19  max =  123.38  avg =  109.45
```

Qualcomm MSM8994 Snapdragon 810 (Cortex-A57 2.0GHz x 4 + Cortex-A53 1.55GHz x 4)
```
angler:/data/local/tmp $ ./benchncnn 8 8 0
loop_count = 8
num_threads = 8
powersave = 0
      squeezenet  min =   57.82  max =   60.46  avg =   58.79
       mobilenet  min =   62.17  max =   65.12  avg =   62.97
    mobilenet_v2  min =   73.92  max =   77.14  avg =   74.54
      shufflenet  min =   50.36  max =   51.03  avg =   50.65
       googlenet  min =  163.27  max =  190.46  avg =  175.74
        resnet18  min =  167.31  max =  191.60  avg =  180.75
         alexnet  min =  113.60  max =  124.58  avg =  117.97
           vgg16  min =  940.28  max = 1050.45  avg = 1004.43
  squeezenet-ssd  min =  169.12  max =  200.12  avg =  184.83
   mobilenet-ssd  min =  122.38  max =  165.38  avg =  142.36
```

Qualcomm MSM8916 Snapdragon 410 (Cortex-A53 1.2GHz x 4)
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   80.16  max =   97.73  avg =   89.13
       mobilenet  min =  131.53  max =  151.08  avg =  140.42
    mobilenet_v2  min =  132.14  max =  162.10  avg =  146.50
      shufflenet  min =   57.88  max =   65.62  avg =   61.16
       googlenet  min =  236.11  max =  248.62  avg =  244.74
        resnet18  min =  271.75  max =  291.05  avg =  282.32
         alexnet  min =  278.70  max =  296.65  avg =  285.61
           vgg16  min = 1485.95  max = 1523.02  avg = 1504.54
  squeezenet-ssd  min =  204.62  max =  222.92  avg =  212.41
   mobilenet-ssd  min =  229.69  max =  236.50  avg =  233.48
```

Rockchip RK3399 (Cortex-A72 1.8GHz x 2 + Cortex-A53 1.5GHz x 4)
```
rk3399_firefly_box:/data/local/tmp/ncnn # ./benchncnn 8 6 0 
loop_count = 8
num_threads = 6
powersave = 0
      squeezenet  min =   52.63  max =  143.31  avg =   64.81
       mobilenet  min =   87.87  max =  189.37  avg =  105.93
    mobilenet_v2  min =   79.99  max =  204.02  avg =   99.30
      shufflenet  min =   39.34  max =   42.18  avg =   40.80
       googlenet  min =  156.26  max =  228.16  avg =  175.04
        resnet18  min =  208.08  max =  294.64  avg =  231.67
        resnet50  min =  713.41  max =  862.93  avg =  796.93
         alexnet  min =  501.34  max =  648.37  avg =  561.81
           vgg16  min = 1265.89  max = 1387.90  avg = 1308.65
  squeezenet-ssd  min =  128.86  max =  247.58  avg =  151.40
   mobilenet-ssd  min =  174.78  max =  250.38  avg =  186.19
```

Rockchip RK3288 (Cortex-A17 1.8GHz x 4)
```
root@rk3288:/data/local/tmp/ncnn # ./benchncnn 8 4 0 
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   78.10  max =   78.89  avg =   78.35
       mobilenet  min =  119.14  max =  120.19  avg =  119.66
    mobilenet_v2  min =  145.50  max =  150.22  avg =  146.43
      shufflenet  min =   47.96  max =   48.95  avg =   48.35
       googlenet  min =  233.60  max =  241.46  avg =  234.85
        resnet18  min =  260.83  max =  274.38  avg =  268.66
         alexnet  min =  221.32  max =  246.93  avg =  226.54
           vgg16  min = 2032.24  max = 2347.03  avg = 2152.34
  squeezenet-ssd  min =  121.88  max =  242.83  avg =  141.04
   mobilenet-ssd  min =  129.17  max =  130.86  avg =  129.87
```

HiSilicon Hi3519V101 (Cortex-A17 1.2GHz x 1)
```
root@Hi3519:/ncnn-benchmark # taskset 2 ./benchncnn 4 1 0 
loop_count = 4
num_threads = 1
powersave = 0
      squeezenet  min =  317.23  max =  317.81  avg =  317.47
       mobilenet  min =  567.67  max =  569.52  avg =  568.43
    mobilenet_v2  min =  390.11  max =  392.65  avg =  391.05
      shufflenet  min =  173.85  max =  174.02  avg =  173.94
       googlenet  min = 1190.73  max = 1193.02  avg = 1191.47
        resnet18  min = 1171.72  max = 1173.36  avg = 1172.22
         alexnet  min = 1216.17  max = 1217.52  avg = 1216.93
  squeezenet-ssd  min =  631.80  max =  633.84  avg =  632.66
   mobilenet-ssd  min = 1118.35  max = 1120.58  avg = 1119.41
```

iPhone 5S (Apple A7 1.3GHz x 2)
```
iPhone:~ root# ./benchncnn 8 2 0
loop_count = 8
num_threads = 2
powersave = 0
      squeezenet  min =  154.75  max =  160.97  avg =  158.21
       mobilenet  min =  170.67  max =  179.59  avg =  175.29
    mobilenet_v2  min =  214.20  max =  220.40  avg =  218.17
      shufflenet  min =  128.01  max =  132.98  avg =  130.91
       googlenet  min =  451.22  max =  461.26  avg =  456.23
        resnet18  min =  414.69  max =  431.77  avg =  422.70
         alexnet  min =  365.50  max =  372.15  avg =  368.19
           vgg16  min = 4786.62  max = 4953.06  avg = 4886.71
  squeezenet-ssd  min =  420.91  max =  446.68  avg =  435.90
   mobilenet-ssd  min =  334.42  max =  358.08  avg =  345.02
```

Freescale i.MX7 Dual (Cortex A7 1.0GHz x 2)
```
imx7d_pico:/data/local/tmp # ./benchncnn 8 2 0
loop_count = 8
num_threads = 2
powersave = 0
      squeezenet  min =  305.80  max =  312.74  avg =  308.57
       mobilenet  min =  482.27  max =  489.59  avg =  485.70
    mobilenet_v2  min =  386.86  max =  405.36  avg =  396.69
      shufflenet  min =  199.23  max =  232.69  avg =  205.30
       googlenet  min = 1078.63  max = 1098.36  avg = 1083.29
        resnet18  min = 1156.38  max = 1176.98  avg = 1165.92
         alexnet  min = 1273.29  max = 1283.62  avg = 1277.34
           vgg16  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
  squeezenet-ssd  min =  737.55  max =  773.54  avg =  746.79
   mobilenet-ssd  min =  946.56  max =  957.85  avg =  952.96
```
