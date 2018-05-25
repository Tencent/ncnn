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
      squeezenet  min =   55.02  max =   59.17  avg =   56.19
       mobilenet  min =   64.74  max =   67.88  avg =   65.69
    mobilenet_v2  min =   68.38  max =   69.97  avg =   68.92
      shufflenet  min =   44.27  max =   47.37  avg =   45.12
       googlenet  min =  158.51  max =  177.85  avg =  165.95
        resnet18  min =  179.01  max =  213.72  avg =  193.28
         alexnet  min =  158.99  max =  172.32  avg =  165.38
           vgg16  min = 1245.07  max = 1453.01  avg = 1377.76
  squeezenet-ssd  min =  106.20  max =  126.05  avg =  114.63
   mobilenet-ssd  min =   74.45  max =   86.69  avg =   79.13
```

Qualcomm MSM8916 Snapdragon 410 (Cortex-A53 1.2GHz x 4)
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   84.88  max =   99.62  avg =   91.00
       mobilenet  min =  152.19  max =  176.29  avg =  162.21
    mobilenet_v2  min =  142.31  max =  160.67  avg =  148.78
      shufflenet  min =   62.32  max =   69.69  avg =   64.26
       googlenet  min =  275.33  max =  293.44  avg =  284.08
        resnet18  min =  322.60  max =  342.60  avg =  337.03
         alexnet  min =  336.16  max =  377.78  avg =  362.75
           vgg16  min = 2355.94  max = 2461.82  avg = 2397.03
  squeezenet-ssd  min =  156.24  max =  166.70  avg =  161.97
   mobilenet-ssd  min =  171.26  max =  182.79  avg =  176.83
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
      squeezenet  min =  142.08  max =  149.40  avg =  146.29
       mobilenet  min =  173.65  max =  198.31  avg =  182.60
    mobilenet_v2  min =  190.85  max =  199.66  avg =  195.04
      shufflenet  min =  102.90  max =  107.93  avg =  104.69
       googlenet  min =  444.17  max =  486.42  avg =  456.42
        resnet18  min = 1499.48  max = 1741.99  avg = 1580.66
         alexnet  min =  757.03  max =  922.91  avg =  853.67
           vgg16  min = 5368.56  max = 5487.89  avg = 5408.96
  squeezenet-ssd  min =  248.48  max =  256.78  avg =  253.82
   mobilenet-ssd  min =  210.11  max =  223.64  avg =  216.97
```

Freescale i.MX7 Dual (Cortex A7 1.0GHz x 2)
```
imx7d_pico:/data/local/tmp # ./benchncnn 8 2 0
loop_count = 8
num_threads = 2
powersave = 0
      squeezenet  min =  376.69  max =  387.00  avg =  379.90
       mobilenet  min =  621.43  max =  649.77  avg =  632.97
    mobilenet_v2  min =  461.56  max =  476.45  avg =  469.34
      shufflenet  min =  215.91  max =  224.06  avg =  218.38
       googlenet  min = 1303.39  max = 1336.32  avg = 1311.89
        resnet18  min = 1470.91  max = 2037.16  avg = 1546.38
         alexnet  min = 1513.10  max = 1529.57  avg = 1517.92
           vgg16  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
  squeezenet-ssd  min =  627.03  max =  634.16  avg =  628.70
   mobilenet-ssd  min =  705.97  max =  749.82  avg =  716.50
```
