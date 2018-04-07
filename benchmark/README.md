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
      squeezenet  min =   36.40  max =   39.33  avg =   38.27
       mobilenet  min =   58.53  max =   62.40  avg =   60.18
    mobilenet_v2  min =   59.49  max =   61.35  avg =   60.58
      shufflenet  min =   27.47  max =   28.47  avg =   27.92
       googlenet  min =  127.26  max =  138.16  avg =  130.62
        resnet18  min =  129.63  max =  145.10  avg =  132.80
         alexnet  min =  196.04  max =  222.30  avg =  205.18
           vgg16  min =  701.14  max =  788.79  avg =  747.02
  squeezenet-ssd  min =   74.85  max =   80.77  avg =   77.36
   mobilenet-ssd  min =   79.09  max =  101.17  avg =   84.24
```

Qualcomm MSM8994 Snapdragon 810 (Cortex-A57 2.0GHz x 4 + Cortex-A53 1.55GHz x 4)
```
angler:/data/local/tmp $ ./benchncnn 8 8 0
loop_count = 8
num_threads = 8
powersave = 0
      squeezenet  min =   58.94  max =   64.86  avg =   60.08
       mobilenet  min =   69.22  max =   81.54  avg =   72.96
    mobilenet_v2  min =   94.18  max =  107.57  avg =   98.76
      shufflenet  min =   48.10  max =   64.86  avg =   53.72
       googlenet  min =  192.29  max =  212.10  avg =  201.60
        resnet18  min =  254.64  max =  308.90  avg =  274.12
         alexnet  min =  196.18  max =  226.44  avg =  210.53
           vgg16  min = 1465.17  max = 1561.20  avg = 1501.23
  squeezenet-ssd  min =  136.68  max =  191.48  avg =  163.87
   mobilenet-ssd  min =  123.21  max =  161.13  avg =  140.20
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

iPhone 5S (Apple A7 1.3GHz x 2)
```
iPhone:~ root# ./benchncnn 8 2 0
loop_count = 8
num_threads = 2
powersave = 0
      squeezenet  min =  146.85  max =  160.10  avg =  152.25
       mobilenet  min =  170.99  max =  192.57  avg =  181.07
    mobilenet_v2  min =  230.88  max =  377.16  avg =  260.93
      shufflenet  min =  101.45  max =  113.17  avg =  107.05
       googlenet  min =  446.15  max =  462.75  avg =  453.62
        resnet18  min = 1711.46  max = 1798.15  avg = 1751.01
         alexnet  min = 1476.57  max = 1651.94  avg = 1574.76
           vgg16  min = 5377.98  max = 5493.23  avg = 5433.10
  squeezenet-ssd  min =  256.51  max =  336.59  avg =  282.25
   mobilenet-ssd  min =  215.67  max =  226.62  avg =  221.52
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
