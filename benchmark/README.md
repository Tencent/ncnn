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

Qualcomm MSM8916 Snapdragon 410 (Cortex-A53 1.2GHz x 4)
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   93.58  max =  101.45  avg =   97.25
       mobilenet  min =  161.20  max =  178.63  avg =  172.35
    mobilenet_v2  min =  197.19  max =  208.24  avg =  201.92
      shufflenet  min =   67.94  max =   78.27  avg =   71.46
       googlenet  min =  295.77  max =  307.95  avg =  300.59
        resnet18  min =  397.61  max =  437.82  avg =  409.97
         alexnet  min =  403.48  max =  432.38  avg =  415.66
           vgg16  min = 2284.47  max = 2472.28  avg = 2365.15
  squeezenet-ssd  min =  174.64  max =  265.13  avg =  197.99
   mobilenet-ssd  min =  180.67  max =  200.76  avg =  192.40
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
      squeezenet  min =  380.20  max =  398.50  avg =  387.51
       mobilenet  min =  621.16  max =  654.25  avg =  629.71
    mobilenet_v2  min =  582.39  max =  602.03  avg =  589.80
      shufflenet  min =  209.09  max =  228.76  avg =  213.98
       googlenet  min = 1309.58  max = 1434.97  avg = 1336.70
        resnet18  min = 1665.45  max = 3474.38  avg = 2166.64
         alexnet  min = 1539.43  max = 1640.56  avg = 1558.17
           vgg16  min =    0.14  max =    0.87  avg =    0.42 (FAIL due to out of memory)
  squeezenet-ssd  min =  677.92  max =  693.59  avg =  685.56
   mobilenet-ssd  min =  720.13  max =  729.33  avg =  724.47
```
