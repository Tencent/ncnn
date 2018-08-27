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
      squeezenet  min =   37.02  max =   37.51  avg =   37.16
       mobilenet  min =   50.18  max =   68.79  avg =   52.83
    mobilenet_v2  min =   51.39  max =   52.37  avg =   51.83
      shufflenet  min =   33.21  max =   33.84  avg =   33.47
       googlenet  min =  117.56  max =  135.35  avg =  122.40
        resnet18  min =  109.85  max =  126.95  avg =  114.74
         alexnet  min =  154.68  max =  168.59  avg =  162.60
           vgg16  min =  555.98  max =  586.12  avg =  572.90
  squeezenet-ssd  min =  109.55  max =  120.25  avg =  112.04
   mobilenet-ssd  min =  100.56  max =  117.62  avg =  103.67
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
         mnasnet  min =   37.91  max =   41.75  avg =   39.01
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
         mnasnet  min =   92.60  max =   98.38  avg =   96.00
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
num_threads = 6
powersave = 0
      squeezenet  min =   62.11  max =   85.62  avg =   65.56
       mobilenet  min =   80.87  max =  100.28  avg =   90.23
    mobilenet_v2  min =   79.58  max =  108.56  avg =   89.35
      shufflenet  min =   41.93  max =   55.57  avg =   45.02
       googlenet  min =  180.45  max =  243.66  avg =  200.81
        resnet18  min =  218.08  max =  355.22  avg =  249.49
         alexnet  min =  224.04  max =  328.52  avg =  254.01
           vgg16  min = 1103.06  max = 1244.06  avg = 1153.49
  squeezenet-ssd  min =  178.21  max =  268.74  avg =  190.84
   mobilenet-ssd  min =  150.63  max =  263.85  avg =  168.26

   
rk3399_firefly_box:/data/local/tmp/ncnn # ./benchncnn 8 1 2                    
loop_count = 8
num_threads = 1
powersave = 2
      squeezenet  min =   89.23  max =  100.22  avg =   92.44
       mobilenet  min =  164.15  max =  169.14  avg =  167.07
    mobilenet_v2  min =  129.67  max =  152.38  avg =  139.02
      shufflenet  min =   50.48  max =   52.65  avg =   51.53
       googlenet  min =  318.68  max =  335.91  avg =  324.63
        resnet18  min =  363.30  max =  379.47  avg =  369.61
         alexnet  min =  351.11  max =  378.56  avg =  362.13
           vgg16  min = 1587.16  max = 1736.78  avg = 1658.48
  squeezenet-ssd  min =  221.53  max =  241.31  avg =  227.60
   mobilenet-ssd  min =  294.47  max =  304.44  avg =  298.55   
   
rk3399_firefly_box:/data/local/tmp/ncnn # ./benchncnn 8 1 1                    
loop_count = 8
num_threads = 1
powersave = 1
      squeezenet  min =  181.13  max =  192.09  avg =  187.33
       mobilenet  min =  295.02  max =  308.79  avg =  300.07
    mobilenet_v2  min =  249.80  max =  272.38  avg =  262.47
      shufflenet  min =  117.77  max =  124.93  avg =  118.90
       googlenet  min =  679.51  max =  700.42  avg =  689.34
        resnet18  min =  770.68  max =  790.55  avg =  779.75
         alexnet  min =  892.71  max = 1059.58  avg = 1014.26
           vgg16  min = 3900.64  max = 3978.23  avg = 3950.21
  squeezenet-ssd  min =  469.53  max =  487.11  avg =  476.50
   mobilenet-ssd  min =  562.98  max =  581.68  avg =  569.61   
```

Rockchip RK3288 (Cortex-A17 1.8GHz x 4)
```
root@rk3288:/data/local/tmp/ncnn # ./benchncnn 8 4 0 
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   72.85  max =  105.05  avg =   79.92
       mobilenet  min =   97.22  max =  101.05  avg =   97.99
    mobilenet_v2  min =  105.60  max =  137.71  avg =  112.41
      shufflenet  min =   45.82  max =   68.10  avg =   50.06
       googlenet  min =  214.63  max =  337.57  avg =  250.70
        resnet18  min =  220.41  max =  267.93  avg =  244.83
         alexnet  min =  159.06  max =  222.84  avg =  180.12
           vgg16  min = 1183.97  max = 1609.07  avg = 1361.01
  squeezenet-ssd  min =  173.40  max =  258.01  avg =  198.69
   mobilenet-ssd  min =  186.89  max =  257.15  avg =  215.70
```

HiSilicon Hi3519V101 (Cortex-A17 1.2GHz x 1)
```
root@Hi3519:/ncnn-benchmark # taskset 2 ./benchncnn 8 1 0 
loop_count = 8
num_threads = 1
powersave = 0
      squeezenet  min =  317.07  max =  318.17  avg =  317.62
       mobilenet  min =  523.64  max =  524.64  avg =  524.07
    mobilenet_v2  min =  401.34  max =  403.80  avg =  402.63
      shufflenet  min =  182.06  max =  182.83  avg =  182.50
       googlenet  min = 1158.43  max = 1159.29  avg = 1158.83
        resnet18  min = 1095.06  max = 1098.53  avg = 1096.74
         alexnet  min = 1035.96  max = 1039.39  avg = 1038.38
  squeezenet-ssd  min =  667.85  max =  670.36  avg =  669.19
   mobilenet-ssd  min = 1032.24  max = 1034.63  avg = 1033.69
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
         mnasnet  min =   59.28  max =   64.81  avg =   61.78
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
         mnasnet  min =  291.79  max =  303.68  avg =  298.00
       googlenet  min =  975.95  max =  986.11  avg =  980.51
        resnet18  min = 1016.60  max = 1035.50  avg = 1021.75
         alexnet  min = 1240.54  max = 1254.86  avg = 1247.18
           vgg16  min =    0.00  max =    0.00  avg =    0.00 (FAIL due to out of memory)
  squeezenet-ssd  min =  614.93  max =  623.15  avg =  619.56
   mobilenet-ssd  min =  842.83  max =  884.64  avg =  855.40
  mobilenet-yolo  min = 1772.24  max = 1924.37  avg = 1805.75
```
