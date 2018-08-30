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
num_threads = 6
powersave = 0
      squeezenet  min =   47.28  max =   70.41  avg =   53.37
       mobilenet  min =   68.74  max =  176.25  avg =   82.80
    mobilenet_v2  min =   71.72  max =  180.24  avg =   86.19
      shufflenet  min =   34.90  max =   36.14  avg =   35.54
       googlenet  min =  158.35  max =  301.30  avg =  191.26
        resnet18  min =  190.96  max =  274.38  avg =  214.78
         alexnet  min =  199.21  max =  334.18  avg =  227.98
           vgg16  min =  988.46  max = 1019.90  avg = 1000.14
  squeezenet-ssd  min =  134.83  max =  223.23  avg =  148.35
   mobilenet-ssd  min =  121.47  max =  235.44  avg =  149.53
  mobilenet-yolo  min =  295.01  max =  413.26  avg =  327.84

rk3399_firefly_box:/data/local/tmp/ncnn # ./benchncnn 8 2 2          
loop_count = 8
num_threads = 2
powersave = 2
      squeezenet  min =   51.64  max =   55.08  avg =   52.36
       mobilenet  min =   88.23  max =   91.07  avg =   88.89
    mobilenet_v2  min =   84.98  max =   86.21  avg =   85.74
      shufflenet  min =   36.04  max =   38.40  avg =   36.82
       googlenet  min =  185.42  max =  188.76  avg =  186.77
        resnet18  min =  202.72  max =  212.27  avg =  206.91
         alexnet  min =  203.89  max =  222.28  avg =  215.28
           vgg16  min =  901.60  max = 1013.80  avg =  948.13
  squeezenet-ssd  min =  139.85  max =  147.36  avg =  142.18
   mobilenet-ssd  min =  156.35  max =  161.21  avg =  157.96
  mobilenet-yolo  min =  365.75  max =  380.79  avg =  371.31

rk3399_firefly_box:/data/local/tmp/ncnn # ./benchncnn 8 1 2                    
loop_count = 8
num_threads = 1
powersave = 2
      squeezenet  min =   83.73  max =   86.78  avg =   84.94
       mobilenet  min =  142.90  max =  147.71  avg =  144.64
    mobilenet_v2  min =  119.18  max =  132.26  avg =  123.92
      shufflenet  min =   52.81  max =   55.84  avg =   53.63
       googlenet  min =  316.69  max =  324.03  avg =  319.34
        resnet18  min =  318.96  max =  331.31  avg =  322.68
         alexnet  min =  340.86  max =  365.09  avg =  348.99
           vgg16  min = 1593.88  max = 1611.65  avg = 1602.36
  squeezenet-ssd  min =  199.00  max =  209.26  avg =  204.65
   mobilenet-ssd  min =  268.03  max =  275.70  avg =  270.74
  mobilenet-yolo  min =  589.43  max =  605.75  avg =  595.67
   
rk3399_firefly_box:/data/local/tmp/ncnn # ./benchncnn 8 1 1                    
loop_count = 8
num_threads = 1
powersave = 1
      squeezenet  min =  167.48  max =  173.60  avg =  169.23
       mobilenet  min =  272.88  max =  278.71  avg =  274.73
    mobilenet_v2  min =  235.35  max =  239.87  avg =  237.05
      shufflenet  min =  111.79  max =  127.11  avg =  114.13
       googlenet  min =  669.47  max =  673.68  avg =  671.23
        resnet18  min =  701.96  max =  714.85  avg =  708.56
         alexnet  min =  989.36  max =  990.63  avg =  989.96
           vgg16  min = 3746.20  max = 3835.75  avg = 3788.90
  squeezenet-ssd  min =  445.71  max =  455.03  avg =  449.07
   mobilenet-ssd  min =  511.59  max =  520.00  avg =  514.59
  mobilenet-yolo  min = 1088.56  max = 1093.53  avg = 1090.39 
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
