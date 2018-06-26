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
