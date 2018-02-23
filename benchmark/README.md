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
```
HM2014812:/data/local/tmp # ./benchncnn 8 4 0
loop_count = 8
num_threads = 4
powersave = 0
      squeezenet  min =   98.27  max =  118.75  avg =  105.22
       mobilenet  min =  168.36  max =  178.58  avg =  174.52
    mobilenet_v2  min =  192.38  max =  210.21  avg =  201.79
      shufflenet  min =   66.07  max =   74.64  avg =   70.58
       googlenet  min =  327.53  max =  344.18  avg =  334.36
        resnet18  min =  465.24  max =  479.58  avg =  470.86
         alexnet  min =  380.57  max =  406.64  avg =  397.66
           vgg16  min = 2341.65  max = 2475.65  avg = 2402.14
  squeezenet-ssd  min =  187.64  max =  283.71  avg =  204.29
   mobilenet-ssd  min =  183.96  max =  214.15  avg =  193.30
```
