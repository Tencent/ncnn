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
HM2014812:/data/local/tmp # ./benchncnn 4 4 0
loop_count = 4
num_threads = 4
powersave = 0
      squeezenet  min =   99.34  max =  104.24  avg =  102.83
       mobilenet  min =  171.06  max =  177.08  avg =  173.45
      shufflenet  min =   67.80  max =   73.63  avg =   69.35
       googlenet  min =  337.30  max =  341.33  avg =  339.00
        resnet18  min =  466.41  max =  476.33  avg =  472.07
         alexnet  min =  397.47  max =  421.54  avg =  404.22
           vgg16  min = 2338.67  max = 2984.17  avg = 2557.16
  squeezenet-ssd  min =  201.43  max =  304.06  avg =  236.67
   mobilenet-ssd  min =  198.93  max =  241.33  avg =  214.54
```
