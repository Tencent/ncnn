caffe-android-lib https://github.com/sh1r0/caffe-android-lib

mini-caffe https://github.com/luoyetx/mini-caffe

openblas-0.2.20 https://github.com/xianyi/OpenBLAS

ncnn https://github.com/Tencent/ncnn

***

squeezenet_v1.1 https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1

mobilenet_v1 https://github.com/shicai/MobileNet-Caffe

vgg16 https://gist.github.com/ksimonyan/211839e770f7b538e2d8

***

Host platform and compiler configuration: 

fedora 27, android-ndk-r15c, target arch = arm64-v8a

we manually update openblas package to version 0.2.20 in caffe-android-lib for better performance


***

Device: Nexus 6p

OS: LineageOS 15.1(Android 8.1.0), ROM newly flashed without any third-party APP installed

CPU: Snapdragon 810 (Cortex-A57 2.0GHz x 4 + Cortex-A53 1.55GHz x 4)

RAM: 3G


***

Benchmark method: 

Run squeezenet, mobilenet inference 23 times in a loop, discard the the first three warmup records, and then calculate the average inference time

Run vgg169 times in a loop, discard the first warmup record, and then calculate the average inference time

Since the system may force SOC lowering its frequency when temperature goes high, sleep over 1 minute before each benchmark to prevent this issue.

fps performance: fps = 1000 / avgtime(ms)

cpu usage: take the CPU value in top utility output

memory usage: take the RES value in top utility output

the overall power consumption and performance per watt: 

Disable usb charging: adb shell echo 0 > /sys/class/power_supply/battery/charging_enabled

current(μA) = adb shell cat /sys/class/power_supply/battery/current_now (multiply -1 for 810 chip)

voltage(μV) = adb shell cat /sys/class/power_supply/battery/voltage_now

power consumption(mW) = current / 1000 * voltage / 1000 / 1000

performance per watt(1000fps/W) = fps / power consumption * 1000


***

The binary size after debug stripping

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/1.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/2.jpg)

***

squeezenet

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/3.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/4.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/5.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/6.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/7.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/8.jpg)
***

mobilnet

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/9.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/10.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/11.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/12.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/13.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/14.jpg)
***

vgg16

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/15.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/16.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/17.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/18.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/19.jpg)

![](https://github.com/nihui/ncnn-assets/raw/master/20180413/20.jpg)
