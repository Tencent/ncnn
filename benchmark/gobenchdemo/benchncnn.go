package main

// #cgo LDFLAGS:-L./ -lncnn
// #include <stdio.h>
// #include <stdlib.h>
// #include "c_api.h"
import "C"
import (
	"fmt"
	"strconv"
	"time"
)

func Min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func Max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

var Vi int

func main() {
	ver := C.ncnn_version()
	fmt.Println(C.GoString(ver))

	loop_count := 4
	num_threads := C.int(4)
	// powersave := 0
	// gpu_device := -1
	// cooling_down := 1

	g_enable_cooling_down := true

	g_loop_count := loop_count

	opt := C.ncnn_option_create()
	C.ncnn_option_set_use_vulkan_compute(opt, 1)
	C.ncnn_option_set_num_threads(opt, num_threads)

	fmt.Printf("loop_count =  %d\n", g_loop_count)
	fmt.Printf("num_threads = %d\n", num_threads)
	fmt.Printf("cooling_down = %t\n", g_enable_cooling_down)

	model := [...]string{"squeezenet", "227", "mobilenet", "224", "mobilenet_int8", "224", "mobilenet_v2", "224", "mobilenet_v3", "224",
		"shufflenet", "224", "shufflenet_v2", "224", "mnasnet", "224", "proxylessnasnet", "224",
		"efficientnet_b0", "224", "regnety_400m", "224", "blazeface", "128", "googlenet", "224", "googlenet_int8", "224",
		"resnet18", "224", "resnet18_int8", "224", "alexnet", "227", "vgg16", "224", "vgg16_int8", "224", "resnet50", "224", "resnet50_int8", "224",
		"squeezenet_ssd", "300", "squeezenet_ssd_int8", "300", "mobilenet_ssd", "300", "mobilenet_ssd_int8", "300",
		"mobilenet_yolo", "416", "mobilenetv2_yolov3", "352", "yolov4-tiny", "416"}
	modellen := len(model)
	// fmt.Printf("modellen   %d\n", modellen)
	const INT_MAX = int(^uint(0) >> 1)
	const INT_MIN = ^INT_MAX
	for {
		if Vi >= modellen {
			break //如果count>=10则退出
		}
		mmodel := model[Vi]
		Vi++
		wh, _ := strconv.Atoi(model[Vi])
		Vi++
		// fmt.Println(mmodel, " wh ", wh)
		{
			in := C.ncnn_mat_create(C.int(wh), C.int(wh), 3)
			C.ncnn_mat_fill_float(in, 0.01)

			net := C.ncnn_net_create()
			C.ncnn_net_set_option(net, opt)

			C.ncnn_net_load_param(net, C.CString(mmodel+".param"))
			dr := C.ncnn_DataReader_read_empty()
			C.ncnn_net_load_model_datareader(net, dr)

			if g_enable_cooling_down {
				time.Sleep(time.Duration(5) * time.Second)
			}
			var out C.ncnn_mat_t
			for i := 0; i < 5; i++ {

				ex := C.ncnn_extractor_create(net)
				C.ncnn_extractor_input(ex, C.CString("data"), in)
				C.ncnn_extractor_extract(ex, C.CString("output"), &out)
				// fmt.Printf("modellen   %d\n", C.ncnn_mat_get_w(out))
			}

			time_min := INT_MAX
			time_max := INT_MIN
			time_avg := 0

			for i := 0; i < 10; i++ {
				t1 := time.Now()
				ex := C.ncnn_extractor_create(net)
				C.ncnn_extractor_input(ex, C.CString("data"), in)
				C.ncnn_extractor_extract(ex, C.CString("output"), &out)

				elapsed := int(time.Since(t1) / time.Millisecond)
				time_min = Min(time_min, elapsed)
				time_max = Max(time_max, elapsed)
				time_avg += elapsed
			}
			time_avg /= 10
			fmt.Println(mmodel, "  min = ", time_min, " max ", time_max, " avg ", +time_avg)
			// fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg)
		}
	}

}
