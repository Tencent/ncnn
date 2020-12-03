import sys
import time
import ncnn

class GlobalGpuInstance:
    def __init__(self):
        ncnn.create_gpu_instance()
    def __del__(self):
        ncnn.destroy_gpu_instance()
        
# initialize vulkan runtime before main()
g_global_gpu_instance = GlobalGpuInstance();

g_warmup_loop_count = 8
g_loop_count = 4

g_blob_pool_allocator = ncnn.UnlockedPoolAllocator()
g_workspace_pool_allocator = ncnn.PoolAllocator()

g_vkdev = None
g_blob_vkallocator = None
g_staging_vkallocator = None

def benchmark(comment, _in, opt):
    _in.fill(0.01)

    net = ncnn.Net()
    net.opt = opt

    if net.opt.use_vulkan_compute:
        net.set_vulkan_device(g_vkdev)

    net.load_param("params/" + comment + ".param")

    dr = ncnn.DataReaderFromEmpty()
    net.load_model(dr)

    g_blob_pool_allocator.clear()
    g_workspace_pool_allocator.clear()

    if net.opt.use_vulkan_compute:
        g_blob_vkallocator.clear()
        g_staging_vkallocator.clear()

    out = ncnn.Mat()

    #warm up
    for i in range(g_warmup_loop_count):
        ex = net.create_extractor()
        ex.input("data", _in)
        ex.extract("output", out)

    time_min = sys.float_info.max
    time_max = -sys.float_info.max
    time_avg = 0.0

    for i in range(g_loop_count):
        start = time.time()

        ex = net.create_extractor()
        ex.input("data", _in)
        ex.extract("output", out)

        end = time.time()

        timespan = end - start

        time_min = timespan if timespan < time_min else time_min
        time_max = timespan if timespan > time_max else time_max
        time_avg += timespan

    # extractor need relese manually when build ncnn with vuklan,
    # due to python relese ex after net, but in extractor.destruction use net
    ex = None

    time_avg /= g_loop_count

    print("%20s  min = %7.2f  max = %7.2f  avg = %7.2f"%(comment, time_min * 1000, time_max * 1000, time_avg * 1000))

if __name__ == "__main__":
    loop_count = 4
    num_threads = ncnn.get_cpu_count()
    powersave = 0
    gpu_device = 0

    argc = len(sys.argv)
    if argc >= 2:
        loop_count = int(sys.argv[1])
    if argc >= 3:
        num_threads = int(sys.argv[2])
    if argc >= 4:
        powersave = int(sys.argv[3])
    if argc >= 5:
        gpu_device = int(sys.argv[4])

    if ncnn.build_with_gpu() == False:
        print('ncnn is not build with gpu(NCNN_VULKAN), please rebuild with gpu')
        sys.exit(0)
    

    g_loop_count = loop_count

    g_blob_pool_allocator.set_size_compare_ratio(0.0)
    g_workspace_pool_allocator.set_size_compare_ratio(0.5)

    g_warmup_loop_count = 10

    g_vkdev = ncnn.get_gpu_device(gpu_device)

    g_blob_vkallocator = ncnn.VkBlobBufferAllocator(g_vkdev)
    g_staging_vkallocator = ncnn.VkStagingBufferAllocator(g_vkdev)

    opt = ncnn.Option()
    opt.lightmode = True
    opt.num_threads = num_threads
    opt.blob_allocator = g_blob_pool_allocator
    opt.workspace_allocator = g_workspace_pool_allocator
    opt.blob_vkallocator = g_blob_vkallocator
    opt.workspace_vkallocator = g_blob_vkallocator
    opt.staging_vkallocator = g_staging_vkallocator
    opt.use_winograd_convolution = True
    opt.use_sgemm_convolution = True
    opt.use_int8_inference = True
    opt.use_vulkan_compute = True
    opt.use_fp16_packed = True
    opt.use_fp16_storage = True
    opt.use_fp16_arithmetic = True
    opt.use_int8_storage = True
    opt.use_int8_arithmetic = True
    opt.use_packing_layout = True

    ncnn.set_cpu_powersave(powersave)
    ncnn.set_omp_dynamic(0)
    ncnn.set_omp_num_threads(num_threads)

    print("loop_count = %d"%(loop_count))
    print("num_threads = %d"%(num_threads))
    print("powersave = %d"%(ncnn.get_cpu_powersave()))
    print("gpu_device = %d"%(gpu_device))

    #time.sleep(15)

    #must use named param w, h, c due to python has no size_t(unsigned int) to call the correct overload ncnn.Mat
    benchmark("squeezenet", ncnn.Mat(w=227, h=227, c=3), opt)
    #benchmark("squeezenet_int8", ncnn.Mat(w=227, h=227, c=3), opt)
    benchmark("mobilenet", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("mobilenet_int8", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("mobilenet_v2", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("mobilenet_v3", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("shufflenet", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("shufflenet_v2", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("mnasnet", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("proxylessnasnet", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("googlenet", ncnn.Mat(w=224, h=224, c=3), opt)
    #benchmark("googlenet_int8", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("resnet18", ncnn.Mat(w=224, h=224, c=3), opt)
    #benchmark("resnet18_int8", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("alexnet", ncnn.Mat(w=227, h=227, c=3), opt)
    benchmark("vgg16", ncnn.Mat(w=224, h=224, c=3), opt)
    #benchmark("vgg16_int8", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("resnet50", ncnn.Mat(w=224, h=224, c=3), opt)
    #benchmark("resnet50_int8", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("squeezenet_ssd", ncnn.Mat(w=300, h=300, c=3), opt)
    #benchmark("squeezenet_ssd_int8", ncnn.Mat(w=300, h=300, c=3), opt)
    benchmark("mobilenet_ssd", ncnn.Mat(w=300, h=300, c=3), opt)
    #benchmark("mobilenet_ssd_int8", ncnn.Mat(w=300, h=300, c=3), opt)
    benchmark("mobilenet_yolo", ncnn.Mat(w=416, h=416, c=3), opt)
    benchmark("mobilenetv2_yolov3", ncnn.Mat(w=352, h=352, c=3), opt)



