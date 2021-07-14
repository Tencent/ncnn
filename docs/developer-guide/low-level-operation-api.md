# implement elementwise addition with/without broadcast using BinaryOp operation

* input must be fp32 storage without packing
* output is expected to be fp32 storage without packing

```cpp
void binary_add(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("BinaryOp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// op_type

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}
```

# implement 3x3 box blur on three channel image using ConvolutionDepthWise operation

* input must be fp32 storage without packing
* output is expected to be fp32 storage without packing

```cpp
void convolution_3x3_boxblur_RGB(const ncnn::Mat& rgb, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("ConvolutionDepthWise");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 3);// num_output
    pd.set(1, 3);// kernel_w
    pd.set(5, 0);// bias_term
    pd.set(6, 3*3*3);// weight_data_size
    pd.set(7, 3);// group

    op->load_param(pd);

    // set weights
    ncnn::Mat weights[1];
    weights[0].create(3*3*3);// weight_data

    for (int i=0; i<3*3*3; i++)
    {
        weights[0][i] = 1.f / 9;
    }

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    // forward
    op->forward(rgb, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
```
# transpose Mat, chw to cwh

* input must be fp32 storage with/without packing
* output is expected to be fp32 storage packed

```cpp
void transpose(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = true;

    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 1);// order_type

    op->load_param(pd);

    op->create_pipeline(opt);

    ncnn::Mat in_packed = in;
    {
        // resolve dst_elempack
        int dims = in.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = in.elempack * in.w;
        if (dims == 2) elemcount = in.elempack * in.h;
        if (dims == 3) elemcount = in.elempack * in.c;

        int dst_elempack = 1;
        if (layer->support_packing)
        {
            if (elemcount % 8 == 0 && (ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx()))
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
        }

        if (in.elempack != dst_elempack)
        {
            convert_packing(in, in_packed, dst_elempack, opt);
        }
    }

    // forward
    op->forward(in_packed, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
```
# apply instance normalization
// x = (x - mean) / sqrt(var)

* input can be fp32/fp16 storage with/without packing
* output is expected to be fp16 storage packed when supported, or fp32 storage packed otherwise

```cpp
void normalize(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = true;
    opt.use_packing_layout = true;

    ncnn::Layer* op = ncnn::create_layer("InstanceNorm");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, in.c);// channels
    pd.set(1, 0.f);// eps

    op->load_param(pd);

    // set weights
    ncnn::Mat weights[2];
    weights[0].create(in.c);// gamma_data
    weights[1].create(in.c);// beta_data

    weights[0].fill(1.f);
    weights[1].fill(0.f);

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    ncnn::Mat in_fp16 = in;
    if (in.elembits() == 32 && op->support_fp16_storage)
    {
        cast_float32_to_float16(in, in_fp16, opt);
    }
    if (in.elembits() == 16 && !op->support_fp16_storage)
    {
        cast_float16_to_float32(in, in_fp16, opt);
    }

    ncnn::Mat in_fp16_packed = in_fp16;
    {
        // resolve dst_elempack
        int dims = in_fp16.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = in_fp16.elempack * in_fp16.w;
        if (dims == 2) elemcount = in_fp16.elempack * in_fp16.h;
        if (dims == 3) elemcount = in_fp16.elempack * in_fp16.c;

        int dst_elempack = 1;
        if (layer->support_packing)
        {
            if (elemcount % 8 == 0 && (ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx()))
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
        }

        if (in_fp16.elempack != dst_elempack)
        {
            convert_packing(in_fp16, in_fp16_packed, dst_elempack, opt);
        }
    }

    // forward
    op->forward(in_fp16_packed, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
```

# cpu -> gpu -> forward -> gpu -> cpu

```cpp
ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

ncnn::VkWeightAllocator* weight_vkallocator = new ncnn::VkWeightAllocator(vkdev);
ncnn::VkWeightStagingAllocator* weight_staging_vkallocator = new ncnn::VkWeightStagingAllocator(vkdev);

// create layer
ncnn::Layer* convolution = ncnn::create_layer("Convolution");
convolution->vkdev = vkdev;

// set option
ncnn::Option opt;
opt.num_threads = 4;
opt.use_vulkan_compute = true;
opt.blob_vkallocator = blob_vkallocator;
opt.workspace_vkallocator = blob_vkallocator;
opt.staging_vkallocator = staging_vkallocator;

// load param
{
    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, ksize);
    pd.set(6, outch*inch*ksize*ksize);
    pd.use_vulkan_compute = 1;

    convolution->load_param(pd);
}

// load model
{
    ncnn::Mat weights[2];
    weights[0] = random_mat(outch*inch*ksize*ksize);
    weights[1] = random_mat(outch);

    ncnn::ModelBinFromMatArray mb(weights);
    convolution->load_model(mb);
}

// create pipeline
convolution->create_pipeline(opt);

// upload model
{
    ncnn::VkTransfer cmd(vkdev);

    ncnn::Option opt_upload = opt;
    opt_upload.blob_vkallocator = weight_vkallocator;
    opt_upload.workspace_vkallocator = weight_vkallocator;
    opt_upload.staging_vkallocator = weight_staging_vkallocator;

    convolution->upload_model(cmd, opt_upload);

    cmd.submit_and_wait();
}

ncnn::Mat bottom = random_mat(w, h, inch);

ncnn::Mat top;

// forward
{
    ncnn::VkCompute cmd(vkdev);

    ncnn::VkMat bottom_gpu;
    cmd.record_upload(bottom, bottom_gpu, opt);

    ncnn::VkMat top_gpu;
    convolution->forward(bottom_gpu, top_gpu, cmd, opt);

    cmd.record_download(top_gpu, top, opt);

    cmd.submit_and_wait();
}

convolution->destroy_pipeline(opt);

delete convolution;

vkdev->reclaim_blob_allocator(blob_vkallocator);
vkdev->reclaim_staging_allocator(staging_vkallocator);

weight_vkallocator->clear();
weight_staging_vkallocator->clear();
delete weight_vkallocator;
delete weight_staging_vkallocator;
```

