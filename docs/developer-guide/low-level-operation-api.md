implement elementwise addition with/without broadcast using BinaryOp operation
```cpp
void binary_add(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c)
{
    ncnn::Option opt;
    opt.num_threads = 2;

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

implement 3x3 box blur on three channel image using ConvolutionDepthWise operation
```cpp
void convolution_3x3_boxblur_RGB(const ncnn::Mat& rgb, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 2;

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
transpose Mat, chw to cwh
```cpp
void transpose(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 2;

    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 1);// order_type

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
```
apply instance normalization
// x = (x - mean) / sqrt(var)
```cpp
void normalize(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 2;

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

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
```

# cpu -> gpu -> forward -> gpu -> cpu
```cpp
ncnn::create_gpu_instance();

{
ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
ncnn::VkBlobAllocator g_blob_vkallocator(vkdev);
ncnn::VkStagingAllocator g_staging_vkallocator(vkdev);
ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

// create layer
ncnn::Layer* convolution = ncnn::create_layer("Convolution");
convolution->vkdev = vkdev;

// set option
ncnn::Option opt;
opt.lightmode = true;
opt.num_threads = 4;
opt.blob_allocator = 0;
opt.workspace_allocator = 0;
opt.vulkan_compute = true;
opt.blob_vkallocator = &g_blob_vkallocator;
opt.workspace_vkallocator = &g_blob_vkallocator;
opt.staging_vkallocator = &g_staging_vkallocator;

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
opt_upload.blob_vkallocator = &g_weight_vkallocator;
opt_upload.workspace_vkallocator = &g_weight_vkallocator;
opt_upload.staging_vkallocator = &g_weight_staging_vkallocator;

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

g_weight_vkallocator.clear();
g_blob_vkallocator.clear();
g_staging_vkallocator.clear();
g_weight_staging_vkallocator.clear();
}

ncnn::destroy_gpu_instance();

```

