implement elementwise addition with/without broadcast using BinaryOp operation
```
void binary_add(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c)
{
    ncnn::Layer* op = ncnn::create_layer("BinaryOp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// op_type

    op->load_param(pd);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops);

    c = tops[0];

    delete op;
}
```

implement 3x3 box blur on three channel image using ConvolutionDepthWise operation
```
void convolution_3x3_boxblur_RGB(const ncnn::Mat& rgb, ncnn::Mat& out)
{
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

    // forward
    op->forward(rgb, out);

    delete op;
}
```
transpose Mat, chw to cwh
```
void transpose(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 1);// order_type

    op->load_param(pd);

    // forward
    op->forward(in, out);

    delete op;
}
```
apply instance normalization
// x = (x - mean) / sqrt(var)
```
void normalize(const ncnn::Mat& in, ncnn::Mat& out)
{
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

    // forward
    op->forward(in, out);

    delete op;
}
```

# cpu -> gpu -> forward -> gpu -> cpu
```
ncnn::create_gpu_instance();

{
ncnn::VulkanDevice vkdev;

ncnn::VkWeightBufferAllocator g_weight_vkallocator(&vkdev);
ncnn::VkBlobBufferAllocator g_blob_vkallocator(&vkdev);
ncnn::VkStagingBufferAllocator g_staging_vkallocator(&vkdev);
ncnn::VkWeightStagingBufferAllocator g_weight_staging_vkallocator(&vkdev);

// create layer
ncnn::Layer* convolution = ncnn::create_layer("Convolution");
convolution->vkdev = &vkdev;

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

// upload model
{
ncnn::VkTransfer cmd(&vkdev);
cmd.weight_vkallocator = &g_weight_vkallocator;
cmd.staging_vkallocator = &g_weight_staging_vkallocator;

convolution->upload_model(cmd);

cmd.submit();
cmd.wait();

g_weight_staging_vkallocator.clear();
}

// create pipeline
convolution->create_pipeline();

// set default option
{
ncnn::Option opt = ncnn::get_default_option();

opt.lightmode = true;
opt.num_threads = 4;
opt.blob_allocator = 0;
opt.workspace_allocator = 0;

opt.vulkan_compute = true;
opt.blob_vkallocator = &g_blob_vkallocator;
opt.workspace_vkallocator = &g_blob_vkallocator;
opt.staging_vkallocator = &g_staging_vkallocator;

ncnn::set_default_option(opt);
}

ncnn::Mat bottom = random_mat(w, h, inch);

ncnn::VkMat bottom_gpu;

// copy bottom to bottom_gpu
{
bottom_gpu.create_like(bottom, &g_blob_vkallocator, &g_staging_vkallocator);
bottom_gpu.prepare_staging_buffer();
bottom_gpu.upload(bottom);
}

ncnn::VkMat top_gpu;

// forward
{
ncnn::VkCompute cmd(&vkdev);

cmd.record_upload(bottom_gpu);

convolution->forward(bottom_gpu, top_gpu, cmd);

top_gpu.prepare_staging_buffer();

cmd.record_download(top_gpu);

cmd.submit();
cmd.wait();
}

ncnn::Mat top;

// copy top_gpu to top
{
top.create_like(top_gpu);
top_gpu.download(top);
}

delete convolution;

g_weight_vkallocator.clear();
g_blob_vkallocator.clear();
g_staging_vkallocator.clear();
g_weight_staging_vkallocator.clear();
}

ncnn::destroy_gpu_instance();

```

