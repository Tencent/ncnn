### caffemodel should be row-major

`caffe2ncnn` tool assumes the caffemodel is row-major (produced by c++ caffe train command).

The kernel 3x3 weights should be stored as
```
a b c
d e f
g h i
```

However, matlab caffe produced col-major caffemodel.

You have to transpose all the kernel weights by yourself or re-training using c++ caffe train command.

Besides, you may interest in https://github.com/conanhujinming/matcaffe2caffe

### check input is RGB or BGR

If your caffemodel is trained using c++ caffe and opencv, then the input image should be BGR order.

If your model is trained using matlab caffe or pytorch or mxnet or tensorflow, the input image would probably be RGB order.

The channel order can be changed on-the-fly through proper pixel type enum
```
// construct RGB blob from rgb image
ncnn::Mat in_rgb = ncnn::Mat::from_pixels(rgb_data, ncnn::Mat::PIXEL_RGB, w, h);

// construct BGR blob from bgr image
ncnn::Mat in_bgr = ncnn::Mat::from_pixels(bgr_data, ncnn::Mat::PIXEL_BGR, w, h);

// construct BGR blob from rgb image
ncnn::Mat in_bgr = ncnn::Mat::from_pixels(rgb_data, ncnn::Mat::PIXEL_RGB2BGR, w, h);

// construct RGB blob from bgr image
ncnn::Mat in_rgb = ncnn::Mat::from_pixels(bgr_data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
```


### image decoding

JPEG(`.jpg`,`.jpeg`) is loss compression, people may get different pixel value for same image on same position. 

`.bmp` images are recommended instead.

### interpolation / resizing

There are several image resizing methods, which may generate different result for same input image.

Even we specify same interpolation method, different frameworks/libraries and their various versions may also introduce difference.

A good practice is feed same size image as the input layer expected, e.g. read a 224x244 bmp image when input layer need 224x224 size.


### Mat::from_pixels/from_pixels_resize assume that the pixel data is continuous

You shall pass continuous pixel buffer to from_pixels family.

If your image is an opencv submat from an image roi, call clone() to get a continuous one.
```
cv::Mat image;// the image
cv::Rect facerect;// the face rectangle

cv::Mat faceimage = image(facerect).clone();// get a continuous sub image

ncnn::Mat in = ncnn::Mat::from_pixels(faceimage.data, ncnn::Mat::PIXEL_BGR, faceimage.cols, faceimage.rows);
```

### pre process
Apply pre process according to your training configuration

Different model has different pre process config, you may find the following transform config in Data layer section
```
transform_param {
    mean_value: 103.94
    mean_value: 116.78
    mean_value: 123.68
    scale: 0.017
}
```
Then the corresponding code for ncnn pre process is
```cpp
const float mean_vals[3] = { 103.94f, 116.78f, 123.68f };
const float norm_vals[3] = { 0.017f, 0.017f, 0.017f };
in.substract_mean_normalize(mean_vals, norm_vals);
```

Mean file is not supported currently

So you have to pre process the input data by yourself (use opencv or something)
```
transform_param {
    mean_file: "imagenet_mean.binaryproto"
}
```

For pytorch or mxnet-gluon
```python
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
```
Then the corresponding code for ncnn pre process is
```cpp
// R' = (R / 255 - 0.485) / 0.229 = (R - 0.485 * 255) / 0.229 / 255
// G' = (G / 255 - 0.456) / 0.224 = (G - 0.456 * 255) / 0.224 / 255
// B' = (B / 255 - 0.406) / 0.225 = (B - 0.406 * 255) / 0.225 / 255
const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
in.substract_mean_normalize(mean_vals, norm_vals);
```

### use the desired blob
The blob names for input and extract are differ among models.

For example, squeezenet v1.1 use "data" as input blob and "prob" as output blob while mobilenet-ssd use "data" as input blob and "detection_out" as output blob.

Some models may need multiple input or produce multiple output.

```cpp
ncnn::Extractor ex = net.create_extractor();

ex.input("data", in);// change "data" to yours
ex.input("mask", mask);// change "mask" to yours

ex.extract("output1", out1);// change "output1" to yours
ex.extract("output2", out2);// change "output2" to yours
```

### blob may have channel gap
Each channel pointer is aligned by 128bit in ncnn Mat structure.

blob may have gaps between channels if (width x height) can not divided exactly by 4

Prefer using ncnn::Mat::from_pixels or ncnn::Mat::from_pixels_resize for constructing input blob from image data

If you do need a continuous blob buffer, reshape the output.
```cpp
// out is the output blob extracted
ncnn::Mat flattened_out = out.reshape(out.w * out.h * out.c);

// plain array, C-H-W
const float* outptr = flattened_out;
```

### create new Extractor for each image
The `ncnn::Extractor` object is stateful, if you reuse for different input, you will always get exact the same result cached inside.

Always create new Extractor to process images in loop unless you do know how the stateful Extractor works.
```cpp
for (int i=0; i<count; i++)
{
    // always create Extractor
    // it's cheap and almost instantly !
    ncnn::Extractor ex = net.create_extractor();

    // use
    ex.input(your_data[i]);
}
```

### use proper loading api

If you want to load plain param file buffer, you shall use Net::load_param_mem instead of Net::load_param.

For more information about the ncnn model load api, see [ncnn-load-model](ncnn-load-model)

```cpp
ncnn::Net net;

// param_buffer is the content buffe of XYZ.param file
net.load_param_mem(param_buffer);
```
