### opencv to ncnn

* cv::Mat CV_8UC3 -> ncnn::Mat 3 channel + swap RGB/BGR

```cpp
// cv::Mat a(h, w, CV_8UC3);
ncnn::Mat in = ncnn::Mat::from_pixels(a.data, ncnn::Mat::PIXEL_BGR2RGB, a.cols, a.rows);
```

* cv::Mat CV_8UC3 -> ncnn::Mat 3 channel + keep RGB/BGR order

```cpp
// cv::Mat a(h, w, CV_8UC3);
ncnn::Mat in = ncnn::Mat::from_pixels(a.data, ncnn::Mat::PIXEL_RGB, a.cols, a.rows);
```

* cv::Mat CV_8UC3 -> ncnn::Mat 1 channel + do RGB2GRAY/BGR2GRAY

```cpp
// cv::Mat rgb(h, w, CV_8UC3);
ncnn::Mat inrgb = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB2GRAY, rgb.cols, rgb.rows);

// cv::Mat bgr(h, w, CV_8UC3);
ncnn::Mat inbgr = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2GRAY, bgr.cols, bgr.rows);
```

* cv::Mat CV_8UC1 -> ncnn::Mat 1 channel

```cpp
// cv::Mat a(h, w, CV_8UC1);
ncnn::Mat in = ncnn::Mat::from_pixels(a.data, ncnn::Mat::PIXEL_GRAY, a.cols, a.rows);
```

* cv::Mat CV_32FC1 -> ncnn::Mat 1 channel

  * **You could construct ncnn::Mat and fill data into it directly to avoid data copy**

```cpp
// cv::Mat a(h, w, CV_32FC1);
ncnn::Mat in(a.cols, a.rows, 1, (void*)a.data);
in = in.clone();
```

* cv::Mat CV_32FC3 -> ncnn::Mat 3 channel

  * **You could construct ncnn::Mat and fill data into it directly to avoid data copy**

```cpp
// cv::Mat a(h, w, CV_32FC3);
ncnn::Mat in_pack3(a.cols, a.rows, 1, (void*)a.data, (size_t)4u * 3, 3);
ncnn::Mat in;
ncnn::convert_packing(in_pack3, in, 1);
```

* std::vector < cv::Mat > + CV_32FC1 -> ncnn::Mat multiple channels

  * **You could construct ncnn::Mat and fill data into it directly to avoid data copy**

```cpp
// std::vector<cv::Mat> a(channels, cv::Mat(h, w, CV_32FC1));
int channels = a.size();
ncnn::Mat in(a[0].cols, a[0].rows, channels);
for (int p=0; p<in.c; p++)
{
    memcpy(in.channel(p), (const uchar*)a[p].data, in.w * in.h * sizeof(float));
}
```

### ncnn to opencv

* ncnn::Mat 3 channel -> cv::Mat CV_8UC3 + swap RGB/BGR

  * **You may need to call in.substract_mean_normalize() first to scale values from 0..1 to 0..255**

```cpp
// ncnn::Mat in(w, h, 3);
cv::Mat a(in.h, in.w, CV_8UC3);
in.to_pixels(a.data, ncnn::Mat::PIXEL_BGR2RGB);
```

* ncnn::Mat 3 channel -> cv::Mat CV_8UC3 + keep RGB/BGR order

  * **You may need to call in.substract_mean_normalize() first to scale values from 0..1 to 0..255**

```cpp
// ncnn::Mat in(w, h, 3);
cv::Mat a(in.h, in.w, CV_8UC3);
in.to_pixels(a.data, ncnn::Mat::PIXEL_RGB);
```

* ncnn::Mat 1 channel -> cv::Mat CV_8UC1

  * **You may need to call in.substract_mean_normalize() first to scale values from 0..1 to 0..255**

```cpp
// ncnn::Mat in(w, h, 1);
cv::Mat a(in.h, in.w, CV_8UC1);
in.to_pixels(a.data, ncnn::Mat::PIXEL_GRAY);
```

* ncnn::Mat 1 channel -> cv::Mat CV_32FC1

  * **You could consume or manipulate ncnn::Mat data directly to avoid data copy**

```cpp
// ncnn::Mat in;
cv::Mat a(in.h, in.w, CV_32FC1);
memcpy((uchar*)a.data, in.data, in.w * in.h * sizeof(float));
```

* ncnn::Mat 3 channel -> cv::Mat CV_32FC3

  * **You could consume or manipulate ncnn::Mat data directly to avoid data copy**

```cpp
// ncnn::Mat in(w, h, 3);
ncnn::Mat in_pack3;
ncnn::convert_packing(in, in_pack3, 3);
cv::Mat a(in.h, in.w, CV_32FC3);
memcpy((uchar*)a.data, in_pack3.data, in.w * in.h * 3 * sizeof(float));
```

* ncnn::Mat multiple channels -> std::vector < cv::Mat > + CV_32FC1

  * **You could consume or manipulate ncnn::Mat data directly to avoid data copy**

```cpp
// ncnn::Mat in(w, h, channels);
std::vector<cv::Mat> a(in.c);
for (int p=0; p<in.c; p++)
{
    a[p] = cv::Mat(in.h, in.w, CV_32FC1);
    memcpy((uchar*)a[p].data, in.channel(p), in.w * in.h * sizeof(float));
}
```
