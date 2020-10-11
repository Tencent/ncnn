
### image roi crop + convert to ncnn::Mat

```
+--------------+
|   y          |           /-------/
| x +-------+  |          +-------+|
|   |     roih |im_h  =>  |      roih
|   +-roiw--+  |          +-roiw--+/
|              |
+-----im_w-----+
```
```cpp
ncnn::Mat in = ncnn::Mat::from_pixels_roi(im.data, ncnn::PIXEL_RGB, im_w, im_h, x, y, roiw, roih);
```

### image roi crop + resize + convert to ncnn::Mat

```
+--------------+
|   y          |           /----/
| x +-------+  |          +----+|
|   |     roih |im_h  =>  |  target_h
|   +-roiw--+  |          |    ||
|              |          +----+/
+-----im_w-----+         target_w
```
```cpp
ncnn::Mat in = ncnn::Mat::from_pixels_roi_resize(im.data, ncnn::PIXEL_RGB, im_w, im_h, x, y, roiw, roih, target_w, target_h);
```

### ncnn::Mat export image + offset paste

```
                +--------------+
 /-------/      |   y          |
+-------+|      | x +-------+  |
|       h|  =>  |   |       h  |im_h
+---w---+/      |   +---w---+  |
                |              |
                +-----im_w-----+
```
```cpp
const unsigned char* data = im.data + (y * im_w + x) * 3;
out.to_pixels(data, ncnn::PIXEL_RGB, im_w * 3);
```

### ncnn::Mat export image + resize + roi paste

```
            +--------------+
 /----/     |   y          |
+----+|     | x +-------+  |
|    h| =>  |   |      roih|im_h
|    ||     |   +-roiw--+  |
+-w--+/     |              |
            +-----im_w-----+
```
```cpp
const unsigned char* data = im.data + (y * im_w + x) * 3;
out.to_pixels_resize(data, ncnn::PIXEL_RGB, roiw, roih, im_w * 3);
```

### image roi crop + resize
```
+--------------+
|   y          |
| x +-------+  |          +----+
|   |      roih|im_h  =>  |  target_h
|   +-roiw--+  |          |    |
|              |          +----+
+-----im_w-----+         target_w
```
```cpp
const unsigned char* data = im.data + (y * im_w + x) * 3;
ncnn::resize_bilinear_c3(data, roiw, roih, im_w * 3, outdata, target_w, target_h, target_w * 3);
```

### image resize + offset paste
```
            +--------------+
            |   y          |
+----+      | x +-------+  |
|    h  =>  |   |     roih |im_h
|    |      |   +-roiw--+  |
+-w--+      |              |
            +-----im_w-----+
```
```cpp
unsigned char* outdata = im.data + (y * im_w + x) * 3;
ncnn::resize_bilinear_c3(data, w, h, w * 3, outdata, roiw, roih, im_w * 3);
```

### image roi crop + resize + roi paste
```
+--------------+         +-----------------+
|   y          |         |  roiy           |
| x +-------+  |         |roix----------+  |
|   |       h  |im_h  => |   |     target_h|outim_h
|   +---w---+  |         |   |          |  |
|              |         |   +-target_w-+  |
+-----im_w-----+         +-----outim_w-----+
```
```cpp
const unsigned char* data = im.data + (y * im_w + x) * 3;
unsigned char* outdata = outim.data + (roiy * outim_w + roix) * 3;
ncnn::resize_bilinear_c3(data, w, h, im_w * 3, outdata, target_w, target_h, outim_w * 3);
```

### image roi crop + rotate
```
+--------------+
|   y          |
| x +-------+  |          +---+
|   |  < <  h  |im_h  =>  | ^ |w
|   +---w---+  |          | ^ |
|              |          +---+
+-----im_w-----+            h
```
```cpp
const unsigned char* data = im.data + (y * im_w + x) * 3;
ncnn::kanna_rotate_c3(data, w, h, im_w * 3, outdata, h, w, h * 3, 6);
```

### image rotate + offset paste
```
             +--------------+
             |   y          |
 +---+       | x +-------+  |
 | ^ |h  =>  |   |  < <  w  |im_h
 | ^ |       |   +---h---+  |
 +---+       |              |
   w         +-----im_w-----+
```
```cpp
unsigned char* outdata = im.data + (y * im_w + x) * 3;
ncnn::kanna_rotate_c3(data, w, h, w * 3, outdata, h, w, im_w * 3, 7);
```

### image roi crop + rotate + roi paste
```
+--------------+         +-----------------+
|   y          |         |        roiy     |
| x +-------+  |         |   roix  +---+   |
|   |  < <  h  |im_h  => |         | ^ w   |outim_h
|   +---w---+  |         |         | ^ |   |
|              |         |         +-h-+   |
+-----im_w-----+         +-----outim_w-----+
```
```cpp
const unsigned char* data = im.data + (y * im_w + x) * 3;
unsigned char* outdata = outim.data + (roiy * outim_w + roix) * 3;
ncnn::kanna_rotate_c3(data, w, h, im_w * 3, outdata, h, w, outim_w * 3, 6);
```
