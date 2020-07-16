Mat structure is now allocator-aware via an extra allocator parameter with default zero value.

The good-old ncnn::fastMalloc()/ncnn::fastFree() will be used for a null allocator.

You could pass a custom allocator to delegate all memory allocation and deallocation.

```cpp
class Allocator
{
public:
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};
```

ncnn has already implemented two simple pooled Allocator class, with mutex lock or without it.

```cpp
ncnn::PoolAllocator locked_mempool;
ncnn::UnlockedPoolAllocator unlocked_mempool;
```

the two allocator types in ncnn

* blob allocator

    used to allocate memory for all named blobs, which you could retrieve by Extractor::extract()
* workspace allocator

    used to allocate memory for internal temporary use in layer implementation, such as the temp blob after padding in convolution

by default, all Extractor instance use the two allocator in the default option
You can alter them by ncnn::set_default_option()
or you can set them per Extractor by Extractor::set_blob_allocator()/Extractor::set_workspace_allocator()

blob allocator is guaranteed to be called in-order in layer implementation during each Extractor lifecycle
while workspace allocator may be called synchronously

the practical usage

* one network, one-by-one inference

    shared unlocked blob allocator for all Extractor

    shared locked workspace allocator for all Extractor

* one network, concurrent inference

    shared unlocked blob allocator for all Extractor in each thread

    shared locked workspace allocator for all Extractor among all threads

* concurrent multiple networks, one-by-one inference for each network

    shared unlocked blob allocator for all Extractor of each network

    shared locked workspace allocator for all Extractor among all networks (for saving memory)

* concurrent multiple networks, concurrent inference for each network

    shared unlocked blob allocator for all Extractor of each network in each thread

    shared locked workspace allocator for all Extractor among all networks (for saving memory)
