#ifndef PYBIND11_NCNN_LAYER_H
#define PYBIND11_NCNN_LAYER_H

#include <layer.h>
#include <pybind11/functional.h>

class LayerImpl : public ncnn::Layer
{
public:
};

class Layer : public ncnn::Layer
{
public:
    Layer(std::function<LayerImpl *()> creator)
    {
        _impl = creator();
    }

    virtual int load_param(const ncnn::ParamDict &pd)
    {
        return _impl->load_param(pd);
    }

    virtual int load_model(const ncnn::ModelBin &mb)
    {
        return _impl->load_model(mb);
    }

    virtual int create_pipeline(const ncnn::Option &opt)
    {
        return _impl->create_pipeline(opt);
    }

    virtual int destroy_pipeline(const ncnn::Option &opt)
    {
        return _impl->destroy_pipeline(opt);
    }

public:
    virtual int forward(const std::vector<ncnn::Mat> &bottom_blobs, std::vector<ncnn::Mat> &top_blobs, const ncnn::Option &opt) const
    {
        return _impl->forward(bottom_blobs, top_blobs, opt);
    }
    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const
    {
        return _impl->forward(bottom_blob, top_blob, opt);
    }

    virtual int forward_inplace(std::vector<ncnn::Mat> &bottom_top_blobs, const ncnn::Option &opt) const
    {
        return _impl->forward_inplace(bottom_top_blobs, opt);
    }
    virtual int forward_inplace(ncnn::Mat &bottom_top_blob, const ncnn::Option &opt) const
    {
        return _impl->forward_inplace(bottom_top_blob, opt);
    }

#if NCNN_VULKAN
public:
    virtual int upload_model(ncnn::VkTransfer &cmd, const ncnn::Option &opt)
    {
        return _impl->upload_model(cmd, opt);
    }

public:
    virtual int forward(const std::vector<ncnn::VkMat> &bottom_blobs, std::vector<ncnn::VkMat> &top_blobs, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        return _impl->forward(bottom_blobs, top_blobs, cmd, opt);
    }
    virtual int forward(const ncnn::VkMat &bottom_blob, ncnn::VkMat &top_blob, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        return _impl->forward(bottom_blob, top_blob, cmd, opt);
    }

    virtual int forward_inplace(std::vector<ncnn::VkMat> &bottom_top_blobs, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        return _impl->forward_inplace(bottom_top_blobs, cmd, opt);
    }
    virtual int forward_inplace(ncnn::VkMat &bottom_top_blob, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        return _impl->forward_inplace(bottom_top_blob, cmd, opt);
    }
#endif // NCNN_VULKAN
protected:
    LayerImpl *_impl;
};

class PyLayer : public LayerImpl
{
public:
    using LayerImpl::LayerImpl;

    virtual int load_param(const ncnn::ParamDict &pd)
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            load_param,
            pd);
    }

    virtual int load_model(const ncnn::ModelBin &mb)
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            load_model,
            mb);
    }

    virtual int create_pipeline(const ncnn::Option &opt)
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            create_pipeline,
            opt);
    }

    virtual int destroy_pipeline(const ncnn::Option &opt)
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            destroy_pipeline,
            opt);
    }

public:
    virtual int forward(const std::vector<ncnn::Mat> &bottom_blobs, std::vector<ncnn::Mat> &top_blobs, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward,
            bottom_blobs,
            top_blobs,
            opt);
    }
    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward,
            bottom_blob,
            top_blob,
            opt);
    }

    virtual int forward_inplace(std::vector<ncnn::Mat> &bottom_top_blobs, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward_inplace,
            bottom_top_blobs,
            opt);
    }
    virtual int forward_inplace(ncnn::Mat &bottom_top_blob, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward_inplace,
            bottom_top_blob,
            opt);
    }

#if NCNN_VULKAN
public:
    virtual int upload_model(ncnn::VkTransfer &cmd, const ncnn::Option &opt)
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            upload_model,
            cmd,
            opt);
    }

public:
    virtual int forward(const std::vector<ncnn::VkMat> &bottom_blobs, std::vector<ncnn::VkMat> &top_blobs, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward,
            bottom_blobs,
            top_blobs,
            cmd,
            opt);
    }
    virtual int forward(const ncnn::VkMat &bottom_blob, ncnn::VkMat &top_blob, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward,
            bottom_blob,
            top_blob,
            cmd,
            opt);
    }

    virtual int forward_inplace(std::vector<ncnn::VkMat> &bottom_top_blobs, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward_inplace,
            bottom_top_blobs,
            cmd,
            opt);
    }
    virtual int forward_inplace(ncnn::VkMat &bottom_top_blob, ncnn::VkCompute &cmd, const ncnn::Option &opt) const
    {
        PYBIND11_OVERLOAD(
            int,
            LayerImpl,
            forward_inplace,
            bottom_top_blob,
            cmd,
            opt);
    }
#endif // NCNN_VULKAN
};

#endif