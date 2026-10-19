// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "caffe_python.h"

namespace pnnx {

// clang-format off
// *INDENT-OFF*

void export_caffe_python(const Graph& graph, FILE* pyfp)
{
    bool priorbox = false;
    bool detectionoutput = false;
    bool roi = false;
    bool proposal = false;
    bool yolo = false;
    bool argmax = false;
    bool power = false;
    for (const Operator* op : graph.ops)
    {
        priorbox |= op->type == "caffe.PriorBox";
        detectionoutput |= op->type == "caffe.DetectionOutput";
        roi |= op->type == "caffe.ROIPooling" || op->type == "caffe.PSROIPooling" || op->type == "caffe.ROIAlign";
        proposal |= op->type == "caffe.Proposal";
        yolo |= op->type == "caffe.YoloDetectionOutput" || op->type == "caffe.Yolov3DetectionOutput";
        argmax |= op->type == "caffe.ArgMax";
        power |= op->type == "caffe.Power";
    }
    if (!priorbox && !detectionoutput && !roi && !proposal && !yolo && !argmax && !power)
        return;

    fprintf(pyfp, "class caffe:\n");
    if (power)
    {
        fputs(R"PY(    @staticmethod
    def Power(input, power, scale, shift):
        return torch.pow(shift + scale * input, power)

)PY", pyfp);
    }
    if (argmax)
    {
        fputs(R"PY(    @staticmethod
    def ArgMax(input, top_k, mode, shape):
        values = input.reshape(-1)
        # Caffe sorts (value, index) pairs descending, including index ties
        # stable sorting of the reversed input gives the same larger-index rule
        reversed_order = torch.argsort(values.flip(0), descending=True, stable=True)[:top_k]
        indices = values.numel() - 1 - reversed_order
        if mode == 0:
            output = indices.to(input.dtype)
        elif mode == 1:
            output = values[indices]
        else:
            output = torch.cat((indices.to(input.dtype), values[indices]))
        return output.reshape(shape)

)PY", pyfp);
    }
    if (priorbox)
    {
        fputs(R"PY(    @staticmethod
    def PriorBox(feature, image, min_sizes, max_sizes, aspect_ratios, variance, clip, image_h, image_w, step_h, step_w, offset):
        height, width = feature.shape[-2:]
        box_sizes = []
        for i, size in enumerate(min_sizes):
            box_sizes.append((size, size))
            if max_sizes:
                side = (size * max_sizes[i]) ** 0.5
                box_sizes.append((side, side))
            for ratio in aspect_ratios:
                box_sizes.append((size * ratio ** 0.5, size / ratio ** 0.5))
        sizes = feature.new_tensor(box_sizes)
        yy, xx = torch.meshgrid(torch.arange(height, device=feature.device, dtype=feature.dtype), torch.arange(width, device=feature.device, dtype=feature.dtype), indexing='ij')
        centers = torch.stack(((xx + offset) * step_w, (yy + offset) * step_h), dim=-1).unsqueeze(-2)
        boxes = torch.cat((centers - sizes * 0.5, centers + sizes * 0.5), dim=-1)
        boxes = boxes / feature.new_tensor((image_w, image_h, image_w, image_h))
        if clip:
            boxes = boxes.clamp(0, 1)
        boxes = boxes.reshape(-1)
        variances = feature.new_tensor(variance).repeat(boxes.numel() // 4)
        return torch.stack((boxes, variances), dim=0).unsqueeze(0)

)PY", pyfp);
    }
    if (detectionoutput)
    {
        fputs(R"PY(    @staticmethod
    def DetectionOutput(location, confidence, priorbox, num_classes, nms_threshold, top_k, keep_top_k, confidence_threshold):
        location = location.reshape(1, -1, 4)
        confidence = confidence.reshape(1, -1, num_classes)
        prior, variance = priorbox.reshape(2, -1, 4)
        prior_size = prior[:, 2:] - prior[:, :2]
        prior_center = (prior[:, 2:] + prior[:, :2]) * 0.5
        center = variance[:, :2] * location[0, :, :2] * prior_size + prior_center
        size = torch.exp(variance[:, 2:] * location[0, :, 2:]) * prior_size
        boxes = torch.cat((center - size * 0.5, center + size * 0.5), dim=1)
        detections = []
        for label in range(1, num_classes):
            scores = confidence[0, :, label]
            indices = torch.nonzero(scores > confidence_threshold, as_tuple=False).flatten()
            indices = indices[torch.argsort(scores[indices], descending=True, stable=True)]
            if top_k >= 0:
                indices = indices[:top_k]
            kept = []
            while indices.numel():
                index = indices[0]
                kept.append(index)
                rest = indices[1:]
                lower = torch.maximum(boxes[index, :2], boxes[rest, :2])
                upper = torch.minimum(boxes[index, 2:], boxes[rest, 2:])
                intersection = (upper - lower).clamp(min=0).prod(dim=1)
                area = (boxes[index, 2:] - boxes[index, :2]).prod()
                areas = (boxes[rest, 2:] - boxes[rest, :2]).prod(dim=1)
                union = area + areas - intersection
                overlap = torch.where(union > 0, intersection / union, torch.zeros_like(union))
                indices = rest[overlap <= nms_threshold]
            if kept:
                selected = torch.stack(kept)
                rows = torch.cat((boxes.new_zeros((len(kept), 1)), boxes.new_full((len(kept), 1), label), scores[selected, None], boxes[selected]), dim=1)
                detections.append(rows)
        if not detections:
            output = boxes.new_full((1, 1, 1, 7), -1)
            output[0, 0, 0, 0] = 0
            return output
        output = torch.cat(detections)
        if keep_top_k >= 0 and output.shape[0] > keep_top_k:
            order = torch.argsort(output[:, 2], descending=True, stable=True)[:keep_top_k]
            output = output[order]
            output = output[torch.argsort(output[:, 1], stable=True)]
        if output.shape[0] == 0:
            output = boxes.new_full((1, 7), -1)
            output[0, 0] = 0
        return output.reshape(1, 1, -1, 7)

)PY", pyfp);
    }
    if (roi)
    {
        fputs(R"PY(    @staticmethod
    def _roi_pool(feature, roi, pooled_h, pooled_w, spatial_scale, mode, output_dim=0):
        import math
        if feature.shape[0] != 1 or roi.numel() != 5 or float(roi[0]) != 0:
            raise ValueError('Caffe ROI operators require a single image and ROI batch index 0')
        def nearest(value):
            return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
        x1, y1, x2, y2 = [float(v) for v in roi[1:]]
        if mode == 'max':
            x1, y1, x2, y2 = [nearest(v * spatial_scale) for v in (x1, y1, x2, y2)]
            roi_w, roi_h = max(x2 - x1 + 1, 1), max(y2 - y1 + 1, 1)
        elif mode == 'position':
            x1, y1 = nearest(x1) * spatial_scale, nearest(y1) * spatial_scale
            x2, y2 = (nearest(x2) + 1) * spatial_scale, (nearest(y2) + 1) * spatial_scale
            roi_w, roi_h = max(x2 - x1, 0.1), max(y2 - y1, 0.1)
        else:
            x1, y1, x2, y2 = [v * spatial_scale for v in (x1, y1, x2, y2)]
            roi_w, roi_h = max(x2 - x1, 1), max(y2 - y1, 1)
        channels = output_dim if mode == 'position' else feature.shape[1]
        height, width = feature.shape[-2:]
        result = feature.new_zeros((channels, pooled_h, pooled_w))
        bin_h, bin_w = roi_h / pooled_h, roi_w / pooled_w
        for ph in range(pooled_h):
            for pw in range(pooled_w):
                if mode == 'align':
                    hs, ws = max(0., min(y1 + ph * bin_h, height)), max(0., min(x1 + pw * bin_w, width))
                    he, we = max(0., min(y1 + (ph + 1) * bin_h, height)), max(0., min(x1 + (pw + 1) * bin_w, width))
                    gh, gw = math.ceil(he - hs), math.ceil(we - ws)
                    if gh <= 0 or gw <= 0:
                        continue
                    for iy in range(gh):
                        y = max(0., min(hs + (iy + 0.5) * bin_h / gh, height - 1))
                        yl, yh = int(y), min(int(y) + 1, height - 1)
                        for ix in range(gw):
                            x = max(0., min(ws + (ix + 0.5) * bin_w / gw, width - 1))
                            xl, xh = int(x), min(int(x) + 1, width - 1)
                            dy, dx = y - yl, x - xl
                            result[:, ph, pw] += (1 - dy) * ((1 - dx) * feature[0, :, yl, xl] + dx * feature[0, :, yl, xh]) + dy * ((1 - dx) * feature[0, :, yh, xl] + dx * feature[0, :, yh, xh])
                    result[:, ph, pw] /= gh * gw
                    continue
                if mode == 'max':
                    hs, ws = y1 + math.floor(ph * bin_h), x1 + math.floor(pw * bin_w)
                    he, we = y1 + math.ceil((ph + 1) * bin_h), x1 + math.ceil((pw + 1) * bin_w)
                else:
                    hs, ws = math.floor(y1 + ph * bin_h), math.floor(x1 + pw * bin_w)
                    he, we = math.ceil(y1 + (ph + 1) * bin_h), math.ceil(x1 + (pw + 1) * bin_w)
                hs, he = max(0, min(hs, height)), max(0, min(he, height))
                ws, we = max(0, min(ws, width)), max(0, min(we, width))
                if he <= hs or we <= ws:
                    continue
                if mode == 'max':
                    result[:, ph, pw] = feature[0, :, hs:he, ws:we].amax(dim=(-2, -1))
                else:
                    indices = (torch.arange(channels, device=feature.device) * pooled_h + ph) * pooled_w + pw
                    result[:, ph, pw] = feature[0, indices, hs:he, ws:we].mean(dim=(-2, -1))
        return result

    @staticmethod
    def ROIPooling(feature, roi, pooled_h, pooled_w, spatial_scale):
        return caffe._roi_pool(feature, roi, pooled_h, pooled_w, spatial_scale, 'max')

    @staticmethod
    def PSROIPooling(feature, roi, pooled_h, pooled_w, spatial_scale, output_dim):
        return caffe._roi_pool(feature, roi, pooled_h, pooled_w, spatial_scale, 'position', output_dim)

    @staticmethod
    def ROIAlign(feature, roi, pooled_h, pooled_w, spatial_scale):
        return caffe._roi_pool(feature, roi, pooled_h, pooled_w, spatial_scale, 'align')


)PY", pyfp);
    }
    if (proposal)
    {
        fputs(R"PY(    @staticmethod
    def Proposal(scores, deltas, im_info, feat_stride, return_scores):
        import math
        height, width = scores.shape[-2:]
        im_h, im_w, im_scale = [float(v) for v in im_info.reshape(-1)]
        if im_h <= 0 or im_w <= 0 or im_scale <= 0:
            raise ValueError('Caffe Proposal requires positive image height, width and scale')
        sizes = []
        for ratio in (0.5, 1., 2.):
            rw = math.floor(16 / math.sqrt(ratio) + 0.5)
            rh = math.floor(rw * ratio + 0.5)
            for scale in (8., 16., 32.):
                sizes.append((rw * scale, rh * scale))
        sizes = scores.new_tensor(sizes).reshape(9, 1, 1, 2)
        yy, xx = torch.meshgrid(torch.arange(height, dtype=scores.dtype, device=scores.device), torch.arange(width, dtype=scores.dtype, device=scores.device), indexing='ij')
        centers = torch.stack((8 + xx * feat_stride, 8 + yy * feat_stride), dim=-1)
        deltas = deltas.reshape(9, 4, height, width).permute(0, 2, 3, 1)
        centers = centers + deltas[..., :2] * sizes
        sizes = torch.exp(deltas[..., 2:]) * sizes
        boxes = torch.cat((centers - sizes * 0.5, centers + sizes * 0.5), dim=-1).reshape(-1, 4)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, im_w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, im_h - 1)
        scores = scores[0, 9:].reshape(-1)
        keep = ((boxes[:, 2:] - boxes[:, :2] + 1) >= 16 * im_scale).all(dim=1)
        boxes, scores = boxes[keep], scores[keep]
        order = torch.argsort(scores, descending=True, stable=True)[:6000]
        kept = []
        while order.numel() and len(kept) < 300:
            index = order[0]
            kept.append(index)
            rest = order[1:]
            lower = torch.maximum(boxes[index, :2], boxes[rest, :2])
            upper = torch.minimum(boxes[index, 2:], boxes[rest, 2:])
            intersection = (upper - lower + 1).clamp(min=0).prod(dim=1)
            area = (boxes[index, 2:] - boxes[index, :2] + 1).prod()
            areas = (boxes[rest, 2:] - boxes[rest, :2] + 1).prod(dim=1)
            order = rest[intersection / (area + areas - intersection) <= 0.7]
        indices = torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)
        rois = torch.cat((boxes.new_zeros((indices.numel(), 1)), boxes[indices]), dim=1)
        if return_scores:
            return rois, scores[indices, None]
        return rois

)PY", pyfp);
    }
    if (yolo)
    {
        fputs(R"PY(    @staticmethod
    def _yolo(heads, num_classes, num_box, confidence_threshold, nms_threshold, biases, mask=None, anchors_scale=None):
        all_boxes, all_scores, all_labels, all_areas = [], [], [], []
        for head_index, head in enumerate(heads):
            height, width = head.shape[-2:]
            values = head.reshape(num_box, num_classes + 5, height, width).permute(0, 2, 3, 1)
            yy, xx = torch.meshgrid(torch.arange(height, dtype=head.dtype, device=head.device), torch.arange(width, dtype=head.dtype, device=head.device), indexing='ij')
            grid = torch.stack((xx, yy), dim=-1)
            centers = (values[..., :2].sigmoid() + grid) / head.new_tensor((width, height))
            anchors = head.new_tensor(biases).reshape(-1, 2)
            if mask is None:
                sizes = values[..., 2:4].exp() * anchors.reshape(num_box, 1, 1, 2) / head.new_tensor((width, height))
                class_scores, labels = values[..., 5:].softmax(dim=-1).max(dim=-1)
                scores = values[..., 4].sigmoid() * class_scores
            else:
                indices = torch.tensor(mask[head_index * num_box:(head_index + 1) * num_box], dtype=torch.long, device=head.device)
                anchors = anchors[indices].reshape(num_box, 1, 1, 2)
                net_shape = (head.new_tensor(anchors_scale[head_index]) * head.new_tensor((width, height))).trunc()
                sizes = values[..., 2:4].exp() * anchors / net_shape
                class_scores, labels = values[..., 5:].max(dim=-1)
                # preserve the historical ncnn / Caffe deployment-layer formula
                scores = 1 / (1 + torch.exp(-values[..., 4]) * (1 + torch.exp(-class_scores)))
            boxes = torch.cat((centers - sizes * 0.5, centers + sizes * 0.5), dim=-1)
            areas = sizes.prod(dim=-1) if mask is not None else (boxes[..., 2:] - boxes[..., :2]).prod(dim=-1)
            keep = scores >= confidence_threshold
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep] + 1)
            all_areas.append(areas[keep])
        boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
        labels, areas = torch.cat(all_labels), torch.cat(all_areas)
        order = torch.argsort(scores, descending=True, stable=True)
        kept = []
        while order.numel():
            index = order[0]
            kept.append(index)
            rest = order[1:]
            lower = torch.maximum(boxes[index, :2], boxes[rest, :2])
            upper = torch.minimum(boxes[index, 2:], boxes[rest, 2:])
            intersection = (upper - lower).clamp(min=0).prod(dim=-1)
            union = areas[index] + areas[rest] - intersection
            # these legacy layers suppress boxes across classes
            suppress = intersection > nms_threshold * union if mask is not None else intersection / union > nms_threshold
            order = rest[~suppress]
        indices = torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)
        return torch.cat((labels[indices, None].to(boxes.dtype), scores[indices, None], boxes[indices]), dim=1)

    @staticmethod
    def YoloDetectionOutput(*heads, num_classes, num_box, confidence_threshold, nms_threshold, biases):
        return caffe._yolo(heads, num_classes, num_box, confidence_threshold, nms_threshold, biases)

    @staticmethod
    def Yolov3DetectionOutput(*heads, num_classes, num_box, confidence_threshold, nms_threshold, biases, mask, anchors_scale):
        return caffe._yolo(heads, num_classes, num_box, confidence_threshold, nms_threshold, biases, mask, anchors_scale)

)PY", pyfp);
    }
    fprintf(pyfp, "\n");
}

// *INDENT-ON*
// clang-format on

} // namespace pnnx
