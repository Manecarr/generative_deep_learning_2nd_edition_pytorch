def get_meta_lib(): ...
def register_meta(op_name, overload_name: str = "default"): ...
def meta_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned): ...
def meta_roi_align_backward(
    grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned
): ...
def meta_ps_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio): ...
def meta_ps_roi_align_backward(
    grad,
    rois,
    channel_mapping,
    spatial_scale,
    pooled_height,
    pooled_width,
    sampling_ratio,
    batch_size,
    channels,
    height,
    width,
): ...
def meta_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width): ...
def meta_roi_pool_backward(
    grad, rois, argmax, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width
): ...
def meta_ps_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width): ...
def meta_ps_roi_pool_backward(
    grad, rois, channel_mapping, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width
): ...
def meta_nms(dets, scores, iou_threshold): ...
def meta_deform_conv2d(
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    n_weight_grps,
    n_offset_grps,
    use_mask,
): ...
def meta_deform_conv2d_backward(
    grad,
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    offset_groups,
    use_mask,
): ...
