/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/image_utils.cc
 */
#include "vlm_utils.h"

#include <cmath>

namespace mlc {
namespace llm {

void CalculateResizeShape(tvm::runtime::Tensor image_data, std::string model_type,
                          int* p_target_height, int* p_target_width) {
  ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  int height = image_data->shape[1];
  int width = image_data->shape[2];

  if ("phi3_v" == model_type) {
    const int hd_num = 4;
    double ratio = static_cast<double>(width) / height;
    int scale = 1;
    while (scale * std::ceil(scale / ratio) <= hd_num) {
      scale += 1;
    }
    scale -= 1;
    *p_target_width = static_cast<int>(scale * 336);
    *p_target_height = static_cast<int>(*p_target_width / ratio);
  } else if ("qwen3_vl" == model_type) {
    // Qwen3-VL uses 16x16 patches with spatial_merge_size=2
    // Resize to nearest multiple of 32 (16 * 2)
    const int patch_size = 16;
    const int merge_size = 2;
    const int unit = patch_size * merge_size;  // 32
    
    // Keep aspect ratio, resize to multiple of 32
    int target_h = ((height + unit - 1) / unit) * unit;
    int target_w = ((width + unit - 1) / unit) * unit;
    
    // Limit max size to prevent memory issues
    const int max_size = 1024;
    if (target_h > max_size) target_h = max_size;
    if (target_w > max_size) target_w = max_size;
    
    *p_target_height = target_h;
    *p_target_width = target_w;
  }
}

void CalculatePadShape(tvm::runtime::Tensor image_data, std::string model_type, int* p_pad_height,
                       int* p_pad_width) {
  ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  if ("phi3_v" == model_type) {
    int resized_height = 0, resized_width = 0;
    CalculateResizeShape(image_data, model_type, &resized_height, &resized_width);
    int tar = (int)(ceil(resized_height / 336.0) * 336);
    int top_padding = (int)((tar - resized_height) / 2);
    int bottom_padding = tar - resized_height - top_padding;
    ICHECK_EQ(tar, resized_height + top_padding + bottom_padding) << "Padding size not equal!";
    *p_pad_height = tar;
    *p_pad_width = resized_width;
  } else if ("qwen3_vl" == model_type) {
    // Qwen3-VL: use resized dimensions directly (no additional padding)
    CalculateResizeShape(image_data, model_type, p_pad_height, p_pad_width);
  }
}

void CalculateCropShape(tvm::runtime::Tensor image_data, std::string model_type, int* p_crop_height,
                        int* p_crop_width) {
  ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  if ("phi3_v" == model_type) {
    int pad_h = 0, pad_w = 0;
    CalculatePadShape(image_data, model_type, &pad_h, &pad_w);
    *p_crop_height = pad_h / 336;
    *p_crop_width = pad_w / 336;
  } else if ("qwen3_vl" == model_type) {
    // Qwen3-VL: crop dimensions in 32x32 units (16 * 2)
    int pad_h = 0, pad_w = 0;
    CalculatePadShape(image_data, model_type, &pad_h, &pad_w);
    *p_crop_height = pad_h / 32;
    *p_crop_width = pad_w / 32;
  }
}

}  // namespace llm
}  // namespace mlc
