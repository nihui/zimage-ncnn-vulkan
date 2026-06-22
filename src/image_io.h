// z-image implemented with ncnn library

#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <vector>

#include "mat.h"
#include "filesystem_utils.h"

namespace ImageIO {

int load_image(const path_t& path, ncnn::Mat& image);

int save_image(const path_t& outpath, const ncnn::Mat& outimage, int b, int batch);

} // namespace ImageIO

#endif // IMAGE_IO_H
