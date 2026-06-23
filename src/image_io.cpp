// z-image implemented with ncnn library

#include "image_io.h"

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if _WIN32
#include <objbase.h>
#include "wic_image.h"
#endif
#include "jpeg_image.h"
#include "png_image.h"
#include "webp_image.h"

namespace ImageIO {

static path_t lowercase_path(path_t s)
{
    for (size_t i = 0; i < s.size(); i++)
    {
#if _WIN32
        if (s[i] >= L'A' && s[i] <= L'Z')
            s[i] = s[i] - L'A' + L'a';
#else
        if (s[i] >= 'A' && s[i] <= 'Z')
            s[i] = s[i] - 'A' + 'a';
#endif
    }
    return s;
}

static int read_file_bytes(const path_t& path, std::vector<unsigned char>& bytes)
{
#if _WIN32
    FILE* fp = _wfopen(path.c_str(), L"rb");
#else
    FILE* fp = fopen(path.c_str(), "rb");
#endif
    if (!fp)
        return -1;

    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (len <= 0)
    {
        fclose(fp);
        return -1;
    }

    bytes.resize((size_t)len);
    size_t nread = fread(bytes.data(), 1, bytes.size(), fp);
    fclose(fp);

    return nread == bytes.size() ? 0 : -1;
}

#if _WIN32
static void ensure_wic_initialized()
{
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
}
#endif

static void convert_bgr_to_rgb(ncnn::Mat& image)
{
    if (image.elempack != 3 && image.elempack != 4)
        return;

    for (int i = 0; i < image.w * image.h; i++)
    {
        unsigned char* p = (unsigned char*)image.data + (size_t)i * image.elempack;
        std::swap(p[0], p[2]);
    }
}

static void assign_pixels(unsigned char* pixeldata, int w, int h, int c, ncnn::Mat& image)
{
    image.create(w, h, (size_t)c, c);
    memcpy(image.data, pixeldata, (size_t)w * h * c);
}

int load_image(const path_t& path, ncnn::Mat& image)
{
    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char* pixeldata = 0;

    path_t ext = lowercase_path(get_file_extension(path));

#if _WIN32
    if (ext != PATHSTR("webp"))
    {
        ensure_wic_initialized();

        pixeldata = wic_decode_image(path.c_str(), &w, &h, &c);
        if (pixeldata)
        {
            assign_pixels(pixeldata, w, h, c, image);
            free(pixeldata);
            convert_bgr_to_rgb(image);
            return 0;
        }
    }
#endif

    std::vector<unsigned char> bytes;
    if (read_file_bytes(path, bytes) != 0)
        return -1;

    if (ext == PATHSTR("png"))
        pixeldata = png_load(bytes.data(), (int)bytes.size(), &w, &h, &c);
    else if (ext == PATHSTR("jpg") || ext == PATHSTR("jpeg"))
        pixeldata = jpeg_load(bytes.data(), (int)bytes.size(), &w, &h, &c);
    else if (ext == PATHSTR("webp"))
        pixeldata = webp_load(bytes.data(), (int)bytes.size(), &w, &h, &c);

    if (!pixeldata)
        pixeldata = png_load(bytes.data(), (int)bytes.size(), &w, &h, &c);
    if (!pixeldata)
        pixeldata = jpeg_load(bytes.data(), (int)bytes.size(), &w, &h, &c);
    if (!pixeldata)
        pixeldata = webp_load(bytes.data(), (int)bytes.size(), &w, &h, &c);

    if (!pixeldata)
        return -1;

    assign_pixels(pixeldata, w, h, c, image);
    free(pixeldata);

#if _WIN32
    if (ext == PATHSTR("webp"))
        convert_bgr_to_rgb(image);
#endif

    return 0;
}

int save_image(const path_t& outpath, const ncnn::Mat& outimage, int b, int batch)
{
    path_t ext = get_file_extension(outpath);
    path_t ext_lower = lowercase_path(ext);

    path_t outpath_b = outpath;
    if (batch > 1)
    {
        path_t filename = get_file_name_without_extension(outpath);

#if _WIN32
        wchar_t hnd[256];
        swprintf(hnd, 256, L"-%d.", b);
#else
        char hnd[256];
        sprintf(hnd, "-%d.", b);
#endif
        outpath_b = filename + hnd + ext;
    }

    if (ext_lower == PATHSTR("webp"))
        return webp_save(outpath_b.c_str(), outimage.w, outimage.h, outimage.elempack, (const unsigned char*)outimage.data);

    if (ext_lower == PATHSTR("png"))
    {
#if _WIN32
        ensure_wic_initialized();
        return wic_encode_image(outpath_b.c_str(), outimage.w, outimage.h, outimage.elempack, outimage.data);
#else
        return png_save(outpath_b.c_str(), outimage.w, outimage.h, outimage.elempack, (const unsigned char*)outimage.data);
#endif
    }

    if (ext_lower == PATHSTR("jpg") || ext_lower == PATHSTR("jpeg"))
    {
#if _WIN32
        ensure_wic_initialized();
        return wic_encode_jpeg_image(outpath_b.c_str(), outimage.w, outimage.h, outimage.elempack, outimage.data);
#else
        return jpeg_save(outpath_b.c_str(), outimage.w, outimage.h, outimage.elempack, (const unsigned char*)outimage.data);
#endif
    }

    return 0;
}

} // namespace ImageIO
