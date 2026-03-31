#include "../include/libvideo.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/select.h>
#endif

static void sleep_ms(int ms) {
    if (ms <= 0) return;
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    (void)select(0, NULL, NULL, NULL, &tv);
#endif
}

static int image_to_ascii_gray(const JCVImage *image,
                               int out_w,
                               unsigned char **out_gray,
                               int *out_h) {
    JCVImage gray_src;
    JCVImage resized;
    int target_h;

    if (!image || !out_gray || !out_h || !image->data) return -1;
    if (image->width <= 0 || image->height <= 0 || (image->channels != 1 && image->channels != 3)) return -1;
    if (out_w <= 0) return -1;

    target_h = (image->height * out_w * 55) / (image->width * 100);
    if (target_h < 1) target_h = 1;

    gray_src.width = 0;
    gray_src.height = 0;
    gray_src.channels = 0;
    gray_src.data = NULL;
    resized.width = 0;
    resized.height = 0;
    resized.channels = 0;
    resized.data = NULL;

    if (image->channels == 1) {
        if (jcv_resize(image, out_w, target_h, JCV_INTER_LINEAR, &resized) != 0) {
            return -1;
        }
    } else {
        if (jcv_convert_channels(image, 1, &gray_src) != 0) {
            return -1;
        }
        if (jcv_resize(&gray_src, out_w, target_h, JCV_INTER_LINEAR, &resized) != 0) {
            jcv_image_free(&gray_src);
            return -1;
        }
        jcv_image_free(&gray_src);
    }

    *out_gray = resized.data;
    *out_h = resized.height;
    return 0;
}

int libvideo_show_image_ascii(const JCVImage *image, int output_width) {
    static const char *palette = " .:-=+*#%@";
    const int palette_max = 9;
    unsigned char *gray = NULL;
    int out_h = 0;

    if (image_to_ascii_gray(image, output_width, &gray, &out_h) != 0) {
        return -1;
    }

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < output_width; ++x) {
            unsigned char v = gray[y * output_width + x];
            int idx = (v * palette_max) / 255;
            putchar(palette[idx]);
        }
        putchar('\n');
    }

    free(gray);
    return 0;
}

int libvideo_show_image_file_ascii(const char *file_path,
                                   JCVImreadMode mode,
                                   int output_width) {
    JCVImage image;
    int rc;

    image.width = 0;
    image.height = 0;
    image.channels = 0;
    image.data = NULL;

    if (!file_path) return -1;

    if (jcv_imread(file_path, mode, &image) != 0) {
        return -1;
    }

    rc = libvideo_show_image_ascii(&image, output_width);
    jcv_image_free(&image);
    return rc;
}

int libvideo_play_image_sequence_ascii(const char **file_paths,
                                       int frame_count,
                                       JCVImreadMode mode,
                                       int output_width,
                                       int fps,
                                       int loops) {
    int frame_delay_ms;

    if (!file_paths || frame_count <= 0 || output_width <= 0 || fps <= 0 || loops <= 0) {
        return -1;
    }

    frame_delay_ms = 1000 / fps;
    if (frame_delay_ms <= 0) frame_delay_ms = 1;

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < frame_count; ++i) {
            if (!file_paths[i]) return -1;

            /* Clear terminal and place cursor at top-left before each frame. */
            fputs("\033[2J\033[H", stdout);

            if (libvideo_show_image_file_ascii(file_paths[i], mode, output_width) != 0) {
                return -1;
            }

            fflush(stdout);
            sleep_ms(frame_delay_ms);
        }
    }

    return 0;
}
