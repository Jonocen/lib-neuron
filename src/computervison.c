#include "../include/computervison.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void jcv_image_reset(JCVImage *image) {
    if (!image) return;
    image->width = 0;
    image->height = 0;
    image->channels = 0;
    image->data = NULL;
}

void jcv_image_free(JCVImage *image) {
    if (!image) return;
    free(image->data);
    jcv_image_reset(image);
}

static int read_ppm_token(FILE *fp, char *buf, size_t buf_size) {
    int ch;
    size_t pos = 0;

    if (!fp || !buf || buf_size == 0) return -1;

    do {
        ch = fgetc(fp);
        if (ch == '#') {
            do {
                ch = fgetc(fp);
            } while (ch != '\n' && ch != EOF);
        }
    } while (ch != EOF && isspace((unsigned char)ch));

    if (ch == EOF) return -1;

    do {
        if (pos + 1 >= buf_size) return -1;
        buf[pos++] = (char)ch;
        ch = fgetc(fp);
    } while (ch != EOF && !isspace((unsigned char)ch));

    buf[pos] = '\0';
    return 0;
}

static int convert_to_gray(const unsigned char *src, int src_channels, int pixels, unsigned char *dst) {
    if (!src || !dst || pixels <= 0) return -1;

    if (src_channels == 1) {
        memcpy(dst, src, (size_t)pixels);
        return 0;
    }

    if (src_channels != 3) return -1;

    for (int i = 0; i < pixels; ++i) {
        const int base = i * 3;
        float r = (float)src[base];
        float g = (float)src[base + 1];
        float b = (float)src[base + 2];
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        if (y < 0.0f) y = 0.0f;
        if (y > 255.0f) y = 255.0f;
        dst[i] = (unsigned char)(y + 0.5f);
    }

    return 0;
}

static int convert_to_rgb(const unsigned char *src, int src_channels, int pixels, unsigned char *dst) {
    if (!src || !dst || pixels <= 0) return -1;

    if (src_channels == 3) {
        memcpy(dst, src, (size_t)pixels * 3u);
        return 0;
    }

    if (src_channels != 1) return -1;

    for (int i = 0; i < pixels; ++i) {
        unsigned char v = src[i];
        int base = i * 3;
        dst[base] = v;
        dst[base + 1] = v;
        dst[base + 2] = v;
    }

    return 0;
}

int jcv_convert_channels(const JCVImage *src, int out_channels, JCVImage *out_image) {
    int pixels;
    size_t bytes;
    unsigned char *dst_data;

    if (!src || !out_image || !src->data || src->width <= 0 || src->height <= 0) return -1;
    if (src->channels != 1 && src->channels != 3) return -1;
    if (out_channels != 1 && out_channels != 3) return -1;

    pixels = src->width * src->height;
    bytes = (size_t)pixels * (size_t)out_channels;
    dst_data = (unsigned char *)malloc(bytes);
    if (!dst_data) return -1;

    if (out_channels == 1) {
        if (convert_to_gray(src->data, src->channels, pixels, dst_data) != 0) {
            free(dst_data);
            return -1;
        }
    } else {
        if (convert_to_rgb(src->data, src->channels, pixels, dst_data) != 0) {
            free(dst_data);
            return -1;
        }
    }

    out_image->width = src->width;
    out_image->height = src->height;
    out_image->channels = out_channels;
    out_image->data = dst_data;
    return 0;
}

static int jcv_imread_pnm_file(const char *file_path, JCVImreadMode mode, JCVImage *out_image) {
    FILE *fp;
    char token[64];
    int width;
    int height;
    int max_val;
    int channels;
    int pixels;
    size_t raw_size;
    unsigned char *raw = NULL;
    JCVImage tmp;
    JCVImage converted;

    if (!file_path || !out_image) return -1;
    jcv_image_reset(out_image);

    fp = fopen(file_path, "rb");
    if (!fp) return -1;

    if (read_ppm_token(fp, token, sizeof(token)) != 0) {
        fclose(fp);
        return -1;
    }

    if (strcmp(token, "P5") == 0) {
        channels = 1;
    } else if (strcmp(token, "P6") == 0) {
        channels = 3;
    } else {
        fclose(fp);
        return -1;
    }

    if (read_ppm_token(fp, token, sizeof(token)) != 0) {
        fclose(fp);
        return -1;
    }
    width = atoi(token);

    if (read_ppm_token(fp, token, sizeof(token)) != 0) {
        fclose(fp);
        return -1;
    }
    height = atoi(token);

    if (read_ppm_token(fp, token, sizeof(token)) != 0) {
        fclose(fp);
        return -1;
    }
    max_val = atoi(token);

    if (width <= 0 || height <= 0 || max_val <= 0 || max_val > 255) {
        fclose(fp);
        return -1;
    }

    pixels = width * height;
    raw_size = (size_t)pixels * (size_t)channels;
    raw = (unsigned char *)malloc(raw_size);
    if (!raw) {
        fclose(fp);
        return -1;
    }

    if (fread(raw, 1, raw_size, fp) != raw_size) {
        free(raw);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    tmp.width = width;
    tmp.height = height;
    tmp.channels = channels;
    tmp.data = raw;

    if (mode == JCV_IMREAD_UNCHANGED) {
        *out_image = tmp;
        return 0;
    }

    jcv_image_reset(&converted);
    if (mode == JCV_IMREAD_GRAYSCALE) {
        if (jcv_convert_channels(&tmp, 1, &converted) != 0) {
            jcv_image_free(&tmp);
            return -1;
        }
    } else {
        if (jcv_convert_channels(&tmp, 3, &converted) != 0) {
            jcv_image_free(&tmp);
            return -1;
        }
    }

    jcv_image_free(&tmp);
    *out_image = converted;
    return 0;
}

static int shell_quote_single(const char *src, char *dst, size_t dst_size) {
    size_t j = 0;

    if (!src || !dst || dst_size < 3) return -1;

    dst[j++] = '\'';
    for (size_t i = 0; src[i] != '\0'; ++i) {
        if (src[i] == '\'') {
            if (j + 4 >= dst_size) return -1;
            dst[j++] = '\'';
            dst[j++] = '\\';
            dst[j++] = '\'';
            dst[j++] = '\'';
        } else {
            if (j + 1 >= dst_size) return -1;
            dst[j++] = src[i];
        }
    }

    if (j + 2 > dst_size) return -1;
    dst[j++] = '\'';
    dst[j] = '\0';
    return 0;
}

static int build_temp_ppm_path(char *out_path, size_t out_size) {
    static int seeded = 0;

    if (!out_path || out_size == 0) return -1;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    for (int attempt = 0; attempt < 32; ++attempt) {
        unsigned int r = ((unsigned int)rand() << 16) ^ (unsigned int)rand();
        if (snprintf(out_path, out_size, "/tmp/libneuron_jcv_%u.ppm", r) <= 0) return -1;

        FILE *probe = fopen(out_path, "rb");
        if (!probe) return 0;
        fclose(probe);
    }

    return -1;
}

static int try_convert_to_ppm(const char *input_path, const char *ppm_path) {
    char q_in[1024];
    char q_out[1024];
    char cmd[2300];
    int rc;

    if (!input_path || !ppm_path) return -1;
    if (shell_quote_single(input_path, q_in, sizeof(q_in)) != 0) return -1;
    if (shell_quote_single(ppm_path, q_out, sizeof(q_out)) != 0) return -1;

    (void)snprintf(cmd, sizeof(cmd), "magick %s %s > /dev/null 2>&1", q_in, q_out);
    rc = system(cmd);
    if (rc == 0) return 0;

    (void)snprintf(cmd, sizeof(cmd), "convert %s %s > /dev/null 2>&1", q_in, q_out);
    rc = system(cmd);
    if (rc == 0) return 0;

    (void)snprintf(cmd, sizeof(cmd), "ffmpeg -y -loglevel error -i %s %s > /dev/null 2>&1", q_in, q_out);
    rc = system(cmd);
    if (rc == 0) return 0;

    return -1;
}

int jcv_imread(const char *file_path, JCVImreadMode mode, JCVImage *out_image) {
    char temp_ppm[256];
    int rc;

    if (!file_path || !out_image) return -1;

    rc = jcv_imread_pnm_file(file_path, mode, out_image);
    if (rc == 0) return 0;

    if (build_temp_ppm_path(temp_ppm, sizeof(temp_ppm)) != 0) {
        return -1;
    }

    if (try_convert_to_ppm(file_path, temp_ppm) != 0) {
        remove(temp_ppm);
        return -1;
    }

    rc = jcv_imread_pnm_file(temp_ppm, mode, out_image);
    remove(temp_ppm);
    return rc;
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static unsigned char bilinear_sample(const JCVImage *src, int channel, float sx, float sy) {
    int x0;
    int y0;
    int x1;
    int y1;
    float dx;
    float dy;
    int idx00;
    int idx01;
    int idx10;
    int idx11;
    float v00;
    float v01;
    float v10;
    float v11;
    float top;
    float bottom;
    float out;

    sx = clampf(sx, 0.0f, (float)(src->width - 1));
    sy = clampf(sy, 0.0f, (float)(src->height - 1));

    x0 = (int)sx;
    y0 = (int)sy;
    x1 = (x0 + 1 < src->width) ? x0 + 1 : x0;
    y1 = (y0 + 1 < src->height) ? y0 + 1 : y0;

    dx = sx - (float)x0;
    dy = sy - (float)y0;

    idx00 = (y0 * src->width + x0) * src->channels + channel;
    idx01 = (y0 * src->width + x1) * src->channels + channel;
    idx10 = (y1 * src->width + x0) * src->channels + channel;
    idx11 = (y1 * src->width + x1) * src->channels + channel;

    v00 = (float)src->data[idx00];
    v01 = (float)src->data[idx01];
    v10 = (float)src->data[idx10];
    v11 = (float)src->data[idx11];

    top = v00 + dx * (v01 - v00);
    bottom = v10 + dx * (v11 - v10);
    out = top + dy * (bottom - top);

    if (out < 0.0f) out = 0.0f;
    if (out > 255.0f) out = 255.0f;
    return (unsigned char)(out + 0.5f);
}

int jcv_resize(const JCVImage *src,
               int new_width,
               int new_height,
               JCVInterpolation interpolation,
               JCVImage *out_image) {
    unsigned char *dst_data;
    float scale_x;
    float scale_y;

    if (!src || !out_image || !src->data) return -1;
    if (src->channels <= 0 || new_width <= 0 || new_height <= 0) return -1;
    if (interpolation != JCV_INTER_NEAREST && interpolation != JCV_INTER_LINEAR) return -1;

    dst_data = (unsigned char *)malloc((size_t)new_width * (size_t)new_height * (size_t)src->channels);
    if (!dst_data) return -1;

    scale_x = (float)src->width / (float)new_width;
    scale_y = (float)src->height / (float)new_height;

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            float sx = ((float)x + 0.5f) * scale_x - 0.5f;
            float sy = ((float)y + 0.5f) * scale_y - 0.5f;

            for (int c = 0; c < src->channels; ++c) {
                int out_idx = (y * new_width + x) * src->channels + c;

                if (interpolation == JCV_INTER_NEAREST) {
                    int nx = (int)(sx + 0.5f);
                    int ny = (int)(sy + 0.5f);
                    if (nx < 0) nx = 0;
                    if (ny < 0) ny = 0;
                    if (nx >= src->width) nx = src->width - 1;
                    if (ny >= src->height) ny = src->height - 1;
                    dst_data[out_idx] = src->data[(ny * src->width + nx) * src->channels + c];
                } else {
                    dst_data[out_idx] = bilinear_sample(src, c, sx, sy);
                }
            }
        }
    }

    out_image->width = new_width;
    out_image->height = new_height;
    out_image->channels = src->channels;
    out_image->data = dst_data;
    return 0;
}

int jcv_image_to_chw_float(const JCVImage *image,
                           int normalize_01,
                           float **out_data,
                           int *out_size) {
    int spatial_size;
    int total_size;
    float scale;
    float *dst;

    if (!image || !out_data || !out_size || !image->data) return -1;
    if (image->width <= 0 || image->height <= 0 || image->channels <= 0) return -1;

    spatial_size = image->width * image->height;
    total_size = spatial_size * image->channels;
    scale = normalize_01 ? (1.0f / 255.0f) : 1.0f;

    dst = (float *)malloc((size_t)total_size * sizeof(float));
    if (!dst) return -1;

    for (int c = 0; c < image->channels; ++c) {
        for (int i = 0; i < spatial_size; ++i) {
            int src_idx = i * image->channels + c;
            int dst_idx = c * spatial_size + i;
            dst[dst_idx] = (float)image->data[src_idx] * scale;
        }
    }

    *out_data = dst;
    *out_size = total_size;
    return 0;
}

int jcv_load_image_for_model(const char *file_path,
                             JCVImreadMode mode,
                             int target_width,
                             int target_height,
                             JCVInterpolation interpolation,
                             int normalize_01,
                             float **out_input,
                             int *out_size,
                             int *out_channels) {
    JCVImage loaded;
    int rc;

    if (!file_path || !out_input || !out_size || !out_channels) return -1;

    *out_input = NULL;
    *out_size = 0;
    *out_channels = 0;

    jcv_image_reset(&loaded);
    rc = jcv_imread(file_path, mode, &loaded);
    if (rc != 0) return -1;

    rc = jcv_format_image_for_model(&loaded,
                                    loaded.channels,
                                    target_width,
                                    target_height,
                                    interpolation,
                                    JCV_CHANNEL_ORDER_RGB,
                                    normalize_01,
                                    out_input,
                                    out_size);
    if (rc != 0) {
        jcv_image_free(&loaded);
        return -1;
    }

    *out_channels = loaded.channels;
    jcv_image_free(&loaded);
    return 0;
}

int jcv_format_image_for_model(const JCVImage *src,
                               int out_channels,
                               int target_width,
                               int target_height,
                               JCVInterpolation interpolation,
                               JCVChannelOrder channel_order,
                               int normalize_01,
                               float **out_input,
                               int *out_size) {
    JCVImage converted;
    JCVImage resized;
    JCVImage reordered;
    const JCVImage *active;
    int rc;

    if (!src || !out_input || !out_size || !src->data) return -1;
    if (src->width <= 0 || src->height <= 0) return -1;
    if (out_channels != 1 && out_channels != 3) return -1;
    if (channel_order != JCV_CHANNEL_ORDER_RGB && channel_order != JCV_CHANNEL_ORDER_BGR) return -1;

    *out_input = NULL;
    *out_size = 0;

    jcv_image_reset(&converted);
    jcv_image_reset(&resized);
    jcv_image_reset(&reordered);

    active = src;

    if (active->channels != out_channels) {
        rc = jcv_convert_channels(active, out_channels, &converted);
        if (rc != 0) return -1;
        active = &converted;
    }

    if (target_width > 0 && target_height > 0 &&
        (active->width != target_width || active->height != target_height)) {
        rc = jcv_resize(active, target_width, target_height, interpolation, &resized);
        if (rc != 0) {
            jcv_image_free(&converted);
            return -1;
        }
        active = &resized;
    }

    if (channel_order == JCV_CHANNEL_ORDER_BGR && active->channels == 3) {
        int pixels = active->width * active->height;
        size_t bytes = (size_t)pixels * 3u;
        reordered.width = active->width;
        reordered.height = active->height;
        reordered.channels = 3;
        reordered.data = (unsigned char *)malloc(bytes);
        if (!reordered.data) {
            jcv_image_free(&resized);
            jcv_image_free(&converted);
            return -1;
        }

        for (int i = 0; i < pixels; ++i) {
            int base = i * 3;
            reordered.data[base] = active->data[base + 2];
            reordered.data[base + 1] = active->data[base + 1];
            reordered.data[base + 2] = active->data[base];
        }
        active = &reordered;
    }

    rc = jcv_image_to_chw_float(active, normalize_01, out_input, out_size);

    jcv_image_free(&reordered);
    jcv_image_free(&resized);
    jcv_image_free(&converted);
    return rc;
}
