#include "../include/dataset.h"

#include <stdlib.h>
#include <string.h>

static char *lnn_strdup(const char *s) {
    size_t len;
    char *dup;

    if (!s) return NULL;
    len = strlen(s);
    dup = (char *)malloc(len + 1);
    if (!dup) return NULL;
    memcpy(dup, s, len + 1);
    return dup;
}

static int dataset_input_size(const ImageDataset *dataset) {
    return dataset->channels * dataset->width * dataset->height;
}

int image_dataset_init(ImageDataset *dataset,
                       int width,
                       int height,
                       int channels,
                       int class_count,
                       JCVImreadMode mode,
                       JCVInterpolation interpolation) {
    if (!dataset || width <= 0 || height <= 0 || class_count <= 0) return -1;
    if (channels != 1 && channels != 3) return -1;

    memset(dataset, 0, sizeof(*dataset));
    dataset->width = width;
    dataset->height = height;
    dataset->channels = channels;
    dataset->class_count = class_count;
    dataset->imread_mode = mode;
    dataset->interpolation = interpolation;
    dataset->channel_order = JCV_CHANNEL_ORDER_RGB;
    dataset->normalize_01 = 0;
    dataset->batch_size = 1;
    dataset->cache_enabled = 0;
    dataset->prefetch_count = 0;
    return 0;
}

void image_dataset_free(ImageDataset *dataset) {
    if (!dataset) return;

    for (int i = 0; i < dataset->sample_count; ++i) {
        free(dataset->samples[i].path);
    }
    free(dataset->samples);
    free(dataset->cached_inputs);
    free(dataset->cached_targets);

    memset(dataset, 0, sizeof(*dataset));
}

int image_dataset_add(ImageDataset *dataset, const char *image_path, int label) {
    char *path_dup;

    if (!dataset || !image_path) return -1;
    if (label < 0 || label >= dataset->class_count) return -1;

    if (dataset->sample_count == dataset->sample_capacity) {
        int new_capacity = (dataset->sample_capacity == 0) ? 16 : dataset->sample_capacity * 2;
        ImageDatasetSample *new_samples =
            (ImageDatasetSample *)realloc(dataset->samples, (size_t)new_capacity * sizeof(ImageDatasetSample));
        if (!new_samples) return -1;
        dataset->samples = new_samples;
        dataset->sample_capacity = new_capacity;
    }

    path_dup = lnn_strdup(image_path);
    if (!path_dup) return -1;

    dataset->samples[dataset->sample_count].path = path_dup;
    dataset->samples[dataset->sample_count].label = label;
    dataset->sample_count++;

    free(dataset->cached_inputs);
    free(dataset->cached_targets);
    dataset->cached_inputs = NULL;
    dataset->cached_targets = NULL;
    dataset->cached_input_size = 0;

    return 0;
}

int image_dataset_map_normalize(ImageDataset *dataset) {
    if (!dataset) return -1;
    dataset->normalize_01 = 1;
    return 0;
}

int image_dataset_batch(ImageDataset *dataset, int batch_size) {
    if (!dataset || batch_size <= 0) return -1;
    dataset->batch_size = batch_size;
    return 0;
}

int image_dataset_cache(ImageDataset *dataset) {
    if (!dataset) return -1;
    dataset->cache_enabled = 1;
    return 0;
}

int image_dataset_prefetch(ImageDataset *dataset, int prefetch_count) {
    if (!dataset || prefetch_count < 0) return -1;
    dataset->prefetch_count = prefetch_count;
    return 0;
}

int image_dataset_map_channel_order(ImageDataset *dataset, JCVChannelOrder channel_order) {
    if (!dataset) return -1;
    if (channel_order != JCV_CHANNEL_ORDER_RGB && channel_order != JCV_CHANNEL_ORDER_BGR) return -1;
    dataset->channel_order = channel_order;
    return 0;
}

int image_dataset_build_cache(ImageDataset *dataset) {
    int input_size;

    if (!dataset || dataset->sample_count <= 0) return -1;

    input_size = dataset_input_size(dataset);

    free(dataset->cached_inputs);
    free(dataset->cached_targets);
    dataset->cached_inputs = NULL;
    dataset->cached_targets = NULL;

    dataset->cached_inputs = (float *)malloc((size_t)dataset->sample_count * (size_t)input_size * sizeof(float));
    dataset->cached_targets = (float *)calloc((size_t)dataset->sample_count * (size_t)dataset->class_count,
                                              sizeof(float));
    if (!dataset->cached_inputs || !dataset->cached_targets) {
        free(dataset->cached_inputs);
        free(dataset->cached_targets);
        dataset->cached_inputs = NULL;
        dataset->cached_targets = NULL;
        return -1;
    }

    for (int i = 0; i < dataset->sample_count; ++i) {
        JCVImage image;
        float *formatted = NULL;
        int formatted_size = 0;

        image.width = 0;
        image.height = 0;
        image.channels = 0;
        image.data = NULL;

        if (jcv_imread(dataset->samples[i].path, dataset->imread_mode, &image) != 0) {
            goto fail;
        }

        if (jcv_format_image_for_model(&image,
                                       dataset->channels,
                                       dataset->width,
                                       dataset->height,
                                       dataset->interpolation,
                                       dataset->channel_order,
                                       dataset->normalize_01,
                                       &formatted,
                                       &formatted_size) != 0) {
            jcv_image_free(&image);
            goto fail;
        }

        jcv_image_free(&image);

        if (formatted_size != input_size) {
            free(formatted);
            goto fail;
        }

        memcpy(&dataset->cached_inputs[i * input_size], formatted, (size_t)input_size * sizeof(float));
        dataset->cached_targets[i * dataset->class_count + dataset->samples[i].label] = 1.0f;
        free(formatted);
    }

    dataset->cached_input_size = input_size;
    return 0;

fail:
    free(dataset->cached_inputs);
    free(dataset->cached_targets);
    dataset->cached_inputs = NULL;
    dataset->cached_targets = NULL;
    dataset->cached_input_size = 0;
    return -1;
}

int image_dataset_train(SequentialModel *model,
                        ImageDataset *dataset,
                        int epochs,
                        float *final_loss_out) {
    int effective_batch;

    if (!model || !dataset || epochs <= 0) return -1;
    if (dataset->sample_count <= 0) return -1;

    if (!dataset->cached_inputs || !dataset->cached_targets || dataset->cached_input_size <= 0) {
        if (image_dataset_build_cache(dataset) != 0) return -1;
    }

    effective_batch = dataset->batch_size;
    if (effective_batch <= 0) effective_batch = 1;
    if (effective_batch > dataset->sample_count) effective_batch = dataset->sample_count;

    (void)dataset->prefetch_count; /* Reserved for future async implementation. */

    return sequential_model_train(model,
                                  dataset->cached_inputs,
                                  dataset->cached_targets,
                                  dataset->sample_count,
                                  dataset->cached_input_size,
                                  dataset->class_count,
                                  epochs,
                                  effective_batch,
                                  final_loss_out);
}
