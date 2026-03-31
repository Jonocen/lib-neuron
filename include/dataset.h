#ifndef DATASET_H
#define DATASET_H

#include "computervison.h"
#include "models.h"

typedef struct {
    char *path;
    int label;
} ImageDatasetSample;

typedef struct {
    ImageDatasetSample *samples;
    int sample_count;
    int sample_capacity;

    int width;
    int height;
    int channels;
    int class_count;

    JCVImreadMode imread_mode;
    JCVInterpolation interpolation;
    JCVChannelOrder channel_order;

    int normalize_01;
    int batch_size;
    int cache_enabled;
    int prefetch_count;

    float *cached_inputs;
    float *cached_targets;
    int cached_input_size;
} ImageDataset;

/* Initialize an empty image dataset container. */
int image_dataset_init(ImageDataset *dataset,
                       int width,
                       int height,
                       int channels,
                       int class_count,
                       JCVImreadMode mode,
                       JCVInterpolation interpolation);

/* Free all owned memory and reset struct fields. */
void image_dataset_free(ImageDataset *dataset);

/* Add one labeled image sample. */
int image_dataset_add(ImageDataset *dataset, const char *image_path, int label);

/* tf.data-like helpers: configure preprocessing/training behavior. */
int image_dataset_map_normalize(ImageDataset *dataset);
int image_dataset_batch(ImageDataset *dataset, int batch_size);
int image_dataset_cache(ImageDataset *dataset);
int image_dataset_prefetch(ImageDataset *dataset, int prefetch_count);
int image_dataset_map_channel_order(ImageDataset *dataset, JCVChannelOrder channel_order);

/* Build contiguous cached input/target arrays from sample list. */
int image_dataset_build_cache(ImageDataset *dataset);

/* Train a compiled sequential model using dataset settings. */
int image_dataset_train(SequentialModel *model,
                        ImageDataset *dataset,
                        int epochs,
                        float *final_loss_out);

#endif /* DATASET_H */
