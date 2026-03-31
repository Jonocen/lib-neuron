#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/lib-neuron.h"

static int parse_int_min(const char *s, int min_value, int *out) {
    char *end = NULL;
    long v;
    if (!s || !out) return -1;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < (long)min_value || v > 1000000L) return -1;
    *out = (int)v;
    return 0;
}

int main(int argc, char **argv) {
    int width;
    int height;
    int epochs;
    int class_count;
    int sample_count;
    int input_size;
    ImageDataset dataset;
    SequentialModel model;
    float final_loss = 0.0f;

    if (argc < 6) {
        fprintf(stderr,
                "Usage:\n"
                "  %s <width> <height> <epochs> <class_count> <label:image> [label:image ...]\n"
                "\n"
                "Example:\n"
                "  %s 28 28 200 2 0:data/cat1.jpg 0:data/cat2.jpg 1:data/dog1.jpg 1:data/dog2.jpg\n",
                argv[0],
                argv[0]);
        return 1;
    }

    if (parse_int_min(argv[1], 1, &width) != 0 ||
        parse_int_min(argv[2], 1, &height) != 0 ||
        parse_int_min(argv[3], 1, &epochs) != 0 ||
        parse_int_min(argv[4], 1, &class_count) != 0) {
        fprintf(stderr, "Invalid numeric arguments\n");
        return 1;
    }

    sample_count = argc - 5;

    if (image_dataset_init(&dataset,
                           width,
                           height,
                           1,
                           class_count,
                           JCV_IMREAD_GRAYSCALE,
                           JCV_INTER_LINEAR) != 0) {
        fprintf(stderr, "Failed to initialize dataset\n");
        return 1;
    }

    for (int i = 0; i < sample_count; ++i) {
        const char *pair = argv[5 + i];
        const char *sep = strchr(pair, ':');
        int label;
        const char *image_path;

        if (!sep || sep == pair || *(sep + 1) == '\0') {
            fprintf(stderr, "Invalid sample format: %s (expected label:path)\n", pair);
            image_dataset_free(&dataset);
            return 1;
        }

        {
            char label_buf[32];
            size_t label_len = (size_t)(sep - pair);
            if (label_len >= sizeof(label_buf)) {
                fprintf(stderr, "Label too long in sample: %s\n", pair);
                image_dataset_free(&dataset);
                return 1;
            }
            memcpy(label_buf, pair, label_len);
            label_buf[label_len] = '\0';
            if (parse_int_min(label_buf, 0, &label) != 0 || label >= class_count) {
                fprintf(stderr, "Invalid label in sample: %s\n", pair);
                image_dataset_free(&dataset);
                return 1;
            }
        }

        image_path = sep + 1;
        if (image_dataset_add(&dataset, image_path, label) != 0) {
            fprintf(stderr, "Failed to add dataset sample: %s\n", pair);
            image_dataset_free(&dataset);
            return 1;
        }
    }

    image_dataset_map_normalize(&dataset);
    image_dataset_batch(&dataset, sample_count < 8 ? sample_count : 8);
    image_dataset_cache(&dataset);
    image_dataset_prefetch(&dataset, 2);

    if (image_dataset_build_cache(&dataset) != 0) {
        fprintf(stderr, "Failed to build dataset cache\n");
        image_dataset_free(&dataset);
        return 1;
    }

    input_size = dataset.cached_input_size;

    if (sequential_model_init(&model, 3) != 0) {
        fprintf(stderr, "Failed to initialize model\n");
        image_dataset_free(&dataset);
        return 1;
    }

    if (sequential_model_add_dense(&model, input_size, 64, ACT_RELU) != 0 ||
        sequential_model_add_dense(&model, 64, class_count, ACT_SIGMOID) != 0) {
        fprintf(stderr, "Failed to add layers\n");
        sequential_model_free(&model);
        image_dataset_free(&dataset);
        return 1;
    }

    sequential_model_randomize(&model, 0.1f);

    if (sequential_model_compile(&model,
                                 LOSS_BCE,
                                 OPTIMIZER_SGD,
                                 0.01f,
                                 0.9f,
                                 0.999f) != 0) {
        fprintf(stderr, "Compile failed\n");
        sequential_model_free(&model);
        image_dataset_free(&dataset);
        return 1;
    }

    if (image_dataset_train(&model, &dataset, epochs, &final_loss) != 0) {
        fprintf(stderr, "Training failed\n");
        sequential_model_free(&model);
        image_dataset_free(&dataset);
        return 1;
    }

    printf("Training complete. samples=%d classes=%d epochs=%d final_loss=%f\n",
           dataset.sample_count,
           class_count,
           epochs,
           final_loss);

    sequential_model_free(&model);
    image_dataset_free(&dataset);
    return 0;
}
