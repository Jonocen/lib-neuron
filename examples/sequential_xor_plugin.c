#include <lib-neuron.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    float x[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float y[4][1] = {{0}, {1}, {1}, {0}};
    float out[1], loss;
    SequentialModel model;

    if (sequential_model_init(&model, 2) != 0 ||
        sequential_model_add_dense(&model, 2, 4, ACT_RELU) != 0 ||
        sequential_model_add_dense(&model, 4, 1, ACT_SIGMOID) != 0) return 1;

    for (int l = 0; l < model.num_layers; l++) {
        float *w = model.layers[l].weights(model.layers[l].ctx);
        int nw = model.layers[l].weights_size(model.layers[l].ctx);
        for (int i = 0; i < nw; i++) w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }

    if (sequential_model_compile(&model, LOSS_MSE, OPTIMIZER_ADAM, 0.005f, 0.9f, 0.999f) != 0) return 1;
    if (sequential_model_train(&model, &x[0][0], &y[0][0], 4, 2, 1, 10000, &loss) != 0) return 1;
    printf("final loss = %.6f\n", loss);
    puts("predictions:");
    for (int i = 0; i < 4; i++) {
        if (sequential_model_predict(&model, x[i], out) != 0) return 1;
        printf("[%g, %g] -> %.6f\n", x[i][0], x[i][1], out[0]);
    }

    sequential_model_free(&model);
    return 0;
}
