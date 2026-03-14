#include <lib-neuron.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    Layer layers[2];
    float x[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float y[4][1] = {{0}, {1}, {1}, {0}};
    float out[1], loss, epoch_loss;
    float gw0[2 * 4], gb0[4], gw1[4], gb1[1];
    float *grads_w[2] = {gw0, gw1};
    float *grads_b[2] = {gb0, gb1};

    if (layer_init(&layers[0], 2, 4, ACT_RELU) != 0 ||
        layer_init(&layers[1], 4, 1, ACT_SIGMOID) != 0) return 1;

    for (int i = 0; i < 8; i++) layers[0].weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;
    for (int i = 0; i < 4; i++) layers[1].weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;

    for (int epoch = 0; epoch < 5000; epoch++) {
        epoch_loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            if (sequential_train_step_with_loss(layers,
                                                2,
                                                x[i],
                                                y[i],
                                                out,
                                                grads_w,
                                                grads_b,
                                                LOSS_MSE,
                                                OPTIMIZER_SGD,
                                                0.05f,
                                                NULL,
                                                &loss) != 0) return 1;
            epoch_loss += loss;
        }
        if (epoch % 500 == 0) printf("epoch %d loss = %.6f\n", epoch, epoch_loss / 4.0f);
    }

    puts("predictions:");
    for (int i = 0; i < 4; i++) {
        if (sequential_forward(layers, 2, x[i], out) != 0) return 1;
        printf("[%g, %g] -> %.6f\n", x[i][0], x[i][1], out[0]);
    }

    layer_free(&layers[0]);
    layer_free(&layers[1]);
    return 0;
}