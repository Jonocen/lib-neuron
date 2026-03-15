#include <lib-neuron.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    float x[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float y[4][1] = {{0}, {1}, {1}, {0}};
    float out[1], final_loss;
    SequentialModel model;

    if (sequential_model_init(&model, 2) != 0 ||
        sequential_model_add_dense(&model, 2, 4, ACT_RELU) != 0 ||
        sequential_model_add_dense(&model, 4, 1, ACT_SIGMOID) != 0) return 1;

    if (sequential_model_randomize(&model, 0.4f) != 0) return 1;

    if (sequential_model_compile(&model, LOSS_HUBER, OPTIMIZER_ADAGRAD, 0.05f, 0.9f, 0.999f) != 0) return 1;
    if (sequential_model_train_with_progress(&model, &x[0][0], &y[0][0], 4, 2, 1, 5000, 1, 10, &final_loss) != 0) return 1;
    printf("final loss = %.6f\n", final_loss);

    puts("predictions:");
    for (int i = 0; i < 4; i++) {
        if (sequential_model_predict(&model, x[i], out) != 0) return 1;
        printf("[%g, %g] -> %.6f\n", x[i][0], x[i][1], out[0]);
    }

    sequential_model_free(&model);
    return 0;
}
