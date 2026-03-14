#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

typedef enum {
    OPTIMIZER_SGD = 0,
    OPTIMIZER_ADAM = 1,
    OPTIMIZER_RMSPROP = 2
} OptimizerType;

/*
 * Adam optimizer.
 * Updates weights in-place and keeps per-parameter first/second moments.
 * Returns 0 on success, -1 on invalid input.
 */
int adam_optimizer(float *weights, float *grads, float *m, float *v,
                   float beta1, float beta2, float learning_rate,
                   int t, int size);

/*
 * Stochastic Gradient Descent (SGD).
 * Updates weights in-place.
 * Returns 0 on success, -1 on invalid input.
 */
int sgd_optimizer(float *weights, float *grads, float learning_rate, int size);

int rmsprop_optimizer(float *weights, float *grads, float *cache, float beta, float learning_rate, int size);

#endif /* OPTIMIZERS_H */
