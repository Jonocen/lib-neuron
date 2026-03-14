#include "../include/optimizers.h"
#include <math.h>
#include <stdlib.h>

int adam_optimizer(float *weights, float *grads, float *m, float *v,
                   float beta1, float beta2, float learning_rate,
                   int t, int size) {
    if (!weights || !grads || !m || !v || size <= 0 || t <= 0) return -1;
    if (learning_rate <= 0.0f) return -1;
    if (beta1 <= 0.0f || beta1 >= 1.0f || beta2 <= 0.0f || beta2 >= 1.0f) return -1;

    float beta1_correction = 1.0f - powf(beta1, (float)t);
    float beta2_correction = 1.0f - powf(beta2, (float)t);
    if (beta1_correction == 0.0f || beta2_correction == 0.0f) return -1;

    for (int i = 0; i < size; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];

        float m_hat = m[i] / beta1_correction;
        float v_hat = v[i] / beta2_correction;

        weights[i] -= learning_rate * m_hat / (sqrtf(v_hat) + 1e-8f);
    }

    return 0;
}

int sgd_optimizer(float *weights, float *grads, float learning_rate, int size) {
    if (!weights || !grads || size <= 0 || learning_rate <= 0.0f) return -1;

    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * grads[i];
    }

    return 0;
}

int rmsprop_optimizer(float *weights, float *grads, float *cache,
                       float beta, float learning_rate, int size) {
    if (!weights || !grads || !cache || size <= 0) return -1;
    if (learning_rate <= 0.0f) return -1;
    if (beta <= 0.0f || beta >= 1.0f) return -1;

    for (int i = 0; i < size; i++) {
        cache[i] = beta * cache[i] + (1.0f - beta) * grads[i] * grads[i];
        weights[i] -= learning_rate * grads[i] / (sqrtf(cache[i]) + 1e-8f);
    }

    return 0;
}

int adagrad_optimizer(float *weights, float *grads, float *accumulator,
                      float learning_rate, int size) {
    if (!weights || !grads || !accumulator || size <= 0) return -1;
    if (learning_rate <= 0.0f) return -1;

    for (int i = 0; i < size; i++) {
        accumulator[i] += grads[i] * grads[i];
        weights[i] -= learning_rate * grads[i] / (sqrtf(accumulator[i]) + 1e-8f);
    }

    return 0;
}