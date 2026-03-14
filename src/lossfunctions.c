#include "../include/lossfunctions.h"
#include <math.h>
#include <stdlib.h>

//Mean Squared Error: L = sum((pred - target)^2) / size

float loss_mse(const float *pred, const float *target, int size) {
	if (!pred || !target || size <= 0) return -1.0f;
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		float d = pred[i] - target[i];
		sum += d * d;
	}
	return sum / (float)size;
}

int loss_mse_grad(const float *pred, const float *target, int size, float *grad_out) {
	if (!pred || !target || !grad_out || size <= 0) return -1;
	for (int i = 0; i < size; i++)
		grad_out[i] = 2.0f * (pred[i] - target[i]) / (float)size;
	return 0;
}

//Binary Cross-Entropy: L = -sum(t*log(p) + (1-t)*log(1-p)) / size

float loss_bce(const float *pred, const float *target, int size) {
	if (!pred || !target || size <= 0) return -1.0f;
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		float p = pred[i];
		/* clamp to avoid log(0) */
		if (p < 1e-7f) p = 1e-7f;
		if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
		sum += -(target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p));
	}
	return sum / (float)size;
}

int loss_bce_grad(const float *pred, const float *target, int size, float *grad_out) {
	if (!pred || !target || !grad_out || size <= 0) return -1;
	for (int i = 0; i < size; i++) {
		float p = pred[i];
		if (p < 1e-7f) p = 1e-7f;
		if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
		grad_out[i] = (-target[i] / p + (1.0f - target[i]) / (1.0f - p)) / (float)size;
	}
	return 0;
}
