#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H

typedef enum {
	LOSS_MSE = 0,
	LOSS_BCE = 1
} LossFunctionType;

/* Mean Squared Error (MSE). */
float loss_mse(const float *pred, const float *target, int size);

/* Gradient of MSE with respect to predictions. */
int   loss_mse_grad(const float *pred, const float *target, int size, float *grad_out);

/* Binary Cross-Entropy (BCE). */
float loss_bce(const float *pred, const float *target, int size);

/* Gradient of BCE with respect to predictions. */
int   loss_bce_grad(const float *pred, const float *target, int size, float *grad_out);

#endif /* LOSSFUNCTIONS_H */
