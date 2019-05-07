#include <inttypes.h>
#include <math.h>

// typedef uint16_t fixed_point_t;

// double fixed_to_float(fixed_point_t input, uint8_t frac_bits);
// fixed_point_t float_to_fixed(double input, uint8_t frac_bits);

int saturate(int x, uint8_t nBits);


void basic_convolve(const int8_t *a, // activation, input, image
					const int8_t *w, //weights
					const int8_t *b, //bias
					const uint16_t dim_X_in,
					const uint16_t dim_Y_in,
					const uint16_t ch_in,
					const uint16_t ch_out,
					const uint8_t KX,
					const uint8_t KY,
					const uint8_t STRIDE_X,
					const uint8_t STRIDE_Y,
					const uint8_t PAD_X,
					const uint8_t PAD_Y,
					const uint16_t dim_X_out,
					const uint16_t dim_Y_out,
					const uint16_t frac_bits,
					const uint8_t bias_shift,
					const uint8_t out_shift,
					uint8_t *im_out,
					int *bufferA,
					int *bufferB);

void depthwise_convolve(const int8_t *a, // activation, input, image
						const int8_t *w, //weights
						const int8_t *b, //bias
						const uint16_t dim_X_in,
						const uint16_t dim_Y_in,
						const uint16_t ch_in,
						const uint16_t ch_out,
						const uint8_t KX,
						const uint8_t KY,
						const uint8_t STRIDE_X,
						const uint8_t STRIDE_Y,
						const uint8_t PAD_X,
						const uint8_t PAD_Y,
						const uint16_t dim_X_out,
						const uint16_t dim_Y_out,
						const uint16_t frac_bits,
						const uint8_t bias_shift,
						const uint8_t out_shift,
						uint8_t *im_out,
						int *bufferA, //16bit
						int *bufferB); //8bit

void relu(int8_t *a, const uint32_t in_size);

void fully_connected(const int8_t *a,
					 const int8_t *w, 
					 const int8_t *b,
					 const uint32_t ch_in,
					 const uint32_t ch_out,
					 const uint8_t bias_shift,
					 const uint8_t out_shift,
					 int8_t *out);

void sigmoid(int8_t *a, const uint32_t in_size);