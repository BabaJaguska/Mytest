#include <stdio.h>
#include "typeTest.h"
#include "weights.h"
#include "activationShifts.h"
#include "parameters.h"
#include "sampleInput.h"
//#include <stdint.h>

//#define FIXED_FRAC_BITS 5
//#define PRIu16 "hu"

//typedef uint16_t fixed_point_t;


static int8_t inputImage[CONV0_IN_X * CONV0_IN_Y * CONV0_IN_CH] = SAMPLE_INPUT;
static int8_t conv0_wt[CONV0_IN_CH * CONV0_OUT_CH * CONV0_KX * CONV0_KY] = CONV0_WT;
static int8_t conv0_bias[CONV0_OUT_CH] = CONV0_BIAS;
static int8_t conv1_ds_wt[CONV1_DS_IN_CH * CONV1_DS_KX * CONV1_DS_KY] = CONV1_DS_WT;
static int8_t conv1_ds_bias[CONV1_DS_OUT_CH] = CONV1_DS_BIAS;
static int8_t conv2_pw_wt[CONV2_PW_IN_CH*CONV2_PW_OUT_CH * CONV2_PW_KX * CONV2_PW_KY] = CONV2_PW_WT;
static int8_t conv2_pw_bias[CONV2_PW_OUT_CH] = CONV2_PW_BIAS;
static int8_t conv3_ds_wt[CONV3_DS_IN_CH * CONV3_DS_KX * CONV3_DS_KY] = CONV3_DS_WT;
static int8_t conv3_ds_bias[CONV3_DS_OUT_CH] = CONV3_DS_BIAS;
static int8_t conv4_pw_wt[CONV4_PW_IN_CH * CONV4_PW_OUT_CH * CONV4_PW_KX * CONV4_PW_KY] = CONV4_PW_WT;
static int8_t conv4_pw_bias[CONV4_PW_OUT_CH] = CONV4_PW_BIAS;
static int8_t fc5_wt[FC5_IN_CH * FC5_OUT_CH] = FC5_WT;
static int8_t fc5_bias[FC5_OUT_CH] = FC5_BIAS;
static int8_t fc6_wt[FC6_IN_CH * FC6_OUT_CH] = FC6_WT;
static int8_t fc6_bias[FC6_OUT_CH] = FC6_BIAS;

int8_t output1[CONV0_OUT_X * CONV0_OUT_Y * CONV0_OUT_CH*5];
int8_t output2[CONV1_DS_OUT_X * CONV1_DS_OUT_Y * CONV1_DS_OUT_CH*5];


int *b;
int *c;



// conversion fixed - float - fixed - float - fixed - 

// inline double fixed_to_float(fixed_point_t input, uint8_t frac_bits){

// 	return((double)input/(double)(1<<frac_bits));

// }

// inline fixed_point_t float_to_fixed(double input, uint8_t frac_bits){

// 	return((fixed_point_t)(input*(1<<frac_bits)));
// }


inline int saturate(int x, uint8_t nBits)
{
	if (x >=1<<(nBits-1))
	{
		return (1<<(nBits-1))-1;

	}
	if (x < 1>>(nBits-1))
	{
		return 1>>(nBits-1);
	}
	return x;
}


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
					int *bufferB){


	uint16_t i, j, k, l, m, n;
	int8_t clipped;

	int32_t conv_out;
	signed char in_row, in_col;

	// iterate over the output 
	for (i = 0; i < ch_out; i++)
	{
		for (j = 0; j < dim_Y_out; j++)
		{
			for (k = 0; k < dim_X_out; k++)
			{
				// so this is one value in the output
				// accumulator is uint32! or..erm..fixed point 32
				conv_out =  ((uint32_t)b[i]<<bias_shift) + out_shift; //ali zasto + out_shift??

				//iterate over the kernel
				for (m = 0; m < KY; m++)
				{
					for (n = 0; n < KX; n++)
					{
						// where are you currently in the image within a channel
						in_row = STRIDE_Y * j + m - PAD_Y;
						in_col = STRIDE_X * k + n - PAD_X;

						//don't add the padded zeros for no reason
						if (in_row >= 0 && in_row <= dim_Y_in && in_col >= 0 && in_col <= dim_X_in)
						{
							//iterate the input channels
							for (l = 0; l < ch_in; l++)
							{
								// MAC
								// seti se ovo ti je 1D niz i to razvijen kao: red1,col1 pa po svim kanalima,
								// red1, col2 pa po svim kanalima,
								// red1, col3 pa po svim kanalima...
								// a razvijen je ulaz iz HWC

								conv_out += a[(in_row * dim_X_in + in_col) * ch_in + l] * 
											w[i * ch_in * KY * KX + (m * KX + n)*ch_in + l];
											//a tezine ce u .h fajlovima biti reorganizovane u format outCH * KX * KY * inCH

							}

						}

					}

				}

				// e jos treba da ga klipujes i assignujes
				conv_out = (conv_out >> out_shift);
				clipped =  (int8_t)saturate(conv_out, 8);
				im_out[i + (j *  dim_X_out + k) * ch_out] = clipped;

			}

		}

	}	

	return;
}


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
						int *bufferB) //8bit

{

	uint16_t i, j, k, l, m, n;
	int8_t clipped;

	int32_t conv_out;
	signed char in_row, in_col;

	// iterate over the output 
	for (i = 0; i < ch_out; i++)
	{
		for (j = 0; j < dim_Y_out; j++)
		{
			for (k = 0; k < dim_X_out; k++)
			{
				// so this is one value in the output
				// accumulator is uint32! or..erm..fixed point 32
				conv_out =  ((uint32_t)b[i]<<bias_shift) + out_shift; //ali zasto + out_shift??

				//iterate over the kernel
				for (m = 0; m < KY; m++)
				{
					for (n = 0; n < KX; n++)
					{
						// where are you currently in the image within a channel
						in_row = STRIDE_Y * j + m - PAD_Y;
						in_col = STRIDE_X * k + n - PAD_X;

						//don't add the padded zeros for no reason
						if (in_row >= 0 && in_row <= dim_Y_in && in_col >= 0 && in_col <= dim_X_in)
						{
							//iterate the input channels
							for (l = 0; l < ch_in; l++)
							{
								// MAC
								// seti se ovo ti je 1D niz i to razvijen kao: red1,col1 pa po svim kanalima,
								// red1, col2 pa po svim kanalima,
								// red1, col3 pa po svim kanalima...
								// a razvijen je ulaz iz HWC

								conv_out += a[(in_row * dim_X_in + in_col) * ch_in + l] * 
											w[(m * KX + n) * ch_out + i];
											//a tezine ce u .h fajlovima biti reorganizovane u format 1 * KX * KY * inCH
											
                    
               

							}

						}

					}

				}

				// e jos treba da ga klipujes i assignujes
				conv_out = (conv_out >> out_shift);
				clipped =  (int8_t)saturate(conv_out, 8);
				im_out[i + (j *  dim_X_out + k) * ch_out] = clipped;

			}

		}

	}	

return;


}

void relu(int8_t *a, const uint32_t in_size)
{	int i;

	for (i = 0; i < in_size; i++)
	{
		if (a[i] < 0){
			a[i] = 0;
		}
	} 
	return;
}

void fully_connected(const int8_t *a,
					 const int8_t *w, 
					 const int8_t *b,
					 const uint32_t ch_in,
					 const uint32_t ch_out,
					 const uint8_t bias_shift,
					 const uint8_t out_shift,
					 int8_t *out)
{
    int i, j, ip_out;
    int8_t clipped;

    for (i = 0; i < ch_out; i++)
    {
        // accumulator
        ip_out = ((int32_t)(b[i]) << bias_shift) + out_shift; //a sto PLUS i sta je NN_ROUND(out_shift)

        for (j = 0; j < ch_in; j++)
        {
            ip_out += a[j] * w[i * ch_in + j];
        }
        		
        ip_out = (ip_out >> out_shift);
		clipped =  (int8_t)saturate(ip_out, 8);
		out[i] = clipped;
    }
    return;
}


void sigmoid(int8_t *a, const uint32_t in_size)
{
	int i;
	int8_t temp;

	for (i = 0; i< in_size; i++)
	{
		a[i]<0 ? temp = -a[i]: a[i];
		a[i] = a[i]/(1 + temp);
	}


}

int main(){

	uint16_t n = 12;
	double t = 13.44;
	uint8_t fracs = 3;
	//fixed_point_t r;
	double vrati;
	uint8_t *out1 = output1;
	uint8_t *out2 = output2;
	int i;



	// printf("So here's your number: %" PRIu16 "\n",n);
	// printf("And again: %u\n", n);

	// r = float_to_fixed(t, fracs);

	// printf("Eo ti jean fix: %u\n", r);

	// vrati = fixed_to_float(r,fracs);

	// printf("A kad vratis: %f\n", vrati);


	printf("Broj ulaznih kanala: %u \n", CONV0_IN_CH);
	printf("Broj izlaznih kanala iz prvog sloja: %u \n", CONV0_OUT_CH);
	printf("X dim slike: %u \n", CONV0_IN_X);
	printf("Y dim slike: %u \n", CONV0_IN_Y);

	

	basic_convolve(inputImage, //image input
					conv0_wt,
					conv0_bias, //bias
					CONV0_IN_X,
					CONV0_IN_Y,
					CONV0_IN_CH,
					CONV0_OUT_CH,
					CONV0_KX,
					CONV0_KY,
					CONV0_STRIDE_X,
					CONV0_STRIDE_Y,
					CONV0_PADDING_X,
					CONV0_PADDING_Y,
					CONV0_OUT_X,
					CONV0_OUT_Y,
					2,
					CONV0_BIAS_LSHIFT,
					CONV0_OUT_RSHIFT,
					out1,
					b,
					c);

	relu(out1, CONV0_OUT_X * CONV0_OUT_Y * CONV0_OUT_CH);


	depthwise_convolve(out1, //image input
					conv1_ds_wt,
					conv1_ds_bias, //bias
					CONV1_DS_IN_X,
					CONV1_DS_IN_Y,
					CONV1_DS_IN_CH,
					CONV1_DS_OUT_CH,
					CONV1_DS_KX,
					CONV1_DS_KY,
					CONV1_DS_STRIDE_X,
					CONV1_DS_STRIDE_Y,
					CONV1_DS_PADDING_X,
					CONV1_DS_PADDING_Y,
					CONV1_DS_OUT_X,
					CONV1_DS_OUT_Y,
					2,
					CONV1_DS_BIAS_LSHIFT,
					CONV1_DS_OUT_RSHIFT,
					out2,
					b,
					c);

	basic_convolve(out2, //image input
					conv2_pw_wt,
					conv2_pw_bias, //bias
					CONV2_PW_IN_X,
					CONV2_PW_IN_Y,
					CONV2_PW_IN_CH,
					CONV2_PW_OUT_CH,
					CONV2_PW_KX,
					CONV2_PW_KY,
					CONV2_PW_STRIDE_X,
					CONV2_PW_STRIDE_Y,
					CONV2_PW_PADDING_X,
					CONV2_PW_PADDING_Y,
					CONV2_PW_OUT_X,
					CONV2_PW_OUT_Y,
					2,
					CONV2_PW_BIAS_LSHIFT,
					CONV2_PW_OUT_RSHIFT,
					out1,
					b,
					c);

	relu(out1, CONV2_PW_OUT_X * CONV2_PW_OUT_Y * CONV2_PW_OUT_CH);


	depthwise_convolve(out1, //image input
					conv3_ds_wt,
					conv3_ds_bias, //bias
					CONV3_DS_IN_X,
					CONV3_DS_IN_Y,
					CONV3_DS_IN_CH,
					CONV3_DS_OUT_CH,
					CONV3_DS_KX,
					CONV3_DS_KY,
					CONV3_DS_STRIDE_X,
					CONV3_DS_STRIDE_Y,
					CONV3_DS_PADDING_X,
					CONV3_DS_PADDING_Y,
					CONV3_DS_OUT_X,
					CONV3_DS_OUT_Y,
					2,
					CONV3_DS_BIAS_LSHIFT,
					CONV3_DS_OUT_RSHIFT,
					out2,
					b,
					c);

	basic_convolve(out2, //image input
					conv4_pw_wt,
					conv4_pw_bias, //bias
					CONV4_PW_IN_X,
					CONV4_PW_IN_Y,
					CONV4_PW_IN_CH,
					CONV4_PW_OUT_CH,
					CONV4_PW_KX,
					CONV4_PW_KY,
					CONV4_PW_STRIDE_X,
					CONV4_PW_STRIDE_Y,
					CONV4_PW_PADDING_X,
					CONV4_PW_PADDING_Y,
					CONV4_PW_OUT_X,
					CONV4_PW_OUT_Y,
					2,
					CONV4_PW_BIAS_LSHIFT,
					CONV4_PW_OUT_RSHIFT,
					out1,
					b,
					c);

	relu(out1, CONV4_PW_OUT_X * CONV4_PW_OUT_Y * CONV4_PW_OUT_CH);

	fully_connected(out1,
					fc5_wt, 
					fc5_bias,
					FC5_IN_CH,
					FC5_OUT_CH,
					FC5_BIAS_LSHIFT,
					FC5_OUT_RSHIFT,
					out2);

	relu(out2, FC5_OUT_CH);

	fully_connected(out2,
					fc6_wt, 
					fc6_bias,
					FC6_IN_CH,
					FC6_OUT_CH,
					FC6_BIAS_LSHIFT,
					FC6_OUT_RSHIFT,
					out1);

	sigmoid(out1, FC6_OUT_CH);


	for (i = 0; i< FC6_OUT_CH; i++)
	{
		printf("%d, ", out1[i]);
	} 
	printf("\n");


return 1;

}
