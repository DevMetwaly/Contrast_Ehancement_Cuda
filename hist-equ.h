#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

typedef struct{
    int w;
    int h;
    unsigned char * img_r;
    unsigned char * img_g;
    unsigned char * img_b;
} PPM_IMG;

typedef struct{
    int w;
    int h;
    unsigned char * img_y;
    unsigned char * img_u;
    unsigned char * img_v;
} YUV_IMG;


typedef struct
{
    int width;
    int height;
    float * h;
    float * s;
    unsigned char * l;
} HSL_IMG;

    

PPM_IMG read_ppm(const char * path);
void write_ppm(PPM_IMG img, const char * path);
void free_ppm(PPM_IMG img);

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

HSL_IMG rgb2hsl(PPM_IMG img_in);
PPM_IMG hsl2rgb(HSL_IMG img_in);

YUV_IMG rgb2yuv(PPM_IMG img_in);
PPM_IMG yuv2rgb(YUV_IMG img_in);    

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in);
void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);
__global__ void histogram_kernel(int * hist_out, unsigned char * img_in, int img_size);
__global__ void clean_kernel(int *hist_out,int size);

__global__ void genLUT_kernel(int* LUT, int* CDF, int CDFmin, int imgSize, int L);
__global__ void genResultImg_kernel(unsigned char* outimg, unsigned char* img, int* LUT, int imgSize, int L);


//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

//Contrast enhancement for color images
PPM_IMG contrast_enhancement_c_rgb(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in);

// Contrast enhacement using GPU
PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in);

__device__
float Hue_2_RGB_gpu( float v1, float v2, float vH );

// Transformation kernels
__global__ void rgb2yuv_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    unsigned char *img_y, unsigned char *img_u, unsigned char *img_v);
__global__ void yuv2rgb_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    unsigned char *img_y, unsigned char *img_u, unsigned char *img_v);
__global__ void rgb2hsl_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    float *img_h, float *img_s, unsigned char *img_l);
__global__ void hsl2rgb_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    float *img_h, float *img_s, unsigned char *img_l);
	

#endif
