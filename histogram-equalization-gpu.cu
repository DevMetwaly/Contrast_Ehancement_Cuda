#include "hist-equ.h"
using namespace std;


__global__ 
void histogram_kernal(int * hist_out, unsigned char * img_in, int img_size){   
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < img_size) atomicAdd(&hist_out[img_in[i]], 1);    
}


__global__
void clean_kernal(int *hist_out,int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < size) hist_out[i] = 0;    
}


void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    unsigned char *d_img_in;
    int *d_hist_out;

    cudaMalloc(&d_img_in, img_size*sizeof(unsigned char));
    cudaMalloc(&d_hist_out, nbr_bin*sizeof(int));
    cudaMemcpy(d_img_in, img_in, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256;
    int blocksPerGrid =(nbr_bin + threadsPerBlock - 1) / threadsPerBlock;
    clean_kernal<<<blocksPerGrid, threadsPerBlock>>>(d_hist_out, nbr_bin);
    
    cudaDeviceSynchronize();
    
    threadsPerBlock = 256;
    blocksPerGrid =(img_size + threadsPerBlock - 1) / threadsPerBlock;
    histogram_kernal<<<blocksPerGrid, threadsPerBlock>>>(d_hist_out, d_img_in, img_size);

    cudaDeviceSynchronize();

    cudaMemcpy(hist_out, d_hist_out, sizeof(int)*nbr_bin, cudaMemcpyDeviceToHost);
    cudaFree(d_img_in);
    cudaFree(d_hist_out);
    return;
}






__global__
void genLUT_kernel(int* LUT, int* CDF, int CDFmin, int imgSize, int L){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < L){

        LUT[idx] = (int)(((float)CDF[idx] - CDFmin) * (L-1) / (imgSize - CDFmin) + 0.5);

        if(LUT[idx] < 0)
            LUT[idx] = 0;
        else if(LUT[idx] > L)
            LUT[idx] = L;
    }
}


__global__
void genResultImg_kernel(unsigned char* outimg, unsigned char* img, int* LUT, int imgSize){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < imgSize) outimg[idx] = (unsigned char)LUT[img[idx]];

}


void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, 
                             int * hist_in, int img_size, int nbr_bin){

    int *CDF = (int *) malloc(sizeof(int) * nbr_bin);
    CDF[0] = hist_in[0];
    for(int i = 1; i < nbr_bin; i++) CDF[i] = CDF[i-1] + hist_in[i];

    int i = 0,CDFmin = 0;
    while(CDFmin == 0) CDFmin = hist_in[i++];


    int *d_LUT;
    int *d_CDF;
    unsigned char *d_outimg;
    unsigned char *d_img;

    // Allocate memory for device variables
    cudaMalloc(&d_LUT, sizeof(int) * nbr_bin);
    cudaMalloc(&d_CDF, sizeof(int) * nbr_bin);
    cudaMalloc(&d_outimg, sizeof(unsigned char) * img_size);
    cudaMalloc(&d_img, sizeof(unsigned char) * img_size);

    // Copy data to device memory
    cudaMemcpy(d_CDF, CDF, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(nbr_bin + threadsPerBlock - 1) / threadsPerBlock;
    
    /// Call genLUT kernel here
    genLUT_kernel<<<blocksPerGrid , threadsPerBlock>>>(d_LUT, d_CDF, CDFmin, img_size, nbr_bin);
        
    cudaDeviceSynchronize();
    
    int *lut =  (int *)malloc(nbr_bin * sizeof(int));
    cudaMemcpy(lut, d_LUT, sizeof(int) * nbr_bin, cudaMemcpyDeviceToHost);

    threadsPerBlock = 1024;
    blocksPerGrid = (img_size + threadsPerBlock - 1) / threadsPerBlock;
    
    /// Call genResultImg kernel here
    genResultImg_kernel<<<blocksPerGrid , threadsPerBlock>>>(d_outimg, d_img, d_LUT, img_size);

    // Copy result from device to host memory
    cudaMemcpy(img_out, d_outimg, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);

    // Free up host memory
    free(CDF);

    // Free up device memory
    cudaFree(d_CDF); 
    cudaFree(d_LUT);
    cudaFree(d_img); 
    cudaFree(d_outimg);
}