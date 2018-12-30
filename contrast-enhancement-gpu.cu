#include "hist-equ.h"   

PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram_gpu(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization_gpu(result.img,img_in.img,hist,result.w*result.h, 256);
    return result;
}