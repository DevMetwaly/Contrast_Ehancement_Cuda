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


PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
	unsigned char * d_y;
    unsigned char * d_u;
    unsigned char * d_v;

    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;
	
	unsigned char s = img_in.w * img_in.h ; 
	
    unsigned char * y_equ;
    int hist[256];
    
	// HOST YUV ALLOCATION
    yuv_med.img_y = (unsigned char *)malloc(sizeof(unsigned char)*s);
    yuv_med.img_u = (unsigned char *)malloc(sizeof(unsigned char)*s);
    yuv_med.img_v = (unsigned char *)malloc(sizeof(unsigned char)*s);
	
	// DEVICE YUV ALLOCATION
	cudaMalloc(&d_y, s * sizeof(unsigned char));
    cudaMalloc(&d_u, s * sizeof(unsigned char));
    cudaMalloc(&d_v, s * sizeof(unsigned char));
    // DEVICE RGB ALLOCATION
    cudaMalloc(&d_r, s * sizeof(unsigned char));
    cudaMalloc(&d_g, s * sizeof(unsigned char));
    cudaMalloc(&d_b, s * sizeof(unsigned char));

	cudaMemcpy(d_r, &img_in.img_r, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, &img_in.img_g, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &img_in.img_b, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 256;
    int blocksPerGrid = (256 + threadsPerBlock - 1) / threadsPerBlock;
	
	rgb2yuv_kernel<<<blocksPerGrid , threadsPerBlock>>>(s, d_r, d_g, d_b, d_y, d_u, d_v);
	
	cudaMemcpy(&yuv_med.img_y, d_y, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
	y_equ = (unsigned char *)malloc(s * sizeof(unsigned char));
	
    histogram_gpu(hist, yuv_med.img_y, s, 256);
    histogram_equalization_gpu(y_equ, yuv_med.img_y, hist, s, 256);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    cudaMemcpy(d_y, &yuv_med.img_y, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
	
    yuv2rgb_kernel<<<blocksPerGrid , threadsPerBlock>>>(s, d_r, d_g, d_b, d_y, d_u, d_v);
	cudaMemcpy(&result.img_r, d_r, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.img_g, d_g, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.img_b, d_b, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
	
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
    cudaFree(d_y); cudaFree(d_u); cudaFree(d_v);
    cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
	
    return result;
}


PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    float * d_h;
    float * d_s;
    unsigned char * d_l;

    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;

    unsigned char s = img_in.w * img_in.h ;

    unsigned char * l_equ;
    int hist[256];

    // HOST HSL ALLOCATION
    hsl_med.h = (float *)malloc(sizeof(float)*s);
    hsl_med.s = (float *)malloc(sizeof(float)*s);
    hsl_med.l = (unsigned char *)malloc(sizeof(unsigned char)*s);
    
    // DEVICE HSL ALLOCATION
    cudaMalloc(&d_h, s * sizeof(float));
    cudaMalloc(&d_s, s * sizeof(float));
    cudaMalloc(&d_l, s * sizeof(unsigned char));
    // DEVICE RGB ALLOCATION
    cudaMalloc(&d_r, s * sizeof(unsigned char));
    cudaMalloc(&d_g, s * sizeof(unsigned char));
    cudaMalloc(&d_b, s * sizeof(unsigned char));

    cudaMemcpy(d_r, &img_in.img_r, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, &img_in.img_g, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &img_in.img_b, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (256 + threadsPerBlock - 1) / threadsPerBlock;
    
    rgb2hsl_kernel<<<blocksPerGrid , threadsPerBlock>>>(s, d_r, d_g, d_b, d_h, d_s, d_l);
    
    cudaMemcpy(&hsl_med.l, d_l, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
    l_equ = (unsigned char *)malloc(s * sizeof(unsigned char));
    
    histogram_gpu(hist, hsl_med.l, s, 256);
    histogram_equalization_gpu(l_equ, hsl_med.l, hist, s, 256);

    free(hsl_med.l);
    hsl_med.l = l_equ;
    cudaMemcpy(d_l, &hsl_med.l, sizeof(unsigned char) * s, cudaMemcpyHostToDevice);
    
    hsl2rgb_kernel<<<blocksPerGrid , threadsPerBlock>>>(s, d_r, d_g, d_b, d_h, d_s, d_l);
    cudaMemcpy(&result.img_r, d_r, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.img_g, d_g, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.img_b, d_b, sizeof(unsigned char) * s, cudaMemcpyDeviceToHost);
    
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    cudaFree(d_h); cudaFree(d_s); cudaFree(d_l);
    cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
    
    return result;
}


__global__ void rgb2yuv_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    unsigned char *img_y, unsigned char *img_u, unsigned char *img_v) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < s){
        int r, g, b;
        r = img_r[i];
        g = img_g[i];
        b = img_b[i];

        img_y[i] = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        img_u[i] = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        img_v[i] = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
    }
}

__global__ void yuv2rgb_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    unsigned char *img_y, unsigned char *img_u, unsigned char *img_v){

        int i = threadIdx.x + blockDim.x*blockIdx.x;
        unsigned char y, cb, cr;
        
        if(i < s){

            y  = img_y[i];
            cb = img_u[i] - 128;
            cr = img_v[i] - 128;

            img_r[i] = ( y + 1.402 * cr);
            img_g[i] = ( y - 0.344 * cb - 0.714 * cr);
            img_b[i] = ( y + 1.772 * cb);

        }
}


__global__ void rgb2hsl_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    unsigned char *img_h, unsigned char *img_s, unsigned char *img_l)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float H, S, L;
    
    float var_r = ( (float)img_r[i]/255 );//Convert RGB to [0,1]
    float var_g = ( (float)img_g[i]/255 );
    float var_b = ( (float)img_b[i]/255 );
    float var_min = (var_r < var_g) ? var_r : var_g;
    var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
    float var_max = (var_r > var_g) ? var_r : var_g;
    var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
    float del_max = var_max - var_min;               //Delta RGB value
    
    L = ( var_max + var_min ) / 2;
    if ( del_max == 0 )//This is a gray, no chroma...
    {
        H = 0;         
        S = 0;    
    }
    else                                    //Chromatic data...
    {
        if ( L < 0.5 )
            S = del_max/(var_max+var_min);
        else
            S = del_max/(2-var_max-var_min );

        float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
        float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
        float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
        if( var_r == var_max ){
            H = del_b - del_g;
        }
        else{       
            if( var_g == var_max ){
                H = (1.0/3.0) + del_r - del_b;
            }
            else{
                    H = (2.0/3.0) + del_g - del_r;
            }   
        }
        
    }
    
    if ( H < 0 )
        H += 1;
    if ( H > 1 )
        H -= 1;

    img_h[i] = H;
    img_s[i] = S;
    img_l[i] = (unsigned char)(L*255);
}


//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]

__global__ void hsl2rgb_kernel(int s, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, 
    unsigned char *img_h, unsigned char *img_s, unsigned char *img_l)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    float H = img_h[i];
    float S = img_s[i];
    float L = img_l[i]/255.0f;
    float var_1, var_2;
    
    unsigned char r, g, b;
    
    if ( S == 0 )
    {
        r = L * 255;
        g = L * 255;
        b = L * 255;
    }
    else
    {
        
        if ( L < 0.5 )
            var_2 = L * ( 1 + S );
        else
            var_2 = ( L + S ) - ( S * L );

        var_1 = 2 * L - var_2;
        r = 255 * Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
        g = 255 * Hue_2_RGB( var_1, var_2, H );
        b = 255 * Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
    }
    img_r[i] = r;
    img_g[i] = g;
    img_b[i] = b;
}


float Hue_2_RGB( float v1, float v2, float vH )
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}