#include "gfx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>

const int width = 640; 
const int height = 480; 

__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{
	double factor = exp(x.x);
	return make_cuDoubleComplex(factor * cos(x.y), factor * sin(2));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCadd(cuDoubleComplex x, double y)
{
	return make_cuDoubleComplex(cuCreal(x) + y, cuCimag(x));
}

__device__ static int compute_point(double x, double y, int max)
{
    cuDoubleComplex z = make_cuDoubleComplex(0, 0);
    cuDoubleComplex alpha = make_cuDoubleComplex(x, y);
    int iter = 0;

    while (cuCabs(z) < 4 && iter < max)
    {
        z = cuCexp(z);
        z = cuCadd(z, alpha);
        iter++;
    }

    return iter;
}

__global__ void compute_image(double xmin, double xmax, double ymin, double ymax, int maxiter, int* result)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < height)
    {
        double x = xmin + row * (xmax - xmin) / width;
        double y = ymin + col * (ymax - ymin) / height;

        int iter = compute_point(x, y, maxiter);
        int gray = 255 * iter / maxiter;
        result[col * width + row] = gray;
    }
}

void update_position(char key, double *xmin, double *xmax, double *ymin, double *ymax)
{
    double xstep = (*xmax - *xmin) / 10;
    double ystep = (*ymax - *ymin) / 10;

    switch (key)
    {
    case 'w':
        *ymin -= ystep;
        *ymax -= ystep;
        break;
    case 'a':
        *xmin -= xstep;
        *xmax -= xstep;
        break;
    case 's':
        *ymin += ystep;
        *ymax += ystep;
        break;
    case 'd':
        *xmin += xstep;
        *xmax += xstep;
        break;
    }
}

int main(int argc, char *argv[])
{
    int maxiter = 500;
    int* result = new int[width * height];
    int* d_result;
    cudaMalloc((void**)&d_result, width * height * sizeof(int));

    double xmin = -1.5;
    double xmax = 0.5;
    double ymin = -1.0;
    double ymax = 1.0;

    gfx_open(width, height, "Mandelbrot Fractal");

    printf("coordinates: %lf %lf %lf %lf\n", xmin, xmax, ymin, ymax);

    gfx_clear_color(0, 0, 255);
    gfx_clear();

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    compute_image<<<gridSize, blockSize>>>(xmin, xmax, ymin, ymax, maxiter, d_result);

    cudaMemcpy(result, d_result, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    for (int col = 0; col < height; ++col)
    {
        for (int row = 0; row < width; ++row)
        {
            int gray = result[col * width + row];
            gfx_color(gray, gray, gray);
            gfx_point(row, col);
        }
    }

    while (1)
    {
        int c = gfx_wait();

        if (c == 'q')
            break;
        else if (c == 'w' || c == 'a' || c == 's' || c == 'd')
        {
            update_position(c, &xmin, &xmax, &ymin, &ymax);
            compute_image<<<gridSize, blockSize>>>(xmin, xmax, ymin, ymax, maxiter, d_result);
            cudaMemcpy(result, d_result, width * height * sizeof(int), cudaMemcpyDeviceToHost);
            gfx_clear();
            for (int col = 0; col < height; ++col)
            {
                for (int row = 0; row < width; ++row)
                {
                    int gray = result[col * width + row];
                    gfx_color(gray, gray, gray);
                    gfx_point(row, col);
                }
            }
        }
    }
    cudaFree(d_result);
    delete[] result;

    return 0;
}
