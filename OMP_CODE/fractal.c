#include "gfx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include <omp.h>

// Variables to store the current position
double xmin = -1.5;
double xmax = 0.5;
double ymin = -1.0;
double ymax = 1.0;
double scale = 3;

static int compute_point(double x, double y, int max)
{
    double complex z = 0;
    double complex alpha = x + I * y;

    int iter = 0;

    while (cabs(z) < 4 && iter < max)
    {
        z = cpow(z, 2) + alpha;
        iter++;
    }

    return iter;
}

// Function to compute the fractal image
void compute_image(double xmin, double xmax, double ymin, double ymax, int maxiter, double zoom)
{
    int i, j;

    int width = gfx_xsize();
    int height = gfx_ysize();

    static int counter = 0;
    counter++;

    double start_time = omp_get_wtime();

    #pragma omp parallel for private(i) shared(width, height, xmin, xmax, ymin, ymax, maxiter, zoom)
    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i++)
        {
            double x = xmin + i * (xmax - xmin) / (width * zoom);
            double y = ymin + j * (ymax - ymin) / (height * zoom);

            int iter = compute_point(x, y, maxiter);
            //int gray = 255 * iter / maxiter;
			int b = (iter * 10) % 255;
            int r = (iter * 5) % 255;
            int g = (iter * 20) % 255;

            #pragma omp critical
            {
                gfx_color(r, g, b);
                gfx_point(i, j);
            }
        }
    }
    double end_time = omp_get_wtime();  // End measuring the execution time

    double execution_time = end_time - start_time;
    printf("Run: %d, Execution Time: %f seconds\n", counter, execution_time);
}
void update_position(char key, double *xmin, double *xmax, double *ymin, double *ymax, double *zoom)
{
    double step = scale / 16;
    double xstep = (*xmax - *xmin) / 10 * step;
    double ystep = (*ymax - *ymin) / 10 * step;

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
    case '+':
        *zoom *= 0.9; // Zoom in by reducing the zoom factor
		scale *= 1.05;
        break;
    case '-':
        *zoom /= 0.9; // Zoom out by increasing the zoom factor
		scale *= 0.95;
        break;
    }
}

int main(int argc, char *argv[])
{
    int maxiter = 500;
    int screen_width = 640;
    int screen_height = 480;
	double xcen = -0.5;
    double ycen = 0.5;

    double xmin = xcen - (scale/2);
    double ymin = ycen - (scale/2);
    double zoom = 1.0;
    double xmax = 0.5;
    double ymax = 1.0;

    if (argc >= 2) 
        screen_width = atoi(argv[1]);
    if (argc >= 3) 
        screen_height = atoi(argv[2]);
    if (argc >= 4) 
        maxiter = atoi(argv[3]);
    if (argc > 4) {
        printf("Usage: %s <WIDTH> <HEIGHT> <MAX ITER>\n", argv[0]);
        return 1;
    }

    gfx_open(screen_width, screen_height, "Mandelbrot Fractal");

    gfx_clear_color(0, 0, 255);
    gfx_clear();

    compute_image(xmin, xmax, ymin, ymax, maxiter, zoom);

    while (1)
    {
        int c = gfx_wait();

        if (c == 'q')
            exit(0);
        else if (c == 'w' || c == 'a' || c == 's' || c == 'd' || c == '+' || c == '-')
        {
            update_position(c, &xmin, &xmax, &ymin, &ymax, &zoom);
            gfx_clear();
            compute_image(xmin, xmax, ymin, ymax, maxiter, zoom);
        }

		
    }

    return 0;
}
