#include "gfx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <omp.h>


// Variables to store the current position
double xmin = -1.5;
double xmax = 0.5;
double ymin = -1.0;
double ymax = 1.0;




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
void compute_image(double xmin, double xmax, double ymin, double ymax, int maxiter)
{
    int i, j;

    int width = gfx_xsize();
    int height = gfx_ysize();

  //  #pragma omp parallel for private(i) shared(width, height, xmin, xmax, ymin, ymax, maxiter)
    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i++)
        {
            double x = xmin + i * (xmax - xmin) / width;
            double y = ymin + j * (ymax - ymin) / height;

            int iter = compute_point(x, y, maxiter);
            int gray = 255 * iter / maxiter;
            gfx_color(gray, gray, gray);
            gfx_point(i, j);
        }
    }
}

// Function to update the position based on the key input
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
 

    gfx_open(640, 480, "Mandelbrot Fractal");

    printf("coordinates: %lf %lf %lf %lf\n", xmin, xmax, ymin, ymax);

    gfx_clear_color(0, 0, 255);
    gfx_clear();

    compute_image(xmin, xmax, ymin, ymax, maxiter);

    while (1)
    {
        int c = gfx_wait();

        if (c == 'q')
            exit(0);
        else if (c == 'w' || c == 'a' || c == 's' || c == 'd')
        {
            update_position(c, &xmin, &xmax, &ymin, &ymax);
        }

        gfx_clear();
        compute_image(xmin, xmax, ymin, ymax, maxiter);
    }

    return 0;
}