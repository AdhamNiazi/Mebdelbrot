#include "gfx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <omp.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XTest.h>


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

	static int counter = 0;
	counter++;

	double start_time = omp_get_wtime();
	
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
	double end_time = omp_get_wtime();  // End measuring the execution time

	double execution_time = end_time - start_time;
	printf("Run: %d, Execution Time: %f seconds\n", counter, execution_time);
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

// Function to handle scroll events
void handle_scroll_event(XEvent *event)
{
    int direction = event->xbutton.button;

    // Zoom in or out based on scroll direction
    if (direction == Button4) // Scroll up
    {
        double xCenter = (xmin + xmax) / 2;
        double yCenter = (ymin + ymax) / 2;

        xmin = (xmin - xCenter) * 0.9 + xCenter;
        xmax = (xmax - xCenter) * 0.9 + xCenter;
        ymin = (ymin - yCenter) * 0.9 + yCenter;
        ymax = (ymax - yCenter) * 0.9 + yCenter;
    }
    else if (direction == Button5) // Scroll down
    {
        double xCenter = (xmin + xmax) / 2;
        double yCenter = (ymin + ymax) / 2;

        xmin = (xmin - xCenter) * 1.1 + xCenter;
        xmax = (xmax - xCenter) * 1.1 + xCenter;
        ymin = (ymin - yCenter) * 1.1 + yCenter;
        ymax = (ymax - yCenter) * 1.1 + yCenter;
    }
}

int main(int argc, char *argv[])
{
	int maxiter = 500;
	int screen_width = 640;
	int screen_height = 480;

	if (argc >= 3) {
		screen_width = atoi(argv[1]);
		screen_height = atoi(argv[2]);
	}

	gfx_open(screen_width, screen_height, "Mandelbrot Fractal");

	gfx_clear_color(0, 0, 255);
	gfx_clear();

	compute_image(xmin, xmax, ymin, ymax, maxiter);

    // Initialize X11 display and select input events
    Display *display = XOpenDisplay(NULL);
    Window root = DefaultRootWindow(display);
    XEvent event;
    XSelectInput(display, root, ButtonPressMask);

	while (1)
	{
		int c = gfx_wait();

		if (c == 'q')
	    		exit(0);
		else if (c == 'w' || c == 'a' || c == 's' || c == 'd')
		{
	    		update_position(c, &xmin, &xmax, &ymin, &ymax);
		}

        // Handle mouse scroll events
        while (XCheckTypedEvent(display, ButtonPress, &event))
        {
            if (event.xbutton.button == Button4 || event.xbutton.button == Button5)
            {
                handle_scroll_event(&event);
            }
        }

		gfx_clear();
			compute_image(xmin, xmax, ymin, ymax, maxiter);
	}

    XCloseDisplay(display);
	return 0;
}
