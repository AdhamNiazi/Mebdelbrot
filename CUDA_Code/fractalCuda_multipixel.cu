#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <complex.h>
#include <cuComplex.h>
#include <omp.h>
#include <time.h>

// #define WIDTH 640
// #define HEIGHT 480

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

/*
gfx_open creates several X11 objects, and stores them in globals
for use by the other functions in the library.
*/

static Display *gfx_display=0;
static Window  gfx_window;
static GC      gfx_gc;
static Colormap gfx_colormap;
static int      gfx_fast_color_mode = 0;
int WIDTH = 640;
int HEIGHT = 480;
int dim = 16;
double scale = 3;
/* These values are saved by gfx_wait then retrieved later by gfx_xpos and gfx_ypos. */

static int saved_xpos = 0;
static int saved_ypos = 0;
static int saved_xsize = 0;
static int saved_ysize = 0;

/* Flush all previous output to the window. */

void gfx_flush()
{
	XFlush(gfx_display);
}

/* Open a new graphics window. */

void gfx_open( int width, int height, const char *title )
{
	gfx_display = XOpenDisplay(0);
	if(!gfx_display) {
		fprintf(stderr,"gfx_open: unable to open the graphics window.\n");
		exit(1);
	}

	Visual *visual = DefaultVisual(gfx_display,0);
	if(visual && visual->c_class == TrueColor) {
		gfx_fast_color_mode = 1;
	} else {
		gfx_fast_color_mode = 0;
	}

	int blackColor = BlackPixel(gfx_display, DefaultScreen(gfx_display));
	int whiteColor = WhitePixel(gfx_display, DefaultScreen(gfx_display));

	gfx_window = XCreateSimpleWindow(gfx_display, DefaultRootWindow(gfx_display), 0, 0, width, height, 0, blackColor, blackColor);

	XSetWindowAttributes attr;
	attr.backing_store = Always;

	XChangeWindowAttributes(gfx_display,gfx_window,CWBackingStore,&attr);

	XStoreName(gfx_display,gfx_window,title);

	XSelectInput(gfx_display, gfx_window, StructureNotifyMask|KeyPressMask|ButtonPressMask);

	XMapWindow(gfx_display,gfx_window);

	gfx_gc = XCreateGC(gfx_display, gfx_window, 0, 0);

	gfx_colormap = DefaultColormap(gfx_display,0);

	XSetForeground(gfx_display, gfx_gc, whiteColor);

	// Wait for the MapNotify event

	for(;;) {
		XEvent e;
		XNextEvent(gfx_display, &e);
		if (e.type == MapNotify)
			break;
	}

	saved_xsize = width;
	saved_ysize = height;
}

/* Draw a single point at (x,y) */

void gfx_point( int x, int y )
{
	XDrawPoint(gfx_display,gfx_window,gfx_gc,x,y);
}

/* Draw a line from (x1,y1) to (x2,y2) */

void gfx_line( int x1, int y1, int x2, int y2 )
{
	XDrawLine(gfx_display,gfx_window,gfx_gc,x1,y1,x2,y2);
}

/* Change the current drawing color. */

void gfx_color( int r, int g, int b )
{
	XColor color;

	if(gfx_fast_color_mode) {
		/* If this is a truecolor display, we can just pick the color directly. */
		color.pixel = ((b&0xff) | ((g&0xff)<<8) | ((r&0xff)<<16) );
	} else {
		/* Otherwise, we have to allocate it from the colormap of the display. */
		color.pixel = 0;
		color.red = r<<8;
		color.green = g<<8;
		color.blue = b<<8;
		XAllocColor(gfx_display,gfx_colormap,&color);
	}

	XSetForeground(gfx_display, gfx_gc, color.pixel);
}

/* Clear the graphics window to the background color. */

void gfx_clear()
{
	XClearWindow(gfx_display,gfx_window);
}

/* Change the current background color. */

void gfx_clear_color( int r, int g, int b )
{
	XColor color;
	color.pixel = 0;
	color.red = r<<8;
	color.green = g<<8;
	color.blue = b<<8;
	XAllocColor(gfx_display,gfx_colormap,&color);

	XSetWindowAttributes attr;
	attr.background_pixel = color.pixel;
	XChangeWindowAttributes(gfx_display,gfx_window,CWBackPixel,&attr);
}

int gfx_event_waiting()
{
       XEvent event;

       gfx_flush();

       while (1) {
               if(XCheckMaskEvent(gfx_display,-1,&event)) {
                       if(event.type==KeyPress) {
                               XPutBackEvent(gfx_display,&event);
                               return 1;
                       } else if (event.type==ButtonPress) {
                               XPutBackEvent(gfx_display,&event);
                               return 1;
                       } else {
                               return 0;
                       }
               } else {
                       return 0;
               }
       }
}

/* Wait for the user to press a key or mouse button. */

int gfx_wait()
{
	XEvent event;

	gfx_flush();

	while(1) {
		XNextEvent(gfx_display,&event);

		if(event.type==KeyPress) {
			saved_xpos = event.xkey.x;
			saved_ypos = event.xkey.y;

			/* If the key sequence maps to an ascii character, return that. */
			KeySym symbol;
			char str[4];
			int r = XLookupString(&event.xkey,str,sizeof(str),&symbol,0);
			if(r==1) return str[0];

			/* Special case for navigation keys, return codes above 129. */
			if(symbol>=0xff50 && symbol<=0xff58) {
				return 129 + (symbol-0xff50);
			}

		} else if(event.type==ButtonPress) {
			saved_xpos = event.xkey.x;
			saved_ypos = event.xkey.y;
			return event.xbutton.button;
		} else if(event.type==ConfigureNotify) {
			saved_xsize = event.xconfigure.width;
			saved_ysize = event.xconfigure.height;
		}
	}
}

/* Return the X and Y coordinates of the last event. */

int gfx_xpos()
{
	return saved_xpos;
}

int gfx_ypos()
{
	return saved_ypos;
}

int gfx_xsize()
{
	return saved_xsize;
}

int gfx_ysize()
{
	return saved_ysize;
}

__global__ void compute_image_kernel(double xmin, double xmax, double ymin, double ymax, int maxiter, int* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int x = i; x < WIDTH; x += stride_x)
    {
        for (int y = j; y < HEIGHT; y += stride_y)
        {
            double px = xmin + x * (xmax - xmin) / WIDTH;
            double py = ymin + y * (ymax - ymin) / HEIGHT;

            cuDoubleComplex z = make_cuDoubleComplex(0, 0);
            cuDoubleComplex alpha = make_cuDoubleComplex(px, py);

            int iter = 0;
            while (cuCabs(z) < 4 && iter < maxiter)
            {
                z = cuCadd(cuCmul(z, z), alpha);
                iter++;
            }

            int index = y * WIDTH + x;
            output[index] = iter;
        }
    }
}

// Function to compute the fractal image using CUDA
void compute_image_cuda(double xmin, double xmax, double ymin, double ymax, int maxiter, int* output)
{
    int* d_output;
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(int));

    dim3 threadsPerBlock(dim, dim);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    printf("Blocks: %d, Threads per Block: %d, Size: %d*%d, Depth: %d ", numBlocks.x * numBlocks.y * numBlocks.z, threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z, WIDTH, HEIGHT, maxiter);
    compute_image_kernel<<<numBlocks, threadsPerBlock>>>(xmin, xmax, ymin, ymax, maxiter, d_output, WIDTH, HEIGHT);
    cudaMemcpy(output, d_output, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}

// Function to update the position based on the key input
void update_position(char key, double* xmin, double* xmax, double* ymin, double* ymax)
{
    double step = scale / dim;
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
    }
}

int main(int argc, char* argv[])
{
    clock_t t;
    double xcen = -0.5;
    double ycen = 0.5;

    double xmin = xcen - (scale/2);
    double ymin = ycen - (scale/2);
    double step = scale / dim;

    double xmax = 0.5;

    double ymax = 1.0;
    int maxiter = 500;
    if (argc >= 2) 
        WIDTH = atoi(argv[1]);
    if (argc >= 3) 
        HEIGHT = atoi(argv[2]);
    if (argc >= 4) 
        dim = atoi(argv[3]);
    if (argc == 5)
        maxiter = atoi(argv[4]);
    
    if (argc > 5) {
        printf("Usage: %s <WIDTH> <HEIGHT> <NUM THREADS> <MAX ITER>\n", argv[0]);
        return 1;
    }

    gfx_open(WIDTH, HEIGHT, "Mandelbrot Fractal");

    gfx_clear_color(0, 0, 255);
    gfx_clear();

    int* output = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    t = clock();
    compute_image_cuda(xmin, xmax, ymin, ymax, maxiter, output);
    t = clock() - t;
    
	printf("Execution Time: %f seconds\n",  ((float)t) / CLOCKS_PER_SEC);
    while (1)
    {
        int c = gfx_wait();

        if (c == 'q')
            break;
        else if (c == 'w' || c == 'a' || c == 's' || c == 'd')
        {
            update_position(c, &xmin, &xmax, &ymin, &ymax);
            gfx_clear();
            t = clock();
            compute_image_cuda(xmin, xmax, ymin, ymax, maxiter, output);
            t = clock() - t;
            printf("Execution Time: %f seconds\n",  ((float)t) / CLOCKS_PER_SEC);
        } else if (c == 'z') {
            scale *= 1.25;
        } else if (c == 'x') {
            scale *= .80;
        }

        gfx_clear();
        for (int j = 0; j < HEIGHT; j++)
        {
            for (int i = 0; i < WIDTH; i++)
            {
                int iter = output[j * WIDTH + i];

                // Compute color based on iteration count
                int b = (iter * 10) % 255;
                int r = (iter * 5) % 255;
                int g = (iter * 20) % 255;

                gfx_color(r, g, b);
                gfx_point(i, j);
            }
        }
    }

    free(output);
    gfx_clear();

    return 0;
}