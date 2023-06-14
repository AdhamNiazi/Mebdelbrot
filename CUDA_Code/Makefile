all: fractalCuda

fractalCuda: fractalCuda.cu gfx.c
	nvcc -o fractal fractalCuda.cu -lX11
	nvcc -o fractal2 fractalCuda_multipixel.cu -lX11
