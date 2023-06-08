all: fractalCuda

fractalCuda: fractalCuda.cu gfx.c
	nvcc -o fractal fractalCuda.cu -lX11
