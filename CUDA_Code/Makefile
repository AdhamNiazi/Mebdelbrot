all: fractalCuda

fractalCuda: fractalCuda.cu gfx.c
	gcc -c -fPIC gfx.c -o gfx.o
	gcc -c -fPIC gfx.c -o gfx.o -lX11
	gcc -shared -o libgfx.so gfx.o -lX11
	nvcc fractalCuda.cu -o fractal --expt-relaxed-constexpr -L. -lgfx 
