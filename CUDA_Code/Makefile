all: fractalCuda

fractalCuda: fractalCuda.cu gfx.c
	nvcc fractalCuda.cu gfx.c -o fractal --expt-relaxed-constexpr


