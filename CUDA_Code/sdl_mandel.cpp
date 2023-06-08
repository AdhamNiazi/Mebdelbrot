#include <SDL2/SDL.h>

const int WIDTH = 800;
const int HEIGHT = 600;
const double X_MIN = -2.0;
const double X_MAX = 1.0;
const double Y_MIN = -1.5;
const double Y_MAX = 1.5;
const int MAX_ITER = 1000;

void computeMandelbrot(int* pixels)
{
    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            double cx = (double)x / WIDTH * (X_MAX - X_MIN) + X_MIN;
            double cy = (double)y / HEIGHT * (Y_MAX - Y_MIN) + Y_MIN;

            double zx = 0.0;
            double zy = 0.0;
            int iter = 0;

            while (zx * zx + zy * zy < 4.0 && iter < MAX_ITER)
            {
                double temp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = temp;
                iter++;
            }

            // Map the iteration count to a color value
            int r = iter % 256;
            int g = (iter * 7) % 256;
            int b = (iter * 13) % 256;

            int index = y * WIDTH + x;
            pixels[index] = (r << 16) | (g << 8) | b;
        }
    }
}

int main(int argc, char* argv[])
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Mandelbrot Fractal", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);

    int* pixels = new int[WIDTH * HEIGHT];

    computeMandelbrot(pixels);

    SDL_UpdateTexture(texture, nullptr, pixels, WIDTH * sizeof(Uint32));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    bool quit = false;
    SDL_Event event;

    while (!quit)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                quit = true;
        }
    }

    delete[] pixels;

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
