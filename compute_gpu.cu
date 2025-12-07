extern "C" {
    #include <math.h>
    #include "vector.h"
    #include "config.h"
}

#include <cuda_runtime.h>
#define BLOCK_X 16
#define BLOCK_Y 16

static vector3 *d_pos  = NULL;
static vector3 *d_vel  = NULL;
static vector3 *d_acc  = NULL;
static double  *d_mass = NULL;
static int      d_n    = 0;

__global__ void kernel_pairwise(const vector3 *pos,
                                const double  *mass,
                                vector3       *accels,
                                int            n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row

    __shared__ vector3 sharedPosI[BLOCK_Y];   // one per row in this tile
    __shared__ vector3 sharedPosJ[BLOCK_X];   // one per column in this tile
    __shared__ double  sharedMassJ[BLOCK_X];  // masses for columns

    if (threadIdx.x == 0) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < n) {
            sharedPosI[threadIdx.y][0] = pos[row][0];
            sharedPosI[threadIdx.y][1] = pos[row][1];
            sharedPosI[threadIdx.y][2] = pos[row][2];
        }
    }

    if (threadIdx.y == 0) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < n) {
            sharedPosJ[threadIdx.x][0] = pos[col][0];
            sharedPosJ[threadIdx.x][1] = pos[col][1];
            sharedPosJ[threadIdx.x][2] = pos[col][2];
            sharedMassJ[threadIdx.x]   = mass[col];
        }
    }

    __syncthreads();
    if (i >= n || j >= n) return;

    vector3 *cell = &accels[i * n + j];

    if (i == j) {
        FILL_VECTOR((*cell), 0.0, 0.0, 0.0);
        return;
    }

    vector3 distance;
    distance[0] = sharedPosI[threadIdx.y][0] - sharedPosJ[threadIdx.x][0];
    distance[1] = sharedPosI[threadIdx.y][1] - sharedPosJ[threadIdx.x][1];
    distance[2] = sharedPosI[threadIdx.y][2] - sharedPosJ[threadIdx.x][2];

    double magnitude_sq =
        distance[0]*distance[0] +
        distance[1]*distance[1] +
        distance[2]*distance[2];

    double magnitude = sqrt(magnitude_sq);
    double accelmag  = -1.0 * GRAV_CONSTANT * sharedMassJ[threadIdx.x]
                       / magnitude_sq;

    FILL_VECTOR((*cell),
        accelmag * distance[0] / magnitude,
        accelmag * distance[1] / magnitude,
        accelmag * distance[2] / magnitude);
}

// Kernel 2: for each body i, sum row i of accels and update vel & pos
__global__ void kernel_sum_and_update(vector3       *pos,
                                      vector3       *vel,
                                      const vector3 *accels,
                                      int            n,
                                      double         interval)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vector3 accel_sum = {0.0, 0.0, 0.0};

    for (int j = 0; j < n; j++) {
        const vector3 *cell = &accels[i * n + j];
        accel_sum[0] += (*cell)[0];
        accel_sum[1] += (*cell)[1];
        accel_sum[2] += (*cell)[2];
    }

    for (int k = 0; k < 3; k++) {
        vel[i][k] += accel_sum[k] * interval;
        pos[i][k] += vel[i][k]   * interval;
    }
}

extern "C" void compute()
{
    int n = NUMENTITIES;

    if (d_pos == NULL) {
        d_n = n;

        cudaMalloc(&d_pos,  sizeof(vector3) * n);
        cudaMalloc(&d_vel,  sizeof(vector3) * n);
        cudaMalloc(&d_acc,  sizeof(vector3) * n * n);
        cudaMalloc(&d_mass, sizeof(double)  * n);

        cudaMemcpy(d_mass, mass, sizeof(double) * n, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_pos, hPos, sizeof(vector3) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, hVel, sizeof(vector3) * n, cudaMemcpyHostToDevice);

    dim3 blockDim1(BLOCK_X, BLOCK_Y);
    dim3 gridDim1(
        (n + BLOCK_X - 1) / BLOCK_X,
        (n + BLOCK_Y - 1) / BLOCK_Y
    );

    kernel_pairwise<<<gridDim1, blockDim1>>>(d_pos, d_mass, d_acc, n);
    cudaDeviceSynchronize();    

    int blockSize2 = 256;
    int gridSize2  = (n + blockSize2 - 1) / blockSize2;

    kernel_sum_and_update<<<gridSize2, blockSize2>>>(d_pos, d_vel, d_acc, n, INTERVAL);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, d_pos, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_vel, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
}
