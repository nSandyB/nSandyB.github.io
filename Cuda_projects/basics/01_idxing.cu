#include <stdio.h>

__global__ void ids(void) {
	int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
	int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;
	int thread_offset = threadIdx.x + threadIdx.y* blockDim.x + threadIdx.z* blockDim.x*blockDim.y;

	int id = block_offset + thread_offset;

	printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
		id,
		blockIdx.x, blockIdx.y, blockIdx.z, block_id,
		threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);

}

int main(int argc, char **argv) {
	const int bx = 2, by = 3, bz = 4;
	const int tx = 4, ty = 4, tz = 4;

	int blocks_per_grid = bx * by*bz;
	int threads_per_block = tx * ty*tz;

	printf("%d total threads\n", blocks_per_grid * threads_per_block);

	dim3 blocksPerGrid(bx, by, bz);
	dim3 threadsPerBlock(tx, ty, tz);

	ids <<<blockPerGrid, threadsPerblock >>> ();
	cudaDeviceSynchronize;
}