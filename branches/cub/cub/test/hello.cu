

#include <stdio.h>


/**
 * Main
 */
int main(int argc, const char**argv)
{
	typedef int T;

	cudaError_t error;

	// Set device
	int current_device = 0;
	if (argc > 1) {
		current_device = atoi(argv[1]);
	}
	if (error = cudaSetDevice(current_device)) {
		printf("cudaSetDevice failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	cudaDeviceProp device_props;
	if (error = cudaGetDeviceProperties(&device_props, current_device)) {
		printf("cudaGetDeviceProperties failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}
	printf("Device(%s), UVA(%d)\n", device_props.name, device_props.unifiedAddressing);

	// Allocate device data
	T *d_data;
	if (error = cudaMalloc((void**) &d_data, sizeof(T))) {
		printf("cudaMalloc failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	cudaPointerAttributes pointer_attrs;
	if (error = cudaPointerGetAttributes(&pointer_attrs, d_data)) {
		printf("cudaPointerGetAttributes1 failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	printf("Type(%s), device(%d)\n",
		(pointer_attrs.memoryType == cudaMemoryTypeDevice) ? "Device" : "Host",
		pointer_attrs.device);

	int h_data[5];

	if (error = cudaPointerGetAttributes(&pointer_attrs, h_data)) {
		printf("cudaPointerGetAttributes2 failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	printf("Type(%s), device(%d)\n",
		(pointer_attrs.memoryType == cudaMemoryTypeDevice) ? "Device" : "Host",
		pointer_attrs.device);

	return 0;
}
