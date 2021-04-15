#include <math.h>
#include <iostream>
#include <stdexcept>

using namespace std;

texture<float, cudaTextureType3D, cudaReadModeElementType> texImg(
		0, cudaFilterModeLinear, cudaAddressModeClamp);
texture<float, cudaTextureType3D, cudaReadModeElementType> texGuide(
		0, cudaFilterModeLinear, cudaAddressModeClamp);

__device__ int Address3D(int ix, int iy, int iz, int3 sz)
{
	return iz * sz.x * sz.y + iy * sz.x + ix;
}

__device__ float GaussianDistance2(
		const float* gaussian, int3 pt0, int3 pt1, int3 ksz)
{
	float dist = 0;
	for (int ix = 0; ix < ksz.x; ix++)
	{
		for (int iy = 0; iy < ksz.y; iy++)
		{
			for (int iz = 0; iz < ksz.z; iz++)
			{
				float weight = gaussian[Address3D(ix, iy, iz, ksz)];
				float val0 = tex3D(texGuide,
						pt0.x + ix - ksz.x / 2 + 0.5f,
						pt0.y + iy - ksz.y / 2 + 0.5f,
						pt0.z + iz - ksz.z / 2 + 0.5f);
				float val1 = tex3D(texGuide,
						pt1.x + ix - ksz.x / 2 + 0.5f,
						pt1.y + iy - ksz.y / 2 + 0.5f,
						pt1.z + iz - ksz.z / 2 + 0.5f);
				dist += weight * (val0 - val1) * (val0 - val1);
			}
		}
	}

	return dist;
}

// res - the result
// texImg - the input
// texGuide - the guide image
// gaussian - the gaussian weighting window for distance calculation
// imgSz - size of image
// sSz - size of search window
// kSz - size of gaussian kernel
// d2 - the square of d for the nlm
__global__ void HYPR_NLM_Kernel(float* res, const float* gaussian,
		int3 imgSz, int3 sSz, int3 kSz, float d2, float eps)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= imgSz.x || iy >= imgSz.y || iz >= imgSz.z)
	{
		return;
	}

	float sumImg = 0;
	float sumGuide = 0;
	int3 pt0 = make_int3(ix, iy, iz);
	for (int isx = 0; isx < sSz.x; isx++)
	{
		for (int isy = 0; isy < sSz.y; isy++)
		{
			for (int isz = 0; isz < sSz.z; isz++)
			{
				int3 pt1 = make_int3(ix - sSz.x / 2 + isx,
						iy - sSz.y / 2 + isy, iz - sSz.z / 2 + isz);
				float dist2 = GaussianDistance2(gaussian, pt0, pt1, kSz);
				float weight = expf(-dist2 / d2);
				sumImg += weight * tex3D(texImg,
						pt1.x + 0.5f, pt1.y + 0.5f, pt1.z + 0.5f);
				sumGuide += weight * tex3D(texGuide,
						pt1.x + 0.5f, pt1.y + 0.5f, pt1.z + 0.5f);
			}
		}
	}

	res[Address3D(ix, iy, iz, imgSz)] = sumImg / (sumGuide + eps)
			* tex3D(texGuide, pt0.x + 0.5f, pt0.y + 0.5f, pt0.z + 0.5f);

}

// res - the result
// texImg - the input
// texGuide - the guide image
// gaussian - the gaussian weighting window for distance calculation
// imgSz - size of image
// sSz - size of search window
// kSz - size of gaussian kernel
// d2 - the square of d for the nlm
__global__ void NLM_Kernel(float* res, const float* gaussian,
		int3 imgSz, int3 sSz, int3 kSz, float d2, float eps)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= imgSz.x || iy >= imgSz.y || iz >= imgSz.z)
	{
		return;
	}

	float sumImg = 0;
	float sumGuide = 0;
	int3 pt0 = make_int3(ix, iy, iz);
	for (int isx = 0; isx < sSz.x; isx++)
	{
		for (int isy = 0; isy < sSz.y; isy++)
		{
			for (int isz = 0; isz < sSz.z; isz++)
			{
				int3 pt1 = make_int3(ix - sSz.x / 2 + isx,
						iy - sSz.y / 2 + isy, iz - sSz.z / 2 + isz);
				float dist2 = GaussianDistance2(gaussian, pt0, pt1, kSz);
				float weight = expf(-dist2 / d2);
				sumImg += weight * tex3D(texImg,
						pt1.x + 0.5f, pt1.y + 0.5f, pt1.z + 0.5f);
				sumGuide += weight;
			}
		}
	}

	res[Address3D(ix, iy, iz, imgSz)] = sumImg / (sumGuide + eps);

}

extern "C" int HYPR_NLM_SetDevice(int i)
{
	return cudaSetDevice(i);
}

extern "C" int HYPR_NLM(float* res, float* img, float* guide, float* gaussian,
		int imgSzx, int imgSzy, int imgSzz,
		int sSzx, int sSzy, int sSzz, int kSzx, int kSzy, int kSzz,
		float d2, float eps)
{
	float* cuRes = NULL;
	float* cuGaussian = NULL;
	cudaArray* arrImg = NULL;
	cudaArray* arrGuide = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&cuRes,
				sizeof(float) * imgSzx * imgSzy * imgSzz))
		{
			throw std::exception();
		}
		if (cudaSuccess != cudaMalloc(&cuGaussian,
				sizeof(float) * kSzx * kSzy * kSzz))
		{
			throw std::exception();
		}
		if (cudaSuccess != cudaMalloc3DArray(&arrImg,
				&texImg.channelDesc,
				make_cudaExtent(imgSzx, imgSzy, imgSzz)))
		{
			throw std::exception();
		}
		if (cudaSuccess != cudaMalloc3DArray(&arrGuide,
				&texGuide.channelDesc,
				make_cudaExtent(imgSzx, imgSzy, imgSzz)))
		{
			throw std::exception();
		}
		cudaBindTextureToArray(texImg, arrImg);
		cudaBindTextureToArray(texGuide, arrGuide);

		cudaMemcpy(cuGaussian, gaussian, sizeof(float) * kSzx * kSzy * kSzz,
				cudaMemcpyHostToDevice);

		cudaMemcpy3DParms params = {0};
		params.dstArray = arrImg;
		params.srcPtr = make_cudaPitchedPtr(
				img, sizeof(float) * imgSzx, imgSzx, imgSzy);
		params.extent = make_cudaExtent(imgSzx, imgSzy, imgSzz);
		params.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&params);

		params.dstArray = arrGuide;
		params.srcPtr = make_cudaPitchedPtr(
				guide, sizeof(float) * imgSzx, imgSzx, imgSzy);
		cudaMemcpy3D(&params);

		dim3 threads(16, 16, 1);
		dim3 blocks((int)ceilf(imgSzx / (float)threads.x),
				(int)ceilf(imgSzy / (float)threads.y),
				(int)ceilf(imgSzz / (float)threads.z));
		HYPR_NLM_Kernel<<<blocks, threads>>>(cuRes, cuGaussian,
				make_int3(imgSzx, imgSzy, imgSzz),
				make_int3(sSzx, sSzy, sSzz),
				make_int3(kSzx, kSzy, kSzz),
				d2, eps);

		cudaMemcpy(res, cuRes, sizeof(float) * imgSzx * imgSzy * imgSzz,
				cudaMemcpyDeviceToHost);
	}
	catch (std::exception &e)
	{
		if (cuRes != NULL)
		{
			cudaFree(cuRes);
		}
		if (cuGaussian != NULL)
		{
			cudaFree(cuGaussian);
		}
		if (arrImg != NULL)
		{
			cudaFreeArray(arrImg);
		}
		if (arrGuide != NULL)
		{
			cudaFreeArray(arrGuide);
		}
		return cudaGetLastError();
	}

	cudaFree(cuRes);
	cudaFree(cuGaussian);
	cudaFreeArray(arrImg);
	cudaFreeArray(arrGuide);

	return cudaGetLastError();

}

extern "C" int NLM(float* res, float* img, float* guide, float* gaussian,
		int imgSzx, int imgSzy, int imgSzz,
		int sSzx, int sSzy, int sSzz, int kSzx, int kSzy, int kSzz,
		float d2, float eps)
{
	float* cuRes = NULL;
	float* cuGaussian = NULL;
	cudaArray* arrImg = NULL;
	cudaArray* arrGuide = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&cuRes,
				sizeof(float) * imgSzx * imgSzy * imgSzz))
		{
			throw std::exception();
		}
		if (cudaSuccess != cudaMalloc(&cuGaussian,
				sizeof(float) * kSzx * kSzy * kSzz))
		{
			throw std::exception();
		}
		if (cudaSuccess != cudaMalloc3DArray(&arrImg,
				&texImg.channelDesc,
				make_cudaExtent(imgSzx, imgSzy, imgSzz)))
		{
			throw std::exception();
		}
		if (cudaSuccess != cudaMalloc3DArray(&arrGuide,
				&texGuide.channelDesc,
				make_cudaExtent(imgSzx, imgSzy, imgSzz)))
		{
			throw std::exception();
		}
		cudaBindTextureToArray(texImg, arrImg);
		cudaBindTextureToArray(texGuide, arrGuide);

		cudaMemcpy(cuGaussian, gaussian, sizeof(float) * kSzx * kSzy * kSzz,
				cudaMemcpyHostToDevice);

		cudaMemcpy3DParms params = {0};
		params.dstArray = arrImg;
		params.srcPtr = make_cudaPitchedPtr(
				img, sizeof(float) * imgSzx, imgSzx, imgSzy);
		params.extent = make_cudaExtent(imgSzx, imgSzy, imgSzz);
		params.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&params);

		params.dstArray = arrGuide;
		params.srcPtr = make_cudaPitchedPtr(
				guide, sizeof(float) * imgSzx, imgSzx, imgSzy);
		cudaMemcpy3D(&params);

		dim3 threads(16, 16, 1);
		dim3 blocks((int)ceilf(imgSzx / (float)threads.x),
				(int)ceilf(imgSzy / (float)threads.y),
				(int)ceilf(imgSzz / (float)threads.z));
		NLM_Kernel<<<blocks, threads>>>(cuRes, cuGaussian,
				make_int3(imgSzx, imgSzy, imgSzz),
				make_int3(sSzx, sSzy, sSzz),
				make_int3(kSzx, kSzy, kSzz),
				d2, eps);

		cudaMemcpy(res, cuRes, sizeof(float) * imgSzx * imgSzy * imgSzz,
				cudaMemcpyDeviceToHost);
	}
	catch (std::exception &e)
	{
		if (cuRes != NULL)
		{
			cudaFree(cuRes);
		}
		if (cuGaussian != NULL)
		{
			cudaFree(cuGaussian);
		}
		if (arrImg != NULL)
		{
			cudaFreeArray(arrImg);
		}
		if (arrGuide != NULL)
		{
			cudaFreeArray(arrGuide);
		}
		return cudaGetLastError();
	}

	cudaFree(cuRes);
	cudaFree(cuGaussian);
	cudaFreeArray(arrImg);
	cudaFreeArray(arrGuide);

	return cudaGetLastError();

}

