#include <math.h>
#include <iostream>
#include <stdexcept>
#include <sstream>

using namespace std;

//time-intensity profile similarity (TIPS) denoising for CT perfusion
//It is a biliteral filtering where the weighting is determined by all the time frames

#define MIN(x,y) (x) < (y) ? (x) : (y)
#define MAX(x,y) (x) > (y) ? (x) : (y)

__device__ int GetIndex(int ix, int iy, int iz, int it, int ny, int nz, int nt)
{
	return ix * ny * nz * nt + iy * nz * nt + iz * nt + it;
}

__global__ void TIPSKernel(float* dst, float* weights, const float* src, int nx, int ny, int nz, int nt,
		int hwx, int hwy, int hwz, float varTips, float varDist, float eps = 1e-6f)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	int ind = GetIndex(ix, iy, iz, 0, ny, nz, nt);
	int xStart = MAX(0, ix - hwx);
	int xEnd = MIN(nx, ix + hwx + 1);
	int yStart = MAX(0, iy - hwy);
	int yEnd = MIN(ny, iy + hwy + 1);
	int zStart = MAX(0, iz - hwz);
	int zEnd = MIN(nz, iz + hwz + 1);

	register float wTips, wDist;

	for (int iix = xStart; iix < xEnd; iix++)
	{
		for (int iiy = yStart; iiy < yEnd; iiy++)
		{
			for (int iiz = zStart; iiz < zEnd; iiz++)
			{
				int iind = GetIndex(iix, iiy, iiz, 0, ny, nz, nt);

				// calculate TIPS weighting
				wTips = 0;
				for (int it = 0; it < nt; it++)
				{
					float v1 = src[iind + it];
					float v2 = src[ind + it];
					wTips += (v1 - v2) * (v1 - v2);
				}
				wTips = __expf(-0.5f * wTips / (nt * varTips));
//
//				// calculate distance weighting
				wDist = (iix - ix) * (iix - ix) + (iiy - iy) * (iiy - iy) + (iiz - iz) * (iiz - iz);
				wDist = __expf(-0.5f * wDist / varDist);
//
//				// TIPS filtering
				for (int it = 0; it < nt; it++)
				{
					dst[ind + it] += wTips * wDist * src[iind + it];
					weights[ind + it] += wTips * wDist;
				}
			}
		}
	}

	for (int it = 0; it < nt; it++)
	{
		dst[ind + it] /= (weights[ind + it] + eps);
	}
}

// for auto estimation of varTips
__global__ void SSDKernel(float* dst, const float* src, int nx, int ny, int nz, int nt,
		int hwx, int hwy, int hwz)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	int ind = GetIndex(ix, iy, iz, 0, ny, nz, nt);
	int xStart = MAX(0, ix - hwx);
	int xEnd = MIN(nx, ix + hwx + 1);
	int yStart = MAX(0, iy - hwy);
	int yEnd = MIN(ny, iy + hwy + 1);
	int zStart = MAX(0, iz - hwz);
	int zEnd = MIN(nz, iz + hwz + 1);

	register float wTips;
	register int n = 0;
	register float val = 0;

	for (int iix = xStart; iix < xEnd; iix++)
	{
		for (int iiy = yStart; iiy < yEnd; iiy++)
		{
			for (int iiz = zStart; iiz < zEnd; iiz++)
			{
				int iind = GetIndex(iix, iiy, iiz, 0, ny, nz, nt);

				// calculate TIPS weighting
				wTips = 0;
				for (int it = 0; it < nt; it++)
				{
					float v1 = src[iind + it];
					float v2 = src[ind + it];
					wTips += (v1 - v2) * (v1 - v2);
				}

				val += wTips / nt;
				n++;
			}
		}
	}

	dst[ind] = val / n;
}

extern "C" int TIPS(float* dst, float* src, int nBatches, int nx, int ny, int nz, int nt,
		int hwx, int hwy, int hwz, float varTips, float varDist, float eps = 1e-6f,
		int ssdOnly = 0)
{
	float* cuDst = NULL;
	float* cuSrc = NULL;
	float* cuWeight = NULL;

	try
	{

		int N = nBatches * nx * ny * nz * nt;
		if (cudaSuccess != cudaMalloc(&cuDst, sizeof(float) * N))
		{
			throw std::runtime_error("cuDst allocation error");
		}
		if (cudaSuccess != cudaMalloc(&cuSrc, sizeof(float) * N))
		{
			throw std::runtime_error("cuSrc allocation error");
		}
		if (cudaSuccess != cudaMalloc(&cuWeight, sizeof(float) * N))
		{
			throw std::runtime_error("cuWeight allocation error");
		}
		cudaMemcpy(cuSrc, src, sizeof(float) * N, cudaMemcpyHostToDevice);
		cudaMemset(cuDst, 0, sizeof(float) * N);
		cudaMemset(cuWeight, 0, sizeof(float) * N);

		dim3 threads(16, 16, 1);
		dim3 blocks((int)ceilf(nx / (float)threads.x),
				(int)ceilf(ny / (float)threads.y),
				(int)ceilf(nz / (float)threads.z));
		for (int ib = 0; ib < nBatches; ib++)
		{
			if (ssdOnly)
			{
				SSDKernel<<<blocks, threads>>>(cuDst + ib * nx * ny * nz * nt,
						cuSrc + ib * nx * ny * nz * nt,
						nx, ny, nz, nt, hwx, hwy, hwz);
			}
			else
			{
				TIPSKernel<<<blocks, threads>>>(cuDst + ib * nx * ny * nz * nt,
						cuWeight + ib * nx * ny * nz * nt,
						cuSrc + ib * nx * ny * nz * nt,
						nx, ny, nz, nt, hwx, hwy, hwz, varTips, varDist, eps);
			}
		}

		cudaMemcpy(dst, cuDst, sizeof(float) * N, cudaMemcpyDeviceToHost);
	}
	catch (std::exception &e)
	{
		if (cuDst != NULL)
		{
			cudaFree(cuDst);
		}
		if (cuSrc != NULL)
		{
			cudaFree(cuSrc);
		}
		if (cuWeight != NULL)
		{
			cudaFree(cuWeight);
		}

		ostringstream oss;
		oss << "TIPS error: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		throw runtime_error(oss.str().c_str());

	}

	cudaFree(cuDst);
	cudaFree(cuSrc);
	cudaFree(cuWeight);

	return cudaGetLastError();

}

