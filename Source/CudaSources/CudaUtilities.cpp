/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Geometry.h"
#include "Logger.h"
#include "CudaUtilities.h"
#include <QString>
#include <cuda_runtime.h>

CCudaTimer::CCudaTimer(void)
{
	StartTimer();
}

CCudaTimer::~CCudaTimer(void)
{
	cudaEventDestroy(m_EventStart);
	cudaEventDestroy(m_EventStop);
}

void CCudaTimer::StartTimer(void)
{
	cudaEventCreate(&m_EventStart);
	cudaEventCreate(&m_EventStop);
	cudaEventRecord(m_EventStart, 0);

	m_Started = true;
}

float CCudaTimer::StopTimer(void)
{
	if (!m_Started)
		return 0.0f;

	cudaEventRecord(m_EventStop, 0);
	cudaEventSynchronize(m_EventStop);

	float TimeDelta = 0.0f;

	cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);
	cudaEventDestroy(m_EventStart);
	cudaEventDestroy(m_EventStop);

	m_Started = false;

	return TimeDelta;
}

float CCudaTimer::ElapsedTime(void)
{
	if (!m_Started)
		return 0.0f;

	cudaEventRecord(m_EventStop, 0);
	cudaEventSynchronize(m_EventStop);

	float TimeDelta = 0.0f;

	cudaEventElapsedTime(&TimeDelta, m_EventStart, m_EventStop);

	m_Started = false;

	return TimeDelta;
}

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void GetCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
	CUresult error = 	cuDeviceGetAttribute( attribute, device_attribute, device );

	if( CUDA_SUCCESS != error) {
		fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);
		exit(-1);
	}
}

void HandleCudaError(const cudaError_t CudaError, const char* pDescription /*= ""*/)
{
	if (CudaError == cudaSuccess)
		return;

	//WR Qt4->Qt5
	// Log(QString("Encountered a critical CUDA error: " + QString::fromAscii(pDescription) + " " + QString(cudaGetErrorString(CudaError))));

	// throw new QString("Encountered a critical CUDA error: " + QString::fromAscii(pDescription) + " " + QString(cudaGetErrorString(CudaError)));

	Log(QString("Encountered a critical CUDA error: " + QString::fromLatin1(pDescription) + " " + QString(cudaGetErrorString(CudaError))));

	throw new QString("Encountered a critical CUDA error: " + QString::fromLatin1(pDescription) + " " + QString(cudaGetErrorString(CudaError)));
}

void HandleCudaKernelError(const cudaError_t CudaError, const char* pName /*= ""*/)
{
	if (CudaError == cudaSuccess)
		return;

	//WR Qt4->Qt5
	// Log(QString("The '" + QString::fromAscii(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError))));

	// throw new QString("The '" + QString::fromAscii(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError)));

	Log(QString("The '" + QString::fromLatin1(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError))));

	throw new QString("The '" + QString::fromLatin1(pName) + "' kernel caused the following CUDA runtime error: " + QString(cudaGetErrorString(CudaError)));
}

int GetTotalCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Total;
}

int GetAvailableCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Available;
}

int GetUsedCudaMemory(void)
{
	size_t Available = 0, Total = 0;
	cudaMemGetInfo(&Available, &Total);
	return Total - Available;
}

int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 },
	  { 0x11,  8 },
	  { 0x12,  8 },
	  { 0x13,  8 },
	  { 0x20, 32 },
	  { 0x21, 48 },
	  {   -1, -1 } 
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}

int GetMaxGigaFlopsDeviceID(void)
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = max(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		    sm_per_multiproc = 1;
		} else {
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

bool SetCudaDevice(const int& CudaDeviceID)
{
	const cudaError_t CudaError = cudaSetDevice(CudaDeviceID);

	HandleCudaError(CudaError, "set Cuda device");

	return CudaError == cudaSuccess;
}

void ResetDevice(void)
{
	HandleCudaError(cudaDeviceReset(), "reset device");
}

extern "C" void BindDensityBuffer(short* pBuffer, cudaExtent Extent);
extern "C" void BindGradientMagnitudeBuffer(short* pBuffer, cudaExtent Extent);
extern "C" void UnbindDensityBuffer(void);
extern "C" void UnbindGradientMagnitudeBuffer(void);
extern "C" void BindRenderCanvasView(const CResolution2D& Resolution);
extern "C" void ResetRenderCanvasView(void);
extern "C" void FreeRenderCanvasView(void);
extern "C" unsigned char* GetDisplayEstimate(void);
extern "C" void BindTransferFunctionOpacity(CTransferFunction& TransferFunctionOpacity);
extern "C" void BindTransferFunctionDiffuse(CTransferFunction& TransferFunctionDiffuse);
extern "C" void BindTransferFunctionSpecular(CTransferFunction& TransferFunctionSpecular);
extern "C" void BindTransferFunctionRoughness(CTransferFunction& TransferFunctionRoughness);
extern "C" void BindTransferFunctionEmission(CTransferFunction& TransferFunctionEmission);
extern "C" void UnbindTransferFunctionOpacity(void);
extern "C" void UnbindTransferFunctionDiffuse(void);
extern "C" void UnbindTransferFunctionSpecular(void);
extern "C" void UnbindTransferFunctionRoughness(void);
extern "C" void UnbindTransferFunctionEmission(void);
extern "C" void BindConstants(CScene* pScene);
extern "C" void Render(const int& Type, CScene& Scene, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage);


void CudaUtil::BindDensityBufferExt(short* pBuffer, cudaExtent Extent) {
	BindDensityBuffer(pBuffer,Extent);
}


void CudaUtil::BindGradientMagnitudeBufferExt(short* pBuffer, cudaExtent Extent) {
	BindGradientMagnitudeBuffer(pBuffer, Extent);
}

void CudaUtil::UnbindDensityBufferExt(void) {
	UnbindDensityBuffer();
}

void CudaUtil::UnbindGradientMagnitudeBufferExt(void) {
	UnbindGradientMagnitudeBuffer();
}

void CudaUtil::BindRenderCanvasViewExt(const CResolution2D& Resolution) {
	BindRenderCanvasView(Resolution);
}

void CudaUtil::ResetRenderCanvasViewExt(void) {
	ResetRenderCanvasView();
}

void CudaUtil::FreeRenderCanvasViewExt(void) {
	FreeRenderCanvasView();
}

unsigned char* CudaUtil::GetDisplayEstimateExt(void) {
	return GetDisplayEstimate();
}

void CudaUtil::BindTransferFunctionOpacityExt(CTransferFunction& TransferFunctionOpacity) {
	BindTransferFunctionOpacity(TransferFunctionOpacity);
}

void CudaUtil::BindTransferFunctionDiffuseExt(CTransferFunction& TransferFunctionDiffuse) {
	BindTransferFunctionDiffuse(TransferFunctionDiffuse);
}

void CudaUtil::BindTransferFunctionSpecularExt(CTransferFunction& TransferFunctionSpecular) {
	BindTransferFunctionSpecular(TransferFunctionSpecular);
}

void CudaUtil::BindTransferFunctionRoughnessExt(CTransferFunction& TransferFunctionRoughness) {
	BindTransferFunctionRoughness(TransferFunctionRoughness);
}

void CudaUtil::BindTransferFunctionEmissionExt(CTransferFunction& TransferFunctionEmission) {
	BindTransferFunctionEmission(TransferFunctionEmission);
}

void CudaUtil::UnbindTransferFunctionOpacityExt(void) {
	UnbindTransferFunctionOpacity();
}

void CudaUtil::UnbindTransferFunctionDiffuseExt(void) {
	UnbindTransferFunctionDiffuse();
}

void CudaUtil::UnbindTransferFunctionSpecularExt(void) {
	UnbindTransferFunctionSpecular();
}

void CudaUtil::UnbindTransferFunctionRoughnessExt(void) {
	UnbindTransferFunctionRoughness();
}

void CudaUtil::UnbindTransferFunctionEmissionExt(void) {
	UnbindTransferFunctionEmission();
}

void CudaUtil::BindConstantsExt(CScene* pScene) {
	BindConstants(pScene);
}

void CudaUtil::RenderExt(const int& Type, CScene& Scene, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage) {
	Render(Type, Scene, RenderImage, BlurImage, PostProcessImage, DenoiseImage);
}
