extern "C" __global__ void pixelDiff_kernel
(float** diff0, float* obj_func,
 const float* user_radiance, const int* conf,
 const int L, const int X_Y, const int Y, const int VEC_SIZE,
 const float** radianceMap, const float* energy, const float* brightness_ratio,
 const unsigned int divide)
{
	// pixel area (i,j)
	unsigned int i = threadIdx.x;
//	unsigned int j = blockIdx.x;
	unsigned int j = divide*blockIdx.x + threadIdx.y;
	unsigned int W = blockDim.x;
//	unsigned int H = gridDim.x;
	
	// candidate id
	unsigned int candID = blockIdx.y;
	
	// synth in radiance area & calculate energy dissipation
	float synthRadiance = 0.0;
	float energyTerm    = 0.0;
	for(int n=0;n<L;n++){
		unsigned int b = conf[ candID*(L*VEC_SIZE) + (n*VEC_SIZE+0) ];
		unsigned int x = conf[ candID*(L*VEC_SIZE) + (n*VEC_SIZE+1) ];
		unsigned int y = conf[ candID*(L*VEC_SIZE) + (n*VEC_SIZE+2) ];
		
		unsigned int id = n*(X_Y)+x*Y+y;
		
		synthRadiance += radianceMap[id][i+j*W]*brightness_ratio[b];
		energyTerm    += energy[b];
	}
	
	// difference
	float diff = synthRadiance - user_radiance[i+j*W]; // radiance-based
//	float designTerm = fabs(diff*diff); // L1-norm
	float designTerm = diff*diff ; // L2-norm
	
	// write the result
	diff0[candID][i+j*W] = designTerm;
	if(i==0 && j==0)
		obj_func[candID] = energyTerm;
}


extern "C" __global__ void reduceSum_kernel
(float** g_idata, float** g_odata, unsigned int n)
{
	extern __shared__ float sdata[];
	
	unsigned int tid      = threadIdx.x;
	unsigned int i        = blockDim.x*2*blockIdx.x + tid;
	unsigned int gridSize = blockDim.x*2*gridDim.x;
	
	int id = blockIdx.y;
	
	float mySum = 0.0;
	
	while(i<n){
		mySum += g_idata[id][i];
		if( i + blockDim.x < n)
			mySum += g_idata[id][i+blockDim.x];
		i+=gridSize;
	}
	
	sdata[tid] = mySum;
	__syncthreads();
	
	if(blockDim.x >= 512){ if(tid<256) { sdata[tid] = mySum = mySum + sdata[tid+256]; } __syncthreads(); }
	if(blockDim.x >= 256){ if(tid<128) { sdata[tid] = mySum = mySum + sdata[tid+128]; } __syncthreads(); }
	if(blockDim.x >= 128){ if(tid< 64) { sdata[tid] = mySum = mySum + sdata[tid+ 64]; } __syncthreads(); }
	
	if(tid<32)
	{
		volatile float* smem = sdata;
		if( blockDim.x >= 64 ){ smem[tid] = mySum = mySum + smem[tid+32]; }
		if( blockDim.x >= 32 ){ smem[tid] = mySum = mySum + smem[tid+16]; }
		if( blockDim.x >= 16 ){ smem[tid] = mySum = mySum + smem[tid+ 8]; }
		if( blockDim.x >=  8 ){ smem[tid] = mySum = mySum + smem[tid+ 4]; }
		if( blockDim.x >=  4 ){ smem[tid] = mySum = mySum + smem[tid+ 2]; }
		if( blockDim.x >=  2 ){ smem[tid] = mySum = mySum + smem[tid+ 1]; }
	}

	if(tid==0)
		g_odata[id][blockIdx.x] = sdata[0];
}


extern "C" __global__ void transfer_kernel
(float* obj_func, float** diff1, int W, int H, float alpha)
{
	unsigned int candID = blockIdx.y;
	
	float designTerm = diff1[candID][0];
	float energyTerm = obj_func[candID];
	
	obj_func[candID] = designTerm/((float)W*H) + alpha * energyTerm;
}
