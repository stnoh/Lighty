package model;

import static jcuda.runtime.JCuda.*;

import java.util.Vector;

import jcuda.*;
import jcuda.runtime.*;
import jcuda.utils.KernelLauncher;

////////////////////////////////////////////////////////////////////////////////
//  
////////////////////////////////////////////////////////////////////////////////
public class GPU {
	
	final int VEC_SIZE = 3;   // B, X, Y
	final int MAX_CAND = 72*Optimizer.numOfSeed; // 12*6  (B+,B-,X+,X-,Y+,Y-)
//	final int MAX_CAND = 12*26;  // 12*26  (B+,0,B-) x (X+,0,X-) x (Y+,0,Y-)
//	final int MAX_CAND = 12*124;
	
	private int getID(int n, int x, int y){ return n*(X*Y)+x*Y+y; };
	
	int L,B,X,Y;
	int W,H;
	
	// 1. radiance maps
	Pointer   d_radianceMap;
	Pointer[] d_each_radianceMap;
	
	// 2. user input
	Pointer d_user_radiance;
	
	// 3. configuration
	Pointer d_conf;
	
	// 4. difference
	Pointer   d_diff0;
	Pointer   d_diff1;
	Pointer[] d_each_diff0;
	Pointer[] d_each_diff1;
	
	// 5. energy consumption
	Pointer   d_energy;
	Pointer   d_brightness;
	
	// 6. objective function
	float[] h_obj_func;
	Pointer d_obj_func;
	
	// kernel launcher
	KernelLauncher pixelDiff_kernel;
	KernelLauncher reduceSum_kernel;
	KernelLauncher transfer_kernel;
	
	
	////////////////////////////////////////////////////////////
	// Public methods for SSHC
	////////////////////////////////////////////////////////////
	public float evaluate(int[] conf, float[] user_radiance, float alpha){
		
		// 1. Device <- Host : user_image, user_radiance
		cudaMemcpy(d_user_radiance, Pointer.to(user_radiance),
				W*H*Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 2. Device <- Host : (current) configuration
		cudaMemcpy(d_conf, Pointer.to(conf),
				1*L*VEC_SIZE*Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 3. launch the kernels & get objective function value
		return h_obj_func[getMinCandNum(1, alpha)];
	}
	
	public Tuple searchAround(int[] conf, float alpha, boolean actuated){
		int   numOfCand = 0;
		int[] h_conf = new int[MAX_CAND*L*VEC_SIZE];
		
		// 1. entry the candidate (72 candidates)
		int[] plusElem = {B-1,X-1,Y-1};
		for(int n=0;n<L;n++){
			//*
			int VEC = (actuated) ? VEC_SIZE : 1 ;
			
			for(int e=0;e<VEC;e++){
				// B+, X+, Y+
				if(conf[3*n+e] < plusElem[e]){
					// copy & modify
					for(int i=0;i<L*VEC_SIZE;i++) h_conf[numOfCand*(L*VEC_SIZE) +i] = conf[i];
					h_conf[numOfCand*(L*VEC_SIZE)+(3*n+e)] += 1;
					
					numOfCand++;
				}
				// B-, X-, Y-
				if(conf[3*n+e] > 0){
					// copy & modify
					for(int i=0;i<L*VEC_SIZE;i++) h_conf[numOfCand*(L*VEC_SIZE) +i] = conf[i];
					h_conf[numOfCand*(L*VEC_SIZE)+(3*n+e)] -= 1;
					
					numOfCand++;
				}
			}
			//*/
			
			/*
			for(int b=0;b<B;b++)
			for(int x=0;x<X;x++)
			for(int y=0;y<Y;y++){
				// except the same condition
				if( b==conf[3*n+0] && x==conf[3*n+1] && y==conf[3*n+2] )
					continue;
				
				// copy & modify
				for(int i=0;i<L*VEC_SIZE;i++) h_conf[numOfCand*(L*VEC_SIZE) +i] = conf[i];
				h_conf[numOfCand*(L*3)+(3*n+0)] = b;
				h_conf[numOfCand*(L*3)+(3*n+1)] = x;
				h_conf[numOfCand*(L*3)+(3*n+2)] = y;
				
				numOfCand++;
			}
			//*/
		}
		
		// 2. Device <- Host : (around) configurations
		cudaMemcpy(d_conf, Pointer.to(h_conf),
				numOfCand*L*VEC_SIZE*Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 3. launch the kernels
		int minNum = getMinCandNum(numOfCand, alpha);
		
		// 4. get the minimum configuration
		float minDiff = h_obj_func[minNum];
		int[] minConf = new int[L*VEC_SIZE];
		for(int i=0;i<L*VEC_SIZE;i++) minConf[i] = h_conf[minNum*(L*VEC_SIZE) +i];
		
		return new Tuple(minDiff, minConf);
	}
	
	
	////////////////////////////////////////////////////////////
	// Public methods for MSHC
	////////////////////////////////////////////////////////////
	public float[] evaluateMulti(Vector<int[]> confs, float[] user_radiance){
		
		// 1. Device <- Host : user_image, user_radiance
		cudaMemcpy(d_user_radiance, Pointer.to(user_radiance),
				W*H*Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 2. Device <- Host : (current) configurations
		// copy
		int seeds = Optimizer.numOfSeed;
		int[] h_conf = new int[seeds*L*VEC_SIZE];
		for(int s=0;s<seeds;s++){
			int conf[] = confs.get(s);
			for(int i=0;i<L*VEC_SIZE;i++) h_conf[s*(L*VEC_SIZE) +i] = conf[i];
		}
		cudaMemcpy(d_conf, Pointer.to(h_conf),
				seeds*L*VEC_SIZE*Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 3. launch the kernels & get objective function value
		bulkCalculation(seeds, 0.0f);
		
		// copy
		float[] diff = new float[Optimizer.numOfSeed];
		for(int i=0;i<Optimizer.numOfSeed;i++) diff[i] = h_obj_func[i];
		
		return diff;
	}
	
	public Vector<Tuple> searchAroundMulti(Vector<int[]> confs, boolean[] local_opt, boolean actuated){
		int seeds = Optimizer.numOfSeed;
		int   totalCand = 0;
		int[] h_conf = new int[MAX_CAND*L*VEC_SIZE];
		
		Vector<Integer> numOfCand = new Vector<Integer>();
		
		// 1. entry the candidate ( number of seeds * 72 candidates)
		int[] plusElem = {B-1,X-1,Y-1};
		for(int s=0;s<seeds;s++){
			int conf[] = confs.get(s);
			
			int m = 0;
			if(local_opt[s]){
				numOfCand.add(m);
				continue;
			}
			
			for(int n=0;n<L;n++){
				int VEC = (actuated) ? VEC_SIZE : 1 ;
				
				
				for(int e=0;e<VEC;e++){
					
					// B+, X+, Y+
					if(conf[3*n+e] < plusElem[e]){
						// copy & modify
						for(int i=0;i<L*VEC_SIZE;i++) h_conf[totalCand*(L*VEC_SIZE)+i] = conf[i];
						h_conf[totalCand*(L*VEC_SIZE)+(3*n+e)] += 1;
						
						totalCand++;
						m++;
					}
					// B-, X-, Y-
					if(conf[3*n+e] > 0){
						// copy & modify
						for(int i=0;i<L*VEC_SIZE;i++) h_conf[totalCand*(L*VEC_SIZE)+i] = conf[i];
						h_conf[totalCand*(L*VEC_SIZE)+(3*n+e)] -= 1;
						
						totalCand++;
						m++;
					}
				}
			}
			numOfCand.add(m);
		}
		
		// 2. Device <- Host : (around) configurations
		cudaMemcpy(d_conf, Pointer.to(h_conf),
				totalCand*L*VEC_SIZE*Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 3. launch the kernels
		bulkCalculation(totalCand, 0.0f);
		
		// 4. get the minimum configurations
		Vector<Tuple> tuples = new Vector<Tuple>();
		
		int n = 0;
		for(int s=0;s<seeds;s++){
			float minDiff = Float.MAX_VALUE;
			int[] minConf = Config.downwardsConfig(0);
			
			for(int c=0;c<numOfCand.get(s);c++,n++){
				if(h_obj_func[n]<minDiff){
					minDiff = h_obj_func[n];
					for(int i=0;i<L*VEC_SIZE;i++) minConf[i] = h_conf[n*(L*VEC_SIZE) +i];
				}
			}
			
			tuples.add(new Tuple(minDiff,minConf));
		}
		
		return tuples;
	}
		
	////////////////////////////////////////////////////////////
	// subroutines
	////////////////////////////////////////////////////////////
	int maxThreads = 128;
	int maxBlocks  = 64;
	
	void bulkCalculation(int numOfCand, float alpha){
		// 1. pixel difference & calculate energy ***HOT SPOT***
		final int divide = 1;
		
		pixelDiff_kernel.setup(new dim3(H/divide,numOfCand,1), new dim3(W,divide,1) );
		pixelDiff_kernel.call (
					d_diff0, d_obj_func,
					d_user_radiance, d_conf,
					L, X*Y, Y, VEC_SIZE,
					d_radianceMap, d_energy, d_brightness,
					divide);
		
		// 2. summation by parallel reduction
		int numOfElements = W*H;
		int numOfBlocks   = getNumOfBlocks (numOfElements, maxBlocks, maxThreads);
		int numOfThreads  = getNumOfThreads(numOfElements, maxBlocks, maxThreads);
		reduce(numOfElements, numOfCand, numOfThreads, numOfBlocks);
		
		// 3. copy objective function to one array
		transfer_kernel.setup(new dim3(1, numOfCand, 1), new dim3(1,1,1) );
		transfer_kernel.call (d_obj_func, d_diff1, W, H, alpha );
		
		// 4. Device -> Host : objective function
		cudaMemcpy(Pointer.to(h_obj_func), d_obj_func,
				numOfCand*Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	}
	
	int getMinCandNum(int numOfCand, float alpha){
		bulkCalculation(numOfCand, alpha);
		
		// 5. get the candidate number which has minimum value
		int   minCand = -1;
		float minDiff = Float.MAX_VALUE;
		for(int i=0;i<numOfCand;i++){
			if( h_obj_func[i] < minDiff){
				minCand = i;
				minDiff = h_obj_func[i];
			}
		}
		
		return minCand;
	}
	
	void reduce(int numOfElements, int numOfCand, int numOfThreads, int numOfBlocks){
		reduceSum_kernel.setup(new dim3(numOfBlocks,numOfCand,1), new dim3(numOfThreads,1,1), getSharedMemSize(numOfThreads));
		reduceSum_kernel.call (d_diff0, d_diff1, numOfElements);

		int s = numOfBlocks;
		while(s > 1){
			int threads = getNumOfThreads(s, maxBlocks, maxThreads);
			int blocks  = getNumOfBlocks (s, maxBlocks, maxThreads);
			int memsize = getSharedMemSize(threads);
			
			reduceSum_kernel.setup(new dim3(blocks,numOfCand,1), new dim3(threads,1,1), memsize);
			reduceSum_kernel.call (d_diff1, d_diff1, s);
			
			s = ( s + (threads*2-1) ) / ( threads*2 );
		}
	}
	
	private int getNumOfBlocks(int n, int maxBlocks, int maxThreads){
		int blocks = 0;
		int threads = getNumOfThreads(n, maxBlocks, maxThreads);
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
		blocks = Math.min(maxBlocks, blocks);
		return blocks;
	}
	private int getNumOfThreads(int n, int maxBlocks, int maxThreads){
		int threads = 0;
		threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
		return threads;
	}
	private int getSharedMemSize(int numOfThreads){
        int sharedMemSize = numOfThreads * Sizeof.FLOAT;
        if (numOfThreads <= 32) 
            sharedMemSize *= 2;
        return sharedMemSize;
	}
	private int nextPow2(int x){
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}
	
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	private GPU(){
		
		// get parameters
		L = Config.L_NUM; B = Config.B_NUM; X = Config.X_NUM; Y = Config.Y_NUM;
		W = Lighty.COARSE_W; H = Lighty.COARSE_H;
		
		// initialize & property
		initialize();
		cudaDeviceProp prop = new cudaDeviceProp();
		cudaGetDeviceProperties(prop, 0);
		
		// arrays
		System.out.print("Allocate GPU memory  : ");
		malloc();
		System.out.println("done.");
		
		System.out.print("Data transfer to GPU : ");
		memcpy();
		System.out.println("done.");
		
		// kernels
		pixelDiff_kernel = KernelLauncher.load("CUDA_kernel.cubin", "pixelDiff_kernel");
		reduceSum_kernel = KernelLauncher.load("CUDA_kernel.cubin", "reduceSum_kernel");
		transfer_kernel  = KernelLauncher.load("CUDA_kernel.cubin", "transfer_kernel");
	}
	void malloc(){
		long used_mem = 0;

		// 1. radiance maps
		d_each_radianceMap = new Pointer[L*X*Y];
		for(int n=0;n<L;n++)
		for(int x=0;x<X;x++)
		for(int y=0;y<Y;y++){
			int id = getID(n,x,y);
			d_each_radianceMap[id] = new Pointer();
			
			int mem_size = W*H*Sizeof.FLOAT;
			if( cudaMalloc(d_each_radianceMap[id], mem_size) != 0 )
				System.out.println("error: cannot allocate radiance map ("+n+","+x+","+y+")");
			else
				used_mem += mem_size;
		}
		d_radianceMap = new Pointer();
		cudaMalloc(d_radianceMap, L*X*Y*Sizeof.POINTER);
		cudaMemcpy(d_radianceMap, Pointer.to(d_each_radianceMap),
				L*X*Y*Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 2. user input
		d_user_radiance = new Pointer();
		cudaMalloc(d_user_radiance, W*H*Sizeof.FLOAT);
		
		// 3. configuration
		d_conf = new Pointer();
		cudaMalloc(d_conf, L*VEC_SIZE*MAX_CAND*Sizeof.INT);
		
		// 4-1. difference 0
		d_each_diff0 = new Pointer[MAX_CAND];
		for(int id=0;id<MAX_CAND;id++){
			d_each_diff0[id] = new Pointer();
			
			int mem_size = W*H*Sizeof.FLOAT;
			if( cudaMalloc(d_each_diff0[id], mem_size) != 0 )
				System.out.println("error: cannot allocate diff0 - candidate id: "+id);
			else
				used_mem += mem_size;
		}
		d_diff0 = new Pointer();
		cudaMalloc(d_diff0, MAX_CAND*Sizeof.POINTER);
		cudaMemcpy(d_diff0, Pointer.to(d_each_diff0),
				MAX_CAND*Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 4-2. difference 1 (for reduction)
		d_each_diff1 = new Pointer[MAX_CAND];
		for(int id=0;id<MAX_CAND;id++){
			d_each_diff1[id] = new Pointer();
			
			int mem_size = W*H*Sizeof.FLOAT;
			if( cudaMalloc(d_each_diff1[id], mem_size) != 0 )
				System.out.println("error: cannot allocate diff1 - candidate id: "+id);
			else
				used_mem += mem_size;
		}
		d_diff1 = new Pointer();
		cudaMalloc(d_diff1, MAX_CAND*Sizeof.POINTER);
		cudaMemcpy(d_diff1, Pointer.to(d_each_diff1),
				MAX_CAND*Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 5-1. energy consumption
		d_energy = new Pointer();
		cudaMalloc(d_energy, B*Sizeof.FLOAT);
		
		// 5-2. brightness ratio
		d_brightness = new Pointer();
		cudaMalloc(d_brightness, B*Sizeof.FLOAT);
		
		// 6. objective function
		h_obj_func = new float[MAX_CAND];
		d_obj_func = new Pointer();
		cudaMalloc(d_obj_func, MAX_CAND*Sizeof.FLOAT);
		
		// 
		long mem_as_MB = used_mem/(1024*1024);
		System.out.print(mem_as_MB+"MB ");
	}
	void memcpy(){
		
		// 1. radiance maps
		for(int n=0;n<L;n++)
		for(int x=0;x<X;x++)
		for(int y=0;y<Y;y++){
			int id = getID(n,x,y);
			int mem_size = W*H*Sizeof.FLOAT;
			cudaMemcpy(d_each_radianceMap[id], Pointer.to(Optimizer.radianceMap[n][x][y]),
					mem_size, cudaMemcpyKind.cudaMemcpyHostToDevice);
		}
		
		// 5-1. energy consumption
		cudaMemcpy(d_energy, Pointer.to(Optimizer.energy),
				B*Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		
		// 5-2. brightness ratio
		float[] h_brightness = new float[B];
		for(int b=0;b<B;b++){
			h_brightness[b] = (float)(Config.brightness[b]/255.0f);
		}
		cudaMemcpy(d_brightness, Pointer.to(h_brightness),
				B*Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
	}
	
	
	////////////////////////////////////////////////////////////
	// C++ style SINGLETON for jCUDA
	////////////////////////////////////////////////////////////
	private static GPU instance = null;
	public  static GPU getInstance(){
		if( instance == null )
			instance = new GPU();
		return instance;
	}
}
