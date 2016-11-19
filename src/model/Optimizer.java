package model;

import java.io.*;
import java.util.Vector;

public class Optimizer {
	
	// radiance map [L][X][Y][W*H]
	static float[][][][] radianceMap;
	int L,B,X,Y;
	int W,H;
	
	static float[] energy;
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	public Optimizer(String foldername){
		
		// set size
		L = Config.L_NUM;
		B = Config.B_NUM;
		X = Config.X_NUM;
		Y = Config.Y_NUM;
		W = Lighty.COARSE_W;
		H = Lighty.COARSE_H;
		
		// allocate arrays
		radianceMap = new float[L][X][Y][W*H];
		
		// copy energy consumption
		energy = Config.energy;
		
		// input radiance maps from files
		System.out.print("Load radiance maps - ");
		for(int n=0;n<L;n++){
			System.out.print(" "+n);
			
			for(int x=0;x<X;x++)
			for(int y=0;y<Y;y++){
				
				String filename = new String(foldername+"\\"
						+String.valueOf(n)+"_"
						+String.valueOf(x)+"_"
						+String.valueOf(y)+".txt");
				
				try{
					BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
					for(int j=0;j<H;j++)
					for(int i=0;i<W;i++){
						radianceMap[n][x][y][i+j*W] = Float.valueOf(br.readLine());
					}
					br.close();
				}catch(Exception e){
					e.printStackTrace();
				}
			}
		}
		System.out.println(" - done.");
	}
	
	
	////////////////////////////////////////////////////////////
	// Single-Start Hill Climbing
	////////////////////////////////////////////////////////////
	public Tuple SSHC(int[] conf, float[] solver_radiance, float alpha, boolean actuated){
//		GPU gpu = GPU.getInstance();
		
		// initial evaluation
		float diff =     evaluate(conf, solver_radiance, alpha);
//		float diff = gpu.evaluate(conf, solver_radiance, alpha);
		
		// iteration loop
//		int iter = 0;
		while(true){
//			iter++;
//			System.out.println("iter="+iter+"  "+diff);
			
			// search around c
			Tuple tuple = searchAroundCPU(conf, solver_radiance, alpha, actuated) ;
//			Tuple tuple = gpu.searchAround(conf, alpha, actuated) ;
			
			float min_diff = tuple.getDiff();
			int[] min_conf = tuple.getConf();
			
			// compare
			if(diff <= min_diff) break;
			
			// change the current configuration
			diff = min_diff;
			conf = Config.copyConfig(min_conf);
		}
		
//		System.out.println("iter = "+iter);
		return new Tuple(diff, conf);
	}
	
	
	////////////////////////////////////////////////////////////
	// Multi-Start Hill Climbing
	////////////////////////////////////////////////////////////
	static final int numOfSeed = 3;
	public Tuple MSHC(Vector<int[]> confs, float[] solver_radiance, boolean actuated){
		GPU gpu = GPU.getInstance();
		
		// checking array
		boolean[] local_opt = new boolean[numOfSeed];
		for(int i=0;i<numOfSeed;i++) local_opt[i] = false;
		
//		System.out.println("\nsearch:");
		
		// initial evaluation
		float[] diffs = gpu.evaluateMulti(confs, solver_radiance);
		
		// iteration loop
//		int iter = 0;
		while(true){
//			System.out.println("iter="+(iter++));
			
//			for(int i=0;i<numOfSeed;i++) System.out.print(diffs[i]+"\t");
//			System.out.println();
			
			// search around {c}
			Vector<Tuple> tuples = gpu.searchAroundMulti(confs, local_opt, actuated);
			
			// compare & change
			Vector<int[]> new_confs = new Vector<int[]>();
			for(int i=0;i<numOfSeed;i++){
				int[] conf = Config.copyConfig(confs.get(i));
				
				float new_diff = tuples.get(i).getDiff();
				
				if(!local_opt[i]){
					if( diffs[i] <= new_diff ){
						local_opt[i] = true;
					}
					else{
						diffs[i] = new_diff;
						conf = Config.copyConfig(tuples.get(i).getConf());
					}
				}
				
				new_confs.add(conf);
			}
			confs = new_confs;
			
			// compare 2. all seeds reaches to local minimum
			boolean flag = true;
			for(int i=0;i<numOfSeed;i++){
				if(!local_opt[i]){
					flag = false;
					break;
				}
			}
			if(flag) break; // then break the loop
		}
		
		// return MINIMUM value
		float minDiff = Float.MAX_VALUE;
		int   minIndex = -1;
		for(int i=0;i<numOfSeed;i++){
			if(diffs[i]<minDiff){
				minDiff = diffs[i];
				minIndex = i;
			}
		}
//		System.out.println(minDiff+"\n");
		
		return new Tuple(minDiff, Config.copyConfig(confs.get(minIndex)));
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutines
	////////////////////////////////////////////////////////////
	//*
	public Tuple searchAroundCPU(int[] conf, float[] user_radiance, float alpha, boolean actuated){
		float min_diff = Float.MAX_VALUE;
		int[] min_conf = Config.copyConfig(conf);
		
		int[] plusElem = {B-1, X-1, Y-1};
		
		////////////////////////////////////////
		// 12 (L) x 6 (B+,B-,X+,X-,Y+,Y-)
		////////////////////////////////////////
		for(int n=0;n<L;n++){
			int VEC = (actuated) ? 3 : 1 ;
			
			for(int e=0;e<VEC;e++){
				
				// element +
				if( conf[3*n+e] < plusElem[e] ){
					conf[3*n+e] += 1; // increment
					
					float diff = evaluate(conf, user_radiance, alpha);
					if(diff<min_diff){
						min_diff = diff;
						min_conf = Config.copyConfig(conf);
					}
					
					conf[3*n+e] -= 1; // back to original vector
				}
				
				// element -
				if( conf[3*n+e] > 0 ){
					conf[3*n+e] -= 1; // decrement
					
					float diff = evaluate(conf, user_radiance, alpha);
					if(diff<min_diff){
						min_diff = diff;
						min_conf = Config.copyConfig(conf);
					}
					
					conf[3*n+e] += 1; // back to original vector
				}
			}
		}
		
		return new Tuple(min_diff, min_conf);
	}
	private float evaluate(int[] conf, float[] user_radiance, float alpha){
		
		float design_term = getDesignTerm(conf, user_radiance);
		float energy_term = getEnergyTerm(conf);
		
//		System.out.println("value = "+design_term+" + "+alpha+" * "+energy_term);
		return design_term + alpha * energy_term;
	}
	protected float getDesignTerm(int[] conf, float[] user_radiance){
		
		////////////////////////////////////////
		// sum in radiance domain
		////////////////////////////////////////
		float[] synth_radiance = new float[W*H];
		for(int n=0;n<L;n++){
			int b = conf[3*n+0];
			int x = conf[3*n+1];
			int y = conf[3*n+2];
			
			for(int j=0;j<H;j++)
			for(int i=0;i<W;i++){
				synth_radiance[i+j*W] += radianceMap[n][x][y][i+j*W]/255.0*Config.brightness[b];
			}
		}
		
		////////////////////////////////////////
		// design term
		////////////////////////////////////////
		float design_term = 0.0f;
		for(int j=0;j<H;j++)
		for(int i=0;i<W;i++){
			design_term += L2norm(user_radiance[i+j*W], synth_radiance[i+j*W]);
		}
		design_term /= (float)(W*H);
		
		return design_term;
	}
	protected float getEnergyTerm(int[] conf){
		
		////////////////////////////////////////
		// energy term
		////////////////////////////////////////
		float energy_term = 0.0f;
		for(int n=0;n<L;n++){
			energy_term += energy[ conf[3*n] ];
		}
		
		return energy_term;
	}
	private float L2norm(float a, float b){ return (a-b)*(a-b); }
	//*/
}
