package model;

import java.io.*;
import java.util.*;

////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////
public class Config {
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	public static final int   IMAGE_W = 640;
	public static final int   IMAGE_H = 480;
	public static final int[] exposure_time = {60, 100, 250, 500, 1000, 2000, 4000, 10000};
	
	
	////////////////////////////////////////////////////////////
	// Components
	////////////////////////////////////////////////////////////
	static String comPort;
	
	public static int L_NUM;
	public static int B_NUM;
	public static int X_NUM;
	public static int Y_NUM;
	
	public static int[] brightness;
	protected static int[] x_degree;
	protected static int[] y_degree;
	
	protected static float[] energy;
	
	public static int[] defaultConfig;
	
	
	////////////////////////////////////////////////////////////
	// initialize by text file
	////////////////////////////////////////////////////////////
	public static void init(int num, String configFilename){
		
		// the number of lights
		L_NUM = num;
		
		// input from file
		try{
			BufferedReader br = new BufferedReader(new FileReader( new File(configFilename)));
			
			while(br.ready()){
				String line = br.readLine();
				
				if(line.contains("com")){
					comPort = getComPortFromString(line);
				}
				else if(line.contains("brightness")){
					brightness = getIntValuesFromString(line);
				}
				else if(line.contains("xdeg")){
					x_degree = getIntValuesFromString(line);
				}
				else if(line.contains("ydeg")){
					y_degree = getIntValuesFromString(line);
				}
				else if(line.contains("energy")){
					energy = getFloatValuesFromString(line);
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		
		// set length
		B_NUM = brightness.length;
		X_NUM = x_degree.length;
		Y_NUM = y_degree.length;
		
		// print out for verification
		print();
		System.out.println();
		
		// g_calib
		defaultConfig = downwardsConfig(0);
		try{
			boolean flag = false;
			BufferedReader br = new BufferedReader(new FileReader( new File(configFilename)));
			
			while(br.ready()){
				String line = br.readLine();
				
				if(flag){
					if(line.contains("g_calib")) break;
					
					String strs[] = line.split(" ");
					
					int n = Integer.valueOf(strs[0]);
					int b = Integer.valueOf(strs[1]);
					int x = Integer.valueOf(strs[2]);
					int y = Integer.valueOf(strs[3]);
					
					defaultConfig[3*n+0] = b;
					defaultConfig[3*n+1] = x;
					defaultConfig[3*n+2] = y;
				}
				
				if(line.contains("g_calib")){
					flag = !flag;
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	////////////////////////////////////////////////////////////
	// Method : get predefine configuration
	////////////////////////////////////////////////////////////
	public static int[] downwardsConfig(int brightness){
		int[] conf = new int[3*L_NUM];
		
		for(int i=0;i<L_NUM;i++){
			conf[3*i+0] = brightness; // brightness
			conf[3*i+1] = X_NUM/2;    // angle_x    = 90 deg.
			conf[3*i+2] = Y_NUM/2;    // angle_y    = 90 deg.
		}
		
		return conf;
	}
	public static int[] getRandomConf(){
		Random random = new Random();
		int[] conf = new int[3*L_NUM];
		
		for(int i=0;i<L_NUM;i++){
			conf[3*i+0] = random.nextInt(B_NUM); // brightness
			conf[3*i+1] = random.nextInt(X_NUM); // angle_x
			conf[3*i+1] = random.nextInt(Y_NUM); // angle_y
		}
		
		return conf;
	}
	public static int[] copyConfig(int[] conf){
		int[] newConf = new int[3*L_NUM];
		for(int i=0;i<L_NUM;i++){
			for(int e=0;e<3;e++)
				newConf[3*i+e] = conf[3*i+e];
		}
		
		return newConf;
	}
	public static boolean isSame(int[] conf1, int[] conf2){
		for(int i=0;i<3*L_NUM;i++){
			if(conf1[i]!=conf2[i]) return false;
		}
		
		return true;
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine : for reading the configuration text file
	////////////////////////////////////////////////////////////
	private static String getComPortFromString(String str){
		String strs[] = str.split(" ");
		return strs[1];
	}
	private static int[] getIntValuesFromString(String str){
		// split values
		String strs[] = str.split(" ");
		
		// create array
		int SIZE = strs.length-2;
		int[] values = new int[SIZE];
		for(int i=0;i<SIZE;i++)
			values[i] = Integer.valueOf(strs[i+1]);
		
		return values;
	}
	private static float[] getFloatValuesFromString(String str){
		// split values
		String strs[] = str.split(" ");
		
		// create array
		int SIZE = strs.length-2;
		float[] values = new float[SIZE];
		for(int i=0;i<SIZE;i++)
			values[i] = Float.valueOf(strs[i+1]);
		
		return values;
	}
	
	////////////////////////////////////////////////////////////
	// subroutine : show configuration detail
	////////////////////////////////////////////////////////////
	private static void print(){
		System.out.println("ComPort = " + comPort);
		
		System.out.print("Brightness = ");
		for(int b=0;b<B_NUM;b++) System.out.print(brightness[b]+" ");
		System.out.println();
		
		System.out.print("Energy = ");
		for(int e=0;e<B_NUM;e++) System.out.print(energy[e]+" ");
		System.out.println();
		
		System.out.print("xdeg = ");
		for(int x=0;x<X_NUM;x++) System.out.print(x_degree[x]+" ");
		System.out.println();
		
		System.out.print("ydeg = ");
		for(int y=0;y<Y_NUM;y++) System.out.print(y_degree[y]+" ");
		System.out.println();
	}
}
