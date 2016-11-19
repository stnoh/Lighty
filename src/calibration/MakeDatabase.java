package calibration;

import java.io.*;

import model.*;


public class MakeDatabase {
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	static int L, X, Y;
	static int IMAGE_W, IMAGE_H;
	static int PAINT_W, PAINT_H;
	
	
	////////////////////////////////////////////////////////////
	// launcher
	////////////////////////////////////////////////////////////
	public static void main(String[] args){
		
		// initialization
		Config.init(12, "light_config_params.txt");
		Transform.init("g_YRGB.txt");
		
		// get constants
		L = Config.L_NUM;
		X = Config.X_NUM;
		Y = Config.Y_NUM;
		IMAGE_W = Config.IMAGE_W;
		IMAGE_H = Config.IMAGE_H;
		
		// default arguments
		//*
		String iFoldername  = "calibration\\sample";
		String oFoldername  = ".";
		int i_exposure_time = 60;
		int o_exposure_time = 100;
		PAINT_W = 80;
		PAINT_H = 60;
		/*/
		// input arguments
		String iFoldername = args[0];
		int    i_exposure_time = Integer.valueOf(args[1]);
		int    o_exposure_time = Integer.valueOf(args[2]);
		
		// resolution: ex) 80x60
		String[] res = args[3].split("x");
		PAINT_W = Integer.valueOf(res[0]);
		PAINT_H = Integer.valueOf(res[1]);
		
		// output folder
		String oFoldername = args[4];
		//*/
		
		// is darkroom?
		boolean darkroom = true;
		
		// make folder
		new File(oFoldername).mkdir();
		
		// make database with downscaling & grayscaling
		String radianceMap_str = oFoldername+"\\radianceMap"+PAINT_W+"x"+PAINT_H;
		new MakeDatabase(iFoldername, radianceMap_str, i_exposure_time, darkroom);
		
		// make base radiance map
		new MakeBase(iFoldername, radianceMap_str, oFoldername, "virtual", 
				PAINT_W, PAINT_H, i_exposure_time, o_exposure_time, darkroom);
	}
	
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	MakeDatabase(String iFoldername, String oFoldername, int exposure_time, boolean darkroom){
		
		// make folder
		new File(oFoldername).mkdir();
		
		// calibration process
		for(int n=0;n<L;n++){
			
			// convert OFF image to radiance map
			String light_0_str = n+"_0";
			double[][] id_0 = Transform.Image2Radiance(iFoldername+"\\"+exposure_time+"\\"+light_0_str+".jpg",
								IMAGE_W, IMAGE_H, exposure_time, false);
			
			for(int x=0;x<X;x++)
			for(int y=0;y<Y;y++)
			{
				// convert ON image to radiance map
				String light_1_str = n+"_1_"+x+"_"+y;
				double[][] id_i = Transform.Image2Radiance(iFoldername+"\\"+exposure_time+"\\"+light_1_str+".jpg",
								IMAGE_W, IMAGE_H, exposure_time, false);
				
				// calculate an effect of light parameter
				double[][] light_i = new double[4][IMAGE_W*IMAGE_H];
				for(int c=0;c<4;c++)
				for(int j=0;j<IMAGE_H;j++)
				for(int i=0;i<IMAGE_W;i++){
					light_i[c][i+j*IMAGE_W] = (darkroom)
											? id_i[c][i+j*IMAGE_W]
											: id_i[c][i+j*IMAGE_W] - id_0[c][i+j*IMAGE_W];
				}
				
				// downscaling
				double[][] down = downscaling(light_i);
				
				// export to text file
				String light_i_str = n+"_"+x+"_"+y;
				Transform.writeArrayToFile(oFoldername+"\\"+light_i_str+".txt", down[0], PAINT_W, PAINT_H);
			}
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine : downscaling
	////////////////////////////////////////////////////////////
	protected double[][] downscaling(double[][] double_array){
		double[][] downscaled_array = new double[4][PAINT_W*PAINT_H];
		
		for(int c=0;c<4;c++)
		for(int j=0;j<IMAGE_H;j++)
		for(int i=0;i<IMAGE_W;i++){
			int bi = convert_x(i);
			int bj = convert_y(j);
			
			downscaled_array[c][bi+bj*PAINT_W]
					+= double_array[c][i+j*IMAGE_W] / ( (IMAGE_W/PAINT_W)*(IMAGE_H/PAINT_H) );
		}
		
		return downscaled_array;
	}
	private int convert_x(int x){ return (x*PAINT_W)/IMAGE_W; }
	private int convert_y(int y){ return (y*PAINT_H)/IMAGE_H; }
}
