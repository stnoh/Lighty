package calibration;

import java.awt.image.*;
import java.io.*;

import javax.imageio.ImageIO;

import model.*;

public class MakeBase {
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	int L;
	int IMAGE_W, IMAGE_H;
	int PAINT_W, PAINT_H;
	
	int exposure_time[];
	int brightness[];
	int defaultConfig[];
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	MakeBase(String photographFolder, String radianceMapFolder, String oFoldername, String oFilename,
			int W, int H, int i_exposure_time, int o_exposure_time, boolean darkroom){
		
		// constants
		L = Config.L_NUM;
		IMAGE_W = Config.IMAGE_W; IMAGE_H = Config.IMAGE_H;
		PAINT_W = W; PAINT_H = H;
		exposure_time = Config.exposure_time;
		brightness    = Config.brightness;
		defaultConfig = Config.defaultConfig;
		
		// using text files
		try{
			double[][] synth = synthFromText(radianceMapFolder);
			
			String oFile = oFoldername+"\\base.txt";
			Transform.writeArrayToFile(oFile, synth[0], PAINT_W, PAINT_H);
			
			// for certification (NO IN USE)
			/*
			String oFile = oFoldername+"\\base_luminance.png";
			System.out.println("write: "+oFile);
			BufferedImage im = Transform.Radiance2Image(synth, PAINT_W, PAINT_H, o_exposure_time, false);
			ImageIO.write(im, "png", new File(oFile) );
			//*/
		}catch(Exception e){
			e.printStackTrace();
		}
		
		// using photographs (high resolution)
		try{
			double[][] synth = synthFromImage(photographFolder, i_exposure_time, darkroom);
			
			// for certification (NO IN USE)
			/*
			for(int n=0;n<exposure_time.length;n++){
				String oFile = oFoldername+"\\"+oFilename+"_"+exposure_time[n]+".png";
				System.out.println("write: "+oFile);
				BufferedImage im = Transform.Radiance2Image(synth, IMAGE_W, IMAGE_H, exposure_time[n], true);
				ImageIO.write(im, "png", new File(oFile) );
			}
			//*/
			
			// base image
			String oFile = oFoldername+"\\base.png";
			System.out.println("write: "+oFile);
			BufferedImage im = Transform.Radiance2Image(synth, IMAGE_W, IMAGE_H, o_exposure_time, true);
			ImageIO.write(im, "png", new File(oFile) );
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	
	////////////////////////////////////////////////////////////////////////////////
	// subroutine : synthesize text file
	////////////////////////////////////////////////////////////////////////////////
	double[][] synthFromText(String inFoldername){
		double[][] synth = new double[4][PAINT_W*PAINT_H];
		
		for(int n=0;n<L;n++){
			
			// default configuration of i-th light
			int b = defaultConfig[3*n+0];
			int x = defaultConfig[3*n+1];
			int y = defaultConfig[3*n+2];
			
			// read i-th light effect from text file
			String light_i_str = inFoldername+"\\"+n+"_"+x+"_"+y+".txt";
			double[] light_i = Transform.readArrayFromFile(light_i_str, PAINT_W, PAINT_H);
			
			// synthesize
			for(int j=0 ;j<PAINT_H;j++)
			for(int i=0 ;i<PAINT_W ;i++){
				synth[0][i+j*PAINT_W] += light_i[i+j*PAINT_W]/255.0*brightness[ b ];
			}
		}
		
		return synth;
	}
	
	
	////////////////////////////////////////////////////////////////////////////////
	// subroutine : synthesize images
	////////////////////////////////////////////////////////////////////////////////
	double[][] synthFromImage(String inFoldername, int in_exposure, boolean darkroom){
		double[][] synth = new double[4][IMAGE_W*IMAGE_H];
		
		for(int n=0;n<L;n++){
			
			// default configuration of i-th light
			int b = defaultConfig[3*n+0];
			int x = defaultConfig[3*n+1];
			int y = defaultConfig[3*n+2];
			
			// read i-th light effect from image
			String light_0_str = inFoldername+"\\"+in_exposure+"\\"+n+"_0.jpg";
			String light_1_str = inFoldername+"\\"+in_exposure+"\\"+n+"_1_"+x+"_"+y+".jpg";
			double[][] light_0 = Transform.Image2Radiance(light_0_str, IMAGE_W, IMAGE_H, in_exposure, true);
			double[][] light_1 = Transform.Image2Radiance(light_1_str, IMAGE_W, IMAGE_H, in_exposure, true);
			
			// synthesize
			for(int c=0;c<4;c++)
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W ;i++){
				synth[c][i+j*IMAGE_W] += (darkroom)
									? light_1[c][i+j*IMAGE_W]/255.0*brightness[ b ]
									:(light_1[c][i+j*IMAGE_W]-light_0[c][i+j*IMAGE_W])/255.0*brightness[ b ];
			}
		}
		
		return synth;
	}
}
