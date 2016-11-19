package model;

import java.awt.Graphics;
import java.awt.image.*;
import java.io.*;
import java.util.Vector;

import javax.imageio.*;

@SuppressWarnings("unused")
////////////////////////////////////////////////////////////////////////////////
// Lighty main model
//
// Data   : 10/Sep/2012
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class Lighty {
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	
	// characteristic of camera
	public static int   IMAGE_W;
	public static int   IMAGE_H;
	public static int[] exposure_time;
	
	// for optimizer
	public static int   COARSE_W;
	public static int   COARSE_H;
	
	
	////////////////////////////////////////////////////////////
	// Component
	////////////////////////////////////////////////////////////
//	Capture    capture;
	SerialCom  sc;
	int[]      current_conf;
	
	static CaptureSim captureSim;
	static Optimizer  optimizer;
	
	private int  [][] YRGB_target;
	private float[][] paint_target;
	private float  [] solver_target;
	private float[][] paint_base;
	private float  [] solver_base;
	
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	public Lighty(int W, int H){
		
		// configuration
		Config.init(12, "light_config_params.txt");
		current_conf = Config.downwardsConfig(0);
		IMAGE_W = Config.IMAGE_W; IMAGE_H = Config.IMAGE_H;
		COARSE_W = W; COARSE_H = H;
		exposure_time = Config.exposure_time;
		
		// allocate arrays
		YRGB_target   = new int  [4][IMAGE_W*IMAGE_H];
		paint_target  = new float[4][IMAGE_W*IMAGE_H];
		solver_target = new float   [COARSE_W*COARSE_H];
		paint_base    = new float[4][IMAGE_W*IMAGE_H];
		solver_base   = new float   [COARSE_W*COARSE_H];
		
		// g curve for radiance
		Transform.init("g_YRGB.txt");
		
		// hardware controller & simulator
		sc = new SerialCom(Config.comPort);
		System.out.println( sc.openSerialPort() );
		System.out.println( sc.settingSerialPort() );
		
		// capture
//		capture    = Capture.create(720);
		
		// optimizer
		captureSim = new CaptureSim("calibration\\sample", 320, 240);
		optimizer  = new Optimizer ("radianceMap"+COARSE_W+"x"+COARSE_H);
		
		// set default exposure time
		int id = 1;
		System.out.println("set exposure time = "+exposure_time[id]);
		
		// base painting
		setBasePaint ("base.png", exposure_time[id] );
		setBaseSolver("base.txt");
		setExposureTime(id);
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : hardware control
	////////////////////////////////////////////////////////////
	public void setExposureTime(int exposure_time_id){
		Transform.createInverseGtable(exposure_time[exposure_time_id]);
		
		// control hardware
		sc.write(exposure_time_id);
		captureSim.simulate(current_conf);
	}
	public void setLight(int id, int b, int x, int y){
		current_conf[3*id+0] = b;
		current_conf[3*id+1] = x;
		current_conf[3*id+2] = y;
		
		// control hardware
		sc.write(current_conf);
	}
	
	public void setLights(int b){
		for(int id=0;id<Config.L_NUM;id++){
			current_conf[3*id+0] = b;
			current_conf[3*id+1] = (Config.X_NUM-1)/2;
			current_conf[3*id+2] = (Config.Y_NUM-1)/2;
		}
		
		// control hardware
		sc.write(current_conf);
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : get target image (PIXEL VALUE)
	////////////////////////////////////////////////////////////
	public int[][] getYRGB(float[] beta_fine, boolean treat_saturation){
		
		// radiance value
		for(int j=0;j<IMAGE_H;j++)
		for(int i=0;i<IMAGE_W ;i++){
			float beta = beta_fine[i+j*IMAGE_W];
			
			paint_target[0][i+j*IMAGE_W] = beta * paint_base[0][i+j*IMAGE_W];
			paint_target[1][i+j*IMAGE_W] = beta * paint_base[1][i+j*IMAGE_W];
			paint_target[2][i+j*IMAGE_W] = beta * paint_base[2][i+j*IMAGE_W];
			paint_target[3][i+j*IMAGE_W] = (beta <= 1.0f || !treat_saturation )
					? (       beta) * paint_base[3][i+j*IMAGE_W]
					: (2.0f - beta) * paint_base[3][i+j*IMAGE_W] ; // show saturation by blue channel (OBSOLETE)
		}
		
		// pixel value
		for(int j=0;j<IMAGE_H;j++)
		for(int i=0;i<IMAGE_W ;i++){
			YRGB_target[0][i+j*IMAGE_W] = Transform.inverse_g(paint_target[0][i+j*IMAGE_W], 0);
			YRGB_target[1][i+j*IMAGE_W] = Transform.inverse_g(paint_target[1][i+j*IMAGE_W], 1);
			YRGB_target[2][i+j*IMAGE_W] = Transform.inverse_g(paint_target[2][i+j*IMAGE_W], 2);
			YRGB_target[3][i+j*IMAGE_W] = Transform.inverse_g(paint_target[3][i+j*IMAGE_W], 3);
		}
		
		return YRGB_target;
	}
	public int[]   getSolverTarget(float[] beta_coarse){
		int[] target = new int[COARSE_W*COARSE_H];
		
		for(int j=0;j<COARSE_H;j++)
		for(int i=0;i<COARSE_W;i++){
			float value = beta_coarse[i+j*COARSE_W] * solver_base[i+j*COARSE_W];
			
			target[i+j*COARSE_W] = Transform.inverse_g(value, 0);
		}
		
		return target;
	}
	
	////////////////////////////////////////////////////////////
	// Method : launch optimizer
	////////////////////////////////////////////////////////////
	public void localSearch(float[] beta, float alpha){
		
		// obtain target radiance
		for(int j=0;j<COARSE_H;j++)
		for(int i=0;i<COARSE_W;i++){
			solver_target[i+j*COARSE_W] = beta[i+j*COARSE_W] * solver_base[i+j*COARSE_W];
		}
		
//		long t0 = System.nanoTime();
		
		// search
//		int[] conf1 = launchOptimizer(Config.downwardsConfig(0), solver_target, 0.0f);
		int[] conf1 = launchOptimizer(current_conf, solver_target, 0.0f);
//		int[] conf2 = launchOptimizer(conf1, solver_target, alpha); // consider energy consumption (OBSOLETE)
		
//		long t1 = System.nanoTime();
//		System.out.println("searching time="+(t1-t0)/1000000+"[ms]");
		
		// assign
		current_conf = conf1;
		sc.write(current_conf);
	}
	protected int[] launchOptimizer(int[] init_seed, float[] target, float alpha){
		// Single-Start Hill Climbing
		//*
		return optimizer.SSHC(init_seed, target, alpha, true).getConf();
		
		// Multi-Start Hill Climbing (NO IN USE)
		/*/
		Vector<int[]> init_seeds = new Vector<int[]>();
		init_seeds.add(Config.downwardsConfig(  0             )    ); // (  0,0,0) for all
		init_seeds.add(Config.downwardsConfig( (Config.B_NUM-1)/2) ); // (128,0,0) for all
		init_seeds.add(Config.downwardsConfig(  Config.B_NUM-1)    ); // (255,0,0) for all
		init_seeds.add(Config.copyConfig(current_conf));
		for(int i=init_seeds.size();i<Optimizer.numOfSeed;i++){
			init_seeds.add(Config.getRandomConf());
		}
		return optimizer.MSHC(init_seeds, target, true).getConf();
		//*/
	}
	
	////////////////////////////////////////////////////////////
	// Method : get top-viewed image
	////////////////////////////////////////////////////////////
	public enum LIVE_MODE{CAMERA, SIMULATION_COARSE, SIMULATION_FINE};
	private     LIVE_MODE live_mode = LIVE_MODE.SIMULATION_FINE;
	public void setLiveMode(LIVE_MODE live_mode){ this.live_mode = live_mode; }
	
	public BufferedImage getCapture(){
		BufferedImage img = new BufferedImage(IMAGE_W, IMAGE_H, BufferedImage.TYPE_INT_RGB);;
		Graphics g = img.getGraphics();
		
		// live view by camera : cripping
		if(live_mode==LIVE_MODE.CAMERA){
//			g.drawImage(capture.getImage(), 0, 0, IMAGE_W, IMAGE_H, 8, 0, 712, 480, null);
		}
		
		// simulation by downscaled radiance maps (FAST)
		else if(live_mode==LIVE_MODE.SIMULATION_COARSE){
			g.drawImage(captureSim.getSolverImage(), 0, 0, IMAGE_W, IMAGE_H, null);
		}
		
		// use Capture Simulator
		else if(live_mode==LIVE_MODE.SIMULATION_FINE){
			g.drawImage(captureSim.getImage(), 0, 0, IMAGE_W, IMAGE_H, null);
		}
		
		return img;
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : module for experiment (NO IN USE)
	////////////////////////////////////////////////////////////
	public void startLog(){}
	public void endLog(){}
	
	
	////////////////////////////////////////////////////////////
	// subroutine : set base image
	////////////////////////////////////////////////////////////
	private void setBasePaint(String filename, int exposure_time){
		System.out.println("read: "+filename);
		
		// calculate log exposure time
		double ln_dt = Math.log(1.0/exposure_time);
		
		// read image file & get pixels
		PixelGrabber  pg = null;
		try{
			BufferedImage im = ImageIO.read(new File(filename));
			pg = new PixelGrabber(im, 0, 0, IMAGE_W, IMAGE_H, true);
			pg.grabPixels();
		}catch(Exception e){
			e.printStackTrace();
		}
		int pixels[] = (int[])pg.getPixels();
		
		// obtain radiance value
		for(int j=0;j<IMAGE_H;j++)
		for(int i=0;i<IMAGE_W ;i++){
			int sRGB = pixels[i+j*IMAGE_W];
			
			int R = ( sRGB >> 16 ) & 255;
			int G = ( sRGB >>  8 ) & 255;
			int B = ( sRGB       ) & 255;
			int Y = Transform.getLuminance(sRGB);
			
			// obtain radiance value
			paint_base[0][i+j*IMAGE_W] = (float)Transform.g_value(Y, 0, ln_dt);
			paint_base[1][i+j*IMAGE_W] = (float)Transform.g_value(R, 1, ln_dt);
			paint_base[2][i+j*IMAGE_W] = (float)Transform.g_value(G, 2, ln_dt);
			paint_base[3][i+j*IMAGE_W] = (float)Transform.g_value(B, 3, ln_dt);
		}
	}
	public  void setBaseSolver(String filename){
		System.out.println("read: "+filename);
		
		try{
			BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
			
			for(int j=0;j<COARSE_H;j++)
			for(int i=0;i<COARSE_W;i++){
				solver_base[i+j*COARSE_W] = Float.valueOf(br.readLine());
			}
			
			br.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
