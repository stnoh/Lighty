package model;

import java.awt.*;
import java.awt.image.*;
import java.io.File;

import javax.imageio.ImageIO;

@SuppressWarnings("serial")
////////////////////////////////////////////////////////////////////////////////
// Capture image simulator
//
// Data   : 11/Sep/2012
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class CaptureSim extends Canvas{
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	int L,X,Y;
	int IMAGE_W,IMAGE_H;
	int COARSE_W, COARSE_H;
	
	
	////////////////////////////////////////////////////////////
	// Components
	////////////////////////////////////////////////////////////
	int pixels_solver[], pixels_image[];
	BufferedImage buffered_img;
	float images[][][][][];
	
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	CaptureSim(String foldername, int W, int H){
		super();
		L = Config.L_NUM; X = Config.X_NUM; Y = Config.Y_NUM;
		IMAGE_W  = W; IMAGE_H = H;
		COARSE_W = Lighty.COARSE_W;
		COARSE_H = Lighty.COARSE_H;
		
		// image instance
		pixels_solver = new int[COARSE_W*COARSE_H];
		pixels_image  = new int[ IMAGE_W* IMAGE_H];
		buffered_img  = new BufferedImage(IMAGE_W, IMAGE_H, BufferedImage.TYPE_INT_RGB);
		
		// image-based simulator database
		images = new float[L][X][Y][3][IMAGE_W*IMAGE_H];
		readImages(foldername); // use Capture Simulator
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : get image
	////////////////////////////////////////////////////////////
	BufferedImage getImage(){
		MemoryImageSource ip = new MemoryImageSource(IMAGE_W, IMAGE_H, pixels_image, 0, IMAGE_W);
		Image img = createImage(ip);
		
		Graphics g = buffered_img.getGraphics();
		g.drawImage(img, 0, 0, IMAGE_W, IMAGE_H, null);
		
		return buffered_img;
	}
	BufferedImage getSolverImage(){
		MemoryImageSource ip = new MemoryImageSource(COARSE_W, COARSE_H, pixels_solver, 0, COARSE_W);
		Image img = createImage(ip);
		
		Graphics g = buffered_img.getGraphics();
		g.drawImage(img, 0, 0, IMAGE_W, IMAGE_H, null);
		
		return buffered_img;
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : image-based simulation
	////////////////////////////////////////////////////////////
	void simulate(int[] conf){
		float  [] coarse = new float   [COARSE_W*COARSE_H];
		float[][] area   = new float[3][IMAGE_W *IMAGE_H ];
		
		// synthesize
		for(int n=0;n<L;n++){
			int b = conf[3*n+0];
			int x = conf[3*n+1];
			int y = conf[3*n+2];
			
			// solver resolution
			for(int j=0;j<COARSE_H;j++)
			for(int i=0;i<COARSE_W;i++){
				coarse[i+j*COARSE_W] += Optimizer.radianceMap[n][x][y][i+j*COARSE_W]/255.0*Config.brightness[b];
			}
			
			// resized image resolution
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W;i++){
				area[0][i+j*IMAGE_W] += images[n][x][y][0][i+j*IMAGE_W]/255.0*Config.brightness[b];
				area[1][i+j*IMAGE_W] += images[n][x][y][1][i+j*IMAGE_W]/255.0*Config.brightness[b];
				area[2][i+j*IMAGE_W] += images[n][x][y][2][i+j*IMAGE_W]/255.0*Config.brightness[b];
			}
		}
		
		// solver radiance map
		for(int j=0;j<COARSE_H;j++)
		for(int i=0;i<COARSE_W;i++){
			int Y = Transform.inverse_g(coarse[i+j*COARSE_W], 0);
			pixels_solver[i+j*COARSE_W] = ( 255 << 24 ) | ( Y << 16 ) | ( Y << 8 ) | Y ;
		}
		
		// color image
		for(int j=0;j<IMAGE_H;j++)
		for(int i=0;i<IMAGE_W;i++){
			int R = Transform.inverse_g(area[0][i+j*IMAGE_W], 1);
			int G = Transform.inverse_g(area[1][i+j*IMAGE_W], 2);
			int B = Transform.inverse_g(area[2][i+j*IMAGE_W], 3);
			pixels_image[i+j*IMAGE_W] = ( 255 << 24 ) | ( R << 16 ) | ( G << 8 ) | B ;
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine : read images from files
	////////////////////////////////////////////////////////////
	@SuppressWarnings("unused")
	private void readImages(String foldername){
		int exposure_time = 60;
		double ln_dt = Math.log(1.0/exposure_time);
		
		System.out.print("Load images -");
		
		for(int n=0;n<L;n++){
			System.out.print(" "+n);
			for(int x=0;x<X;x++)
			for(int y=0;y<Y;y++){
				
				// filename
				String filename = new String(foldername+"\\"+exposure_time+"\\"
						+String.valueOf(n)+"_1_"
						+String.valueOf(x)+"_"
						+String.valueOf(y)+".jpg");
				
				// read pixel value
				PixelGrabber  pg = null;
				
				try{
					BufferedImage im1 = ImageIO.read(new File(filename));
					BufferedImage im2 = new BufferedImage(IMAGE_W, IMAGE_H, BufferedImage.TYPE_INT_RGB);
					Graphics2D g = im2.createGraphics();
					g.drawImage(im1, 0, 0, IMAGE_W, IMAGE_H, null);
					
					pg = new PixelGrabber(im2, 0, 0, -1, -1, true);
					pg.grabPixels();
				}catch(Exception e){
					e.printStackTrace();
				}
				int pixels[] = (int[])pg.getPixels();
				
				// convert to radiance value
				for(int j=0;j<IMAGE_H;j++)
				for(int i=0;i<IMAGE_W;i++){
					int aRGB = pixels[i+j*IMAGE_W];
					
					int R = ( aRGB >> 16 ) & 255;
					int G = ( aRGB >>  8 ) & 255;
					int B = ( aRGB       ) & 255;
					
					images[n][x][y][0][i+j*IMAGE_W] = (float)Transform.g_value(R, 1, ln_dt);
					images[n][x][y][1][i+j*IMAGE_W] = (float)Transform.g_value(G, 2, ln_dt);
					images[n][x][y][2][i+j*IMAGE_W] = (float)Transform.g_value(B, 3, ln_dt);
				}
			}
		}
		System.out.println();
	}
}
