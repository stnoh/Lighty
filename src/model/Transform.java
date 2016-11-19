package model;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;
import java.io.*;

import javax.imageio.ImageIO;

////////////////////////////////////////////////////////////////////////////////
// Static functions for transformation among the values
//
// Data   : 10/Sep/2012
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class Transform {
	
	////////////////////////////////////////////////////////////
	// Container : [channel:YRGB][resolution]
	////////////////////////////////////////////////////////////
	static double g_func[][] = new double[4][256];
	
	static int    SIZE;
	static float  STEP;
	static int    inverse_g_table[][];
	
	////////////////////////////////////////////////////////////
	// Initializer
	////////////////////////////////////////////////////////////
	public static void init(String filename){
		
		// read g curve from text file
		try{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			
			for(int c=0;c<4  ;c++)
			for(int i=0;i<256;i++){
				g_func[c][i] = Double.valueOf(br.readLine());
			}
			
			br.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public static void createInverseGtable(int exposure_time){
		double ln_dt = Math.log(1.0/exposure_time);
		
		// create inverse g (hash) table with minimum size.
		SIZE = 256;
		for(int size=0;size<6;size++){
			SIZE*=2; STEP=0.8f;
			for(int step=0;step<5;step++){
				STEP*=0.5f;
				
				if( checkTable(ln_dt) ){
					System.out.println("Inverse g table - done.");
					return ;
				}
			}
		}
		
		System.out.println("Error - could not create inverse table!!!");
	}
	private static boolean checkTable(double ln_dt){
//		System.out.println("trying : size = "+SIZE+"\tstep = "+STEP); // for DEBUG
		inverse_g_table = new int[4][SIZE];
		
		// create the table
		for(int c=0;c<4   ;c++)
		for(int i=0;i<SIZE;i++){
			float radiance = STEP*i;
			int Z = inverse_g_binary(Math.log(radiance)+ln_dt, c);
			
			inverse_g_table[c][i]=Z;
		}
		
		// check monotonicity
		for(int c=0;c<4   ;c++){
			boolean check[] = new boolean[256];
			
			for(int i=0;i<SIZE;i++){
				int Z = inverse_g_table[c][i];
				check[Z] = true;
			}
			
			for(int i=0;i<256;i++){
				if(check[i]==false) return false;
			}
		}

		return true;
	}
	
	////////////////////////////////////////////////////////////
	// Method : Radiance <-> Pixel
	////////////////////////////////////////////////////////////
	public static double g_value(int Z, int ch, double ln_dt){
		if(Z==0) return 0.0;
		return Math.exp( g_func[ch][Z] - ln_dt );
	}
	public static int inverse_g_binary(double lnE, int ch){
		int Z = 0;
		int dz = 128;
		
		for(int i=0;i<8;i++){
			if(lnE >= g_func[ch][Z+dz])
				Z += dz;
			dz /= 2;
		}
		
		return Z;
	}
	public static int inverse_g(float exposure, int ch){
		float hash = (float)(exposure)/STEP;
		
		// out of range (0~255)
		if     (hash>=SIZE) return 255;
		else if(hash<=0.0f) return 0;
		
		// range (0~255)
		return inverse_g_table[ch][(int)hash];
	}
	
	////////////////////////////////////////////////////////////
	// Method : Color, beta weight, etc.
	////////////////////////////////////////////////////////////
	public static int getLuminance(int sRGB){
		int R = ( sRGB >> 16 ) & 255;
		int G = ( sRGB >>  8 ) & 255;
		int B = ( sRGB       ) & 255;
		
		return (int)(0.29891*R + 0.58661*G + 0.11448*B);
	}
	
	// user painting (sRGB) <-> beta (float)
	public static float int2Beta(int sRGB){
		int R = ( sRGB >> 16 ) & 255;
		int B = ( sRGB       ) & 255;
		
		return (2*R-B)/255.f;
	}
	public static int   beta2Int(float beta){
		int R = (beta <= 1.0f) ? (int)(255.0f*beta) : 255;
		int G = (beta <= 1.0f) ? (int)(255.0f*beta) : 255;
		int B = (beta <= 1.0f) ? (int)(255.0f*beta) : (int)(255.0f*(2.0f-beta));
		
		return ( 255 << 24 ) | ( R << 16 ) | ( G << 8 ) | B ;
	}
	
	// colorbar visualization
	public static int beta2HSB(float beta){
		final float min = 0.0f;
		final float max = 1.0f;
		
		float hue;
		if( beta < min )     { hue = 240.f/360.f; }
		else if( beta > max ){ hue = 0.0f; }
		else{
			hue = ( beta > max ) ? 0.0f : ((max-beta)*240f)/((max-min)*360f);
		}
		
		return Color.HSBtoRGB(hue, 1.0f, 1.0f);
	}
	
	////////////////////////////////////////////////////////////
	// Method : up & downscaling [2x2], gaussian [5x5]
	////////////////////////////////////////////////////////////
	private static final int bx = 2;
	private static final int by = 2;
	
	public static float[] upscale2D(float[] pre, int W, int H){
//		System.out.println("upscale  : "+W+"x"+H);
		float[] post = new float[(W*bx)*(H*by)];
		
		for(int j=0;j<H*by;j++)
		for(int i=0;i<W*bx;i++){
			int di = i/bx;
			int dj = j/by;
			
			post[i+j*(W*bx)] = pre[di+dj*W];
		}
		
		return post;
	}
	public static float[] downscale2D(float[] pre, int W, int H){
//		System.out.println("downscale: "+W+"x"+H);
		float[] post = new float[(W/bx)*(H/by)];
		
		final float ratio = (1.0f/bx)*(1.0f/by);
		
		for(int j=0;j<H;j++)
		for(int i=0;i<W;i++){
			int di = i/bx;
			int dj = j/by;
			
			post[di+dj*(W/bx)] += ratio*pre[i+j*W];
		}
		
		return post;
	}
	public static float[] gaussian(float[] area, int W, int H, int iter){
//		System.out.println("gaussian: "+W+"x"+H+" iter = "+iter);
		
		// 1/16 * [ 0.9  3.9  6.4  3.9  0.9 ]
		final float[] weight = {0.9f/16.f, 3.9f/16.f, 6.4f/16.f, 3.9f/16.f, 0.9f/16.f};
		
		for(int n=0;n<iter;n++){
			float[] new_area_x = new float[W*H];
			float[] new_area_y = new float[W*H];
			
			// x-direction gaussian
			for(int j=0;j<H;j++)
			for(int i=0;i<W;i++){
				
				// align in i-coordinate
				int im1 = (i  ==  0) ? i   : i  -1;
				int im2 = (im1==  0) ? im1 : im1-1;
				int ip1 = (i  ==W-1) ? i   : i  +1;
				int ip2 = (ip1==W-1) ? ip1 : ip1+1;
				
				// obtain
				float m2 = area[im2+j*W];
				float m1 = area[im1+j*W];
				float c  = area[i  +j*W];
				float p1 = area[ip1+j*W];
				float p2 = area[ip2+j*W];
				
				// calculate
				new_area_x[i+j*W] = m2*weight[0]+m1*weight[1]+c*weight[2]+p1*weight[3]+p2*weight[4];
			}
			
			// y-direction gaussian
			for(int j=0;j<H;j++)
			for(int i=0;i<W;i++){
				
				// j-coordinate
				int jm1 = (j  ==  0) ? j   : j  -1;
				int jm2 = (jm1==  0) ? jm1 : jm1-1;
				int jp1 = (j  ==H-1) ? j   : j  +1;
				int jp2 = (jp1==H-1) ? jp1 : jp1+1;
				
				// obtain
				float m2 = new_area_x[i+(jm2)*W];
				float m1 = new_area_x[i+(jm1)*W];
				float c  = new_area_x[i+(j  )*W];
				float p1 = new_area_x[i+(jp1)*W];
				float p2 = new_area_x[i+(jp2)*W];
				
				// calculate
				new_area_y[i+j*W] = m2*weight[0]+m1*weight[1]+c*weight[2]+p1*weight[3]+p2*weight[4];
			}
			
			area = new_area_y;
		}
		
		return area;
	}
	
	////////////////////////////////////////////////////////////////////////////////
	// Method : convert image to radiance map
	////////////////////////////////////////////////////////////////////////////////
	public static double[][] Image2Radiance(String filename, int W, int H,
			int exposure_time, boolean color){
//		System.out.println("read: "+filename);
		double ln_dt = Math.log(1.0/(double)exposure_time);
		
		double[][] radianceMap = new double[4][W*H];
		
		// read image file & get pixels
		PixelGrabber pg = null;
		try{
			BufferedImage inputImage = ImageIO.read(new File(filename));
			pg = new PixelGrabber(inputImage , 0 , 0 , -1 , -1 , true);
			pg.grabPixels();
		}catch(Exception e){
			e.printStackTrace();
		}
		int pixels[] = (int[])pg.getPixels();
		
		// get YRGB
		for(int j=0;j<H;j++)
		for(int i=0;i<W ;i++){
			int sRGB = pixels[i+j*W];
			
			// RGB -> Y
			int Y = Transform.getLuminance(sRGB);
			
			// pixel value -> radiance value
			radianceMap[0][i+j*W] = Transform.g_value(Y, 0, ln_dt);
			
			// color
			if(color){
				int R = ( sRGB >> 16 ) & 255;
				int G = ( sRGB >>  8 ) & 255;
				int B = ( sRGB       ) & 255;
				radianceMap[1][i+j*W] = Transform.g_value(R, 1, ln_dt);
				radianceMap[2][i+j*W] = Transform.g_value(G, 2, ln_dt);
				radianceMap[3][i+j*W] = Transform.g_value(B, 3, ln_dt);
			}
		}
		
		// return radiance
		return radianceMap;
	}
	
	////////////////////////////////////////////////////////////////////////////////
	// Method : invert radiance map to image
	////////////////////////////////////////////////////////////////////////////////
	public static BufferedImage Radiance2Image(double[][] radianceMap, int W, int H,
			int exposure_time, boolean color){
		double ln_dt = Math.log(1.0/exposure_time);
		
		BufferedImage image = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
		
		// radiance -> image
		for(int j=0;j<H;j++)
		for(int i=0;i<W ;i++){
				
			// radiance[YRGB] -> sYYY
			if(!color){
				int Y = Transform.inverse_g_binary( Math.log( radianceMap[0][i+j*W] )+ln_dt, 0);
				int sYYY = (255 << 24) | ( Y << 16 ) | ( Y << 8 ) | Y ;
				
				image.setRGB(i, j, sYYY);
			}
			
			// radiance[YRGB] -> sRGB
			else{
				int R = Transform.inverse_g_binary( Math.log( radianceMap[1][i+j*W] )+ln_dt, 1);
				int G = Transform.inverse_g_binary( Math.log( radianceMap[2][i+j*W] )+ln_dt, 2);
				int B = Transform.inverse_g_binary( Math.log( radianceMap[3][i+j*W] )+ln_dt, 3);
				int sRGB = (255 << 24) | ( R << 16 ) | ( G << 8 ) | B ;
				
				image.setRGB(i, j, sRGB);
			}
		}
		
		return image;
	}
	
	////////////////////////////////////////////////////////////////////////////////
	// Method : read array from text file
	////////////////////////////////////////////////////////////////////////////////
	public static double[] readArrayFromFile(String filename, int W, int H){
//		System.out.println("read: "+filename);
		
		double[] double_array = new double[W*H];
		
		try{
			BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
			
			for(int j=0;j<H;j++)
			for(int i=0;i<W ;i++){
				double_array[i+j*W] = Double.valueOf(br.readLine());
			}
			
			br.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		
		return double_array;
	}
	
	////////////////////////////////////////////////////////////////////////////////
	// Method : export array to text file
	////////////////////////////////////////////////////////////////////////////////
	public static void writeArrayToFile(String filename, double[] double_array, int W, int H){
		System.out.println("write: "+filename);
		
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
			
			for(int j=0;j<H;j++)
			for(int i=0;i<W ;i++){
				bw.write( Double.toString(double_array[i+j*W]) );
				bw.newLine();
			}
			
			bw.flush();
			bw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
