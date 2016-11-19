package calibration;

import java.awt.image.*;
import java.io.*;
import java.util.*;

import javax.imageio.ImageIO;

import model.*;

public class GetGcurve {
	
	////////////////////////////////////////////////////////////
	// Constants, Components
	////////////////////////////////////////////////////////////
	static int[] exposure_time;
	static int   IMAGE_W, IMAGE_H;
	ArrayList<Integer> quasiRandom;
	
	// related to SVD
	final double lambda      = 8.0; // smoothing factor
	final int    numOfPixels = 128; // sample pixels
	final int    numOfPhotos = 6;   // sample photos
	
	
	////////////////////////////////////////////////////////////
	// launcher
	////////////////////////////////////////////////////////////
	public static void main(String[] args){
		
		exposure_time = Config.exposure_time;
		IMAGE_W       = Config.IMAGE_W;
		IMAGE_H       = Config.IMAGE_H;
		
		// default arguments
		//*
		String iFoldername = "calibration\\calc_g";
		String oFilename   = "g_YRGB.txt";
		/*/
		// input arguments
		String iFoldername = args[0];
		String oFilename   = args[1];
		//*/
		
		// print out for certification
		System.out.println("input  folder name : "+iFoldername);
		System.out.println("output file   name : "+oFilename);
		
		// obtain response curve G
		new GetGcurve(iFoldername, oFilename);
	}
	
	
	////////////////////////////////////////////////////////////
	// Constructor : obtain g curve by SVD
	////////////////////////////////////////////////////////////
	public GetGcurve(String inputFolderName, String outputFileName){
		
		// 1. make quasi-random list for sampling
		quasiRandom = makeQuasiRandomList(IMAGE_W*IMAGE_H, 5);
		
		// 2. sampling : using Luma 0 / Red 1 / Green 2 / Blue 3
		int[][][] samples = new int[4][numOfPhotos][numOfPixels];
		
		int start_id = 0;
		for(int j=0;j<numOfPhotos;j++){
			
			// color IMAGE -> int[] YRGB
			int[][] value = getValue(inputFolderName, exposure_time[j+start_id]);
			
			// sampling pixels
			for(int c=0;c<4;c++)
			for(int i=0;i<numOfPixels;i++){
				samples[c][j][i] = value[c][quasiRandom.get(i)];
			}
		}
		
		// 3. calculate g curve by SVD
		double[][] g_func = new double[4][256];
		for(int c=0;c<4;c++){
			System.out.println("channel = "+c);
			g_func[c] = calc_g_func(samples[c], numOfPhotos, numOfPixels);
		}
		
		// 4. export g curve to text file
		export_g(outputFileName, g_func);
		
		// 5. make pixel log
//		makeSamplePixelLog(inputFolderName, samples[0], numOfPhotos, numOfPixels, g_func[0]);
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine 1 : make quasi random list by shuffling
	////////////////////////////////////////////////////////////
	protected ArrayList<Integer> makeQuasiRandomList(int size, int shuffle){
		
		ArrayList<Integer> randomList = new ArrayList<Integer>();
		
		for(int i=0;i<size;i++)
			randomList.add(i);
		
		for(int i=0;i<shuffle;i++)
			Collections.shuffle(randomList);
		
		return randomList;
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine 2 : obtain [YRGB] from image file
	////////////////////////////////////////////////////////////
	int[][] getValue(String iFoldername, int shutterSpeed){
		
		// file name
		String iFilename = iFoldername + "\\"      + Integer.toString(shutterSpeed)+ ".jpg";
//		String oFilename = iFoldername + "\\gray_" + Integer.toString(shutterSpeed)+ ".jpg";
		
		// read image file & get pixels
		PixelGrabber pg = null;
		try{
			BufferedImage inputImage = ImageIO.read(new File(iFilename));
			pg = new PixelGrabber(inputImage , 0 , 0 , -1 , -1 , true);
			pg.grabPixels();
		}catch(Exception e){
			e.printStackTrace();
		}
		int pixels[] = (int[])pg.getPixels();
		
		// RGB -> YRGB
		int[][] value = new int[4][IMAGE_W*IMAGE_H];
		BufferedImage outputImage = new BufferedImage(IMAGE_W, IMAGE_H, BufferedImage.TYPE_INT_RGB);
		
		for(int j=0;j<IMAGE_H;j++)
		for(int i=0;i<IMAGE_W;i++){
			int sRGB = pixels[i+j*IMAGE_W];

			int R = ( sRGB >> 16 ) & 255 ;
			int G = ( sRGB >>  8 ) & 255 ;
			int B = ( sRGB       ) & 255 ;
			
			int Y = Transform.getLuminance(sRGB);
			int rgb = ( Y << 16 ) | ( Y << 8 ) | Y ;
			
			value[0][i+j*IMAGE_W] = Y;
			value[1][i+j*IMAGE_W] = R;
			value[2][i+j*IMAGE_W] = G;
			value[3][i+j*IMAGE_W] = B;
			
			outputImage.setRGB(i, j, rgb);
		}
		
		// export grayscale image (DEPRECATED)
		/*
		try{
			ImageIO.write(outputImage, "jpg", new File(oFilename));
		}catch(Exception e){
			e.printStackTrace();
		}
		//*/
		
		return value;
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine 3 : get g curve by SVD [Debevec and Malik 97]
	////////////////////////////////////////////////////////////
	double[] calc_g_func(int[][] samples, int numOfPhotos, int numOfPixels){
		
		// matrix [A] and vector {b}
		int n = 256;
		double[][] A = new double[numOfPixels*numOfPhotos + n+1][n + numOfPixels];
		double[]   b = new double[numOfPixels*numOfPhotos + n+1];
		
		// make matrix [A]
		int k = 0;
		for(int i=0; i<numOfPixels; i++)
			for(int j=0; j<numOfPhotos; j++){
				double wij = w(samples[j][i]+1);
				A[k][samples[j][i]+1]=wij;
				A[k][n+i] = -wij;
				b[k] = wij*Math.log(1.0/(double)exposure_time[j]);
				k++;
			}
		
		A[k][129] = 1;
		k++;
		
		for(int i=0; i< n-2; i++){
			A[k][i] = lambda * w(i+1);
			A[k][i+1] = -2*lambda*w(i+1);
			A[k][i+2] = lambda*w(i+1);
			k++;
		}
		
		double[] x = new double[n + numOfPixels];
		
		System.out.print("solve [A]{x} = {b} by SVD: ");
		try{
			VisualNumerics.math.DoubleSVD doubleSVD = new VisualNumerics.math.DoubleSVD(A);
			x = VisualNumerics.math.DoubleMatrix.multiply(doubleSVD.inverse(), b);
			System.out.println("Solved");
		}catch(Exception e){
			System.out.println("Solve failed! "+e);
		}
		
		double[] g = new double[256];
		for(int i=0; i<256; i++){
			g[i] = x[i];	
		}
		return g;
	}
	private double w(int z){
		if (z<128)
			return z;
		else
			return 255-z;
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine 4 : 
	////////////////////////////////////////////////////////////
	void export_g(String outputFileName, double[][] g_func){
		System.out.println("write: "+outputFileName);
		
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(outputFileName));
			
			// write
			for(int c=0; c<4  ; c++)
			for(int i=0; i<256; i++){
				bw.write(""+g_func[c][i]);
				bw.newLine();
			}
			
			bw.flush();
			bw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine 5 : make pixel log (DEPRECATED)
	////////////////////////////////////////////////////////////
	void makeSamplePixelLog(String inputFolderName, int[][] samples, int numOfPhotos, int numOfPixels, double[] g_func){
		String outputFileName = inputFolderName + "\\pixelLog.txt";
		
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(outputFileName));
			
			for(int i=0;i<numOfPixels;i++){
				int pI = quasiRandom.get(i)%IMAGE_W;
				int pJ = quasiRandom.get(i)/IMAGE_W;
				
				bw.write("pixel("+pI+","+pJ+")"); bw.newLine();
				
				for(int j=0;j<numOfPhotos;j++){
					
					int Y = samples[j][i];
					double dt     = 1.0/(double)exposure_time[j];
					double unitDt = 1.0/(double)exposure_time[numOfPhotos/2];
					
					double y = Math.log(dt) - Math.log(unitDt) ;
					
					String str = Integer.toString(Y)+ "\t" + Double.toString(y) + "\t" + g_func[Y];
					
					bw.write(str); bw.newLine();
				}
				
				bw.newLine();
			}
			
			bw.flush();
			bw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
