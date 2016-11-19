package ui;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;

import model.*;

@SuppressWarnings("serial")
////////////////////////////////////////////////////////////////////////////////
// Painting canvas
//
// Data   : 11/Sep/2012
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class PCanvas extends Canvas implements ActionListener, MouseListener, MouseMotionListener {
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	int IMAGE_W , IMAGE_H ;
	int COARSE_W, COARSE_H;
	final int FPS = 60;
	
	////////////////////////////////////////////////////////////
	// Components
	////////////////////////////////////////////////////////////
	UndoRedoImageList undoRedo;
	ImageProducer     ip;
	
	int  [] pixels;
	float[] grid_fine;
	float[] grid_coarse;
	
	BufferedImage captured_img;
	Image show_image;
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	PCanvas() {
		super();
		IMAGE_W  = Lighty.IMAGE_W;
		IMAGE_H  = Lighty.IMAGE_H;
		COARSE_W = Lighty.COARSE_W;
		COARSE_H = Lighty.COARSE_H;
		
		// thread for real-time updating captured image
		(new javax.swing.Timer(1000/FPS, this)).start();
		captured_img = new BufferedImage(IMAGE_W, IMAGE_H, BufferedImage.TYPE_INT_RGB);
		
		// [ PAINT_W x PAINT_H ]
		undoRedo  = new UndoRedoImageList(10+1); // 11 status = 10 undo/redo
		pixels    = new int  [IMAGE_W*IMAGE_H];
		grid_fine = new float[IMAGE_W*IMAGE_H];
		ip        = new MemoryImageSource(IMAGE_W, IMAGE_H, pixels, 0, IMAGE_W); // for double buffering
		
		// [ COARSE_W x COARSE_H ]
		grid_coarse = new float[COARSE_W*COARSE_H];
	}
	
	
	////////////////////////////////////////////////////////////
	// Joint with other objects
	////////////////////////////////////////////////////////////
	protected boolean isBrush (){ return InputStatus.isBrush (); }
	protected boolean isSquirt(){ return InputStatus.isSquirt(); }
	protected Color   getBrushColor (){ return InputStatus.getBrushColor();  }
	protected void    setBrushColor (Color c){ InputStatus.setBrushColor(c); }
	protected int     getBrushRadius(){ return InputStatus.getBrushRadius(); }
	
	protected float   getAlphaValue(){ return InputStatus.getAlphaValue(); }
	
	protected void    renewButtons(){ InputStatus.renewButtons(); }
	
	protected void    addLog(String log){ InputStatus.addLog(log); }
	
	
	////////////////////////////////////////////////////////////
	// Method : manipulation
	////////////////////////////////////////////////////////////
	public BufferedImage getBaseImage(){
		return undoRedo.base_image[undoRedo.getCurrentIndex()];
	}
	
	public boolean undoable(){ return undoRedo.undoable(); }
	public boolean redoable(){ return undoRedo.redoable(); }
	public void    undo()    { undoRedo.undo(); edited = true; repaint(); }
	public void    redo()    { undoRedo.redo(); edited = true; repaint(); }
	
	public void    clear(boolean stack){
		if(stack) undoRedo.stack();
		edited = true;
		
		Graphics g = getBaseImage().getGraphics();
		g.setColor( getBrushColor() );
		g.fillRect(0, 0, IMAGE_W, IMAGE_H);
		g.dispose();
	}
	
	public void load(Image img){
		undoRedo.stack();
		edited = true;
		
		Graphics g = getBaseImage().getGraphics();
		g.drawImage(img, 0, 0, IMAGE_W, IMAGE_H, this);
		g.dispose();
	}
	
	public BufferedImage getTeaser(){
		BufferedImage img = new BufferedImage(ToolPanel.ICON_W, ToolPanel.ICON_H, BufferedImage.TYPE_INT_RGB);
		
		Graphics2D g = img.createGraphics();
		g.drawImage(Main.lighty.getCapture(), 0, 0, ToolPanel.ICON_W, ToolPanel.ICON_H, null);
		
		return img;
	}
	
	////////////////////////////////////////////////////////////
	// Listener : Action
	////////////////////////////////////////////////////////////
	public void actionPerformed(ActionEvent arg0) {
		captured_img = Main.lighty.getCapture();
		repaint();
	}
	
	
	////////////////////////////////////////////////////////////
	// Listener : Mouse & MouseMotion
	////////////////////////////////////////////////////////////
	private boolean edited        = false;
	private boolean mouse_entered = false;
	private boolean mouse_button1 = false;
	private Point   prev_p        = new Point(-1,-1);
	
	public void mouseDragged(MouseEvent arg0) {
		Point mouse_p = arg0.getPoint();
		
		if( isBrush() && mouse_button1 ){
			edited = true;
			
			Color color = getBrushColor();
			int radius  = getBrushRadius();
			
			int RADIUS = ( convert_x(2*radius) < convert_y(2*radius) ) ? convert_x(2*radius) : convert_y(2*radius);

			Graphics2D g = getBaseImage().createGraphics();
			g.setStroke(new BasicStroke( RADIUS, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND ) );
			g.setColor( color );
			g.drawLine( convert_x(mouse_p.x), convert_y(mouse_p.y), convert_x(prev_p.x), convert_y(prev_p.y) );
			g.fillOval( convert_x(mouse_p.x-radius), convert_y(mouse_p.y-radius), convert_x(2*radius), convert_y(2*radius) );
			g.dispose();
		}
		
		prev_p = mouse_p;
	}
	public void mouseMoved(MouseEvent arg0) {
		Point mouse_p = arg0.getPoint();
		prev_p = mouse_p;
	}
	
	public void mouseEntered(MouseEvent arg0) {
		Point mouse_p = arg0.getPoint();
		setCursor(new Cursor(Cursor.CROSSHAIR_CURSOR));
		addLog("pen entered: ");
		
		mouse_entered = true;
		prev_p = mouse_p;
	}
	public void mouseExited(MouseEvent arg0) {
		Point mouse_p = arg0.getPoint();
		setCursor(new Cursor(Cursor.DEFAULT_CURSOR));
		
		addLog("pen exited: ");
		mouse_entered = false;
		prev_p = mouse_p;
	}
	
	public void mousePressed(MouseEvent arg0) {
		Point mouse_p = arg0.getPoint();
		
		if( isBrush() && arg0.getButton()==MouseEvent.BUTTON1){
			undoRedo.stack(); edited = true; // for undo/redo
			mouse_button1 = true;
			
			Color color = getBrushColor();
			int radius  = getBrushRadius();
			
			Graphics2D g = getBaseImage().createGraphics();
			g.setColor( color );
			g.fillOval( convert_x(mouse_p.x-radius), convert_y(mouse_p.y-radius), convert_x(2*radius), convert_y(2*radius) );
			g.dispose();
			
			addLog("drawing started: ");
		}
		
		prev_p = mouse_p;
	}
	public void mouseReleased(MouseEvent arg0) {
		Point mouse_p = arg0.getPoint();
		
		if( isSquirt() ){
			int i = convert_x(mouse_p.x);
			int j = convert_y(mouse_p.y);
			int c = getBaseImage().getRGB(i, j);
			setBrushColor(new Color(c));
			
			addLog("squirt: brightness = "+Transform.int2Beta(c) );
		}
		else addLog("drawing ended  : ");
		
		renewButtons();
		
		mouse_button1 = false;		
		prev_p = mouse_p;
	}
	public void mouseClicked(MouseEvent arg0) {} // no in use
	
	private int convert_x(int x){ return (x*IMAGE_W)/getWidth (); }
	private int convert_y(int y){ return (y*IMAGE_H)/getHeight(); }
	
	
	////////////////////////////////////////////////////////////
	// Method : drawing
	////////////////////////////////////////////////////////////
	public enum DRAW_MODE{CAPTURE, TARGET_COARSE, TARGET_FINE, BETA_COARSE, BETA_FINE, COLORBAR_COARSE, COLORBAR_FINE};
	private     DRAW_MODE draw_mode = DRAW_MODE.CAPTURE;
	protected void setDrawMode(DRAW_MODE draw_mode){ this.draw_mode = draw_mode; }
	
	public void update(Graphics g){
		paint(g);
		g.dispose();
		
		if( mouse_entered ){
			edited = true;
			Main.lighty.localSearch(grid_coarse, getAlphaValue());
			return ;
		}
		if( edited ){
			edited = false;
			Main.lighty.localSearch(grid_coarse, getAlphaValue());
		}
	}
	public void paint(Graphics g){
		if( show_image == null ){
			show_image  = createImage(IMAGE_W, IMAGE_H);
		}
		doubleBuffering();
		
		g.drawImage(show_image, 0, 0, getWidth(), getHeight(), null);
		g.dispose();
	}
	
	private boolean contour = true;
	public  boolean getContourMode(){ return contour; }
	public  void    setContourMode(boolean contour){
		addLog("set contour: "+contour);
		this.contour = contour;
		repaint();
	}
	
	
	////////////////////////////////////////////////////////////
	// subroutine : drawing
	////////////////////////////////////////////////////////////
	private void doubleBuffering(){
		
		// diffuse beta area
		float[] non_smoothed = getRawBetaFromPaint();
		grid_fine = MG(non_smoothed, IMAGE_W, IMAGE_H, 1);
		
		// draw on buffer
		drawBuffer();
		
		// live image or canvas visualization
		Graphics g = show_image.getGraphics();
		if(draw_mode==DRAW_MODE.CAPTURE)
			g.drawImage(captured_img   , 0, 0, IMAGE_W, IMAGE_H, this);
		else
			g.drawImage(createImage(ip), 0, 0, IMAGE_W, IMAGE_H, this);
		
		// contour for painting visualization
		if( getContourMode() ){
			g.setColor( Color.blue  );
			drawContour((Graphics2D)g, IMAGE_W, IMAGE_H, grid_coarse, COARSE_W, COARSE_H, 0.125f);
			g.setColor( Color.green );
			drawContour((Graphics2D)g, IMAGE_W, IMAGE_H, grid_coarse, COARSE_W, COARSE_H, 0.375f);
			g.setColor( Color.yellow);
			drawContour((Graphics2D)g, IMAGE_W, IMAGE_H, grid_coarse, COARSE_W, COARSE_H, 0.625f);
			g.setColor( Color.red   );
			drawContour((Graphics2D)g, IMAGE_W, IMAGE_H, grid_coarse, COARSE_W, COARSE_H, 0.875f);
		}
		
		// circle for brush visualization
		if( !getContourMode() && mouse_entered && isBrush() ){
			int radius = getBrushRadius();
			
			// draw circle only
			g.setColor(Color.red);
			g.drawOval( convert_x(prev_p.x-radius), convert_y(prev_p.y-radius),
					convert_x(2*radius), convert_y(2*radius) );
		}
	}
	private float[] getRawBetaFromPaint(){
		float[] beta  = new float[IMAGE_W*IMAGE_H];
		
		// get pixels from base image
		int[]    pixels = new int[IMAGE_W*IMAGE_H];
		PixelGrabber pg = new PixelGrabber(getBaseImage(), 0, 0, IMAGE_W, IMAGE_H, true);
		
		try{
			pg.grabPixels();
		}catch(Exception e){
			e.printStackTrace();
		}
		pixels = (int[])pg.getPixels();
		
		// get float[] from Image & mouse cursor
		int   radius = getBrushRadius();
		float rad_x = convert_x(radius);
		float rad_y = convert_y(radius);
		
		// get float[] from Image & mouse cursor
		for(int j=0;j<IMAGE_H;j++)
		for(int i=0;i<IMAGE_W;i++){
			float m_i   = convert_x(prev_p.x);
			float m_j   = convert_y(prev_p.y);
			float ellip = (m_i-i)*(m_i-i)*(rad_y*rad_y) + (m_j-j)*(m_j-j)*(rad_x*rad_x);
			
			if( mouse_entered && isBrush() && (ellip <= rad_x*rad_x*rad_y*rad_y) ){
				beta[i+j*IMAGE_W] = Transform.int2Beta(getBrushColor().getRGB());
			}
			else{
				beta[i+j*IMAGE_W] = Transform.int2Beta(pixels[i+j*IMAGE_W]);
			}
		}
		
		return beta;
	}
	private float[] MG(float[] area, int W, int H, int iter){
		
		if( W<=COARSE_W || H<=COARSE_H ){
			grid_coarse = Transform.gaussian(area, W, H, iter);
			return grid_coarse;
		}
		
		// downscale
		float[] downscaled = MG(Transform.downscale2D(area, W, H), W/2, H/2, iter+1);
		
		// upscale & smoothing
		return Transform.gaussian(Transform.upscale2D(downscaled, W/2, H/2), W, H, iter);
	}
	private void drawBuffer(){
		int DOWN_X = IMAGE_W/COARSE_W;
		int DOWN_Y = IMAGE_H/COARSE_H;
		
		if(draw_mode==DRAW_MODE.TARGET_COARSE){
			int[] target_luminance = Main.lighty.getSolverTarget(grid_coarse);
			
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W;i++){
				int di = i/DOWN_X;
				int dj = j/DOWN_Y;
				int Y = target_luminance[di+dj*COARSE_W];
				
				pixels[i+j*IMAGE_W] = ( 255 << 24 ) | ( Y << 16 ) | ( Y << 8 ) | Y ; // aYYY;
			}
		}
		else if(draw_mode==DRAW_MODE.TARGET_FINE){
			int[][] target_YRGB = Main.lighty.getYRGB(grid_fine, true);
			
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W;i++){
				int R = target_YRGB[1][i+j*IMAGE_W];
				int G = target_YRGB[2][i+j*IMAGE_W];
				int B = target_YRGB[3][i+j*IMAGE_W];
				
				pixels[i+j*IMAGE_W] = ( 255 << 24 ) | ( R << 16 ) | ( G << 8 ) | B ; // aRGB';
			}
		}
		else if(draw_mode==DRAW_MODE.BETA_COARSE){
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W;i++){
				int di = i/DOWN_X;
				int dj = j/DOWN_Y;
				
				int pixel = Transform.beta2Int(grid_coarse[di+dj*COARSE_W]);
				pixels[i+j*IMAGE_W] = pixel;
			}
		}
		else if(draw_mode==DRAW_MODE.BETA_FINE){
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W;i++){
				int pixel = Transform.beta2Int(grid_fine[i+j*IMAGE_W]);
				pixels[i+j*IMAGE_W] = pixel;
			}
		}
		else if(draw_mode==DRAW_MODE.COLORBAR_COARSE){
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W;i++){
				int di = i/DOWN_X;
				int dj = j/DOWN_Y;
				
				int pixel = Transform.beta2HSB(grid_coarse[di+dj*COARSE_W]);
				pixels[i+j*IMAGE_W] = pixel;
			}
		}
		else if(draw_mode==DRAW_MODE.COLORBAR_FINE){
			for(int j=0;j<IMAGE_H;j++)
			for(int i=0;i<IMAGE_W;i++){
				int pixel = Transform.beta2HSB(grid_fine[i+j*IMAGE_W]);
				pixels[i+j*IMAGE_W] = pixel;
			}
		}
	}
	private void drawContour(Graphics2D g, int gW, int gH, float[] grid, int W, int H, float bound){
		g.setStroke(new BasicStroke(2.0f));
		
		// detect the grid case (1 line or 2 crossed lines)
		for(int j=-1;j<H;j++)
		for(int i=-1;i<W;i++){
			
			// get coordinate as CLAMP
			float v00 = (i<0    || j<0    ) ? -bound : grid[(i  )+(j  )*W] - bound ;
			float v01 = (i<0    || j>=H-1 ) ? -bound : grid[(i  )+(j+1)*W] - bound ;
			float v10 = (i>=W-1 || j<0    ) ? -bound : grid[(i+1)+(j  )*W] - bound ;
			float v11 = (i>=W-1 || j==H-1 ) ? -bound : grid[(i+1)+(j+1)*W] - bound ;
			
			// case : two crossed lines
			int g_x0 = (int)((i+0.5f)*gW/W);
			int g_x1 = (int)((i+1.5f)*gW/W);
			int g_y0 = (int)((j+0.5f)*gH/H);
			int g_y1 = (int)((j+1.5f)*gH/H);
			
			// smoothed boundary
			int g_vx0 = g_x0-(int)(v00/(v10-v00)*(g_x1-g_x0));
			int g_vx1 = g_x0-(int)(v01/(v11-v01)*(g_x1-g_x0));
			int g_vy0 = g_y0-(int)(v00/(v01-v00)*(g_y1-g_y0));
			int g_vy1 = g_y0-(int)(v10/(v11-v10)*(g_y1-g_y0));
			
			// vertex 1, 2, 3, 4
			if( v00*v10<0.0 && v00*v01<0.0 && v10*v11<0.0 && v01*v11<0.0 ){
				g.drawLine(g_x0 , g_vy0, g_x1 , g_vy1);
				g.drawLine(g_vx0, g_y0 , g_vx1, g_y1 );
				
				continue;
			}
			
			// vertex 1 & 2
			if( v00*v10<0.0 && v00*v01<0.0 ){
				g.drawLine(g_vx0, g_y0, g_x0, g_vy0);
			}
			// vertex 1 & 3
			if( v00*v10<0.0 && v10*v11<0.0 ){
				g.drawLine(g_vx0, g_y0, g_x1, g_vy1);
			}
			// vertex 1 & 4
			if( v00*v10<0.0 && v01*v11<0.0 ){
				g.drawLine(g_vx0, g_y0, g_vx1, g_y1);
			}
			// vertex 2 & 3
			if( v00*v01<0.0 && v10*v11<0.0 ){
				g.drawLine(g_x0, g_vy0, g_x1, g_vy1);
			}
			// vertex 2 & 4
			if( v00*v01<0.0 && v01*v11<0.0 ){
				g.drawLine(g_x0, g_vy0, g_vx1, g_y1);
			}
			// vertex 3 & 4
			if( v10*v11<0.0 && v01*v11<0.0 ){
				g.drawLine(g_x1, g_vy1, g_vx1, g_y1);
			}
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// Inner class : Undo/Redo manager
	////////////////////////////////////////////////////////////
	class UndoRedoImageList extends UndoRedoManager{
		BufferedImage[] base_image;
		
		UndoRedoImageList(int max_status) {
			super(max_status);
			
			base_image = new BufferedImage[MAX_STATUS];
			for(int i=0;i<MAX_STATUS;i++)
				base_image[i] = new BufferedImage(Lighty.IMAGE_W, Lighty.IMAGE_H, BufferedImage.TYPE_INT_RGB);
		}
		
		void copy(){
			Graphics g = base_image[getNextIndex()].getGraphics();
			g.drawImage(base_image[getCurrentIndex()], 0, 0, null);
		}
	}
}
