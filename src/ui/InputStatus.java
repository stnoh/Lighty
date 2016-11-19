package ui;

import java.awt.Color;

import model.*;

////////////////////////////////////////////////////////////////////////////////
// GUI Status of Lighty
//
// Data   : 10/Sep/2012
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class InputStatus {
	public static final boolean LOG = false;
	
	////////////////////////////////////////////////////////////
	// Input log
	////////////////////////////////////////////////////////////
//	public static void startLog(){} // OBSOLETE
//	public static void endLog(){}   // OBSOLETE
	public static void addLog(String log){
		if(LOG){
			System.out.println(log);
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// Brush type
	////////////////////////////////////////////////////////////
	protected static enum INPUT_MODE { BRUSH, SQUIRT };
	private   static INPUT_MODE mode = INPUT_MODE.BRUSH;
	public    static boolean isBrush (){ return ( mode==INPUT_MODE.BRUSH  ) ? true : false ; }
	public    static boolean isSquirt(){ return ( mode==INPUT_MODE.SQUIRT ) ? true : false ; }
	public    static void setInputMode(INPUT_MODE _mode){
		addLog("set: "+_mode.toString());
		mode = _mode;
	}
	
	
	////////////////////////////////////////////////////////////
	// Brush color & size 
	////////////////////////////////////////////////////////////
	private static Color brush_color = Color.black;
	public  static Color getBrushColor(){ return brush_color; }
	public  static void  setBrushColor(Color c){
		addLog("set brightness: "+model.Transform.int2Beta(c.getRGB()) );
		brush_color = c;
	}
	
	public  static final int MIN_RADIUS = 60;
	public  static final int MAX_RADIUS = 120;
	private static int brush_radius = (MIN_RADIUS+MAX_RADIUS)/2;
	public  static int  getBrushRadius(){ return brush_radius; }
	public  static void setBrushRadius(int radius){
		addLog("set radius: "+radius);
		brush_radius = radius;
	}
	
	
	////////////////////////////////////////////////////////////
	// canvas editing
	////////////////////////////////////////////////////////////
	public static void clear(boolean stack){
		addLog("clear: "+Transform.int2Beta(getBrushColor().getRGB()));
		
		int b = (int)( (Config.B_NUM-1) * Transform.int2Beta(getBrushColor().getRGB()) );
		Main.lighty.setLights(b);
		Main.canvas.clear(stack);
	}
	
	public static void undo(){
		addLog("undo:");
		Main.canvas.undo();
	}
	public static void redo(){
		addLog("redo:");
		Main.canvas.redo();
	}
	public static boolean undoable(){ return Main.canvas.undoable(); }
	public static boolean redoable(){ return Main.canvas.redoable(); }
	
	
	////////////////////////////////////////////////////////////
	// Alpha value (OBSOLETE)
	////////////////////////////////////////////////////////////
	public  static final float MIN_ALPHA = 0.0f;
	public  static final float MAX_ALPHA = 20.0f;
	private static float alpha = 0.0f;
	public  static float getAlphaValue(){ return alpha; }
	public  static void  setAlphaValue(float alpha_value){
		addLog("set alpha:"+alpha_value);
		alpha = alpha_value;
	}
	
	
	////////////////////////////////////////////////////////////
	// misc.
	////////////////////////////////////////////////////////////
	public static boolean getContourMode(){ return Main.canvas.getContourMode(); }
	public static void    setContourMode(boolean mode){
		addLog("set contour: "+mode);
		Main.canvas.setContourMode(mode);
	}
	public static void renewButtons(){ ToolPanel.renewButtons(); }
}
