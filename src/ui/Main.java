package ui;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;

import ui.PCanvas.DRAW_MODE;

import model.*;
import model.Lighty.LIVE_MODE;

////////////////////////////////////////////////////////////////////////////////
// Lighty launcher
//
// Data   : 10/Sep/2012
//          13/Oct/2016 modified for 1920x1080 screen
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class Main implements ActionListener {
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	static Point     margin;
	static Dimension screen, window;
	
	
	////////////////////////////////////////////////////////////
	// Components
	////////////////////////////////////////////////////////////
	static Lighty  lighty;
	static PCanvas canvas;
	public static DirectControl direct;
	
	JFrame    frame_main, frame_canvas, frame_live, frame_tool, frame_direct;
	ToolPanel toolPanel;
	
	
	////////////////////////////////////////////////////////////
	// main launcher
	////////////////////////////////////////////////////////////
	public static void main(String[] args) {
		
		// default arguments
		//*
		int PAINT_W = 80;
		int PAINT_H = 60;
		screen = new Dimension(1920,1080);
		margin = new Point(1920-320,1080-230);
		window = new Dimension(1320,990);
		/*/
		// input
		String[] res = args[0].split("x");
		int PAINT_W = Integer.valueOf(res[0]);
		int PAINT_H = Integer.valueOf(res[1]);
		
		String[] margin_xy = args[1].split("x");
		margin = new Point(Integer.valueOf(margin_xy[0]),Integer.valueOf(margin_xy[1]));
		
		String[] screen_xy = args[2].split("x");
		screen = new Dimension(Integer.valueOf(screen_xy[0]),Integer.valueOf(screen_xy[1]));
		
		String[] window_xy = args[3].split("x");
		window = new Dimension(Integer.valueOf(window_xy[0]),Integer.valueOf(window_xy[1]));
		//*/
		// model instance
		lighty = new Lighty(PAINT_W, PAINT_H);
		
		new Main();
	}
	
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	Main() {
		
		////////////////////////////////////////////////////////////
		// main frame
		////////////////////////////////////////////////////////////
		frame_main = new JFrame("Main window");
		frame_main.setSize(320,180);
		frame_main.setLocation(margin);
		frame_main.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setMenubar();
		
		////////////////////////////////////////////////////////////
		// painting canvas
		////////////////////////////////////////////////////////////
		frame_canvas = new JFrame("Painting canvas");
		canvas = new PCanvas();
		canvas.addMouseListener(canvas);
		canvas.addMouseMotionListener(canvas);
		frame_canvas.add(canvas);
		
		////////////////////////////////////////////////////////////
		// direct manipulation
		////////////////////////////////////////////////////////////
		frame_direct = new JFrame("Direct manipulation");
		direct = new DirectControl();
		frame_direct.add(direct);
		
		////////////////////////////////////////////////////////////
		// live view : reusing canvas as live view panel
		////////////////////////////////////////////////////////////
		frame_live = new JFrame("Live view");
		PCanvas live = new PCanvas();
		live.setDrawMode(DRAW_MODE.CAPTURE);
		live.setContourMode(false);
		frame_live.add(live);
		
		////////////////////////////////////////////////////////////
		// tool panel
		////////////////////////////////////////////////////////////
		frame_tool = new JFrame("Tool panel");
		toolPanel = new ToolPanel();
		frame_tool.add(toolPanel);
		
		////////////////////////////////////////////////////////////
		// 
		////////////////////////////////////////////////////////////
		frame_main.setVisible(true);
		doLayoutPainting();
		InputStatus.clear(false);
	}
	
	
	////////////////////////////////////////////////////////////
	// Method & Listener : user control
	////////////////////////////////////////////////////////////
	String[] select_ui   = {"painting", "direct", "merged", "preliminary"};
	String[] select_mode = {"capture", "target:coarse", "target:fine", "beta:coarse", "beta:fine", "colorbar:coarse", "colorbar:fine"};
	String[] select_live = {"camera", "simulation:coarse", "simulation:fine"};
	String[] select_time = {"1/60","1/100","1/250","1/500","1/1000","1/2000"};
	
	void setMenubar(){
		
		JMenuBar menubar = new JMenuBar();
		frame_main.setJMenuBar(menubar);
		
		// menu item
		JMenu selectUI   = new JMenu("UI");
		JMenu selectMode = new JMenu("Mode");
		JMenu selectLive = new JMenu("Live");
		JMenu selectTime = new JMenu("Shutter");
		menubar.add(selectUI);
		menubar.add(selectMode);
		menubar.add(selectLive);
		menubar.add(selectTime);
		
		// menu item 1 : UI
		for(int i=0;i<select_ui.length;i++){
			JMenuItem ui = new JMenuItem(select_ui[i]);
			ui.addActionListener(this);
			selectUI.add(ui);
		}
		
		// menu item 2 : visualization mode
		for(int i=0;i<select_mode.length;i++){
			JMenuItem mode = new JMenuItem(select_mode[i]);
			mode.addActionListener(this);
			selectMode.add(mode);
		}
		
		// menu item 3 : camera view or simulation
		for(int i=0;i<select_live.length;i++){
			JMenuItem live = new JMenuItem(select_live[i]);
			live.addActionListener(this);
			selectLive.add(live);
		}
		
		// menu item 4 : camera exposure time
		for(int i=0;i<select_time.length;i++){
			JMenuItem time = new JMenuItem(select_time[i]);
			time.addActionListener(this);
			selectTime.add(time);
		}
	}
	
	public void actionPerformed(ActionEvent arg0) {
		String cmd = arg0.getActionCommand();
		
		// window layout
		if( cmd==select_ui[0] ) doLayoutPainting();
		if( cmd==select_ui[1] ) doLayoutDirect();
		if( cmd==select_ui[2] ) doLayoutMerged();
		if( cmd==select_ui[3] ) doLayoutPrelim();
		
		// visualize mode
		if( cmd==select_mode[0] ) canvas.setDrawMode(DRAW_MODE.CAPTURE);
		if( cmd==select_mode[1] ) canvas.setDrawMode(DRAW_MODE.TARGET_COARSE);
		if( cmd==select_mode[2] ) canvas.setDrawMode(DRAW_MODE.TARGET_FINE);
		if( cmd==select_mode[3] ) canvas.setDrawMode(DRAW_MODE.BETA_COARSE);
		if( cmd==select_mode[4] ) canvas.setDrawMode(DRAW_MODE.BETA_FINE);
		if( cmd==select_mode[5] ) canvas.setDrawMode(DRAW_MODE.COLORBAR_COARSE);
		if( cmd==select_mode[6] ) canvas.setDrawMode(DRAW_MODE.COLORBAR_FINE);
		
		// live view
		if( cmd==select_live[0] ) lighty.setLiveMode(LIVE_MODE.CAMERA);
		if( cmd==select_live[1] ) lighty.setLiveMode(LIVE_MODE.SIMULATION_COARSE);
		if( cmd==select_live[2] ) lighty.setLiveMode(LIVE_MODE.SIMULATION_FINE);
		
		// camera exposure time
		if( cmd==select_time[0] ) lighty.setExposureTime(0);
		if( cmd==select_time[1] ) lighty.setExposureTime(1);
		if( cmd==select_time[2] ) lighty.setExposureTime(2);
		if( cmd==select_time[3] ) lighty.setExposureTime(3);
		if( cmd==select_time[4] ) lighty.setExposureTime(4);
		if( cmd==select_time[5] ) lighty.setExposureTime(5);
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : 
	////////////////////////////////////////////////////////////
	void doLayoutPainting(){
		
		// painting canvas
		frame_canvas.setSize(window);
		frame_canvas.setLocation(0,0);
		canvas.setDrawMode(DRAW_MODE.CAPTURE);
		canvas.setContourMode(true);
		
		// tool panel
		int w = screen.width - window.width - 200;
		int h = window.height - 350;
		frame_tool.setSize(w,h);
		frame_tool.setLocation(window.width+(screen.width-window.width-w)/2, (window.height-h)/2);
		toolPanel.layoutVertical();
		
		// visibility
		frame_canvas.setVisible(true);
		frame_direct.setVisible(false);
		frame_live.setVisible(false);
		frame_tool.setVisible(true);
	}
	void doLayoutDirect(){
		
		// live view
		frame_live.setSize(window);
		frame_live.setLocation(0,0);
		
		// direct manipulation
		int w = screen.width - window.width;
		int h = w*3/4;
		frame_direct.setSize(w,h);
		frame_direct.setLocation(window.width, (window.height-h)/2);
		
		// visibility
		frame_canvas.setVisible(false);
		frame_direct.setVisible(true);
		frame_live.setVisible(true);
		frame_tool.setVisible(false);
	}
	void doLayoutMerged(){
		
		// for two-screens
		int w = screen.width/2;
		int h = w*3/4;
		Dimension window = new Dimension(w,h);
		
		// painting canvas
		frame_canvas.setSize(window);
		frame_canvas.setLocation(0,0);
		canvas.setDrawMode(DRAW_MODE.CAPTURE);
		canvas.setContourMode(true);

		// direct manipulation
		frame_direct.setSize(window);
		frame_direct.setLocation(window.width, 0);
		
		// tool panel
		int tool_w = screen.width-320;
		frame_tool.setSize(tool_w,screen.height-window.height-50);
		frame_tool.setLocation(0, window.height);
		toolPanel.layoutHorizontal();

		// visibility
		frame_canvas.setVisible(true);
		frame_direct.setVisible(true);
		frame_live.setVisible(false);
		frame_tool.setVisible(true);
	}
	void doLayoutPrelim(){
		
		// for two-screens
		int w = screen.width/2;
		int h = w*3/4;
		Dimension window = new Dimension(w,h);
		
		// live view
		frame_live.setSize(window);
		frame_live.setLocation(0,0);
		
		// painting canvas
		frame_canvas.setSize(window);
		frame_canvas.setLocation(window.width, 0);
		canvas.setDrawMode(DRAW_MODE.TARGET_FINE);
		canvas.setContourMode(false);
		
		// tool panel
		int tool_w = screen.width-320;
		frame_tool.setSize(tool_w,screen.height-window.height-50);
		frame_tool.setLocation(0, window.height);
		toolPanel.layoutHorizontal();
		
		// visibility
		frame_canvas.setVisible(true);
		frame_direct.setVisible(false);
		frame_live.setVisible(true);
		frame_tool.setVisible(true);
	}
}
