package ui;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;

import javax.imageio.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

import ui.InputStatus.INPUT_MODE;

import model.*;

@SuppressWarnings("serial")
////////////////////////////////////////////////////////////////////////////////
// Tool panel
//
// Data   : 11/Sep/2012
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class ToolPanel extends JPanel{
	
	////////////////////////////////////////////////////////////
	// Constants, components
	////////////////////////////////////////////////////////////
	static final int ICON_W = 160;
	static final int ICON_H = 120;
	
	static OvalCanvas ovalCanvas;
	PresetPanel comp0;
	ColorPanel  comp1;
	SizePanel   comp2;
	
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	ToolPanel(){
		super();
		
		comp0 = new PresetPanel();
		comp1 = new ColorPanel();
		comp2 = new SizePanel();
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : layout again
	////////////////////////////////////////////////////////////
	public void layoutVertical(){
		removeAll();
		
		setLayout(new GridLayout(2,1));
		add(comp1); add(comp2);
	}
	public void layoutHorizontal(){
		removeAll();
		
		setLayout(new GridLayout(1,3));
		add(comp0); add(comp1); add(comp2);
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : for redraw buttons
	////////////////////////////////////////////////////////////
	public static void renewButtons(){
		
		// brush & squirt
		if(InputStatus.isSquirt()){
			type_button[0].setBackground(DISABLE_COLOR);
			type_button[1].setBackground(ENABLE_COLOR);
		}
		else{
			type_button[0].setBackground(ENABLE_COLOR);
			type_button[1].setBackground(DISABLE_COLOR);
		}
		
		// clear button
		float beta = Transform.int2Beta(InputStatus.getBrushColor().getRGB());
		if(beta>1.0f)
			type_button[2].setEnabled(false);
		else
			type_button[2].setEnabled(true);
		
		// undo button
		if( InputStatus.undoable() )
			type_button[3].setEnabled(true);
		else
			type_button[3].setEnabled(false);
		
		// redo button
		if( InputStatus.redoable() )
			type_button[4].setEnabled(true);
		else{
			type_button[4].setEnabled(false);
		}
		
		// contour mode
		if( InputStatus.getContourMode() )
			type_button[5].setBackground(ENABLE_COLOR);
		else
			type_button[5].setBackground(DISABLE_COLOR);
		
		// brush color
		float diff = Float.MAX_VALUE;
		int selected = -1;
			
		for(int i=0;i<NUM_OF_COLORS;i++){
			color_button[i].setBorder( LineBorder.createGrayLineBorder() );
			
			if(diff > Math.abs(beta-beta_value[i]) ){
				selected = i;
				diff = Math.abs(beta-beta_value[i]);
			}
		}
		color_button[selected].setBorder( new LineBorder(Color.RED , 5) );
	}
	
	
	////////////////////////////////////////////////////////////
	// Inner class : color selection
	////////////////////////////////////////////////////////////
	// select type
	static final  Color ENABLE_COLOR  = Color.yellow;
	static final  Color DISABLE_COLOR = Color.white;
	static JButton[] type_button      = new JButton[6];
	static String [] type_button_icon = {"brush","squirt","clear","undo","redo","contour"};
	
	// select color
	public static final int NUM_OF_COLORS = 5;
	public static final float  [] beta_value   = {0.0f, 0.255f, 0.505f, 0.755f, 1.00f}; // exact floating point causes error.
	static JButton[] color_button = new JButton[NUM_OF_COLORS];
	
	JSlider colorSlider;
	
	class ColorPanel extends JPanel implements ActionListener{
		ColorPanel(){
			super();
			
			////////////////////////////////////////
			// 1) brush type
			////////////////////////////////////////
			JPanel typePanel  = new JPanel();
			typePanel.setLayout(new GridLayout(1,6));
			
			for(int i=0;i<6;i++){
				type_button[i] = new JButton();
				type_button[i].setActionCommand(type_button_icon[i]);
				type_button[i].addActionListener(this);
				
				// read image
				Image img = null;
				try{
					img = ImageIO.read(new File("icon\\"+type_button_icon[i]+".png"));
				}catch(Exception e){
					e.printStackTrace();
				}
				
				// set the icon
				type_button[i].setIcon( new ImageIcon(img) );
				type_button[i].setBackground(Color.WHITE);
				
				// set the pressed icon
				if(i>=2){
					BufferedImage img2 = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_RGB);
					Graphics2D g = img2.createGraphics();
					g.drawImage(img, 0, 0, ENABLE_COLOR, null);
					g.dispose();
					
					type_button[i].setPressedIcon( new ImageIcon(img2) );
				}
				
				typePanel.add(type_button[i]);
			}
			
			
			////////////////////////////////////////
			// 2) brush color
			////////////////////////////////////////
			// construct color buttons
			for(int i=0;i<NUM_OF_COLORS;i++){
				color_button[i] = new JButton();
				color_button[i].setActionCommand( Float.toString(beta_value[i]) );
				color_button[i].addActionListener(this);
				
				// set the color
				color_button[i].setBackground(new Color(Transform.beta2Int(beta_value[i])));
				color_button[i].setFocusable(false);
			}
			
			// background change
			renewButtons();
			
			// discrete color
			JPanel colorPanel = new JPanel();
			
			FlowLayout flowLayout = new FlowLayout(FlowLayout.LEFT);
			flowLayout.setHgap(0);flowLayout.setVgap(0);
			colorPanel.setLayout(flowLayout);
			
			for(int i=0;i<NUM_OF_COLORS;i++){
				color_button[i].setPreferredSize(new Dimension(55,50) );
				color_button[i].setForeground(Color.white);
				colorPanel.add(color_button[i]);
			}
			
			////////////////////////////////////////
			// this part is hard-coded ************************************************************
			////////////////////////////////////////
			typePanel.setPreferredSize( new Dimension(320, 64) );
			add(typePanel);
			
			ImagePanel colorbarPanel1 = new ImagePanel(280,50);
			ImagePanel colorbarPanel2 = new ImagePanel(280,50);
			BufferedImage img = new BufferedImage(280,120,BufferedImage.TYPE_INT_RGB);
			for(int i=0;i<280;i++)
			for(int j=0;j<120;j++){
				img.setRGB(i,j,getBackground().getRGB());
			}
			for(int i=0;i<5 ;i++)
			for(int j=0;j<120;j++){
				img.setRGB(i+ 55,j,Color.blue.getRGB());
				img.setRGB(i+110,j,Color.green.getRGB());
				img.setRGB(i+165,j,Color.yellow.getRGB());
				img.setRGB(i+220,j,Color.red.getRGB());
			}
			
			colorbarPanel1.setImage(img);
			colorbarPanel2.setImage(img);
			add(colorbarPanel1);
			
			add(colorPanel);
			add(colorbarPanel2);
			////////////////////////////////////////
			// this part is hard-coded ************************************************************
			////////////////////////////////////////
		}

		public void actionPerformed(ActionEvent e) {
			String cmd = e.getActionCommand();
			
			if(cmd=="brush")        InputStatus.setInputMode(INPUT_MODE.BRUSH);
			else if(cmd=="squirt")  InputStatus.setInputMode(INPUT_MODE.SQUIRT);
			else if(cmd=="clear")   InputStatus.clear(true);
			else if(cmd=="undo")    InputStatus.undo();
			else if(cmd=="redo")    InputStatus.redo();
			else if(cmd=="contour") InputStatus.setContourMode(!InputStatus.getContourMode());
			
			else{
				float beta = Float.valueOf(e.getActionCommand());
				
				InputStatus.setBrushColor( new Color(Transform.beta2Int(beta)) );
				ovalCanvas.repaint();
			}
			
			renewButtons();
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// Inner class : preset save/load
	////////////////////////////////////////////////////////////
	class PresetPanel extends JPanel implements ActionListener, MouseListener{
		boolean save_mode = false;
		
		JButton   modeButton;
		JButton[] slotButton;
		
		PresetPanel(){
			super();
			
			// 1. mode select button : select
			modeButton = new JButton();
			modeButton.setFont(new Font(Font.SANS_SERIF,Font.BOLD,14));
			modeButton.setActionCommand("mode_change");
			modeButton.addActionListener(this);
			setModeText();
			
			// 2. save/load slots
			JPanel slotPanel = new JPanel();
			slotPanel.setLayout(new GridLayout(2,2,8,2));
			
			slotButton = new JButton[4];
			for(int i=0;i<4;i++){
				slotButton[i] = new JButton();
				slotButton[i].setName( String.valueOf(i) );
				slotButton[i].setPreferredSize(new Dimension(ICON_W, ICON_H));
				slotButton[i].setContentAreaFilled(false);
				slotButton[i].setBorder(null);
				
				slotButton[i].addMouseListener(this);
				slotPanel.add(slotButton[i]);
				setSlot(i);
			}
			
			add(modeButton);
			add(slotPanel);
		}
		
		// sub-routine : text
		private void setModeText(){
			if(save_mode)
				modeButton.setText("Cancel");
			else
				modeButton.setText("Register preset");
		}
		
		// sub-routine : 
		private void setSlot(int slot_num){
			Image img = null;
			try{
				img = ImageIO.read(new File("save\\teaser_"+slot_num+".png"));
			}catch(Exception e){
				e.printStackTrace();
			}
			
			// reset the icon
			slotButton[slot_num].setIcon( new ImageIcon(img) );
		}
		
		// listener : action
		public void actionPerformed(ActionEvent e){
			String str = e.getActionCommand();
			
			if(str=="mode_change"){
				save_mode = !save_mode;
				setModeText();
			}
		}
		
		// listener : mouse
		public void mouseEntered (MouseEvent arg0){
			int slot_num = Integer.valueOf( arg0.getComponent().getName() );
			
			if(save_mode){
				BufferedImage teaser = Main.canvas.getTeaser();
				Graphics2D g = (Graphics2D)teaser.getGraphics();
				g.setFont(new Font(Font.SANS_SERIF,Font.BOLD,14));
				g.setColor(Color.cyan);
				g.drawString("Tab to register", 10, 20);
				g.dispose();
				
				slotButton[slot_num].setIcon(new ImageIcon(teaser));
				slotButton[slot_num].setBorder( new LineBorder(Color.BLUE , 5) );
			}
			else{
				slotButton[slot_num].setBorder( new LineBorder(Color.RED , 5) );
			}
		}
		public void mouseExited  (MouseEvent arg0){
			int slot_num = Integer.valueOf( arg0.getComponent().getName() );
			
			if(save_mode){
				setSlot(slot_num);
			}
			slotButton[slot_num].setBorder( null );
		}
		public void mousePressed (MouseEvent arg0){
			int slot_num = Integer.valueOf( arg0.getComponent().getName() );
			
			// save
			if(save_mode){
				try{
					ImageIO.write(Main.canvas.getBaseImage(), "bmp", new File("save\\beta_"+slot_num+".bmp"));
					ImageIO.write(Main.canvas.getTeaser()   , "png", new File("save\\teaser_"+slot_num+".png"));
				}catch(Exception e){
					e.printStackTrace();
				}
				
				// change save mode also.
				save_mode = !save_mode;
				slotButton[slot_num].setBorder(null);
				
				setModeText();
				setSlot(slot_num);
			}
			// load
			else{
				Image img = null;
				try{
					img = ImageIO.read(new File("save\\beta_"+slot_num+".bmp"));
				}catch(Exception e){
					e.printStackTrace();
				}
				
				Main.canvas.load(img);
			}
		}
		public void mouseClicked (MouseEvent arg0){} // no in use
		public void mouseReleased(MouseEvent arg0){} // no in use
	}
	
	
	////////////////////////////////////////////////////////////
	// Inner class : brush size
	////////////////////////////////////////////////////////////
	JSlider    sizeSlider;
	class SizePanel extends JPanel implements ChangeListener{
		
		// constructor
		SizePanel(){
			super();
			
			// component
			ovalCanvas = new OvalCanvas();
			sizeSlider = new JSlider(JSlider.VERTICAL, InputStatus.MIN_RADIUS, InputStatus.MAX_RADIUS, InputStatus.getBrushRadius());
			sizeSlider.setPreferredSize(new Dimension(50,250));
			sizeSlider.addChangeListener(this);
			
			// layout
			setLayout(new BorderLayout());
			add(ovalCanvas, BorderLayout.CENTER);
			add(sizeSlider, BorderLayout.EAST  );
		}
		
		// listener
		public void stateChanged(ChangeEvent e){
			InputStatus.setBrushRadius( sizeSlider.getValue() );
			
			ovalCanvas.repaint();
			Main.canvas.repaint();
		}
	}
	
	////////////////////////////////////////////////////////////
	// Inner class : oval canvas 
	////////////////////////////////////////////////////////////
	class OvalCanvas extends Canvas{
		
		// constructor
		OvalCanvas(){
			super();
		}
		
		// listener
		public void paint(Graphics g){
			Point center = new Point(getSize().width/2, getSize().height/2);
			int   radius = InputStatus.getBrushRadius();
			
			// filled circle with color & red outline
			g.setColor(InputStatus.getBrushColor());
			g.fillOval(center.x-radius, center.y-radius, 2*radius, 2*radius );
			g.setColor(Color.red);
			g.drawOval(center.x-radius, center.y-radius, 2*radius, 2*radius );
			
			g.dispose();
		}
	}
	
	////////////////////////////////////////////////////////////
	// Inner class : image panel
	////////////////////////////////////////////////////////////
	public class ImagePanel extends Canvas{
		
		// Components
		BufferedImage img;
		
		// Constructor
		ImagePanel(int W, int H){
			super();
			setPreferredSize(new Dimension(W,H));
			img = new BufferedImage(W,H,BufferedImage.TYPE_INT_RGB);
		}
		
		// set image
		public void setImage(BufferedImage img){
			this.img = img;
			repaint();
		}
		public void paint(Graphics g){
			g.drawImage(img, 0, 0, getWidth(), getHeight(), 
					0, 0, img.getWidth(), img.getHeight(), this);
			g.dispose();
		}
	}
}
