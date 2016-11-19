package ui;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

import model.*;

@SuppressWarnings("serial")
////////////////////////////////////////////////////////////////////////////////
// GUI Status of Lighty
//
// Data   : 11/Sep/2012
// Author : Seung-tak Noh
////////////////////////////////////////////////////////////////////////////////
public class DirectControl extends JPanel{
	
	////////////////////////////////////////////////////////////
	// Components
	////////////////////////////////////////////////////////////
	LightPanel[] light;
	JSlider[] brightness;
	OrPanel[] orientation;
	
	
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	DirectControl(){
		super();
		setLayout(new GridLayout(3, 4));
		setPreferredSize(new Dimension(640,480));
		
		// container
		brightness  = new JSlider[Config.L_NUM];
		orientation = new OrPanel[Config.L_NUM];
		
		light = new LightPanel[Config.L_NUM];
		for(int id=1;id<=Config.L_NUM;id++){
			light[Config.L_NUM-id] = new LightPanel(Config.L_NUM-id);
			
			add(light[Config.L_NUM-id]);
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// Method : for feedback
	////////////////////////////////////////////////////////////
	public void setConf(int id, int b, int x, int y){
		brightness [id].setValue(b);
		orientation[id].x = x;
		orientation[id].y = y;
		
		brightness [id].repaint();
		orientation[id].repaint();
	}
	
	public int[] getConf(){
		int[] current_conf = new int[Config.L_NUM*3];
		
		for(int id=0;id<Config.L_NUM;id++){
			current_conf[3*id+0] = brightness [id].getValue();
			current_conf[3*id+1] = orientation[id].x;
			current_conf[3*id+2] = orientation[id].y;
		}
		
		return current_conf;
	}
	
	
	////////////////////////////////////////////////////////////
	// inner class : each light panel = (canvas+slider)
	////////////////////////////////////////////////////////////
	class LightPanel extends JPanel implements ChangeListener{
		int id;
		
		////////////////////////////////////////
		// Constructor
		////////////////////////////////////////
		LightPanel(int id){
			super();
			
			// write id
			this.id = id;
			
			// panel : orientation
			orientation[id] = new OrPanel(id);
			
			// slider: brightness
			brightness[id] = new JSlider(JSlider.VERTICAL, 0, Config.B_NUM-1, 0);
			brightness[id].setMajorTickSpacing(1);
			brightness[id].setSnapToTicks(true);
			brightness[id].setPaintTicks(true);
			
			// this panel
			setBackground(Color.gray);
			setBorder(new EmptyBorder(10,2,10,2));
			setLayout(new BorderLayout());
			add(orientation[id],BorderLayout.CENTER);
			add(brightness [id],BorderLayout.EAST);
			
			brightness[id].addChangeListener(this);
		}

		////////////////////////////////////////
		// Constructor
		////////////////////////////////////////
		public void stateChanged(ChangeEvent arg0) {
			orientation[id].repaint();
		}
	}
	
	
	////////////////////////////////////////////////////////////
	// inner class : each light
	////////////////////////////////////////////////////////////
	class OrPanel extends Canvas implements MouseListener, MouseMotionListener{
		int id;
		
		int x = (Config.X_NUM-1)/2;
		int y = (Config.Y_NUM-1)/2;
		
		////////////////////////////////////////
		// Constructor
		////////////////////////////////////////
		OrPanel(int id){
			super();
			
			this.id = id;
			
			setBackground(Color.black);
			
			addMouseListener(this);
			addMouseMotionListener(this);
		}
		
		////////////////////////////////////////
		// Listener : draw
		////////////////////////////////////////
		public void update(Graphics g){
			paint(g);
			
			// direct manipulation
			int b = brightness[id].getValue();
			Main.lighty.setLight(id,b,x,y);
		}
		public void paint(Graphics g){
			Color c = new Color(Transform.beta2Int( (float)brightness[id].getValue() /(float)(Config.B_NUM-1) ));
			
			int w_d = getWidth ()/Config.X_NUM;
			int h_d = getHeight()/Config.Y_NUM;
			
			g.clearRect(0, 0, getWidth(), getHeight());
			g.setColor(Color.white);
			g.fillOval( (x-1)*w_d-2, (y-1)*h_d-2, 3*w_d+4, 3*h_d+4 );
			g.setColor(c);
			g.fillOval( (x-1)*w_d  , (y-1)*h_d  , 3*w_d  , 3*h_d   );
			g.dispose();
		}
		
		
		////////////////////////////////////////
		// Listener : Mouse
		////////////////////////////////////////
		public void mouseDragged(MouseEvent arg0) {
			Point p = arg0.getPoint();
			
			int w_d = getWidth() /Config.X_NUM;
			int h_d = getHeight()/Config.Y_NUM;
			
			// stopper : x
			x = p.x / w_d;
			if(x>=Config.X_NUM) x = Config.X_NUM-1;
			else if(x<0) x = 0;
			
			// stopper : y
			y = p.y / h_d;
			if(y>=Config.Y_NUM) y = Config.Y_NUM-1;
			else if(y<0) y = 0;
			
			repaint();
		}
		public void mousePressed(MouseEvent arg0){
			Point p = arg0.getPoint();
			
			int w_d = getWidth() /Config.X_NUM;
			int h_d = getHeight()/Config.Y_NUM;
			
			// stopper : x
			x = p.x / w_d;
			if(x>=Config.X_NUM) x = Config.X_NUM-1;
			else if(x<0) x = 0;
			
			// stopper : y
			y = p.y / h_d;
			if(y>=Config.Y_NUM) y = Config.Y_NUM-1;
			else if(y<0) y = 0;
			
			repaint();
		}
		public void mouseMoved(MouseEvent arg0)   {} // no in use
		public void mouseClicked(MouseEvent arg0) {} // no in use
		public void mouseEntered(MouseEvent arg0) {} // no in use
		public void mouseExited(MouseEvent arg0)  {} // no in use
		public void mouseReleased(MouseEvent arg0){} // no in use
	}
}
