package model;

import java.io.*;

import gnu.io.*;

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
public class SerialCom {
	
	////////////////////////////////////////////////////////////
	// Constants
	////////////////////////////////////////////////////////////
	private static String COM;
	private CommPortIdentifier portId;
	private SerialPort port;
	
	
	////////////////////////////////////////////////////////////
	// Components
	////////////////////////////////////////////////////////////
	private int[] current_conf;
    
	////////////////////////////////////////////////////////////
	// Constructor
	////////////////////////////////////////////////////////////
	SerialCom(String com_num){
		COM = com_num;
		current_conf = Config.downwardsConfig(0);
	}
	protected boolean openSerialPort() {
		/*
        try {
        	System.out.println(COM);
        	portId = CommPortIdentifier.getPortIdentifier(COM);
            port = (SerialPort)portId.open("RadianceMap", 2000);
        } catch (NoSuchPortException e) {
            e.printStackTrace();
            return false;
        } catch (PortInUseException e) {
            e.printStackTrace();
            return false;
        }
        //*/
        return true;
    }
	protected boolean settingSerialPort(){
		/*
		try {
            port.setSerialPortParams(
                9600,                  // bps
                SerialPort.DATABITS_8, // 8
                SerialPort.STOPBITS_1, // stop
                SerialPort.PARITY_NONE // parity
            );
            port.setFlowControlMode(SerialPort.FLOWCONTROL_NONE);
        } catch (UnsupportedCommOperationException e) {
            e.printStackTrace();
            return false;
        }
        port.setDTR(true);
        port.setRTS(false);
        //*/
        return true;
	}
	
	////////////////////////////////////////////////////////////
	// Method : control
	////////////////////////////////////////////////////////////
	void write(int[] configuration){
		try{
//			PrintWriter pw = new PrintWriter( port.getOutputStream(), true);
			for(int l=0; l<Config.L_NUM; l++){
				String setting = "i"+String.valueOf(l)+" ";
				
				switch(l){
				case 0: case 1: case 4: case 5: case 8: case 9:
					setting += "x" + String.valueOf(      Config.x_degree  [configuration[3*l+1]]) + " ";
					setting += "y" + String.valueOf(180 - Config.y_degree  [configuration[3*l+2]]) + " ";
					setting += "b" + String.valueOf(      Config.brightness[configuration[3*l+0]]) + " s\n\r";
					break;
				case 2: case 3: case 6: case 7: case 10: case 11:
					setting += "x" + String.valueOf(180 - Config.x_degree  [configuration[3*l+1]]) + " ";
					setting += "y" + String.valueOf(      Config.y_degree  [configuration[3*l+2]]) + " ";
					setting += "b" + String.valueOf(      Config.brightness[configuration[3*l+0]]) + " s\n\r";
					break;
				}
				
				// if configuration of i-th light is changed, rewrite the value.
				if(	!(	configuration[3*l  ]==current_conf[3*l  ] &&
						configuration[3*l+1]==current_conf[3*l+1] &&
						configuration[3*l+2]==current_conf[3*l+2] ) ){
//					pw.write(setting+"\n");
					
					int b = configuration[3*l+0];
					int x = configuration[3*l+1];
					int y = configuration[3*l+2];
					ui.Main.direct.setConf(l, b, x, y);
				}
			}
			
//			pw.write("s\n");
//			pw.flush();
//			pw.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
		// change the current configuration
		current_conf = Config.copyConfig(configuration);
		Lighty.captureSim.simulate(current_conf);
	}
	void write(int shutter_speed){
		/*
		try{
			PrintWriter pw = new PrintWriter( port.getOutputStream(), true);
			String setting = "i16 e"+shutter_speed+" s";
			pw.write(setting+"\n");
			pw.flush();
		}
		catch(Exception e){
			e.printStackTrace();
		}
		//*/
	}
}
