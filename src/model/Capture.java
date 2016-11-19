package model;

import de.humatic.dsj.DSCapture;
import de.humatic.dsj.DSFilterInfo;
import de.humatic.dsj.DSFiltergraph;
import de.humatic.dsj.DSMediaType;
import de.humatic.dsj.DSFilterInfo.DSMediaFormat;
import de.humatic.dsj.DSFilterInfo.DSPinInfo;

@SuppressWarnings("serial")
public class Capture extends DSCapture implements java.beans.PropertyChangeListener{
	
	public static int CAPTURE_WIDTH;
	
	Capture(int flags, DSFilterInfo videoDeviceInfo, boolean captureAudioFromVideoDevice, DSFilterInfo audioDeviceInfo, java.beans.PropertyChangeListener pcl) {
		super(flags, videoDeviceInfo, captureAudioFromVideoDevice, audioDeviceInfo, pcl);
	}
	
	public static Capture create(int width){
		CAPTURE_WIDTH = width;
		
		DSFilterInfo dsi = DSCapture.queryDevices()[0][0];
		DSPinInfo pin = getCapturePin(dsi);
		
		if (pin != null) {
			setPinWidth(pin, DSMediaFormat.VST_RGB24, CAPTURE_WIDTH);
		}
		
		Capture capture = new Capture(DSFiltergraph.RENDER_NATIVE, dsi, false,
				DSFilterInfo.doNotRender(), null);
		capture.addPropertyChangeListener(capture);
		
		return capture;
	}
	
	// property change listener
	public void propertyChange(java.beans.PropertyChangeEvent pe) {
		System.out.println("received event or callback from "
				+ pe.getPropagationId() + " with "+ pe.getPropertyName() );
	}
	
	public static DSPinInfo getCapturePin(DSFilterInfo filter) {
		DSPinInfo[] pins = filter.getPins();
		if (pins == null || pins.length != 2)
			return null;

		DSPinInfo supposedSmartTeeCapturePin = null;
		int modes = 0;
		for (DSPinInfo pin : pins) {
			int cur = pin.getFormats().length;
			if (cur > modes) {
				supposedSmartTeeCapturePin = pin;
				modes = cur;
			}
		}
		return supposedSmartTeeCapturePin;
	}

	public static boolean setPinWidth(DSPinInfo pin, int width) {
		return setPinWidth(pin, -1, width);
	}
	
	public static DSMediaType[] getPinAvailableFormats(DSPinInfo pin) {
		return pin.getFormats();
	}
	
	public static boolean setPinWidth(DSPinInfo pin, int type, int width) {
		DSMediaType[] formats = getPinAvailableFormats(pin);
		int index = 0;
		for (DSMediaType format : formats) {
			if ((type == -1 || type == format.getSubType())
					&& format.getWidth() == width) {
				pin.setPreferredFormat(index);
				return true;
			}
			index++;
		}
		return false;
	}
}
