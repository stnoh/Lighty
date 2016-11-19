package model;

////////////////////////////////////////////////////////////
// Tuple class for returning 2 arguments
////////////////////////////////////////////////////////////
public class Tuple{
	private float diff;
	private int[] conf;
	
	public Tuple(float diff, int[] conf){
		this.diff = diff;
		this.conf = conf;
	}
	
	public float getDiff(){ return this.diff; }
	public int[] getConf(){ return this.conf; }
}
