package ui;

////////////////////////////////////////////////////////////////////////////////
// Undo/redo manager using simple array for ring buffer
////////////////////////////////////////////////////////////////////////////////
abstract public class UndoRedoManager {
	
	UndoRedoManager(int max_status){
		MAX_STATUS = max_status;
	}
	
	abstract void copy();
	
	public  void    reset()   { CURRENT_DEPTH=0; MAXIMUM_DEPTH = 0; }
	public  boolean undoable(){ return (CURRENT_DEPTH==0            ) ? false : true; }
	public  boolean redoable(){ return (CURRENT_DEPTH>=MAXIMUM_DEPTH) ? false : true; }
	
	public void undo(){
		if(!undoable()){
			System.out.println("invalid operation: undo");
			return ;
		}
		
		CURRENT_INDEX = (MAX_STATUS + CURRENT_INDEX - 1) % MAX_STATUS;		
		CURRENT_DEPTH--;
	}
	
	public void redo(){
		if( !redoable() ){
			System.out.println("invalid operation: redo");
			return ;
		}
		
		CURRENT_INDEX = (MAX_STATUS + CURRENT_INDEX + 1) % MAX_STATUS;
		CURRENT_DEPTH++;
	}
	
	public void stack(){
		int NEXT = (CURRENT_INDEX+1) % MAX_STATUS;
		
		copy();
		
		CURRENT_INDEX = NEXT;
		CURRENT_DEPTH = (CURRENT_DEPTH+1 == MAX_STATUS) ? CURRENT_DEPTH : CURRENT_DEPTH + 1;
		MAXIMUM_DEPTH = CURRENT_DEPTH;
	}
	
	protected int getNextIndex   (){ return (           CURRENT_INDEX+1) % MAX_STATUS; }
	protected int getPrevIndex   (){ return (MAX_STATUS+CURRENT_INDEX-1) % MAX_STATUS; }
	protected int getCurrentIndex(){ return             CURRENT_INDEX;                 }
	
	protected int MAX_STATUS;
	private   int CURRENT_INDEX = 0;
	private   int CURRENT_DEPTH = 0;
	private   int MAXIMUM_DEPTH = 0;
}
