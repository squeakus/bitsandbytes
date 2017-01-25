package ie.errity.pd.graphics;

import ie.errity.pd.*;
//Import GUI components
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

/**
 * Main program frame, contains a menubar with options to configure rules 
 * settings, manage saved strategies and select the current type of game.
 * @author	Andrew Errity
 */
public class MenuFrame extends JFrame implements ActionListener
{	
	//Swing Components	
   	JMenuBar menuBar;
   	JMenu type, rulesMen, stratMen;
   	JPanel blank;
   	JRadioButtonMenuItem rbMenuItem;
   	JMenuItem menuItem;
   	
   	//Game panels
   	InteractivePanel inter;
   	TournamentPanel tournament;
   	SpatialPanel spatial;
   	
   	//Dialogs
   	PrisonerDialog pDlg;
   	RulesDialog rulesDlg;
   	
   	//Strategy manager variables
   	private Prisoner[] savedPris;
   	private String[] namePris;
   	private int mem;	//max number of saved strategies
   	
   	
   	/**
   	 *Create a new frame
   	 *@param s frame's title
   	 */
	public MenuFrame(String s)
 	{   
 		super(s);
		//Create the menu bar.
	    menuBar = new JMenuBar();
	    setJMenuBar(menuBar);
	    buildMenu();
	    
	    //Blank screen when first run
	    blank = new JPanel();
	    blank.setPreferredSize(new Dimension(800, 600));
		setContentPane(blank);
		
		rulesDlg = new RulesDialog(this);
		
		tournament = new TournamentPanel(this);
		spatial = new SpatialPanel(this);

		//Lists of saved prisoners
		mem = 10;
		savedPris = new Prisoner[mem];
		savedPris[0] = new Prisoner("TFT");
		savedPris[1] = new Prisoner("TFTT");
		savedPris[2] = new Prisoner("ALLC");
		savedPris[3] = new Prisoner("ALLD");
		for(int m = 4; m < mem; m++)
			savedPris[m] = null;
		
		namePris = new String[mem];
		namePris[0] = "TFT";
		namePris[1] = "TFTT";
		namePris[2] = "ALLC";
		namePris[3] = "ALLD";
		for(int m = 4; m < mem; m++)
			namePris[m] = "Empty";
		
		
		pDlg = new PrisonerDialog(this, "Saved Strategies",namePris);
		inter = new InteractivePanel(this,namePris);
	}   

	/**
	 *ActionListener listens for menubar actions
	 */
	public void actionPerformed(ActionEvent e) 
	{
		if ("Rules".equals(e.getActionCommand()))
		{
			rulesDlg.showDlg();	//open rules Dialog
		}
		else if ("Tournament".equals(e.getActionCommand()))
		{
			rulesDlg.type("Tournament");
			spatial.stop();		//ensure all background threads are closed
			setContentPane(tournament);	//switch to Tournament window
			pack();
		}
		else if ("Spatial".equals(e.getActionCommand()))
		{
			rulesDlg.type("Spatial");
			tournament.stop(); 	//ensure all background threads are closed
			setContentPane(spatial);	//switch to Spatial window
			pack();
		}
		else if ("Interactive".equals(e.getActionCommand()))
		{
			rulesDlg.type("Interactive");
			tournament.stop(); 	//ensure all background threads are closed
			spatial.stop();		//ensure all background threads are closed
			setContentPane(inter);		//switch to Interactive window
			pack();
		}
		else if ("Strategy".equals(e.getActionCommand()))
		{
			pDlg.showDlg();		//open strategy manager Dialog
		}
	}
	
	private void buildMenu()
	{
		//Create Game Type menu
	    type = new JMenu("Game Type");
	    type.setMnemonic(KeyEvent.VK_G);
	    type.getAccessibleContext().setAccessibleDescription(
	            "Game Type Menu");
	    menuBar.add(type);
	
	    ButtonGroup group = new ButtonGroup();
	    rbMenuItem = new JRadioButtonMenuItem("Tournament");
	    rbMenuItem.setMnemonic(KeyEvent.VK_T);
	    rbMenuItem.setActionCommand("Tournament");
	    rbMenuItem.addActionListener(this);
	    group.add(rbMenuItem);
	    type.add(rbMenuItem);
	
	    rbMenuItem = new JRadioButtonMenuItem("Spatial Society");
	    rbMenuItem.setMnemonic(KeyEvent.VK_S);
	    rbMenuItem.setActionCommand("Spatial");
	    rbMenuItem.addActionListener(this);
	    group.add(rbMenuItem);
	    type.add(rbMenuItem);
	    
	    rbMenuItem = new JRadioButtonMenuItem("Interactive");
	    rbMenuItem.setMnemonic(KeyEvent.VK_I);
	    rbMenuItem.setActionCommand("Interactive");
	    rbMenuItem.addActionListener(this);
	    group.add(rbMenuItem);
	    type.add(rbMenuItem);
	    
	    
	    //Build the rules menu.
	    rulesMen = new JMenu("Rules");
	    rulesMen.setMnemonic(KeyEvent.VK_R);
	    rulesMen.getAccessibleContext().setAccessibleDescription(
	            "Rules Menu");
	    menuBar.add(rulesMen);

	    menuItem = new JMenuItem("Rule Settings...");
	    menuItem.setActionCommand("Rules");
	    menuItem.setMnemonic(KeyEvent.VK_U);
	    menuItem.addActionListener(this);
	    rulesMen.add(menuItem);
	    
	    //Build the strategy menu.
	    stratMen = new JMenu("Strategies");
	    stratMen.setMnemonic(KeyEvent.VK_A);
	    stratMen.getAccessibleContext().setAccessibleDescription(
	            "Strategy Menu");
	    menuBar.add(stratMen);

	    menuItem = new JMenuItem("Strategy Manager...");
	    menuItem.setActionCommand("Strategy");
	    menuItem.setMnemonic(KeyEvent.VK_M);
	    menuItem.addActionListener(this);
	    stratMen.add(menuItem);
	}
	
	/**
	 *Set the {@link  ie.errity.pd.Rules Rules} in each subwindow
	 *@param r new {@link  ie.errity.pd.Rules Rules}
	 */
	public void setRules(Rules r)
	{
		tournament.setRules(r);
		spatial.setRules(r);
		inter.setRules(r);	
	}

	/**
	 *Removes a saved strategy
	 *@param i index of element to remove
	 */	
	public void delete(int i)
	{
		namePris[i] = "Empty";
		savedPris[i] = null;
		pDlg.delete(i);
		inter.delete(i);
	}
	
	
	/**
	 *Display the selected strategy in a dialog box (tied to a dialog)
	 *@param i 	index of selected strategy
	 *@param d	dialog which opens this dialog
	 */
	public void view(int i, JDialog d)
	{
		if(savedPris[i] != null)	//if empty don't display
		{
			StrategyDialog sDlg = new StrategyDialog(this,d,namePris[i]);
			sDlg.setStrat(savedPris[i]);
			sDlg.showDlg();
		}
	}
	
	/**
	 *Display the selected strategy in a dialog box
	 *@param i 	index of selected strategy
	 */
	public void view(int i)
	{
		if(savedPris[i] != null)
		{
			StrategyDialog sDlg = new StrategyDialog(this,namePris[i]);
			sDlg.setStrat(savedPris[i]);
			sDlg.showDlg();
		}
	}
	
	/**
	 *Save {@link  ie.errity.pd.Prisoner Prisoner} in saved {@link  ie.errity.pd.Prisoner Prisoners} list
	 *@param s	name to save as
	 *@param p	{@link  ie.errity.pd.Prisoner Prisoner} to save
	 */
	public void save(String s, Prisoner p)
	{
		for(int i = 0; i < mem; i++)
			if(savedPris[i] == null)
			{
				savedPris[i] = p;
				namePris[i] = s;
				pDlg.update(i,s);
				inter.update(i,s);
				return;
			}
		//if no free spaces, just ignore
	}
	
	/**
	 * Check if there is a free space in the {@link  ie.errity.pd.Prisoner Prisoners} list
	 *@return true if a free space exists, false if not
	 */
	public boolean freeSpace()
	{
		for(int i = 0; i < mem; i++)
			if(savedPris[i] == null)
			{
				return true;
			}
		return false; //no free spaces
	}
	
	/**
	 *Get {@link  ie.errity.pd.Prisoner Prisoner} from saved {@link  ie.errity.pd.Prisoner Prisoners} list
	 *@param i	index of {@link  ie.errity.pd.Prisoner Prisoner} in list
	 *@return {@link  ie.errity.pd.Prisoner Prisoner} retrieved
	 */
	public Prisoner getPrisoner(int i){	return savedPris[i];}
	
	/**
	 *Get {@link  ie.errity.pd.Prisoner Prisoner} from saved {@link  ie.errity.pd.Prisoner Prisoners} list
	 *@param s 	{@link  ie.errity.pd.Prisoner Prisoner's} name
	 *@return	{@link  ie.errity.pd.Prisoner Prisoner} or <code>null</code> if no match
	 */
	public Prisoner getPrisoner(String s)
	{	
		for(int i = 0; i < mem; i++)
			if(s.equals(namePris[i]))
				return savedPris[i];
		return null;
	}
}