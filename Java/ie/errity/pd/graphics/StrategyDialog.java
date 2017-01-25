package ie.errity.pd.graphics;

import ie.errity.pd.*;
//Import GUI components
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.plaf.metal.*;

/**
 *Dialog displaying a {@link  ie.errity.pd.Prisoner Prisoner's} strategy
 *@author Andrew Errity
 */
public class StrategyDialog extends JPanel
{
	//Swing Components
	JLabel stratLbl;
	JButton closeBtn;
	JDialog stratDlg;
	JButton saveBtn;
	
	private String strat;
	private Prisoner pris;
	
	/**
	 *Create a new Dialog displaying a strategy and the options
	 *to save the strategy or close
	 *@param frame	application master frame
	 *@param title 	dialog's title
	 */
	public StrategyDialog(final JFrame frame, String title)
	{
		// Add border around the panel.
		setBorder(BorderFactory.createEmptyBorder(5,5,5,10));
		setLayout(new GridLayout(0,1));
		
        closeBtn = new JButton("Close");
        closeBtn.setMnemonic(KeyEvent.VK_C);
		closeBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) 
            {
                stratDlg.setVisible(false);
            }
        });
        
        saveBtn = new JButton("Save");
        saveBtn.setMnemonic(KeyEvent.VK_S);
		saveBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) 
            {
            	MenuFrame mf = (MenuFrame)frame;
            	Prisoner p1 = pris; //save current player, dialog will still update according to game
            	//rules but this ensures the player that was visisble when save was clicked is saved
               	
               	if(mf.freeSpace())
               	{
               		SaveDialog sD = new SaveDialog(mf,stratDlg,"Enter a strategy name");
	               	sD.showDlg();
	                String s = sD.getText();
	             	  
	                if(s != null)
	                	mf.save(s,(new Prisoner(s,p1.getStrat())));
	            }
                else	
                	JOptionPane.showMessageDialog(null, "Strategy List is Full.\nPlease delete a saved Strategy in the Strategy manager to free a space.","Strategy List",  JOptionPane.ERROR_MESSAGE); 
            }
        });

		stratLbl = new JLabel(strat);
		add(stratLbl);
		add(saveBtn);
		add(closeBtn);
		
		// THEMES
		// user selected theme
		MetalTheme theme = new EmeraldTheme();  
		// set the chosen theme
		MetalLookAndFeel.setCurrentTheme(theme);		
		try
		{
		    UIManager.setLookAndFeel(
			UIManager.getCrossPlatformLookAndFeelClassName());
		    SwingUtilities.updateComponentTreeUI(frame);
		}
		catch(Exception e){}
			
		JDialog.setDefaultLookAndFeelDecorated(true);
		stratDlg = new JDialog(frame,"Strategy - " + title, false);
		stratDlg.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		stratDlg.setContentPane(this);
	}
	
	/**
	 *Create a new Dialog displaying a strategy
	 *@param frame	application master frame
	 *@param d		dialog which this dialog is opened from
	 *@param title 	dialog's title
	 */
	public StrategyDialog(final JFrame frame, JDialog d, String title)
	{
		// Add border around the panel.
		setBorder(BorderFactory.createEmptyBorder(5,5,5,10));
		setLayout(new GridLayout(0,1));
		
		
        closeBtn = new JButton("Close");
        closeBtn.setMnemonic(KeyEvent.VK_C);
		closeBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) 
            {
                stratDlg.setVisible(false);
            }
        });

		stratLbl = new JLabel(strat);
		this.add(stratLbl);
		this.add(closeBtn);
		
		
		// THEMES
		// user selected theme
		MetalTheme theme = new EmeraldTheme();  
		// set the chosen theme
		MetalLookAndFeel.setCurrentTheme(theme);		
		try
		{
		    UIManager.setLookAndFeel(
			UIManager.getCrossPlatformLookAndFeelClassName());
		    SwingUtilities.updateComponentTreeUI(frame);
		}
		catch(Exception e){}
			
		JDialog.setDefaultLookAndFeelDecorated(true);
		stratDlg = new JDialog(d,"Strategy - " + title, false);
		stratDlg.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		stratDlg.setContentPane(this);
	}


	/**
	 *Set the displayed text
	 *@param p a prisoner whose strategy will be displayed
	 */
	public void setStrat(Prisoner p)
	{ 
		pris = p;
		stratLbl.setText(p.toString());
	}
	
	/**
	 *Opens the dialog
	 */
	public void showDlg()
	{
			stratDlg.pack();
			stratDlg.setVisible(true);
	}
	
	/**
	 *Closes the dialog
	 */
	public void hideDlg()
	{
			stratDlg.setVisible(false);
	}
		

}
	