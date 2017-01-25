// local tester/visualizer/reference solution for TopCoder Marathon Match
// NASA NTL longeron challenge
// by Rustyoldman and mystic_tc

import java.io.* ;
import java.util.* ;
import java.awt.* ;
import java.awt.image.* ;
import javax.swing.* ;
import javax.imageio.* ;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;

import java.text.* ;

// *********************************************




/* ******************************************************************************************
 CCC   OOO  N   N  SSS  TTTTT RRRR   AAA  III N   N TTTTT  SSS   CCC  H   H EEEEE  CCC  K   K EEEEE RRRR  
C     O   O NN  N S       T   R   R A   A  I  NN  N   T   S     C     H   H E     C     K  K  E     R   R 
C     O   O N N N  SSS    T   RRRR  AAAAA  I  N N N   T    SSS  C     HHHHH EEE   C     KK    EEE   RRRR  
C     O   O N  NN     S   T   R R   A   A  I  N  NN   T       S C     H   H E     C     K K   E     R R   
 CCC   OOO  N   N  SSS    T   R  R  A   A III N   N   T    SSS   CCC  H   H EEEEE  CCC  K  K  EEEEE R  R  
 ****************************************************************************************** */



// =============================================
class ConstraintsChecker {
// =============================================
// Minimum and maximum possible beta angles, in degrees.
static final double MIN_BETA = -90.0;
static final double MAX_BETA = 90.0;

// Minimum and maximum valid yaw angles, in degrees.
static final double MIN_YAW = 0.0;
final double MAX_YAW = 7.0;

// Number of intermediate evaluation points along a single orbit.
static final int NUM_STATES = 92;

// Solar array names
static final String[] SOLAR_ARRAY_NAME = new String[] 
   {"1A", "2A", "3A", "4A", "1B", "2B", "3B", "4B"};

// SARJ axe names
static final String[] SARJ_NAME = new String[]
   {"SSARJ", "PSARJ"};

// Time (in seconds) between two consecutive evaluation points.
static final double TIME_PER_STATE = 60.0;

// Maximum BGA/SARJ rotation speed, in degrees per second.
static final double MAX_BGA_SPEED = 0.25;
static final double MAX_SARJ_SPEED = 0.15;

// Maximum BGA/SARJ acceleration, in degrees per second^2.
static final double MAX_BGA_ACC = 0.01;
static final double MAX_SARJ_ACC = 0.005;

// Number of strings and longerons in a single solar array.
static final int NUM_STRINGS = 41 * 2;
static final int NUM_LONGERONS = 4;

// Minimum shadow fraction at which a longeron is considered shadowed.
static final double MIN_LONGERON_SHADOW = 0.1;

// Minimum value of a counter which is considered to be a failure.
static final double MIN_FAILURE_HEAT = 21;

// The multiplier from the power formula, in watt per meter^2.
static final double POWER_MULTIPLIER = 1371.3 * 0.1;

// String area that generates power, in meter^2. Currently assumed 
// to be 400 cells of 8 cm x 8 cm dimensions.
static final double STRING_AREA = 400 * 0.08 * 0.08;

// below is the order of constraints for 1A 2A 3A... ordering
static final double[] MIN_POWER_REQ = new double[] {
   6300.0, 6400.0, 6700.0, 8900.0,
   5500.0, 8100.0, 4400.0, 4300.0
};

// tolerance
static final double EPS = 1e-9;

// end of constants

java.util.List<String> errors = new ArrayList<String>();
java.util.List<String> lowPowerBGAs = new ArrayList<String>();

double beta, yaw;

double[][] angleSARJ = new double[NUM_STATES][2];
double[][] speedSARJ = new double[NUM_STATES][2];
double[][] angleBGA = new double[NUM_STATES][8];
double[][] speedBGA = new double[NUM_STATES][8];

double[][] rotSARJ = new double[NUM_STATES][2];
double[][] rotBGA = new double[NUM_STATES][8];

double[][] cosTheta = new double[NUM_STATES][8];
double[][][] stringShadow = new double[NUM_STATES][8][NUM_STRINGS];
double[][][] longeronShadow = new double[NUM_STATES][8][NUM_LONGERONS];

double[][] power = new double[NUM_STATES][8];
double[] totalPower = new double[8];
int [] maxCounter = new int [8] ;

double[] lastAns = null; // return value from last evaluateSingleState call

// set beta angle, no checks
// =============================================
void setBeta(double beta)
// =============================================
{
if ( beta < MIN_BETA || beta > MAX_BETA )
   {
   errors.add("The value of beta angle must be between " + MIN_BETA +
		   " and " + MAX_BETA + " degrees, inclusive.");
   }
this.beta = beta;
}

// set yaw angle, check validity
// =============================================
void setYaw(double yaw)
// =============================================
{
if ( yaw < MIN_YAW || yaw > MAX_YAW )
   {
   errors.add("The value of yaw angle must be between " + MIN_YAW + 
                    " and " + MAX_YAW + " degrees, inclusive.");
   }

this.yaw = yaw;
}

// Sets data (angles, speeds) for minute = t (0 <= t < NUM_STATES).
// Makes trivial checks.
// =============================================
void setDataForFrame(int t, double[] data)
// =============================================
{
if (data.length != 20) 
   {
   errors.add("Return value from getStateAtMinute must " +
                    "contain exactly 20 elements. Minute = " + t + ".");
   return;
   }
   
for (int i=0; i < data.length; i++)
   if (Double.isNaN(data[i]))
      {
      errors.add("NaN elements in return value are not " +
                       "allowed. Minute = " + t + ". Position (0-based) = " + i + ".");
      return;
      }

angleSARJ[t][0] = data[0]; angleSARJ[t][1] = data[2];
speedSARJ[t][0] = data[1]; speedSARJ[t][1] = data[3];
for (int i=4, j=0; j < 8; j++, i += 2) 
   {
   angleBGA[t][j] = data[i];
   speedBGA[t][j] = data[i+1];
   }
   
for (int i=0; i < angleSARJ[t].length; i++)
   if (angleSARJ[t][i] < 0.0 || angleSARJ[t][i] >= 360.0) 
      {
      errors.add("Each SARJ angle needs to be from 0.0 " +
                       "(inclusive) to 360.0 (exclusive). Minute = " + 
                          t + ". SARJ = " + SARJ_NAME[i] + ". Your angle = " + angleSARJ[t][i] + ".");
      }
   
for (int i=0; i < angleBGA[t].length; i++)
   if (angleBGA[t][i] < 0.0 || angleBGA[t][i] >= 360.0) 
      {
      errors.add("Each BGA angle needs to be from 0.0 " +
                       "(inclusive) to 360.0 (exclusive). Minute = " + 
                       t + ". BGA = " + SOLAR_ARRAY_NAME[i] + ". Your angle = " + angleBGA[t][i] + ".");
      }
   
for (int i=0; i < speedSARJ[t].length; i++)
   if (Math.abs(speedSARJ[t][i]) > MAX_SARJ_SPEED) 
      {
      errors.add("The absolute value of SARJ rotation speed " +
                          "can't exceed " + MAX_SARJ_SPEED + 
                          ". Minute = " + t + ". SARJ = " + SARJ_NAME[i] + ". Your speed = " + speedSARJ[t][i] + ".");
      }
   
for (int i=0; i < speedBGA[t].length; i++)
   if (Math.abs(speedBGA[t][i]) > MAX_BGA_SPEED) 
      {
      errors.add("The absolute value of BGA rotation speed " +
                 " can't exceed " + MAX_BGA_SPEED + 
                 ". Minute = " + t + ". BGA = " + SOLAR_ARRAY_NAME[i] + ". Your speed = " + speedBGA[t][i] + ".");
      }
}

// Checks that angles and speeds are consistent.
// =============================================
void checkAnglesAndSpeeds()
// =============================================
{
for (int t=0; t < NUM_STATES-1; t++) 
   {
   int tt = (t + 1) % NUM_STATES;
   for (int i=0; i < 2; i++) 
      {
      double[] ret = processTransition(angleSARJ[t][i], speedSARJ[t][i], 
                                       angleSARJ[tt][i], speedSARJ[tt][i], 
                                       MAX_SARJ_SPEED, MAX_SARJ_ACC);
      if (ret[0] != 1.0) 
         {
         errors.add("It is impossible to make a transition between" +
                          " your SARJ configurations from minute " + t +
                          " to minute " + tt + ". SARJ = " + SARJ_NAME[i] + ".");
         }
      rotSARJ[t][i] = ret[1];
      }
   
   for (int i=0; i < 8; i++) 
      {
      double[] ret = processTransition(angleBGA[t][i], 
                                       speedBGA[t][i], 
                                       angleBGA[tt][i], 
                                       speedBGA[tt][i], 
                                       MAX_BGA_SPEED, 
                                       MAX_BGA_ACC);
      if (ret[0] != 1.0) 
         {
         errors.add("It is impossible to make a transition" +
                          " between your BGA configurations from minute " + 
                          t + " to minute " + tt + ". BGA = " + 
                    SOLAR_ARRAY_NAME[i] + ".");
         }
      rotBGA[t][i] = ret[1];
      }
   }
}

// Evaluates a particular frame.
// =============================================
void evaluateFrame(int t)
// =============================================
{
evaluateFrame ( t , t * 360.0 / NUM_STATES ) ;
}
// =============================================
void evaluateFrame(int t, double alpha)
// =============================================
{
double[] inp = new double[] 
   {
      t, alpha, beta, yaw,
      angleSARJ[t][0], angleSARJ[t][1],
      angleBGA[t][0], angleBGA[t][1], angleBGA[t][2], angleBGA[t][3],
      angleBGA[t][4], angleBGA[t][5], angleBGA[t][6], angleBGA[t][7]
   };
   
// lt.addFatalError ( "about to call evaluateSingleState\n" ) ;
// =============================================
   double[] out = evaluateSingleState(inp);
// =============================================
//lt.addFatalError ( "returned from evaluateSingleState\n" ) ;
   if ( out == null )
      {
      errors.add ( "returned from evaluateSingleState, " +
                         "Error: result = null\n" ) ;
      return;
      }

   int pos = 0;
   
   for (int i = 0 ; i < 8 ; i++ )
      {
      cosTheta[t][i] = out[pos++];
      //      if ( cosTheta [t][i] <= 0 )
      //         lt.addFatalError ( "Error SAW " + i + " cosine " + 
      //                      cosTheta[t][i] + "\n" ) ;
      // if ( t == 0 ) 
      //    lt.addFatalError ( "SAW " + i + " cosine " + 
      //                       cosTheta[t][i] + "\n" ) ;
      }
   
   
   for ( int i = 0; i < 8 ; i++ )
      for ( int j = 0; j < NUM_STRINGS ; j++ )
         {
         stringShadow[t][i][j] = out[pos++];
         if ( stringShadow[t][i][j] < 0 || stringShadow[t][i][j] > 1 )
            {
            // lt.addFatalError ( "Error: SAW " + i + " string " + j +
            //                    " shadow " + stringShadow[t][i][j] ) ;
            }
         }
   
   for ( int i = 0; i < 8 ; i++ )
      for (int j = 0; j < NUM_LONGERONS ; j++ )
         longeronShadow[t][i][j] = out[pos++];
}

// Checks whether any longerons failed.
// =============================================
int [] checkLongerons()
// =============================================
{
for ( int id = 0 ; id < 8 ; id++ ) 
   {
   int[] add = new int [NUM_STATES] ;
   for ( int t = 0 ; t < NUM_STATES ; t++ ) 
      {
      int shadowCnt = 0 ;
      for ( int i = 0; i < NUM_LONGERONS ; i++ )
         if ( longeronShadow[t][id][i] >= MIN_LONGERON_SHADOW )
            shadowCnt++;
      
      add[t] = ( shadowCnt % 2 == 1 ? 1 : -1 ) ;
      }
   
   int counter = 0 ;
   // simulate 100 rotations around the orbit, since it works quick anyway
   boolean ok = true ;
   for ( int x = 0 ; x < 100 ; x++ )
      {
      for ( int t = 0; t < NUM_STATES ; t++ ) 
         {
         counter += add[t];
         if ( counter < 0 ) counter = 0;
         maxCounter[id] = Math.max ( maxCounter[id] , counter ) ;
         if ( counter >= MIN_FAILURE_HEAT )
            {
            if ( ok ) 
               errors.add ( "Longerons of solar array " +
                            SOLAR_ARRAY_NAME[id] + " failed. Minute = " + 
                            (t + x * NUM_STATES) + ".") ;
            ok = false ;
            }
         }
      }
   }
return maxCounter ;
}
// Evaluates power for frame t.
// =============================================
void evaluatePower ( int t )
// =============================================
{
for ( int id = 0; id < 8 ; id++ ) 
   {
   power[t][id] = 0 ;
   for ( int i = 0; i < NUM_STRINGS ; i++ )
      {
      double cosFactor = Math.max ( 0.0 , cosTheta[t][id] ) ;
      power[t][id] += cosFactor * 
         fractionToFactor(stringShadow[t][id][i]);
      }
   
   power[t][id] *= POWER_MULTIPLIER * STRING_AREA;
   //   ISS.longtest.addFatalError ( power[t][id] + " " ) ;
   }
//ISS.longtest.addFatalError ( "\n" ) ;
}

// Aggregates all powers. Checks minimum power requirements.
// =============================================
void aggregateAndCheckPower()
// =============================================
{
for ( int id = 0; id < 8 ; id++ ) 
   {
   totalPower[id] = 0 ;
   for ( int t = 0 ; t < NUM_STATES ; t++ )
      {
      totalPower[id] += power[t][id];
      }
   
   totalPower[id] /= NUM_STATES ;
   
   if (totalPower[id] < MIN_POWER_REQ[id]) 
      {
      lowPowerBGAs.add(SOLAR_ARRAY_NAME[id]);
      }
   }
}

// Verifies whether a transition from (angle1, speed1) to 
// (angle2, speed2) is possible within TIME_PER_STATE seconds 
// assuming |speed| <= maxSpeed and |acceleration| <= maxAcc.
// If it is possible, element 0 of return will be 1.0, otherwise 0.0.
// Element 1 of return will contain the minimum amount of rotation 
// necessary to make such transition.
// (This is only defined if transition is possible.)
// =============================================
double[] processTransition(double angle1, double speed1, 
                           double angle2, double speed2, 
                           double maxSpeed, double maxAcc) 
// =============================================
{
// make sure speed1 >= 0, to reduce the number of cases
if (speed1 < 0)
   return processTransition ( -angle1, -speed1, 
                              -angle2, -speed2, maxSpeed, maxAcc ) ;

// determine angular shift we need to make
double shift = angle2 - angle1;
if (shift < -180.0) 
   shift += 360.0;
if (shift > 180.0)
   shift -= 360.0;

// check validity
double[] res = new double[] {0.0, 0.0};

if (!canMakeTransition(shift, speed1, speed2, -maxSpeed, maxSpeed, maxAcc))
   return res;

res[0] = 1.0;
res[1] = Math.abs(shift);

// check if some extra rotation is needed
double pos, neg;
if (speed2 >= 0) 
   {
   // can we always keep speed positive?
   if (canMakeTransition(shift, speed1, speed2, 0, maxSpeed, maxAcc))
      return res;
   
   // minimize rotation at positive speeds
   double t1 = speed1 / maxAcc;
   double t2 = speed2 / maxAcc;
   pos = path(t1, 0, maxAcc) + path(t2, 0, maxAcc);
   neg = pos - shift;
   } 
else 
   {
   double t1 = speed1 / maxAcc;
   double t2 = (-speed2) / maxAcc;
   pos = path(t1, 0, maxAcc);
   neg = path(t2, 0, maxAcc);
   double midShift = pos - neg;
   if (shift < midShift) 
      {
      // minimize rotation at positive speeds
      neg = pos - shift;
      }
   else 
      {
      // minimize rotation at negative speeds
      pos = neg + shift;
      }
   }

res[1] = pos + neg;

return res;
}

// Verifies whether it's possible to make an angular shift of "shift" 
// within TIME_PER_STATE seconds
// if initial speed is "speed1", final speed is "speed2", 
// minSpeed <= speed <= maxSpeed and |acceleration| <= maxAcc.
// =============================================
boolean canMakeTransition ( double shift, double speed1, 
                            double speed2, double minSpeed, 
                            double maxSpeed, double maxAcc ) 
// =============================================
{
// simple acceleration check
if (Math.abs((speed2 - speed1) / TIME_PER_STATE) > maxAcc + EPS)
   return false;

double minShift = 0.0, maxShift = 0.0;

// find minimum possible angular shift
double t1 = (speed1 - minSpeed) / maxAcc;
double t2 = TIME_PER_STATE - (speed2 - minSpeed) / maxAcc;
if (t1 <= t2) 
   {
   minShift += path(t1, speed1, -maxAcc);
   minShift += path(t2 - t1, minSpeed, 0);
   minShift += path(TIME_PER_STATE - t2, minSpeed, maxAcc);
   }
else
   {
   double t = (speed1 - speed2 + TIME_PER_STATE * maxAcc) / 2.0 / maxAcc;
   minShift += path(t, speed1, -maxAcc);
   minShift += path(TIME_PER_STATE - t, speed1 - maxAcc * t, maxAcc);
   }

// find maximum possible angular shift
t1 = (maxSpeed - speed1) / maxAcc;
t2 = TIME_PER_STATE - (maxSpeed - speed2) / maxAcc;
if (t1 <= t2) 
   {
   maxShift += path(t1, speed1, maxAcc);
   maxShift += path(t2 - t1, maxSpeed, 0);
   maxShift += path(TIME_PER_STATE - t2, maxSpeed, -maxAcc);
   } 
else
   {
   double t = (speed2 - speed1 + TIME_PER_STATE * maxAcc) / 2.0 / maxAcc;
   maxShift += path(t, speed1, maxAcc);
   maxShift += path(TIME_PER_STATE - t, speed1 + maxAcc * t, -maxAcc);
   }

// validate
return (minShift <= shift && shift <= maxShift +EPS);
}

// Returns path, given time, initial speed and acceleration.
// =============================================
double path(double t, double v0, double a) 
// =============================================
{
return v0 * t + a * t * t / 2.0;
}

// Inputs:
// - Frame ID. 0-based ID of the frame.
// - Alpha angle. This and all subsequent angles are in degrees.
//                Thus at time t, 0 <= t < NUM_STATES, alpha angle is 
//   equal to t / NUM_STATES * 360.0.
// - Beta angle.
// - Yaw angle.
// - 2 SARJ angles.
// - 8 BGA angles (in order: 1A, 2A, 3A, 4A, 1B, 2B, 3B, 4B.
// Outputs:
// - 8 cos(theta) values used for power calculation 
//
// - NUM_STRINGSx8 shadow fractions (from 0.0 to 1.0) of individual strings 
//   (first NUM_STRINGS values for 1A, then for 2A, then for 3A and so on).
//
// - NUM_LONGERONSx8 shadow fractions (from 0.0 to 1.0) of individual 
//   longerons (first NUM_LONGERONS values for 1A, then for 2A, then for 3A and so on).
// =============================================
double[] evaluateSingleState(double[] input) 
// =============================================
{
double minute = input[0] ;                                              
ISSVis.alpha  = input[1] ;
ISSVis.beta   = input[2] ;
ISSVis.yaw    = input[3] ;

for ( int i = 0 ; i < 10 ; i ++ )
   ISSVis.control [i] = input [i+4] ;

// set longerons invisible
ISSVis.longeron_visibility ( ISSVis.m , false ) ;

// set up position of sun
ISSVis.sunTransform = ISSVis.makeSunProjection ( ISSVis.beta , ISSVis.alpha ) ;
ISSVis.inverse = ISSVis.makeInverseSunProjection ( ISSVis.beta , ISSVis.alpha ) ;
ISSVis.toSun = new V ( ) ;
ISSVis.sunTransform.transform ( new V ( 0 , -1 , 0 ) , ISSVis.toSun ) ;

// rotate joints
ISSVis.rotate_SARJ_BGA ( ISSVis.m , ISSVis.control ) ;
ISSVis.m.transform ( ) ;

// go

double [] answer = ISSVis.calculateOnePosition ( ISSVis.m , ISSVis.toSun , 
                                                 ISSVis.inverse ) ;

if ( minute < -1 )
   {
   for ( int saw = 0 ; saw < 8 ; saw ++ )
      {
      double p = 0 ;
      for ( int str = 0 ; str < 82 ; str ++ )
         {
         p += answer[saw] * 104.96/41 * 1371.3 * 0.1 *
            Math.max(0.0,1.0 - answer[8+saw*82+str] * 5 ) ;
         }
      //p /= 82 ;
      }
   }

this.lastAns = answer;

return answer ;
}

// Convert shadow fraction to shadow factor.
// =============================================
double fractionToFactor(double x) 
// =============================================
{
return Math.max(0.0, 1.0 - 5.0 * x);
}
// =============================================
public double totalRotation ( int b ) 
// =============================================
{
double totalRotation = 0 ;
for ( int j = 0 ; j < 92 ; j ++ )
   totalRotation += rotBGA [j][b] ;
return totalRotation ;

}
// =============================================
public double rawScore ( ) 
// =============================================
{
double power = 0 ;
for ( int i = 0 ; i < 8 ; i ++ )
   power += totalPower[i] ;
double maxBGArotation = 0 ;
for ( int i = 0 ; i < 8 ; i ++ )
   {
   maxBGArotation = Math.max ( maxBGArotation , totalRotation(i) ) ;
   }

double score = power * Math.min(1.0 , Math.pow(2.0 , (80.0-maxBGArotation) / 300)) ;
for ( int i = 0 ; i < lowPowerBGAs.size() ; i ++ )
   {
   score /= 2.0;
   }
return score;
}
// =============================================
// *********************************************
} // end class ConstraintsChecker
// *********************************************









/* *********************************************
DDDD  RRRR   AAA  W   W EEEEE RRRR  
D   D R   R A   A W   W E     R   R 
D   D RRRR  AAAAA W W W EEE   RRRR  
D   D R R   A   A WW WW E     R R   
DDDD  R  R  A   A W   W EEEEE R  R  
***********************************************/



// *********************************************
class Drawer extends JFrame 
// *********************************************
{

/*
DDDD  RRRR   AAA  W   W EEEEE RRRR  K   K EEEEE Y   Y L     III  SSS  TTTTT EEEEE N   N TTTTT EEEEE RRRR  
D   D R   R A   A W   W E     R   R K  K  E      Y Y  L      I  S       T   E     NN  N   T   E     R   R 
D   D RRRR  AAAAA W W W EEE   RRRR  KK    EEE     Y   L      I   SSS    T   EEE   N N N   T   EEE   RRRR  
D   D R R   A   A WW WW E     R R   K K   E       Y   L      I      S   T   E     N  NN   T   E     R R   
DDDD  R  R  A   A W   W EEEEE R  R  K  K  EEEEE   Y   LLLLL III  SSS    T   EEEEE N   N   T   EEEEE R  R  
*/
// *********************************************
private class DrawerKeyListener extends KeyAdapter 
// *********************************************
{
// *********************************************
public void keyPressed(KeyEvent e) 
// *********************************************
{
if ( e.getKeyChar() == 'q' ||
     e.getKeyChar() == '' ||
     e.getKeyChar() == 'k' ||
     e.getKeyChar() == 'p') 
   {
   System.exit ( 0 ) ;
   }
if ( e.getKeyChar() == ' ') 
   {
   synchronized (paintMutex) 
      {
      animationMode = !animationMode;
      panel.repaint();
      }
   }
if ( e.getKeyCode() == KeyEvent.VK_LEFT ||
     e.getKeyCode() == KeyEvent.VK_DOWN ||
     e.getKeyChar() == 's' ||
     e.getKeyChar() == 'e' )
   {
   synchronized (paintMutex) 
      {
     animationMode = false ;
     curFrame = ( curFrame - 1 + frameCnt ) % frameCnt ;
     panel.repaint();
      }
   }
if ( e.getKeyCode() == KeyEvent.VK_RIGHT ||
     e.getKeyCode() == KeyEvent.VK_UP ||
     e.getKeyChar() == 'f' ||
     e.getKeyChar() == 'd' )
   
   {
   synchronized (paintMutex) 
      {
      animationMode = false ;
      curFrame = ( curFrame + 1 ) % frameCnt ;
      panel.repaint();
      }
   }
}
}

/*
DDDD  RRRR   AAA  W   W EEEEE RRRR  PPPP   AAA  N   N EEEEE L     
D   D R   R A   A W   W E     R   R P   P A   A NN  N E     L     
D   D RRRR  AAAAA W W W EEE   RRRR  PPPP  AAAAA N N N EEE   L     
D   D R R   A   A WW WW E     R R   P     A   A N  NN E     L     
DDDD  R  R  A   A W   W EEEEE R  R  P     A   A N   N EEEEE LLLLL 
 */
// *********************************************
private class DrawerPanel extends JPanel 
// *********************************************
{
public void paint(Graphics g) 
{
synchronized (paintMutex) 
   {
   if (curFrame < 0 || curFrame >= frameCnt) 
      {
      return;
      }
  
   BufferedImage bi = new BufferedImage(xz, yz, BufferedImage.TYPE_INT_RGB);
   
   byte [][] pix = movie[curFrame];
   for (int y=0; y < yz; ++y)
      for (int x=0; x < xz*3; x+=3) 
         {
         int p = (((int)pix[y][x]) & 255) * 65536 ;
         p +=    (((int)pix[y][x+1]) & 255) * 256 ;
         p +=    (((int)pix[y][x+2]) & 255) ;
         
         bi.setRGB(x/3, y, p);
         }
   
   g.drawImage(bi, 0, 0, null);
   
   g.setFont(new Font("Arial", Font.BOLD, 12));
   
   int s = 0 ;
   int w = 37 ;
   int st = 58 ;
   g.setColor ( new Color ( 250,0,0) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );
   g.setColor ( new Color ( 150,250,250) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );
   g.setColor ( new Color ( 250,100,100) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );
   g.setColor ( new Color ( 0,250,250) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );

   g.setColor ( new Color ( 230,230,150) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );
   g.setColor ( new Color ( 0,0,250) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );
   g.setColor ( new Color ( 230,230,0) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );
   g.setColor ( new Color ( 100,100,250) ) ;
   g.fillRect ( xz/2+st+(s++)*w , 5 , w , 20 );
   

   g.setColor(Color.BLACK);
   for ( int line = 0 ; line < titles[curFrame].length-1 ; ++ line )
      {
      g.drawChars ( titles[curFrame][line].toCharArray(), 
                    0, 
                    titles[curFrame][line].length(), 
                    xz / 2 , 
                    20+20*line);
      }
   if ( titles[curFrame].length > 0 ) 
      g.drawChars ( titles[curFrame][titles[curFrame].length-1].toCharArray() ,
                    0 ,
                    titles[curFrame][titles[curFrame].length-1].length() ,
                    xz / 2 ,
                    yz - 11 ) ;
   
   if ( footer != null )
      {
      for ( int line = 0 ; line < footer.length ; ++line )
         {
         g.drawChars ( footer[line].toCharArray() ,
                       0 ,
                       footer[line].length() ,
                       xz / 2 ,
                       yz - (31+20*line) ) ;
         }
      }
   
   // s = "Animation mode: " + (animationMode ? "on" : "off");
   // g.drawChars(s.toCharArray(), 0, s.length(), xz / 2, 40);
   
   drawer.setTitle(windowTitles[curFrame] );
   }
}
// *********************************************
} // class DrawerPanel
// *********************************************
	
byte [][][] movie = new byte [92][][];
String [][] titles = new String [92][0] ;
String[] windowTitles = new String[92];
String [] footer = null ;
public int frameCnt = 0, curFrame = -1;

private int xz, yz;

public boolean animationMode = true;

private static Drawer drawer;
private static DrawerPanel panel;

// *********************************************
private Drawer() 
// *********************************************
{
super();

panel = new DrawerPanel();
getContentPane().add(panel);

addKeyListener(new DrawerKeyListener());

xz = 2 * ISSVis.res;
yz = ISSVis.res;

setSize(xz+5, yz+30);
setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
setVisible(true);
}
	
public static Object paintMutex = new Object();

// *********************************************
public void setFrame(int frameId, byte[][] data, String [] title ,
                     String [] foot , String windowTitle ) 
// *********************************************
{
synchronized (paintMutex) 
   {   
   titles[frameId] = new String [title.length] ;
   if ( foot != null && foot.length > 0 )
      footer = foot.clone() ;
   for ( int line = 0 ; line < title.length ; ++ line )
      titles[frameId][line] = title[line] ;
   drawer.movie[frameId] = new byte [data.length][];
   for (int i=0; i < data.length; i++) 
      {
      drawer.movie[frameId][i] = data[i].clone();
      }
   drawer.windowTitles[frameId] = windowTitle;
   drawer.frameCnt = frameId + 1;
   drawer.curFrame = frameId;
   panel.repaint();
   }
}
// *********************************************
public void displayFrame(int frameId) 
// *********************************************
{
synchronized (paintMutex) 
   {
   drawer.curFrame = frameId;
   panel.repaint();
   }
}

// *********************************************
public static Drawer getDrawer() 
// *********************************************
{
if (drawer == null) 
   {
   drawer = new Drawer();
   }
return drawer;
}
// ************************************************************
} // end class Drawer
// ************************************************************






/* ************************************************************
BBBB  III 
B   B  I  
BBBB   I  
B   B  I  
BBBB  III 
************************************************************ */

// ************************************************************
class Bi {
// ************************************************************
final BufferedImage bi ;
// *********************************************
Bi ( ) 
// *********************************************
   {
   bi = null ;
   } ;
// *********************************************
Bi ( int res )
   // *********************************************
   {
   bi = new BufferedImage ( res+res , res , BufferedImage.TYPE_INT_RGB ) ;
   }
// *********************************************
} ; // class Bi
// ************************************************************







/* ************************************************************
M   M 
MM MM 
M M M 
M   M 
M   M 
************************************************************ */
// ************************************************************
class M 
// ************************************************************

// ************************************************************
{
double [][] m ; // the current transformation of this instance
static M tempMc ; // compose temporary matrix
static M tempMr ; // rotate and translate temporary matrix
                            // must be different than compose temp matrix
static double [] tempA1 ;   // two temps for Matrix Vector multiply
static double [] tempA2 ;


void print ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post ) ;
}
void print ( String s ) 
{
print ( ) ; 
System.out.print ( s ) ;
}
void println ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post + "\n" ) ;
}
void println ( String s ) 
{
print ( ) ; 
System.out.print ( s + "\n" ) ;
}
void println ( ) 
{
for ( int i = 0 ; i < 4 ; i ++ )
   {
   for ( int j = 0 ; j < 4 ; j ++ )
      System.out.print ( " " + m[i][j] ) ;
   System.out.println ( ) ;
   }
}

void print ( ) 
{
System.out.println ( ) ;
for ( int i = 0 ; i < 4 ; i ++ )
   {
   for ( int j = 0 ; j < 4 ; j ++ )
      System.out.print ( " " + m[i][j] ) ;
   System.out.println ( ) ;
   }
}

void transform ( V in , V out ) 
{
tempA1[0] = in.x ;
tempA1[1] = in.y ;
tempA1[2] = in.z ;
tempA1[3] = 1 ;
for ( int i = 0 ; i < 4 ; ++i )
   {
   tempA2[i] = 0 ;
   for ( int j = 0 ; j < 4 ; ++j )
      tempA2[i] += m[i][j] * tempA1[j] ;
   }
out.x = tempA2[0] ;
out.y = tempA2[1] ;
out.z = tempA2[2] ;
} ;

void transform ( P in , P out ) 
{
//print ( ) ;
tempA1[0] = in.x ;
tempA1[1] = in.y ;
tempA1[2] = in.z ;
tempA1[3] = 1 ;
for ( int i = 0 ; i < 4 ; ++i )
   {
   tempA2[i] = 0 ;
   for ( int j = 0 ; j < 4 ; ++j )
      tempA2[i] += m[i][j] * tempA1[j] ;
   }
out.x = tempA2[0] ;
out.y = tempA2[1] ;
out.z = tempA2[2] ;
//in.print ( ) ;
//out.println ( ) ;
} ;

void compose_pre ( M pre , M out ) 
{
for ( int i = 0 ; i < 4 ; ++i )
   for ( int j = 0 ; j < 4 ; ++j )
      {
      tempMc.m[i][j] = 0 ;
      for ( int k = 0 ; k < 4 ; ++k )
         tempMc.m[i][j] += pre.m[i][k] * m[k][j] ;
      }
for ( int i = 0 ; i < 4 ; ++i )
   for ( int j = 0 ; j < 4 ; ++j )
      out.m[i][j] = tempMc.m[i][j] ;
} ;

void compose_post ( M post , M out ) 
{
for ( int i = 0 ; i < 4 ; ++i )
   for ( int j = 0 ; j < 4 ; ++j )
      {
      tempMc.m[i][j] = 0 ;
      for ( int k = 0 ; k < 4 ; ++k )
         tempMc.m[i][j] += m[i][k] * post.m[k][j] ;
      }
for ( int i = 0 ; i < 4 ; ++i )
   for ( int j = 0 ; j < 4 ; ++j )
      out.m[i][j] = tempMc.m[i][j] ;
} ;


static void translate ( V trans , M out )
{
translate ( trans.x , trans.y , trans.z , out ) ;
} ;

static void translate ( double x , double y , double z , M out )
{
for ( int i = 0 ; i < 4 ; ++i )
   {
   for ( int j = 0 ; j < 4 ; ++j )
      out.m[i][j] = 0 ;
   out.m[i][i] = 1 ;
   }
out.m[0][3] = x ;
out.m[1][3] = y ;
out.m[2][3] = z ;
} ;

static void negative_translate ( V trans , M out ) 
{
M.translate ( -trans.x , -trans.y , -trans.z , out ) ;
} ;

void translate_compose ( double x , double y , double z , M out )
{
M.translate ( x , y , z , tempMr ) ;
compose_pre ( tempMr , out ) ;
} ;

void translate_compose ( V trans , M out ) 
{
M.translate ( trans , tempMr ) ;
compose_pre ( tempMr , out ) ;
} ;

void rotate_x_axis_compose ( double angle , M out )
{
M.rotate_x_axis ( angle , tempMr ) ;
compose_pre ( tempMr , out ) ;
} ;

void rotate_y_axis_compose ( double angle , M out )
{
M.rotate_y_axis ( angle , tempMr ) ;
compose_pre ( tempMr , out ) ;
} ;

void rotate_z_axis_compose ( double angle , M out )
{
M.rotate_z_axis ( angle , tempMr ) ;
compose_pre ( tempMr , out ) ;
} ;

static void rotate_x_axis ( double angle , M out )
{
double ang = angle / 180 * Math.PI ;

for ( int i = 0 ; i < 4 ; ++ i )
   for ( int j = 0 ; j < 4 ; ++ j )
      out.m[i][j] = 0 ;
out.m[1][1] = out.m[2][2] = Math.cos(ang) ;
out.m[2][1] = Math.sin(ang) ;
out.m[1][2] = - out.m[2][1] ;
out.m[0][0] = 1 ;
out.m[3][3] = 1 ;
} ;

static void rotate_y_axis ( double angle , M out )
{
double ang = angle / 180 * Math.PI ;

for ( int i = 0 ; i < 4 ; ++ i )
   for ( int j = 0 ; j < 4 ; ++ j )
      out.m[i][j] = 0 ;
out.m[0][0] = out.m[2][2] = Math.cos(ang) ;
out.m[2][0] = Math.sin(ang) ;
out.m[0][2] = - out.m[2][0] ;
out.m[1][1] = 1 ;
out.m[3][3] = 1 ;
} ;

static void rotate_z_axis ( double angle , M out )
{
double ang = angle / 180 * Math.PI ;

for ( int i = 0 ; i < 4 ; ++ i )
   for ( int j = 0 ; j < 4 ; ++ j )
      out.m[i][j] = 0 ;
out.m[0][0] = out.m[1][1] = Math.cos(ang) ;
out.m[0][1] = Math.sin(ang) ;
out.m[1][0] = - out.m[0][1] ;
out.m[2][2] = 1 ;
out.m[3][3] = 1 ;
} ;

M ( ) 
{
if ( tempMr == null )
   {
   //System.out.println ( "intializing matrix static fields" ) ;
   tempMr = new M ( 1 ) ;
   tempMc = new M ( 1 ) ;
   tempA1 = new double [4] ;
   tempA2 = new double [4] ;
   }
m = new double [4][4] ;
for ( int i = 0 ; i < 4 ; ++i )
   {
   for ( int j = 0 ; j < 4 ; ++j )
      m[i][j] = 0 ;
   m[i][i] = 1 ;
   }
} ;

M ( M orig ) 
{
if ( tempMr == null )
   {
   //System.out.println ( "intializing matrix static fields" ) ;
   tempMr = new M ( 1 ) ;
   tempMc = new M ( 1 ) ;
   tempA1 = new double [4] ;
   tempA2 = new double [4] ;
   }
m = new double [4][4] ;
for ( int i = 0 ; i < 4 ; ++i )
   {
   for ( int j = 0 ; j < 4 ; ++j )
      m[i][j] = orig.m[i][j] ;
   }
} ;

M ( int flag ) 
{
//System.out.println ( "M ( 1 )" ) ;
m = new double [4][4] ;
for ( int i = 0 ; i < 4 ; ++i )
   {
   for ( int j = 0 ; j < 4 ; ++j )
      m[i][j] = 0 ;
   m[i][i] = 1 ;
   }
} ;
// ************************************************************
} ; // end of class M 
// ************************************************************







/* ************************************************************
V   V 
V   V 
 V V  
 V V  
  V  
************************************************************ */


// ************************************************************
class V
// ************************************************************
// ************************************************************
{
double x , y , z ;

void print ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post ) ;
}
void print ( String s ) 
{
print ( ) ; 
System.out.print ( s ) ;
}
void println ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post + "\n" ) ;
}
void println ( String s ) 
{
print ( ) ; 
System.out.print ( s + "\n" ) ;
}
void print ( ) 
{
System.out.print ( "(" + x + ", " + y + ", " + z + ")" ) ;
}

void println ( ) 
{
print ( ) ;
System.out.println ( ) ;
} ;

double dotp ( P v )
{
return x*v.x + y*v.y + z*v.z ;
} ;

double dot ( V v )
{
return x*v.x + y*v.y + z*v.z ;
} ;

V ( double x1 , double x2 , double x3 )
{
x = x1 ;
y = x2 ;
z = x3 ;
}

V ( V o )
{
x = o.x ;
y = o.y ;
z = o.z ;
}

V ( )
{
x = 0 ;
y = 0 ;
z = 0 ;
}
// ************************************************************
} // end class V
// ************************************************************








/* ************************************************************
PPPP  
P   P 
PPPP  
P     
P     
************************************************************ */

// ************************************************************
class P 
// ************************************************************
// ************************************************************
{
double x , y , z , w ;

void print ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post ) ;
}
void print ( String s ) 
{
print ( ) ; 
System.out.print ( s ) ;
}
void println ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post + "\n" ) ;
}
void println ( String s ) 
{
print ( ) ; 
System.out.print ( s + "\n" ) ;
}
void print ( ) 
{
System.out.print ( "(" + x + ", " + y + ", " + z + ")" ) ;
}

void println ( ) 
{
print ( ) ;
System.out.println ( ) ;
} ;

double dist2 ( P a )
{
return ( (x-a.x)*(x-a.x) + (y-a.y)*(y-a.y) + (z-a.z)*(z-a.z) ) ;
}

double dist ( P a )
{
return Math.sqrt ( (x-a.x)*(x-a.x) + (y-a.y)*(y-a.y) + (z-a.z)*(z-a.z) ) ;
}

P ( )
   {
   w = 1 ;
   } ;

P ( double x , double y , double z )
   {
   this.x = x ;
   this.y = y ;
   this.z = z ;
   this.w = 1 ;
   } ;

P ( P o )
   {
   x = o.x ;
   y = o.y ;
   z = o.z ;
   w = o.w ;
   } ;

// ************************************************************
} ; // end of class P
// ************************************************************








/************************************************************
BBBB  BBBB   OOO  X   X 
B   B B   B O   O  X X  
BBBB  BBBB  O   O   X   
B   B B   B O   O  X X  
BBBB  BBBB   OOO  X   X 
************************************************************/

// ************************************************************
class BBox 
// ************************************************************
// ************************************************************
{
double xmin , xmax ;
double ymin , ymax ;
double zmin , zmax ;
static int hit ;

BBox ( ) 
   {
   xmin = 1e50 ;
   xmax = -1e50 ;
   ymin = 1e50 ;
   ymax = -1e50;
   zmin = 1e50;
   zmax = -1e50 ;
   } ;
// ************************************************************
} // end class BBox
// ************************************************************









/************************************************************
PPPP   AAA  RRRR  TTTTT 
P   P A   A R   R   T   
PPPP  AAAAA RRRR    T   
P     A   A R R     T   
P     A   A R  R    T   
************************************************************/

// ************************************************************
class Part 
// ************************************************************
// ************************************************************
{
Prim [] obj ;
Prim [] bound ;
boolean visible ;
// ************************************************************
void makeBound ( ) 
// ************************************************************
{
BBox b = new BBox ( ) ;

bound ( b ) ;
bound = new Prim [1] ;
bound[0] = new Sphere ( new P ( (b.xmin+b.xmax)/2 , 
                             (b.ymin+b.ymax)/2 , 
                             (b.zmin+b.zmax)/2 ) ,
                     0.5 * Math.sqrt ( (b.xmin-b.xmax) * (b.xmin-b.xmax) +
                                       (b.ymin-b.ymax) * (b.ymin-b.ymax) +
                                       (b.zmin-b.zmax) * (b.zmin-b.zmax) ) ) ;
}
// ************************************************************
void bound ( BBox bb ) 
// ************************************************************
{
for ( int i = 0 ; i < obj.length ; ++ i )
   obj[i].bound ( bb ) ;
}
// ************************************************************
void transform ( M m )
// ************************************************************
{
for ( int i = 0 ; i < obj.length ; ++ i )
   obj[i].transform ( m ) ;
for ( int i = 0 ; i < bound.length ; ++ i )
   bound[i].transform ( m ) ;
}
// ************************************************************
Prim closestIntersection ( Ray r ) // part.closestIntersection
// ************************************************************
{
if ( ! visible ) return null ;
boolean dhit = false ;
double closestDistance = 1e100 ;
Prim closestPrim = null ;

if ( bound.length == 0 )
   dhit = true ;
for ( int i = 0 ; i < bound.length ; ++ i )
   {
   if ( bound[i].anyIntersection ( r ) != null )
      {
      dhit = true ;
      ++BBox.hit ;
      }
   }
if ( dhit )
   {
   for ( int i = 0 ; i < obj.length ; ++ i )
      {
      Prim p = obj[i].closestIntersection ( r ) ;
      if ( p != null )
         {
         if ( p.depth < closestDistance ) 
            {
            closestDistance = p.depth ;
            closestPrim = p ;
            }
         }
      }
   }
return closestPrim ;
}
// ************************************************************
Prim anyIntersection ( Ray r ) // part.anyIntersection
// ************************************************************
{
if ( ! visible ) return null ;
boolean dhit = false ;
if ( bound.length == 0 )
   dhit = true ;
for ( int i = 0 ; i < bound.length ; ++ i )
   {
   if ( bound[i].anyIntersection ( r ) != null )
      {
      dhit = true ;
      ++BBox.hit ;
      }
   }

if ( dhit )
   {
   for ( int i = 0 ; i < obj.length ; ++ i )
      {
      Prim p = obj[i].anyIntersection ( r ) ;
      if ( p != null )
         return p ;
      }
   }
return null ;
} ;
// ************************************************************
void allIntersections ( Ray ray , PrimList pl ) 
// ************************************************************
{
if ( ! visible )
   return ;
boolean bbhit = false ;
if ( bound.length == 0 ) bbhit = true ;
for ( int i = 0 ; i < bound.length ; ++i )
   {
   if ( bound[i].anyIntersection ( ray ) != null )
      bbhit = true ;
   }
if ( bbhit )
   {
   for ( int s = 0 ; s < obj.length ; ++ s )
      obj[s].allIntersections ( ray , pl ) ;
   }
}

// ************************************************************
Part ( ) 
// ************************************************************
   {
   obj = new Prim [0] ;
   bound = new Prim [0] ;
   visible = true ;
   } ;
// ************************************************************
} ; // end class Part
// ************************************************************








/************************************************************
III  SSS   SSS      RRRR  EEEEE  AAA  DDDD  EEEEE RRRR  
 I  S     S         R   R E     A   A D   D E     R   R 
 I   SSS   SSS      RRRR  EEE   AAAAA D   D EEE   RRRR  
 I      S     S     R R   E     A   A D   D E     R R   
III  SSS   SSS  ___ R  R  EEEEE A   A DDDD  EEEEE R  R  
************************************************************/

// ************************************************************
class ISS_Reader
// ***********************************************************
// ************************************************************
{
static FileInputStream in ;
static int linenumber ;
static int lineinc ;
static String [] saved_tokens ;
static int token_index ;
static boolean tflag ;

// ************************************************************
static String readLine ( FileInputStream f ) throws Exception 
// ************************************************************
{
int c ;
String line = "" ;
c = f.read ( ) ;
while ( c != '\n' && c != -1 )
   {
   line += ((char)c) ;
   c = f.read ( ) ;
   }
return line ;
}
// ************************************************************
static String readLine ( ) throws Exception 
// ************************************************************
{
String line = "" ;
return line ;
}
// ************************************************************
static String readString ( ) throws Exception
// ************************************************************
{
if ( saved_tokens != null )
   {
   linenumber ++ ;
   return ( saved_tokens[token_index++] ) ;
   }

int c = 0 ;
String val = "" ;

linenumber += lineinc ;
lineinc = 0 ;
c = in.read() ;
while ( c <= 32 || c == ',' ) // commma considered white space
   {
   if ( c == -1 )
      break ;
   if ( c == '\n' )
      ++ linenumber ;
   c = in.read() ;
   }
if ( c == '{' ) // comments
   {
   c = in.read() ;
   while ( c != '}' )
      c = in.read ( ) ;
   return readString ( ) ;
   }

while ( c > 32 && c != ',' )
   {
   val += (char) c ;
   c = in.read() ;
   }
if ( c == '\n' )
   lineinc = 1 ;

//System.out.println ( "  read " + val + " at line " + linenumber ) ;

return val ;
} ;
// ************************************************************
static double readDouble ( ) throws Exception
// ************************************************************
{
double val = 0 ;
String s = "0" ;
try 
   {
   s = readString ( ) ;
   val = Double.parseDouble ( s ) ;
   }
catch ( Exception e )
   {
   System.err.println ( "number format exception (" + s + 
                        ") at line " + linenumber ) ;
   
   }
return val ;
} ;
// ************************************************************
static int readInteger ( ) throws Exception
// ************************************************************
{
return ( Integer.parseInt ( readString ( ) ) ) ;
} ;
// ************************************************************
void expect ( String key ) throws Exception
// ************************************************************
{
String got = readString ( ) ;
expect ( got , key );
} ;
// ************************************************************
void expect ( String got , String key ) throws Exception
// ************************************************************
{
if ( ! got.equals(key) )
   {
   System.err.println ( "parser expected " + key + " but got " + got +
                       " on line " + linenumber ) ;
   }
} ;
// ************************************************************
Prim readSphere ( )  throws Exception
// ************************************************************
{
double x , y ,z ;
expect ( "radius" )  ;
double r = readDouble() ;
expect ( "center" ) ;
x = readDouble ( ) ;
y = readDouble ( ) ;
z = readDouble ( ) ;
P c = new P ( x , y , z ) ;
expect ("endsphere") ;
return new Sphere ( c , r );
}

// ************************************************************
Prim readCylinder ( )  throws Exception
// ************************************************************
{
double x , y ,z ;
expect ( "radius" )  ;
double r = readDouble() ;
expect ( "center" ) ;
x = readDouble ( ) ;
y = readDouble ( ) ;
z = readDouble ( ) ;
P a1 = new P ( x , y , z ) ;
expect ( "center" ) ;
x = readDouble ( ) ;
y = readDouble ( ) ;
z = readDouble ( ) ;
P a2 = new P ( x , y , z ) ;
expect ("endcylinder") ;
return new Cylinder ( a1 , a2 , r );
}

// ************************************************************
Prim readPolygon ( )  throws Exception
// ************************************************************
{
P [] vtx = new P [100] ;
int vc = 0 ;
String vs = readString ( ) ;
while ( vs.equals( "vertex") ) 
   {
   double x = readDouble ( ) ;
   double y = readDouble ( ) ;
   double z = readDouble ( ) ;
   P pt = new P ( x , y , z ) ;
   vtx[vc++] = pt ;
   vs = readString ( ) ;
   }
expect ( vs , "endpolygon" ) ;

Polygon p = new Polygon ( vtx , vc) ;
return p ;
} ;



// ************************************************************
Prim readPrim ( String what ) throws Exception
// ************************************************************
{
Prim p = null ;
if ( what.equals ( "polygon" ) )
   p = readPolygon ( ) ;
else if ( what.equals ( "cylinder" ) )
   p = readCylinder ( ) ;
else if ( what.equals ( "sphere" ) )
   p = readSphere ( ) ;
else
   {
   expect ( what , "polygon, cylinder or sphere" ) ;
   }
return p ;
}

// ************************************************************
Part readShape ( ) throws Exception
// ************************************************************
{
Prim [] pt = new Prim [1000] ;
Part sol = new Part ( ) ;
int pc = 0 ;
String key ;
String name ;
//expect ( "shape" ) ;

name = readString ( ) ;
key = readString ( ) ;
while ( key.equals("cylinder") || key.equals("sphere") || 
        key.equals("polygon" ) )
   {
   pt[pc++] = readPrim ( key ) ; 
   key = readString ( ) ;
   }

sol.obj = new Prim [pc] ;
for ( int i = 0 ; i < pc ; ++ i )
   sol.obj[i] = pt[i] ;
expect ( key , "endshape" ) ;
return sol ;
} ;

// ************************************************************
Structure readStructure ( ) throws Exception
// ************************************************************
{
String key ;
Structure mod = new Structure ( ) ; 
Part [] sa = new Part [100] ;

// expect ( "structure" ) ;
key = readString ( ) ;
String name = key ;
//System.out.println ( "reading structure " + key ) ;

   expect ( "parts" ) ;
      key = readString ( ) ;
      int si = 0 ;
      while ( key.equals ( "shape" ) )
         {
         sa[si++] = readShape ( ) ;
         key = readString ( ) ;
         }
      mod.solid = new Part [si] ;
      for ( int i = 0 ; i < si ; ++ i )
         {
         mod.solid [i] = sa[i] ;
         mod.solid[i].makeBound() ;
         }
      expect ( key ,  "endparts" ) ;

   expect ( "children" ) ;
      int childCount = 0 ;
      Structure [] tc = new Structure [100] ;
      key = readString ( ) ;
      while ( key.equals("structure") )
         {
         tc[childCount++] = readStructure ( ) ;
         key = readString ( ) ;
         }
      
      mod.child = new Structure [childCount] ;
      for ( int i = 0 ; i < childCount ; ++ i )
         mod.child [i] = tc[i] ;
      expect ( key , "endchildren" ) ;
      //System.out.println ( name ) ;
      
expect ( "endstructure" ) ;
mod.makeBound ( ) ;
return mod ;
}


// ************************************************************
Structure readModel ( ) 
// ************************************************************
{
Structure mod = null ;

try
   {
   expect ( "model" ) ;
   ISSVis.model_name = readString ( ) ;
      expect ( "structure" ) ;
      mod = readStructure (  ) ;
   expect ( "endmodel" ) ;
   }
catch ( Exception e )
   {
   System.out.println ( "ERROR: reading input file at line " +
                        linenumber ) ;
   e.printStackTrace ( ) ;
   System.exit ( 666 ) ;
   }
fix_longerons ( mod ) ;
return mod ;
} ;


// ************************************************************
void fix_longeron ( Structure m ,int side , int saw , 
                    int [] offsets , int index ) 
// ************************************************************
{

// SAW model
//     0 bottom arm
//          ...
//     1 top arm
//          ...
//     2 blanket 1
//          0 polygon
//     3 blanket 2
//          0 polygon
//     4 longerons
//          0 cylinder
//          1 cylinder
//          2 cylinder
//          3 cylinder
//     5 bottom cap 
//          0 cylinder
//     6 top cap
//          0 cylinder
Cylinder c = (Cylinder) m.child[side].child[saw].solid[5].obj[0] ;
P axis = c.a1 ;

Cylinder longeron ;

for ( int i = 0 ; i < 4 ; i ++ )
   {
   longeron = (Cylinder) m.child[side].child[saw].solid[4].obj[i] ;

   longeron.a1.y = axis.y + offsets[index + 2*i] ;
   longeron.a1.z = axis.z + offsets[index + 2*i+1] ;
   
   longeron.a2.y = longeron.a1.y ;
   longeron.a2.z = longeron.a1.z ;
   }
}
// ************************************************************
void fix_longerons ( Structure m ) 
// ************************************************************
{
// From NASA memo, y z offsets of longerons from rotation axis
//
// order of table "offsets"
// 1A ( y,z, y,z, y,z, y,z ) 1B ( ... ) ... 3B ( ... ) 4B ( ... )
//
int [] offsets = new int [] 
   {
   -181, -502,
   580, -367,
   444, 394,
   -316, 258,

   386, -637,
   -375, -503,
   -241, 258,
   520, 124,

   183, -622,
   -578, -489,
   -446, 272,
   316, 139,

   -322, -542,
   439, -407,
   305, 354,
   -456, 219,

   296, -607,
   -464, -473,
   -330, 288,
   431, 154,

   -243, -558,
   517, -422,
   381, 339,
   -379, 202,

   -248, -562,
   513, -430,
   380, 332,
   -381, 199,

   246, -566,
   -515, -434,
   -383, 327,
   378, 195
   } ;


fix_longeron ( m , 0 , 0 , offsets , 0 ) ;
fix_longeron ( m , 1 , 0 , offsets , 8 ) ;
fix_longeron ( m , 0 , 1 , offsets , 16 ) ;
fix_longeron ( m , 1 , 1 , offsets , 24 ) ;
fix_longeron ( m , 0 , 2 , offsets , 32 ) ;
fix_longeron ( m , 1 , 2 , offsets , 40 ) ;
fix_longeron ( m , 0 , 3 , offsets , 48 ) ;
fix_longeron ( m , 1 , 3 , offsets , 56 ) ;
}
// ************************************************************
ISS_Reader ( String [] tokes )
// ************************************************************
{
saved_tokens = tokes ;
token_index = 0 ;
tflag = false ;
linenumber = 1 ;
lineinc = 0 ;
}

// ************************************************************
ISS_Reader ( String name )
// ************************************************************
{
try
   {  
   System.out.println ( "reading " + name ) ;
   in = new FileInputStream ( name ) ;
   linenumber =  1 ;
   lineinc = 0 ;
   }
catch ( Exception e )
   {
   System.out.println ( "ERROR: Unable to open file " + name + 
                       " for reading." ) ;
   e.printStackTrace ( ) ;
   System.exit ( 666 ) ;
   }
}
// ************************************************************
} ; // end class IIS_Reader
// ************************************************************








/************************************************************
 SSS  TTTTT RRRR  U   U  CCC  TTTTT U   U RRRR  EEEEE 
S       T   R   R U   U C       T   U   U R   R E     
 SSS    T   RRRR  U   U C       T   U   U RRRR  EEE   
    S   T   R R   U   U C       T   U   U R R   E     
 SSS    T   R  R   UUU   CCC    T    UUU  R  R  EEEEE 
************************************************************/

// ************************************************************
class Structure 
// ************************************************************
// ************************************************************
{
Part [] solid ;
Structure [] child ;
Prim [] bb ; // bounding boxes, (may also be spheres)
boolean visible ;
M transform ;

static Prim last_prim ;
// ************************************************************
void makeBound ( ) 
// ************************************************************
{
BBox b = new BBox ( ) ;

bound ( b ) ;
// System.out.println ( "bbox " + b.xmin + " " +  b.xmax + " " + b.ymin +
//                     " " + b.ymax + " " + b.zmin + " " + b.zmax ) ;
bb = new Prim [1] ;
bb[0] = new Sphere ( new P ( (b.xmin+b.xmax)/2 , 
                             (b.ymin+b.ymax)/2 , 
                             (b.zmin+b.zmax)/2 ) ,
                     0.5 * Math.sqrt ( (b.xmin-b.xmax) * (b.xmin-b.xmax) +
                                       (b.ymin-b.ymax) * (b.ymin-b.ymax) +
                                       (b.zmin-b.zmax) * (b.zmin-b.zmax) ) ) ;
}
// ************************************************************
void bound ( BBox b )
// ************************************************************
{
for ( int i = 0 ; i < solid.length ; ++ i ) // parts
   solid[i].bound ( b ) ;
}
// ************************************************************
void transform ( M m )
// ************************************************************
{
M new_m = new M ( ) ;
transform.compose_pre ( m , new_m ) ;

for ( int i = 0 ; i < solid.length ; ++ i )
   solid[i].transform ( new_m ) ;
for ( int i = 0 ; i < child.length ; ++ i )
   child[i].transform ( new_m ) ;
for ( int i = 0 ; i < bb.length ; ++ i )
   bb[i].transform ( new_m ) ;
}
// ************************************************************
void transform ( )
// ************************************************************
{
M new_m = new M ( ) ;
transform ( new_m ) ;
}
// ************************************************************
Prim closestIntersection ( Ray ray ) // model.closestIntersection
// ************************************************************
{
++ISSVis.closest_cast ;
double closestDistance = 1e100 ;
Prim closestPrim = null ;
if ( ! visible ) return null ;
Prim p = null ;;

boolean bbhit = false ;
for ( int i = 0 ; i < bb.length ; i ++ )
   {
   if ( null != bb[i].anyIntersection ( ray ) ) 
      {
      ++BBox.hit ;
      bbhit = true ;
      }
   }

if ( bbhit )
   {
   p = null ;
   for ( int s = 0 ; s < solid.length ; ++ s )
      {
      p = solid[s].closestIntersection ( ray ) ;
      if ( p != null )
         {
         if ( p.depth < closestDistance ) 
            {
            closestDistance = p.depth ;
            closestPrim = p ;
            }
         }
      }
   }

// cast rays for children always
for ( int c = 0 ; c < child.length ; c++ )
   {
   p = child[c].closestIntersection ( ray ) ;
   if ( p != null )
      {
         if ( p.depth < closestDistance ) 
            {
            closestDistance = p.depth ;
            closestPrim = p ;
            }
      }
   }
if ( closestPrim != null )
   ++ ISSVis.closest_hits ;
return closestPrim ;
}
static Prim hint = null ;
// ************************************************************
Prim anyIntersectionHint ( Ray ray ) // model.anyIntersection
// ************************************************************
{
// I questioned whether this should be any different than
// anyIntersection, since it may lose some coherency which
// would make subdivision less efficient.
// Without the hint, anyIntersection always returns the first
// intersected prim from a fixed order.

Prim hit = null ; 

++ISSVis.any_cast ;
if ( hint != null )
   {
   hit = hint.anyIntersection ( ray ) ;
   }
if ( hit == null )
   {
   ++ ISSVis.multi_cast ;
   hit = anyIntersection ( ray ) ;
   if ( hit != null )
      {
      hint = hit ;
      ++ISSVis.multi_hits ;
      }
   }
else
   ++ISSVis.one_hits ;
return hit ;
}
// ************************************************************
void allIntersections ( Ray ray , PrimList pl ) 
// ************************************************************
{
ISSVis.all_cast ++ ;
if ( ! visible )
   return ;
boolean bbhit = false ;
if ( bb.length == 0 ) bbhit = true ;
for ( int i = 0 ; i < bb.length ; ++i )
   {
   if ( bb[i].anyIntersection ( ray ) != null )
      bbhit = true ;
   }
if ( bbhit )
   {
   for ( int s = 0 ; s < solid.length ; ++ s )
      solid[s].allIntersections ( ray , pl ) ;
   }
for ( int c = 0 ; c < child.length ; c++ )
   child[c].allIntersections ( ray , pl ) ;
ISSVis.all_hits += pl.s ;
}
// ************************************************************
Prim anyIntersection ( Ray ray ) // model.anyIntersection
// ************************************************************
{
if ( ! visible ) return null ;
Prim p = null ;;
/* if ( last_prim != null ) */
/*    p = last_prim.anyIntersection ( ray ) ; */
if ( p != null )
   return p ;

boolean bbhit = false ;
for ( int i = 0 ; i < bb.length ; i ++ )
   {
   if ( null != bb[i].anyIntersection ( ray ) ) 
      {
      ++BBox.hit ;
      bbhit = true ;
      break ; 
      }
   }

if ( bbhit )
   {
   p = null ;
   for ( int s = 0 ; s < solid.length ; ++ s )
      {
      p = solid[s].anyIntersection ( ray ) ;
      if ( p != null )
         {
         last_prim = p ;
         return p ;
         }
      }
   }

// cast rays for children always
for ( int c = 0 ; c < child.length ; c++ )
   {
   p = child[c].anyIntersection ( ray ) ;
   if ( p != null )
      {
      last_prim = p ;
      return p ;
      }
   }
return null ;
} ;
// ************************************************************
Structure ( )
// ************************************************************
   {
   last_prim = null ;
   solid = new Part [0] ;
   child = new Structure [0] ;
   transform = new M ( ) ;
   bb = new Prim [0] ;
   visible = true ;
   } ;
// ************************************************************
} ; // end of class Structure
// ************************************************************



/************************************************************
PPPP  RRRR  III M   M L     III  SSS  TTTTT 
P   P R   R  I  MM MM L      I  S       T   
PPPP  RRRR   I  M M M L      I   SSS    T   
P     R R    I  M   M L      I      S   T   
P     R  R  III M   M LLLLL III  SSS    T   
************************************************************/
// ************************************************************
class PrimList 
// ************************************************************
{
Prim [] prim ;
boolean [] valid ;
int s ;

// ************************************************************
void copy ( PrimList original )
// ************************************************************
{
s = original.s ;
for (int i = 0 ; i < s ; ++i )
   {
   valid[i] = true ;
   prim[i] = original.prim[i] ;
   }
}
// ************************************************************
void reset ( ) 
// ************************************************************
{
for ( int i = 0 ; i < s ; ++i )
   valid [i] = true ;
s = 0 ;
}
// ************************************************************
PrimList ( )
// ************************************************************
{
prim = new Prim [ 600 ] ;
valid = new boolean [ 600 ] ;
s = 0 ;
}
// ************************************************************
} // end of class PrimList
// ************************************************************


/************************************************************
PPPP  RRRR  III M   M 
P   P R   R  I  MM MM 
PPPP  RRRR   I  M M M 
P     R R    I  M   M 
P     R  R  III M   M 
 ************************************************************/

// ************************************************************
abstract class Prim 
// ************************************************************
// ************************************************************
{
boolean visible ;
short red ;
short green ;
short blue ;
short basered ;
short basegreen ;
short baseblue ;
double depth ;
int order ;
static int next_order = 0 ;
V normal ;
abstract Prim closestIntersection ( Ray r ) ;
abstract Prim anyIntersection ( Ray r ) ;
abstract void transform ( M old ) ;
abstract void bound ( BBox b ) ;
abstract void print ( ) ;
// ************************************************************
void print ( String s ) 
// ************************************************************
{
print ( ) ;
System.out.print ( s ) ;
}
// ************************************************************
void print ( String pre , String post ) 
// ************************************************************
{
System.out.print ( pre ) ;
print ( ) ;
System.out.print ( post ) ;
}
// ************************************************************
void allIntersections ( Ray ray , PrimList p ) 
{
Prim x = anyIntersection ( ray ) ;
if ( x != null )
   {
   p.valid[p.s] = true ;
   p.prim[p.s++] = x ;
   }
}
// ************************************************************
Prim ( ) 
   {
   order = next_order ++ ;
   }
// ************************************************************
} ; // end of class Prim
// ************************************************************








/************************************************************
RRRR   AAA  Y   Y 
R   R A   A  Y Y  
RRRR  AAAAA   Y   
R R   A   A   Y   
R  R  A   A   Y   
 ************************************************************/

// ************************************************************
class Ray
// ************************************************************
// ************************************************************
{
P o ;
V d ;

void print ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post ) ;
}
void print ( String s ) 
{
print ( ) ; 
System.out.print ( s ) ;
}
void println ( String pre , String post ) 
{
System.out.print ( pre ) ;
print ( ) ; 
System.out.print ( post + "\n" ) ;
}
void println ( String s ) 
{
print ( ) ; 
System.out.print ( s + "\n" ) ;
}
void print ( ) 
{
System.out.print ( "Ray o: " ) ;
o.print() ;
System.out.print ( " d: " ) ;
d.print ( ) ;
} ;

void println ( ) 
{
print ( ) ;
System.out.println ( ) ;
} ;

Ray ( P origin , V direction ) 
   {
   o = new P ( origin ) ;
   d = new V ( direction ) ;
   } ;
Ray ( ) 
   {
   o = new P ( ) ;
   d = new V ( ) ;
   } ;

// ************************************************************
} ; // end of class Ray
// ************************************************************



/************************************************************
 SSS  PPPP  H   H EEEEE RRRR  EEEEE 
S     P   P H   H E     R   R E     
 SSS  PPPP  HHHHH EEE   RRRR  EEE   
    S P     H   H E     R R   E     
 SSS  P     H   H EEEEE R  R  EEEEE 
************************************************************/

// ************************************************************
class Sphere extends Prim
// ************************************************************
// ************************************************************
{
double rad ;
P      cen , original_cen ;

void bound ( BBox b )
{
if ( cen.x + rad > b.xmax ) b.xmax = cen.x + rad ;
if ( cen.y + rad > b.ymax ) b.ymax = cen.y + rad ;
if ( cen.z + rad > b.zmax ) b.zmax = cen.z + rad ;
if ( cen.x - rad < b.xmin ) b.xmin = cen.x - rad ;
if ( cen.y - rad < b.ymin ) b.ymin = cen.y - rad ;
if ( cen.z - rad < b.zmin ) b.zmin = cen.z - rad ;
}

void transform ( M m ) 
{
m.transform ( original_cen , cen ) ; 
} ;


void print ( )
{
System.out.print ( "Sphere c: " ) ;
cen.print() ;
System.out.print ( " r: " + rad ) ;
} ;

void println ( ) 
{
print() ;
System.out.println ( ) ;
} ;

Prim closestIntersection ( Ray ray ) // sphere.closestIntersection
{
// we don't have spheres in the model, so this is never called
depth = ray.d.dotp(cen) ; 
return anyIntersection ( ray ) ; // we only use them for bounding 
} ;

// ************************************************************
Prim anyIntersection ( Ray ray ) // sphere.anyIntersection
// ************************************************************
{
double a = cen.x - ray.o.x ;
double b = cen.y - ray.o.y ;
double c = cen.z - ray.o.z ;

double B = 2 * ( -a*ray.d.x - b*ray.d.y - c*ray.d.z ) ;

double disc = B * B - 4 * (a*a + b*b + c*c - rad*rad ) ;

++ISSVis.one_cast ;
if ( disc < 0.0 )
   return null ;
// values of t for intersection are -B-sqrt()/2A and -B+sqrt()/2A
// the + one is farther away. Use it since we don't care about normal
// of first visible intersection
double t = 0.5 * Math.sqrt(disc) + a*ray.d.x + b*ray.d.y + c*ray.d.z ;
if ( t <= 0 )
   return null ;
return (Prim) this ; // we hit it
} ;
// ************************************************************

Sphere ( )
{
cen = new P ( ) ;
original_cen = new P ( ) ;
basered = 200 ;
basegreen = 200 ;
baseblue = 200 ;
visible = true ;
normal = new V ( ) ;
}

Sphere ( P c , double r )
{
cen = new P ( c ) ;
original_cen = new P ( c ) ;
rad = r ;
basered = 200 ;
basegreen = 200 ;
baseblue = 200 ;
visible = true ;
normal = new V ( ) ;
}

// ************************************************************
} ; // end of class sphere
// ************************************************************







/************************************************************
 CCC  Y   Y L     III N   N DDDD  EEEEE RRRR  
C      Y Y  L      I  NN  N D   D E     R   R 
C       Y   L      I  N N N D   D EEE   RRRR  
C       Y   L      I  N  NN D   D E     R R   
 CCC    Y   LLLLL III N   N DDDD  EEEEE R  R  
 ************************************************************/

// ************************************************************
class Cylinder extends Prim
// ************************************************************
// ************************************************************
{
double rad ;
P      a1 , original_a1 ;
P      a2 , original_a2 ;
double length ;
V      n ; // unit vector along axis from a1 to a2
double D1 ; // n and d are equation of plane of end cap 1
double D2 ; // resp. end cap 2
double nx , ny , nz ;

void bound ( BBox b )
{
if ( a1.x + rad > b.xmax ) b.xmax = a1.x + rad ;
if ( a1.y + rad > b.ymax ) b.ymax = a1.y + rad ;
if ( a1.z + rad > b.zmax ) b.zmax = a1.z + rad ;
if ( a1.x - rad < b.xmin ) b.xmin = a1.x - rad ;
if ( a1.y - rad < b.ymin ) b.ymin = a1.y - rad ;
if ( a1.z - rad < b.zmin ) b.zmin = a1.z - rad ;

if ( a2.x + rad > b.xmax ) b.xmax = a2.x + rad ;
if ( a2.y + rad > b.ymax ) b.ymax = a2.y + rad ;
if ( a2.z + rad > b.zmax ) b.zmax = a2.z + rad ;
if ( a2.x - rad < b.xmin ) b.xmin = a2.x - rad ;
if ( a2.y - rad < b.ymin ) b.ymin = a2.y - rad ;
if ( a2.z - rad < b.zmin ) b.zmin = a2.z - rad ;
}
void transform ( M m )
{
m.transform ( original_a1 , a1 ) ;
m.transform ( original_a2 , a2 ) ;
n.x = ( a2.x - a1.x ) / length ;
n.y = ( a2.y - a1.y ) / length ;
n.z = ( a2.z - a1.z ) / length ;

D1 = - ( a1.x * n.x + a1.y * n.y + a1.z * n.z ) ;
D2 = - ( a2.x * n.x + a2.y * n.y + a2.z * n.z ) ;

// for rotations and translations length and radius are unchanged
} ;

void print ( )
{
System.out.print ( "Cylinder a1: " ) ;
a1.print() ;
System.out.print ( " a2: " ) ;
a2.print() ;
System.out.print ( " r: " + rad ) ;
} ;

void println ( ) 
{
print() ;
System.out.println ( ) ;
} ;

Cylinder ( P p1 , P p2 , double radius )
{
a1 = new P ( p1 ) ;
a2 = new P ( p2 ) ;
original_a1 = new P ( p1 ) ;
original_a2 = new P ( p2 ) ;
normal = new V ( ) ;
rad = radius ;
length = Math.sqrt ( (a1.x-a2.x)*(a1.x-a2.x) +
                     (a1.y-a2.y)*(a1.y-a2.y) +
                     (a1.z-a2.z)*(a1.z-a2.z) ) ;
n = new V ( (a2.x-a1.x)/length , (a2.y-a1.y)/length , (a2.z-a1.z)/length );
D1 = - ( a1.x * n.x + a1.y * n.y + a1.z * n.z ) ;
D2 = - ( a2.x * n.x + a2.y * n.y + a2.z * n.z ) ;
visible = true ;
basered = (short) (200- (a1.x+50000) / 100000 * 100 );
basegreen = basered ;
baseblue = basered ;
} ;

// ************************************************************
// This part of the code is pretty hairy, although
// algebraically it is fairly straightforward.
// ************************************************************
Prim closestIntersection ( Ray ray ) // cylinder.closestIntersection
// ************************************************************
// We are assuming that the origin of the ray is not in the
// cylinder.
// ************************************************************
{
/*
 * 1, Note the following handy fact from vector algebra: If you have
 * a line passing through the origin and an arbitray point, P, then
 * the vector cross product of (a unit vector in the direction 
 * of the line) and (a vector, V, from the origin to P) gives you a 
 * vector that is in the direction from P to the closest point to P
 * that is on the line. The length of this vector is the distance
 * from this closest point to P. (Actually you don't have to use
 * the origin, any point on the line also works but the algebra is
 * a little simpler using the origin).
 *
 * 2, The defintion of a (mathematical) cylinder is all the points 
 * that are a given distance (radius) from a line.
 *
 * 3, Combining 1 and 2 yields
 * |U-V|^2 = r^2
 *
 * 4, The expression for a point on a ray is (ray.o + ray.d * t )
 * where ray.o is the start (origin) of the ray and ray.d is the 
 * direction vector of the ray.
 *
 * 5 Substitue 4 into 3 giving
 * |U - (ray.o + ray.d * t - O)|^2 = r^2
 * This equation now represnts points of the ray that are also
 * on the cylinder.
 *
 * 6, Solve for t giving a quadratic equation in the variable t
 * t^2 * a + t * b + c = 0
 *
 * 7, if this equation has solutions that are real numbers then
 * the ray hits the sphere and the points are (ray.o + ray.d * t1)
 * and (ray.o + ray.d * t2) where t1 and t2 are the roots of the
 * quadratic equation.
 *
 * The equation has real solutions if the discriminate, 
 * (b^2 - 4 * a * c), is postive.
 *
 *
 * So, procedure: If we translate the ray and cylinder by the same 
 * amount (vector) then the intersection points are also translated
 * by this amount. 
 * So, translate ray and cylinder so that a point on the axis 
 * of the cylinder is at the origin (we know two points that are on
 * the axis, a1 and a2, so this is easy.
 *
 * Solve for the roots t1 and t2 and substitute back in to the 
 * original ray expression to give the intersection points.
 *
 * Since we are dealing with a finite cylinder with circular end
 * caps, we must check that the points are between the end caps.
 *
 * Also we must test the ray for intersection with end caps. This
 * is done by intersecting the ray with the plane of the end cap
 * and then testing if the distance from the intersection point to
 * the cap center is less than the radius.
 * 
 * If any of the intersection points above are valid then they are
 * also tested to be sure that they are in the positive t direction
 * from the ray origin (otherwise the intersection is on the line
 * of the ray, but not on the half of the line which is the ray).
 *
 * If any intersection points are still valid, then the closest one
 * to the ray origin is the winner. In order to do shading you need 
 * to know the normal vector to the surface at the closest intersection 
 * point. For cylinders this is easy, as it is either V/|V| (from 1 above)
 * or the normal to one of the end caps.
 *
 *
 * In traditional ray casting you
 * repeat this procedure for every cylinder (or a similar procedure
 * on other types of geometric primitives, such as polygons) that 
 * potentially intersect the ray and choose the closest one.
 *
 * That is how you do it in traditional ray casting. For shadow 
 * testing you can to a slightly simplified test. You cast a ray from
 * a point on the object to the light source. If you get any
 * valid intersection in the cylinder calculation, you can stop and 
 * know you are in shadow. You do no have to calculate both intersection
 * points. Actually you don't even have to calculate the intersection
 * point, all you have to do is know that there is a valid intersection
 * with the cylinder somewhere. Once any primitive is found to 
 * intersect the ray, then you know you are in shadow and you do not have 
 * to test any more primitives for that ray.
 * 
 *
 *
 *
 */
depth = 2e100 ;
if ( ! visible ) return null ;

double o1 = ray.o.x - a1.x ;
double o2 = ray.o.y - a1.y ;
double o3 = ray.o.z - a1.z ; // ray origin shifted so a1 = (0,0,0)
   
double c1 = n.x ;
double c2 = n.y ;
double c3 = n.z ;  // direction vector for cylinder (precalculated)
double t , d ;
int hits = 0 ; // can have at most 2 hits 

double r_dot_n = ( ray.d.x*n.x + ray.d.y*n.y + ray.d.z*n.z ) ;
if ( r_dot_n < 0.999999999 && r_dot_n > -0.999999999 )
   { // ray not (anti) parallel to cylinder
   
   // (t * ray.d + o) cross c = t(d12+d23+d31) + o12+o23+o31
   
   double d12 = c2*ray.d.x - c1*ray.d.y ;
   double d23 = c3*ray.d.y - c2*ray.d.z ;
   double d31 = c1*ray.d.z - c3*ray.d.x ;
   double o12 = c2*o1 - c1*o2 ;
   double o23 = c3*o2 - c2*o3 ;
   double o31 = c1*o3 - c3*o1 ; // cross product terms
   
   double A = d12*d12 + d23*d23 + d31*d31 ;
   double B = 2 * (d23*o23 + d12*o12 + d31*o31 ) ;
   double C = o12*o12 + o23*o23 + o31*o31 - rad*rad ;
   
   double disc = B*B-4*A*C ;
   
   if ( disc <= 0 ) // a graze is as good as miss for us.
      return null ;

   // hit infinite cylinder 
   // now check for bounds of finite cylinder
   double i1 , i2 , i3 ;
   t = ( -B - Math.sqrt(disc) ) / (2*A) ; // more negative root
   if ( t >= 0 )
      {
      i1 = t*ray.d.x + ray.o.x ;
      i2 = t*ray.d.y + ray.o.y ;
      i3 = t*ray.d.z + ray.o.z ; // (i1, i2, i3) is intersection point
      
      d = i1*c1 + i2*c2 + i3*c3 + D1 ; // signed distance from plane of end cap 1
      if ( d >= 0 && d <= length )
         {
         double dep = i1*ray.d.x + i2*ray.d.y + i3*ray.d.z ;
         depth = dep ; // normalized depth such that the depth
         // of any point on the plane passing through the origin
         // and perpendicular to the ray is zero

         // the following nonsense calculates the normal at the 
         // intersection point and does primitive shading
         double prj = ( ( i1-a1.x ) * n.x + 
                        ( i2-a1.y ) * n.y +
                        ( i3-a1.z ) * n.z ) ;
         double ax = a1.x + n.x*prj ;
         double ay = a1.y + n.y*prj ;
         double az = a1.z + n.z*prj ;
         
         double nx = i1 - ax ;
         double ny = i2 - ay ;
         double nz = i3 - az ;
         double iv = 1.0 / Math.sqrt ( nx*nx + 
                                       ny*ny + 
                                       nz*nz ) ;
         nx *= iv ;
         ny *= iv ;
         nz *= iv ;

         normal.x = nx ;
         normal.y = ny ;
         normal.z = nz ;
         hits = 1 ;
         // farthest intersection is on finite cylinder wall
         }
      }

   t = ( -B + Math.sqrt(disc) ) / (2*A) ; // more positive root 
   if ( t >= 0 )
      {
      i1 = t*ray.d.x + ray.o.x ;
      i2 = t*ray.d.y + ray.o.y ;
      i3 = t*ray.d.z + ray.o.z ;
      
      d = i1*c1 + i2*c2 + i3*c3 + D1 ;
      if ( d >= 0 && d <= length )
         {
         double dep = i1*ray.d.x + i2*ray.d.y + i3*ray.d.z ;
         if ( dep < depth )
            {
            depth = dep ;
            double prj = ( i1-a1.x ) * n.x + ( i2-a1.y ) * n.y +
               ( i3-a1.z ) * n.z  ;
            double ax = a1.x + n.x*prj ;
            double ay = a1.y + n.y*prj ;
            double az = a1.z + n.z*prj ;
            
            double nx = i1 - ax ;
            double ny = i2 - ay ;
            double nz = i3 - az ;
            double iv = 1.0 / Math.sqrt ( nx*nx + 
                                          ny*ny + 
                                          nz*nz ) ;
            nx *= iv ;
            ny *= iv ;
            nz *= iv ;
            normal.x = nx ;
            normal.y = ny ;
            normal.z = nz ;
            }
         // nearest intersection is on finite cylinder wall
         ++ hits ;
         }
      }
   }

// we could possibly hit the end caps without hitting the finite 
// cylinder wall so... (nearly parallel to axis case)
// test end cap a1
if ( hits < 2 ) 
   {
   if ( (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) != 0 )
      { // ray not parallel to end cap
      t = - (c1*ray.o.x + c2*ray.o.y + c3*ray.o.z + D1) / 
         (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) ;
      if ( t >= 0 ) 
         {
         double i1 = t*ray.d.x + ray.o.x;
         double i2 = t*ray.d.y + ray.o.y;
         double i3 = t*ray.d.z + ray.o.z;
         
         d = ( (i1 - a1.x) * (i1 - a1.x) +
               (i2 - a1.y) * (i2 - a1.y) +
               (i3 - a1.z) * (i3 - a1.z) ) ;
         
         if ( d <= rad*rad )
            {
            double dep = i1*ray.d.x + i2*ray.d.y + i3*ray.d.z ;
            if ( dep < depth ) 
               {
               if ( n.x * ISSVis.sray.d.x + 
                    n.y * ISSVis.sray.d.y + 
                    n.z * ISSVis.sray.d.z > 0 ) 
                  {
                  normal.x = n.x ;
                  normal.y = n.y ;
                  normal.z = n.z ;
                  }
               else
                  {
                  normal.x = -n.x ;
                  normal.y = -n.y ;
                  normal.z = -n.z ;
                  }
               depth = dep ; // just to be careful
               }
            ++ hits ;
            }
         }
      }
   }

// test end cap 2 just incase end cap 1 is behind us.
if ( hits < 2 )
   {
   if ( (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) != 0 )
      {
      t = - ( (c1*ray.o.x + c2*ray.o.y + c3*ray.o.z + D2) / 
              (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) ) ;
      if ( t >= 0 )
         {
         double i1 = t*ray.d.x + ray.o.x;
         double i2 = t*ray.d.y + ray.o.y;
         double i3 = t*ray.d.z + ray.o.z;
         
         d = ( (i1 - a2.x) * (i1 - a2.x) +
               (i2 - a2.y) * (i2 - a2.y) +
               (i3 - a2.z) * (i3 - a2.z) ) ;
         
         if ( d <= rad*rad )
            {
            double dep = (i1)*ray.d.x + (i2)*ray.d.y + 
               (i3)*ray.d.z ;
            if ( dep < depth ) 
               {
               if ( n.x * ISSVis.sray.d.x + 
                    n.y * ISSVis.sray.d.y + 
                    n.z * ISSVis.sray.d.z < 0 ) 
                  {
                  normal.x = n.x ;
                  normal.y = n.y ;
                  normal.z = n.z ;
                  }
               else
                  {
                  normal.x = -n.x ;
                  normal.y = -n.y ;
                  normal.z = -n.z ;
                  }
               depth = dep ; // just to be careful
               }
            }
         }
      }
   }

if ( depth < 1e200 )
   return this ;
return null ; // we didn't hit it
} ;

// ************************************************************
Prim anyIntersection ( Ray ray ) // cylinder.anyIntersection
// ************************************************************
{
++ISSVis.one_cast ;
if ( ! visible ) return null ;

double o1 = ray.o.x - a1.x ;
double o2 = ray.o.y - a1.y ;
double o3 = ray.o.z - a1.z ; // ray origin shifted so a1 = (0,0,0)
   
double c1 = n.x ;
double c2 = n.y ;
double c3 = n.z ;  // direction vector for cylinder
double t , d ;

double r_dot_n = ( ray.d.x*n.x + ray.d.y*n.y + ray.d.z*n.z ) ;
if ( r_dot_n < 0.999999999 && r_dot_n > -0.999999999 )
   { // ray not (anti) parallel to cylinder
   
   // (t * ray.d + o) cross c = t(d12+d23+d31) + o12+o23+o31
   
   double d12 = c2*ray.d.x - c1*ray.d.y ;
   double d23 = c3*ray.d.y - c2*ray.d.z ;
   double d31 = c1*ray.d.z - c3*ray.d.x ;
   double o12 = c2*o1 - c1*o2 ;
   double o23 = c3*o2 - c2*o3 ;
   double o31 = c1*o3 - c3*o1 ; // cross product terms
   
   double A = d12*d12 + d23*d23 + d31*d31 ;
   double B = 2 * (d23*o23 + d12*o12 + d31*o31 ) ;
   double C = o12*o12 + o23*o23 + o31*o31 - rad*rad ;
   
   double disc = B*B-4*A*C ;
   
   if ( disc < 0 )
      return null ;

   // hit infinite cylinder or cylinder parallel to ray, 
   // now check for finite cylinder wall
   double i1 , i2 , i3 ;
   t = ( -B - Math.sqrt(disc) ) / (2*A) ;
   if ( t >= 0 )
      {
      i1 = t*ray.d.x + ray.o.x ;
      i2 = t*ray.d.y + ray.o.y ;
      i3 = t*ray.d.z + ray.o.z ; // (i1, i2, i3) is intersection point
      
      d = i1*c1 + i2*c2 + i3*c3 + D1;
      if ( d >= 0 && d <= length )
         return this ; // farthest intersection is on finite cylinder wall
      }

   t = ( -B + Math.sqrt(disc) ) / (2*A) ;
   if ( t >= 0 )
      {
      i1 = t*ray.d.x + ray.o.x ;
      i2 = t*ray.d.y + ray.o.y ;
      i3 = t*ray.d.z + ray.o.z ; // (i1, i2, i3) is intersection point
      
      d = i1*c1 + i2*c2 + i3*c3 + D1;
      if ( d >= 0 && d <= length )
         return this ; // nearest intersection is on finite cylinder wall
      }
   }

// we could possibly hit the end caps without hitting the finite 
// cylinder wall so...
// test end cap a1
if ( (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) != 0 )
   { // ray not parallel to end cap
   t = - (c1*ray.o.x + c2*ray.o.y + c3*ray.o.z + D1) / 
      (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) ;
   if ( t >= 0 ) 
      {
      double i1 = t*ray.d.x + ray.o.x;
      double i2 = t*ray.d.y + ray.o.y;
      double i3 = t*ray.d.z + ray.o.z;
      
      d = ( (i1 - a1.x) * (i1 - a1.x) +
            (i2 - a1.y) * (i2 - a1.y) +
            (i3 - a1.z) * (i3 - a1.z) ) ;
      
      if ( d <= rad*rad )
         return this ; // hit end cap 1
      }
   }

// test end cap 2 just incase end cap 1 is behind us.
if ( (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) != 0 )
   {
   t = - ( (c1*ray.o.x + c2*ray.o.y + c3*ray.o.z + D2) / 
           (c1*ray.d.x + c2*ray.d.y + c3*ray.d.z) ) ;
   if ( t >= 0 )
      {
      double i1 = t*ray.d.x + ray.o.x;
      double i2 = t*ray.d.y + ray.o.y;
      double i3 = t*ray.d.z + ray.o.z;
      
      d = ( (i1 - a1.x) * (i1 - a1.x) +
            (i2 - a1.y) * (i2 - a1.y) +
            (i3 - a1.z) * (i3 - a1.z) ) ;
      
      if ( d <= rad*rad )
         return this ; // hit end cap 2 // probably never happens.
      }
   }

return null ; // we didn't hit it

} ;
// ************************************************************
} ; // end of class cylinder
// ************************************************************






/************************************************************
PPPP   OOO  L     Y   Y  GGG   OOO  N   N 
P   P O   O L      Y Y  G     O   O NN  N 
PPPP  O   O L       Y   G GGG O   O N N N 
P     O   O L       Y   G   G O   O N  NN 
P      OOO  LLLLL   Y    GGGG  OOO  N   N 
 ************************************************************/

// ************************************************************
class Polygon extends Prim // convex only
// ************************************************************
// ************************************************************
{
P [] v ;
P [] original_v ;
double A, B, C, D ;

void bound ( BBox b )
{
for ( int i = 0 ; i < v.length ; ++ i )
   {
   if ( v[i].x > b.xmax ) b.xmax = v[i].x ;
   if ( v[i].y > b.ymax ) b.ymax = v[i].y ;
   if ( v[i].z > b.zmax ) b.zmax = v[i].z ;
   if ( v[i].x < b.xmin ) b.xmin = v[i].x ;
   if ( v[i].y < b.ymin ) b.ymin = v[i].y ;
   if ( v[i].z < b.zmin ) b.zmin = v[i].z ;
   }
}

void transform ( M m )
{
for ( int i = 0 ; i < v.length ; ++ i )
   m.transform ( original_v[i] , v[i] ) ;
setupEquation() ;
//println ( ) ;

} ;

void print () 
{
System.out.print ( "Polygon plane " + A + "x + " + B + "y + " +
                   C + "z + " + D + " = 0" ) ;
for ( int i = 0 ; i < v.length ; ++i )
   {
   v[i].print() ;
   System.out.print ( " " ) ;
   }
} ;

void println ( )
{
System.out.println ( "Polygon plane " + A + "x + " + B + "y + " +
                   C + "z + " + D + " = 0" ) ;
for ( int i = 0 ; i < v.length ; ++i )
   {
   v[i].print() ;
   System.out.print ( " " ) ;
   }
System.out.println ( ) ;
};


Prim closestIntersection ( Ray ray ) // polygon.closestIntersection
{
if ( ISSVis.sray.d.x * A + ISSVis.sray.d.y * B + ISSVis.sray.d.z * C < 0 ) 
   {
   normal.x = -A ;
   normal.y = -B ;
   normal.z = -C ;
   }
else
   {
   normal.x = A ;
   normal.y = B ;
   normal.z = C ;
   }
return anyIntersection ( ray ) ;
} ;

// ************************************************************
Prim anyIntersection ( Ray ray )
 // polygon.simplifedCast
// ************************************************************
{
++ISSVis.one_cast ;
if ( ! visible ) return null ;
if ( Math.abs (A*ray.d.x + B*ray.d.y + C*ray.d.z) <= 0.000001 )
   return null ; // plane parallel to ray 

double t = - ( ( A*ray.o.x + B*ray.o.y + C*ray.o.z + D)  /
               (A*ray.d.x + B*ray.d.y + C*ray.d.z) ) ;
if ( t < 0 ) return null ; // behind us

double i1 = ray.o.x + t * ray.d.x ;
double i2 = ray.o.y + t * ray.d.y ;
double i3 = ray.o.z + t * ray.d.z ;

depth = i1*ray.d.x + i2*ray.d.y + i3*ray.d.z ; // for traditional

int sign = 0 ;
for ( int i = 0 ; i < v.length ; i ++ )
   {
   int j = (i+1)%v.length ;
   double c1 = ( ( v[j].z - v[i].z) * ( v[i].y - i2 ) - 
                 ( v[j].y - v[i].y) * ( v[i].z - i3 ) ) ;
   double c2 = ( ( v[j].x - v[i].x) * ( v[i].z - i3 ) - 
                 ( v[j].z - v[i].z) * ( v[i].x - i1 ) ) ;
   double c3 = ( ( v[j].y - v[i].y) * ( v[i].x - i1 ) - 
                 ( v[j].x - v[i].x) * ( v[i].y - i2 ) ) ;

   double side = ( c1*A + c2*B + c3*C ) ;
   if ( sign == 0 )
      sign = ( side > 0 ) ? 1 : -1 ;
   else if ( sign * side < 0 )
      return null ;
    }
return this ;
} ;
// ************************************************************
Polygon ( P p1 , P p2 , P p3 )
// ************************************************************
{
v = new P [3] ;
original_v = new P [3] ;

v[0] = new P ( p1 ) ;
v[1] = new P ( p2 ) ;
v[2] = new P ( p3 ) ;
original_v[0] = new P ( p1 ) ;
original_v[1] = new P ( p2 ) ;
original_v[2] = new P ( p3 ) ;

normal = new V ( ) ;
setupEquation ( ) ;
visible = true ;
basered = (short) (200 - (v[0].y+50000) / 100000 * 100 );
basegreen = basered ;
baseblue = basered ;
} ;
// ************************************************************
Polygon ( P [] p , int pc )
// ************************************************************
{
v = new P [pc] ;
original_v = new P [pc] ;
for (int i = 0 ; i < pc ; i ++ )
   {
   v[i] = new P ( p[i] ) ;
   original_v[i] = new P ( p[i] ) ;
   }
normal = new V ( ) ;
setupEquation ( )  ;
visible = true ;
basered = (short) (200 - (v[0].y+50000) / 100000 * 150 );
basegreen = basered ;
baseblue = basered ;
} ;

// ************************************************************
void setupEquation ( ) 
// ************************************************************
{
A = (v[0].z-v[1].z) * (v[2].y-v[1].y) - (v[0].y-v[1].y) * (v[2].z-v[1].z) ;
B = (v[0].x-v[1].x) * (v[2].z-v[1].z) - (v[0].z-v[1].z) * (v[2].x-v[1].x) ;
C = (v[0].y-v[1].y) * (v[2].x-v[1].x) - (v[0].x-v[1].x) * (v[2].y-v[1].y) ;
double n = Math.sqrt ( A*A + B*B + C*C ) ;
A /= n ;
B /= n ;
C /= n ;
D = 0 ;
for ( int i = 0 ; i < v.length ; ++ i )
   D -= ( A*v[0].x + B*v[0].y + C*v[0].z ) ; 
D /= v.length ;
normal.x = A ;
normal.y = B;
normal.z = C ;
} ;
// ************************************************************
} ; // end class Polygon 
// ************************************************************


// ************************************************************
/**************************************************************
TTTTT RRRR  III  AAA  N   N  GGG  L     EEEEE 
  T   R   R  I  A   A NN  N G     L     E     
  T   RRRR   I  AAAAA N N N G GGG L     EEE   
  T   R R    I  A   A N  NN G   G L     E     
  T   R  R  III A   A N   N  GGGG LLLLL EEEEE 
*************************************************************/
// ************************************************************
class Triangle {
// ************************************************************
P v1 , v2 , v3 , v12l , v23l , v31l , v12h , v23h , v31h , mid ;
Prim prim1 , prim2 , prim3 ;
int caseno ;
double value ;
double area ;

// class triangle is not used for ray casting, It is only used for
// shadow calculations via subdivision.

Triangle ( )
   {
   v1 = new P ( ) ;
   v2 = new P ( ) ;
   v3 = new P ( ) ;
   v12l = new P ( ) ;
   v23l = new P ( ) ;
   v31l = new P ( ) ;
   v12h = new P ( ) ;
   v23h = new P ( ) ;
   v31h = new P ( ) ;
   mid = new P ( ) ;
   prim1 = null ;
   prim2 = null ;
   prim3 = null ;
   caseno = -1 ;
   }
//************************************************************
} // end class Triangle 
//************************************************************


         





/************************************************************
III  SSS   SSS  V   V III  SSS  
 I  S     S     V   V  I  S     
 I   SSS   SSS   V V   I   SSS  
 I      S     S  V V   I      S 
III  SSS   SSS    V   III  SSS  
 ************************************************************/

// ************************************************************
class ISSVis
// ************************************************************
// ************************************************************
{
static String model_name ;
static V toSun = new V ( );
static Structure m ;
static M sunTransform;
static M inverse ;
//static int ss = 1 ;
static int ss = 1 ; // default
static double i_ambient = 0 ;
static double i_diffuse = 1.0 - i_ambient ;
static double blanketArea = 104.96 ;
static double stringArea = blanketArea / 41 ;
static double solar = 1371.3 ;
static double eff = 0.1 ;
public static int one_cast = 0 ;
public static int one_hits = 0 ;
public static int multi_cast = 0 ;
public static int any_cast = 0 ;
public static int all_cast = 0 ;
public static int closest_cast = 0 ;
public static int multi_hits = 0 ;
public static int any_hits = 0 ;
public static int all_hits = 0 ;
public static int closest_hits = 0 ;
public static int shadow_hits = 0 ;
public static Random alpha_rnd = new Random ( 42 ) ;
static int subdivide_countdown ;
static int temp_max_level ;
static int cutoff_threshold ;

// some random appearing variables are made static, preallocated and
// reused rather than being newed up a million times as local variables
static M [] stack ;
static int sp ;
P oc ; // offset center of sphere
static short [][] raster ;
static byte [][] big_raster ;
//static boolean [][] shadowed ;

static double time ;
static double beta = 75 ;
static double alpha = 0 ;
static double yaw = 0 ;
static int    step_number ;
static double view_beta = 45 ;
static double view_alpha = 60 ;
static boolean longerons_visible = false ;
static boolean show_power_by_step = false ;
static double [] control = new double [10] ;
//static int [] to_c ;
//static int [] a_to_c ;
//static int [] c_to_a ;
//static int [] from_c ;
static double [][] velocity_history ;
static double [][] position_history ;
static JFrame jf = null ;
static Bi bic ;
static double w = 12e4 ; // size of ISS local coordinate space to render
//static double w = 9.5e4 ;
static int rrez ;

static final int SUBDIVISION_LIMIT = 3000 ;
static final int MAX_TLEVEL = 40 ;
static int max_t_level = 10 ;
static int deepest_t_level = 0 ;
static Triangle [] triangle_stack = new Triangle [MAX_TLEVEL];
static PrimList [][] prim_stack = new PrimList [ MAX_TLEVEL ][ 3 ] ;
static int triangle_count = 0 ;
static int zero_vertices = 0 ;
static int one_vertex = 0 ;
static int two_vertices = 0 ;
static int three_vertices = 0 ;
static double one_vertex_area = 0 ;
static double two_vertex_area = 0 ;
static int depth_cutoff = 0 ;
static int size_cutoff = 0 ;
static int null_one_vertex = 0 ;
static int one_vertex_return_1 = 0 ;
static int one_vertex_one_edge = 0 ;
static int one_vertex_two_edges = 0 ;
static int one_edge_bisector_zero = 0 ;
static int one_edge_bisector_non_zero = 0 ;
static int binary_search_rays = 0 ;
static int namec = 0 ;

// adaptive subdivision
static final int MAX_ALEVEL = 40 ;
static final int ARES = 3 ;
static int max_frame = 92 ;
static int max_binary_search_level = 30 ;

static Ray sray = new Ray ( ) ; // direction always to sun

static P mid = new P ( ) ;
static P high = new P ( ) ;
static P low = new P ( ) ;
static P high2 = new P ( ) ;
static P low2 = new P ( ) ;
static Ray vray = new Ray ( ) ; // direction always to view_alpha, beta
static boolean rendering ;
static int res = 501 ;
static String mfile = "ISS_simple.model" ;
static boolean check_constraints = false ;
static int csv_cols = 0 ;
// ************************************************************
// staticly allocate some variables for speed
// ************************************************************
static void allocate ( int res ) 
// ************************************************************
{
stack = new M [100] ;
for ( int i = 0 ; i < 100 ; ++ i )
   stack[i] = new M ( ) ;
sp = 0 ;
raster = new short [res][res*3] ;
big_raster = new byte [res][res*6] ;
///shadowed = new boolean [res][res] ;
bic = new Bi ( res ) ;
}
// ************************************************************
static void play_movie ( String title ) 
// ************************************************************
{
while ( Drawer.getDrawer() != null )
   {
   if ( Drawer.getDrawer().animationMode )
      {
      int i = 0 ;
      synchronized ( Drawer.getDrawer().paintMutex )
         {
         i = (Drawer.getDrawer().curFrame + 1 ) % 
            Drawer.getDrawer().frameCnt ;
         }
      Drawer.getDrawer().displayFrame(i);
      }
   try { Thread.currentThread().sleep(100) ; } catch (Exception e) {} ;
   }

byte [][][] movie = Drawer.getDrawer().movie ;
int xz = movie[0][0].length/3 ;
int yz = movie[0].length ;
int fz = movie.length ;
int [][] oldpix = new int [yz][xz*3] ;
if ( jf == null )
   {
   jf = new JFrame ( ) ;
   jf.setSize ( xz+5 , yz+30 ) ;
   }
jf.setTitle ( "ISS " + title ) ;
for ( int y = 0 ; y < yz ; ++y )
   for ( int x = 0 ; x < xz*3 ; ++x )
      oldpix [y] [x] = 255 ; // based on fillRect below
try
   {
   Graphics2D g = (Graphics2D) bic.bi.getGraphics() ;

   g.setColor ( Color.WHITE ) ;
   g.fillRect ( 0 , 0 , xz-1 , yz-1 ) ;

   while ( true )
      {
      for ( int f = 0 ; f < fz ; ++f )
         {
         byte [][] pix = movie[f] ;
         for ( int y = 0 ; y < yz ; ++y )
            for ( int x = 0 ; x < xz*3 ; x+=3 ) 
               {
               int p = (((int)pix[y][x]) & 255) * 65536 ;
               p += (((int)pix[y][x+1]) & 255) * 256 ;
               p += (int)(pix[y][x+2]) & 255 ;
               if ( p != oldpix[y][x/3] )
                  {
                  bic.bi.setRGB ( x/3, y , p ) ;
                  oldpix[y][x/3] = p ;
                  }
               }
         // -----
         jf.getContentPane().repaint() ;
         Thread.currentThread().sleep(100) ;
         }
      }
   }
catch ( Exception e )
   {
   e.printStackTrace ( ) ;
   System.exit(77) ;
   }
}
// ************************************************************
void ck ( int a , int b , int c ) 
// ************************************************************
{
if ( a < b || a > c )
   {
   System.err.println ( "Error parsing ppm file" ) ;
   System.exit ( 99 ) ;
   }
}
// ************************************************************
void play_movie_file ( String name)
// ************************************************************
// not sure if this works at this point
{
try
   {
   FileInputStream in = new FileInputStream ( name ) ;
   byte [][] frame ;
   int c = 0 ;
   c = in.read ( ) ;
   ck ( c , 'P' , 'P' ) ;
   c = in.read ( ) ;
   ck ( c , '6' , '6' ) ;
   c = in.read ( ) ;
   ck ( c , 0 , ' ' ) ;
   int xr = 0 , yr = 0 , mv = 0 ;
   c = in.read ( ) ;
   while ( c >= '0' && c <= '0' )
      {
      xr = xr * 10 + c - '0' ;
      c = in.read ( ) ;
      }
   while ( c >= '0' && c <= '0' )
      {
      yr = yr * 10 + c - '0' ;
      c = in.read ( ) ;
      }
   while ( c >= '0' && c <= '0' )
      {
      mv = mv * 10 + c - '0' ;
      c = in.read ( ) ;
      }
   frame = new byte [yr][xr] ;
   for ( int f = 0 ; f < 92 ; ++f )
      {
      for ( int y = 0 ; y < yr ; ++y )
         for ( int x = 0 ; x < xr ; ++x )
            {
            c = in.read ( ) ;
            frame[y][x] = (byte) c ;
            }
      Drawer.getDrawer().setFrame ( f , frame , new String [0] , null, "" ) ;
      }
   play_movie ( name ) ;
   }
catch (Exception e)
   {
   System.err.println ( "Error: unable to open ppm file: " + name ) ;
   System.exit ( 999 ) ;
   }
}
// ************************************************************
void minus ( P e , P s , V d )
// ************************************************************
{
d.x = e.x - s.x ;
d.y = e.y - s.y ;
d.z = e.z - s.z ;
} ;
// ************************************************************
static M makeSunProjection ( double beta , double alpha ) 
// ************************************************************
{
// the sun is roughly in the -y direction for positive beta.
// so the vector to the sun is (0,-1,0) and from the sun is (0,1,0)
//
M proj = new M ( ) ;
M.rotate_y_axis ( -alpha , proj ) ;
proj.rotate_x_axis_compose ( 90-beta ,  proj ) ;
proj.rotate_y_axis_compose ( alpha , proj ) ;
return proj ;
}
// ************************************************************
static M makeViewProjection ( double beta , double alpha ) 
// ************************************************************
{
// the sun is roughly in the -y direction for positive beta.
// so the vector to the sun is (0,-1,0) and from the sun is (0,1,0)
//
M proj = new M ( ) ;
if ( beta > 0 ) 
   {
   M.rotate_y_axis ( -alpha , proj ) ;
   proj.rotate_x_axis_compose ( 90-beta ,  proj ) ;
   proj.rotate_y_axis_compose ( alpha , proj ) ;
   }
else
   {
   M.rotate_y_axis ( alpha , proj ) ;
   proj.rotate_x_axis_compose ( (90+beta) ,  proj ) ;
   proj.rotate_y_axis_compose ( -alpha , proj ) ;
   proj.rotate_z_axis_compose ( 180.0 , proj ) ;
   }
return proj ;
}
// ************************************************************
static M makeInverseSunProjection ( double beta , double orbitPos ) 
// ************************************************************
{
M proj = new M ( ) ;
M.rotate_y_axis ( -orbitPos , proj ) ;
proj.rotate_x_axis_compose ( -(90-beta) , proj ) ;
proj.rotate_y_axis_compose ( orbitPos , proj ) ;
return proj ;
}
// ************************************************************
static void color_a_pixel ( short [][] raster , int x , int y , 
                            Prim the_hit , Prim shadower )
// ************************************************************
{
x *= 3 ;
if ( the_hit == null )
   {
   // if you use the BBox line instead of the 255 line
   // it draws the bounding boxes (sort of) transparently
   // fun to do for debugging
   //
   // raster[y][x] = (byte) ( 255 - BBox.hit * 10 ) ;
   raster[y][x] = (short) ( 255 ) ;
   raster[y][x+1] = (short) ( 255 ) ;
   raster[y][x+2] = (short) ( 255 ) ; // white background
   }
else
   {
   if ( shadower == null )
      {
      raster[y][x] = (short) the_hit.red ;
      raster[y][x+1] = (short) the_hit.green ;
      raster[y][x+2] = (short) the_hit.blue ;
      }
   else
      {
      raster[y][x] = (short) (Math.max(the_hit.red - 100, 0)/2) ;
      raster[y][x+1] = (short) (Math.max(the_hit.green - 100, 0)/2) ;
      raster[y][x+2] = (short) (Math.max(the_hit.blue - 100, 0)/2 ) ;
      }
   }
}
// ************************************************************
static void render_from_view ( int res , Structure m , 
                               M view_proj , M inverse_view ,
                               V to_sun )
// ************************************************************
{
// we assume, the viewpoint is in the direction of the -y axis
// transformed by "proj" (for historical reasons)
// so we scan in the x-z plane (transformed by proj)
if ( res < 3 ) return ;
rrez = res ;
V         view     = new V ( 0 , 1 , 0 ) ; // from sun untransfromed (ad hoc)
P         base     = new P ( ) ;
Prim      the_hit  = null ;
Prim      shadower = null ;
P         p        = new P ( ) ;
boolean [][] resample = new boolean [res][res] ;
Prim [][] hitter = new Prim [res][res] ;
double dz = w/(res-1) ;
double dx = w/(res-1) ;

//view_proj.transform ( to_sun , to_sun ) ;
//to_sun.println ( ) ;

view_proj.transform ( view , vray.d ) ; // now transformed

for ( int iy = 0 ; iy < res ; ++iy ) // image coordinates
   {
   double z = -w/2 + iy*dz ;
   int jx = -1 ;
   for ( int ix = 0 ; ix < res; ++ix )
      {
      double x = w/2 - ix*dx ;
      p.x = x ;
      p.y = 0 ;
      p.z = z ;
      view_proj.transform ( p , base ) ;

      vray.o.x = base.x - 1e6*vray.d.x ;
      vray.o.y = base.y - 1e6*vray.d.y ;
      vray.o.z = base.z - 1e6*vray.d.z ;

      BBox.hit = 0 ;
      shadower = null ;
      the_hit = m.closestIntersection ( vray ) ;
      hitter [iy][ix] = the_hit ;
      boolean bw = false ;
      if ( the_hit != null )
         {
         the_hit.visible = false ; // do not shadow self
         sray.o.x = base.x + vray.d.x * the_hit.depth ;
         sray.o.y = base.y + vray.d.y * the_hit.depth ;
         sray.o.z = base.z + vray.d.z * the_hit.depth ;
         shadower = m.anyIntersection ( sray ) ;
         the_hit.visible = true ; // make self visible again

         double diffuse = ( to_sun.x * the_hit.normal.x + 
                            to_sun.y * the_hit.normal.y + 
                            to_sun.z * the_hit.normal.z )  ;
         if ( diffuse < 0 ) 
            diffuse = 0 ;
         diffuse = Math.sqrt(diffuse) ; // gamma
         the_hit.red = (short) ( ( ISSVis.i_diffuse * diffuse + 
                                   ISSVis.i_ambient ) * 
                                 the_hit.basered ) ;
         the_hit.green = (short) ( ( ISSVis.i_diffuse * diffuse + 
                                     ISSVis.i_ambient ) *
                                   the_hit.basegreen ) ;
         the_hit.blue = (short) ( ( ISSVis.i_diffuse * diffuse + 
                                    ISSVis.i_ambient ) *
                                  the_hit.baseblue ) ;
         }
      color_a_pixel ( raster , ix , iy , the_hit , shadower ) ;
      }
   }
Random rnd = new Random ( ) ;

if ( ss > 1 )
   {
   for ( int iy = 1 ; iy < res-1 ; ++iy )
      {
      for ( int ix = 1 ; ix < res-1 ; ++ix )
         {
         if ( hitter[iy][ix] != hitter[iy][ix+1] ||
              hitter[iy][ix] != hitter[iy][ix-1] ||
              hitter[iy][ix] != hitter[iy+1][ix] ||
              hitter[iy][ix] != hitter[iy-1][ix] ||
              
              raster[iy][ix*3-3] != raster[iy][ix*3+0] ||
              raster[iy][ix*3-2] != raster[iy][ix*3+1] ||
              raster[iy][ix*3-1] != raster[iy][ix*3+2] ||
              raster[iy][ix*3+0] != raster[iy][ix*3+3] ||
              raster[iy][ix*3+1] != raster[iy][ix*3+4] ||
              raster[iy][ix*3+2] != raster[iy][ix*3+5] ||
              
              raster[iy-1][ix*3+0] != raster[iy][ix*3+0] ||
              raster[iy-1][ix*3+1] != raster[iy][ix*3+1] ||
              raster[iy-1][ix*3+2] != raster[iy][ix*3+2] ||
              raster[iy][ix*3+0] != raster[iy+1][ix*3+0] ||
              raster[iy][ix*3+1] != raster[iy+1][ix*3+1] ||
              raster[iy][ix*3+2] != raster[iy+1][ix*3+2] )
            {
            resample [iy][ix] = true ;

            resample [iy+1][ix+1] = true ;
            resample [iy+1][ix-1] = true ;
            resample [iy-1][ix+1] = true ;
            resample [iy-1][ix-1] = true ;

            resample [iy][ix+1] = true ;
            resample [iy][ix-1] = true ;
            resample [iy+1][ix] = true ;
            resample [iy-1][ix] = true ;
            }
         //resample [iy][ix] = true ;
         }
      }
   }

if ( ss > 1 ) 
   {
   for ( int iy = 1 ; iy < res-1 ; ++iy )
      {
      for ( int ix = 1 ; ix < res-1 ; ++ix )
         {
         if ( resample[iy][ix] )
            {
            double zs = -w/2 + (iy-0.5)*dz ;
            double xs =  w/2 - (ix-0.5)*dx ;
            double rd = 0 , bd = 0 , gd = 0 ;
            
            for ( int jz = 0 ; jz < ss ; ++jz )
               {
               for ( int jx = 0 ; jx < ss ; ++jx )
                  {
                  double z = zs + dz*(jz+rnd.nextDouble())/ss ;
                  double x = xs + dx*(jx+rnd.nextDouble())/ss ;
                  
                  p.x = x ;
                  p.y = 0 ;
                  p.z = z ;
                  view_proj.transform ( p , base ) ;
                  vray.o.x = base.x - 1e6*vray.d.x ;
                  vray.o.y = base.y - 1e6*vray.d.y ;
                  vray.o.z = base.z - 1e6*vray.d.z ;
                  BBox.hit = 0 ;
                  shadower = null ;
                  the_hit = m.closestIntersection ( vray ) ;
                  boolean bw = false ;
                  
                  if ( the_hit != null )
                     {
                     sray.o.x = base.x + the_hit.depth * vray.d.x ;
                     sray.o.y = base.y + the_hit.depth * vray.d.y ;
                     sray.o.z = base.z + the_hit.depth * vray.d.z ;
                     the_hit.visible = false ;
                     shadower = m.closestIntersection ( sray ) ;
                     the_hit.visible = true ;
                     double diffuse = ( to_sun.x * the_hit.normal.x + 
                                        to_sun.y * the_hit.normal.y + 
                                        to_sun.z * the_hit.normal.z )  ;
                     if ( diffuse < 0 ) 
                        diffuse = 0 ;
                     diffuse = Math.sqrt(diffuse) ; // gamma
                     the_hit.red = (short) ( ( ISSVis.i_diffuse * diffuse + 
                                               ISSVis.i_ambient ) * 
                                             the_hit.basered ) ;
                     the_hit.green = (short) ( ( ISSVis.i_diffuse * diffuse + 
                                                 ISSVis.i_ambient ) *
                                               the_hit.basegreen ) ;
                     the_hit.blue = (short) ( ( ISSVis.i_diffuse * diffuse + 
                                                ISSVis.i_ambient ) *
                                              the_hit.baseblue ) ;
                     }
                  color_a_pixel ( raster , ix , iy , the_hit , shadower ) ;
                  rd += raster[iy][ix*3] ;
                  gd += raster[iy][ix*3+1] ;
                  bd += raster[iy][ix*3+2] ;
                  }
               }
            raster[iy][ix*3]   = (short) (rd/(ss*ss)+0.5) ;
            raster[iy][ix*3+1] = (short) (gd/(ss*ss)+0.5) ;
            raster[iy][ix*3+2] = (short) (bd/(ss*ss)+0.5) ;
            }
         }
      }
   }
} ;
// ************************************************************
static void render_from_sun ( int res , Structure m , M proj )
// ************************************************************
{
// we assume, the sun is roughly in the direction of the -y axis
// for large positive beta angles, transformed by "proj"
// so we scan in the x-z plane (transformed by proj)
if ( res < 3 ) return ;
rrez = res ;
Prim the_hit = null ;

boolean [][] resample = new boolean [res][res] ;
Prim [][] hitter = new Prim [res][res] ;
Random rnd = new Random ( ) ;

V from_sun = new V ( 0 , 1 , 0 ) ; // from sun untransfromed (ad hoc)
proj.transform ( from_sun , vray.d ) ; // now transformed
V to_sun = new V ( 0 , -1 , 0 ) ; // from sun untransfromed (ad hoc)
proj.transform ( to_sun , to_sun ) ; // now transformed
double dz = w/(res-1) ;
double dx = w/(res-1) ;
P p = new P ( 0 , 0 , 0) ;

for ( int iy = 0 ; iy < res ; ++iy ) // image coordinates
   {
   double z = -w/2 + iy*dz ;
   int jx = -1 ;
   for ( int ix = 0 ; ix < res; ++ix )
      {
      double x = w/2 - ix*dx ;

      p.x = x ;
      p.y = 0 ;
      p.z = z ;
      proj.transform ( p , vray.o ) ;

      vray.o.x = vray.o.x - 1e7*vray.d.x ;
      vray.o.y = vray.o.y - 1e7*vray.d.y ;
      vray.o.z = vray.o.z - 1e7*vray.d.z ;
      BBox.hit = 0 ;
      the_hit = m.closestIntersection ( vray ) ; // from
      hitter [iy][ix] = the_hit ;
      if ( the_hit == null )
         {
         // if you use the BBox line instead of the 255 line
         // it draws the bounding boxes (sort of) transparently
         // fun to do for debugging
         //
         // raster[y][x] = (byte) ( 255 - BBox.hit * 10 ) ;
         raster[iy][++jx] = (short) ( 255 ) ;
         raster[iy][++jx] = (short) ( 255 ) ;
         raster[iy][++jx] = (short) ( 255 ) ;
         }
      else
         {
         double diffuse = the_hit.normal.dot(to_sun) ;
         if ( diffuse < 0 )
            diffuse = 0 ;
         diffuse = Math.sqrt ( diffuse ) ;
         raster[iy][++jx] = (short) ( ( ISSVis.i_diffuse * diffuse +
                                        ISSVis.i_ambient ) * 
                                      the_hit.basered ) ;
         raster[iy][++jx] = (short) ( ( ISSVis.i_diffuse * diffuse +
                                        ISSVis.i_ambient ) * 
                                      the_hit.basegreen ) ;
         raster[iy][++jx] = (short) ( ( ISSVis.i_diffuse * diffuse +
                                        ISSVis.i_ambient ) * 
                                      the_hit.baseblue ) ;
         }
      }
   }
int ac = 0 ;
if ( ss > 1 )
   {
   for ( int iy = 1 ; iy < res-1 ; ++iy )
      {
      for ( int ix = 1 ; ix < res-1 ; ++ix )
         {
         if ( hitter[iy][ix] != hitter[iy][ix+1] ||
              hitter[iy][ix] != hitter[iy][ix-1] ||
              hitter[iy][ix] != hitter[iy+1][ix] ||
              hitter[iy][ix] != hitter[iy-1][ix] ||
              
              raster[iy][ix*3-3] != raster[iy][ix*3+0] ||
              raster[iy][ix*3-2] != raster[iy][ix*3+1] ||
              raster[iy][ix*3-1] != raster[iy][ix*3+2] ||
              raster[iy][ix*3+0] != raster[iy][ix*3+3] ||
              raster[iy][ix*3+1] != raster[iy][ix*3+4] ||
              raster[iy][ix*3+2] != raster[iy][ix*3+5] ||
              
              raster[iy-1][ix*3+0] != raster[iy][ix*3+0] ||
              raster[iy-1][ix*3+1] != raster[iy][ix*3+1] ||
              raster[iy-1][ix*3+2] != raster[iy][ix*3+2] ||
              raster[iy][ix*3+0] != raster[iy+1][ix*3+0] ||
              raster[iy][ix*3+1] != raster[iy+1][ix*3+1] ||
              raster[iy][ix*3+2] != raster[iy+1][ix*3+2] )
            {
            resample [iy][ix] = true ;

            resample [iy+1][ix+1] = true ;
            resample [iy+1][ix-1] = true ;
            resample [iy-1][ix+1] = true ;
            resample [iy-1][ix-1] = true ;

            resample [iy][ix+1] = true ;
            resample [iy][ix-1] = true ;
            resample [iy-1][ix] = true ;
            resample [iy+1][ix] = true ;
            }
         //resample [iy][ix] = true ;
         }
      }
   }

if ( ss > 1 ) 
   {
   for ( int iy = 1 ; iy < res-1 ; ++iy )
      {
      for ( int ix = 1 ; ix < res-1 ; ++ix )
         {
         if ( resample[iy][ix] )
            {
            ++ac ;
            double zs = -w/2 + (iy-0.5)*dz ;
            double xs = w/2 - (ix-0.5)*dx ;
            double rd = 0 , bd = 0 , gd = 0 ;
            
            for ( int jz = 0 ; jz < ss ; ++jz )
               {
               for ( int jx = 0 ; jx < ss ; ++jx )
                  {
                  double z = zs + dz*(jz+rnd.nextDouble())/ss ;
                  double x = xs + dx*(jx+rnd.nextDouble())/ss ;
                  
                  p.x = x ;
                  p.y = 0 ;
                  p.z = z ;
                  proj.transform ( p , vray.o ) ;
                  
                  vray.o.x = vray.o.x - 1e7*vray.d.x ;
                  vray.o.y = vray.o.y - 1e7*vray.d.y ;
                  vray.o.z = vray.o.z - 1e7*vray.d.z ;
                  BBox.hit = 0 ;
                  Prim sub_hit = m.closestIntersection ( vray ) ;
                  if ( sub_hit == null )
                     {
                     rd += 255 ;
                     gd += 255 ;
                     bd += 255 ;
                     }
                  else
                     {
                     /* color_a_pixel ( raster , ix , iy , sub_hit , null ) ; */
                     double diffuse = sub_hit.normal.dot(to_sun) ;
                     if ( diffuse < 0 )
                        diffuse = 0 ;
                     diffuse = Math.sqrt ( diffuse ) ;
                     /* rd += raster[iy][ix*3] * diffuse ; */
                     /* gd += raster[iy][ix*3+1] * diffuse ; */
                     /* bd += raster[iy][ix*3+2] * diffuse ; */
                     rd += (short) ( ( ISSVis.i_diffuse * diffuse +
                                        ISSVis.i_ambient ) * 
                                      sub_hit.basered ) ;
                     gd += (short) ( ( ISSVis.i_diffuse * diffuse +
                                        ISSVis.i_ambient ) * 
                                      sub_hit.basegreen ) ;
                     bd += (short) ( ( ISSVis.i_diffuse * diffuse +
                                        ISSVis.i_ambient ) * 
                                      sub_hit.baseblue ) ;
                     }
                  }
               }
            raster[iy][ix*3] = (short ) (rd/(ss*ss)+0.5) ;
            raster[iy][ix*3+1] = (short ) (gd/(ss*ss)+0.5) ;
            raster[iy][ix*3+2] = (short ) (bd/(ss*ss)+0.5) ;
            }
         }
      }
   }
//System.out.println ( ac + " " + ss ) ;
}
// ************************************************************
static void rotate_BGA ( Structure m , V axis ,
                         double angle , int side , int saw ) 
// ************************************************************
{
M trans = m.child[side].child[saw].transform ;
M.negative_translate ( axis , trans ) ;
trans.rotate_x_axis_compose ( angle , trans ) ;
trans.translate_compose ( axis , trans ) ;
}
// ************************************************************
static void rotate_SARJ ( Structure m , V axis ,
                          double angle , int sarj ) 
// ************************************************************
{
M trans = m.child[sarj].transform ;
M.negative_translate ( axis , trans ) ;
trans.rotate_y_axis_compose ( angle , trans ) ;
trans.translate_compose ( axis , trans ) ;
}
// ************************************************************
static void rotate_SARJ_BGA ( Structure m , double [] ang )
// ************************************************************
{
/*
 * assumed structure of ISS model (ISS_simple.model)
 * SAW = Solar Array Wing
 *
 * m = whole ISS
 *
 * m.child[0] = starboard truss with starboard SARJ
 * m.child[1] = port truss with port SARJ
 *
 * m.child[0].child[0] is first SAW in starboard truss with BGA 1A
 * m.child[0].child[1] is second SAW in starboard truss with BGA 3A
 * m.child[0].child[2] is third SAW in starboard truss with BGA 1B
 * m.child[0].child[3] is fourth SAW in starboard truss with BGA 3B
 *
 * m.child[1].child[0] is first SAW in port truss with BGA 2A
 * m.child[1].child[1] is second SAW in port truss with BGA 4A
 * m.child[1].child[2] is third SAW in port truss with BGA 2B
 * m.child[1].child[3] is fourth SAW in port truss with BGA 4B
 *
 * m.child[].child[].solid[2] is one panel
 * m.child[].child[].solid[2].obj[0] is the polygon
 * m.child[].child[].solid[3] is the other panel
 * m.child[].child[].solid[3].obj[0] is the polygon
 *
 * m.child[].child[].solid[4] is the longerons
 * m.child[].child[].solid[4].obj[0-3] are the four cylinders
 * 
 */

// x translation doesn't matter here since BGA rotation axes are 
// parallel to the x axis
// similar to y translation for SARJs
V [] rotation_axis = new V [10 ] ;

rotation_axis [0] = new V ( 5 , 0 , 11 );  // ssarj
rotation_axis [1] = new V ( 5 , 0 , 11 );  // psarj
rotation_axis [2] = new V ( 0 , 33350 , -750 ) ;  // 1A 
rotation_axis [3] = new V ( 0 , -33350 , -750 ) ; // 2A 
rotation_axis [4] = new V ( 0 , 33350 ,  750 ) ;  // 3A 
rotation_axis [5] = new V ( 0 , -33350 , 750 ) ;  // 4A 
rotation_axis [6] = new V ( 0 , 48455 ,  750 ) ;  // 1B 
rotation_axis [7] = new V ( 0 , -48455 , 750 ) ;  // 2B
rotation_axis [8] = new V ( 0 , 48455 , -750 ) ;  // 3B
rotation_axis [9] = new V ( 0 , -48455 , -750 ) ; // 4B 

int [] rotation_direction = new int [10] ;

rotation_direction [0] = 1 ;
rotation_direction [1] = -1 ;
rotation_direction [2] = 1 ;
rotation_direction [3] = -1 ;
rotation_direction [4] = -1 ;
rotation_direction [5] = 1 ;
rotation_direction [6] = -1 ;
rotation_direction [7] = 1 ;
rotation_direction [8] = 1 ;
rotation_direction [9] = -1 ;

int [] rotation_angle_offset = new int [10 ] ;

rotation_angle_offset [0] = 0 ;        // ssarj
rotation_angle_offset [1] = 0 ;        // psarj
rotation_angle_offset [2] = 90 ;       // 1A
rotation_angle_offset [3] = 90 ;       // 2A
rotation_angle_offset [4] = 270 ;      // 3A
rotation_angle_offset [5] = 270 ;      // 4A
rotation_angle_offset [6] = 270 ;      // 1B
rotation_angle_offset [7] = 270 ;      // 2B
rotation_angle_offset [8] = 90 ;       // 3B
rotation_angle_offset [9] = 90 ;       // 4B

// SAWS in model are upside down
for ( int i = 2 ; i < 10 ; i ++ ) 
   rotation_angle_offset[i] += 180 ;

M.rotate_z_axis ( -yaw , m.transform ) ;

for ( int s = 0 ; s < 10 ; s ++ )
   ang[s] = ( ang[s]-rotation_angle_offset[s]) *
      rotation_direction[s] ;

rotate_SARJ ( m , rotation_axis[0] , ang[0] , 0 ) ;     // ssarj
rotate_SARJ ( m , rotation_axis[1] , ang[1] , 1 ) ;     // psarj
rotate_BGA ( m , rotation_axis[2] ,  ang[2] , 0 , 0 ) ; // 1A
rotate_BGA ( m , rotation_axis[3] ,  ang[3] , 1 , 0 ) ; // 2A
rotate_BGA ( m , rotation_axis[4] ,  ang[4] , 0 , 1 ) ; // 3A
rotate_BGA ( m , rotation_axis[5] ,  ang[5] , 1 , 1 ) ; // 4A
rotate_BGA ( m , rotation_axis[6] ,  ang[6] , 0 , 2 ) ; // 1B
rotate_BGA ( m , rotation_axis[7] ,  ang[7] , 1 , 2 ) ; // 2B
rotate_BGA ( m , rotation_axis[8] ,  ang[8] , 0 , 3 ) ; // 3B
rotate_BGA ( m , rotation_axis[9] ,  ang[9] , 1 , 3 ) ; // 4B

V port_radiator_axis       = new V ( -250 , -14600 , -5.5 ) ;
V starboard_radiator_axis  = new V ( -250 , 14694 , -23 ) ;
double port_angle = 45 ;
double starboard_angle = ( ( beta > 0 ) ? 25 : 60 ) ;

/* // 29 30 31 32 */
M trans = m.child[3].transform ;
M.negative_translate ( starboard_radiator_axis , trans ) ;
trans.rotate_x_axis_compose ( starboard_angle , trans ) ;
trans.translate_compose ( starboard_radiator_axis , trans ) ;

trans = m.child[2].transform ;
M.negative_translate ( port_radiator_axis , trans ) ;
trans.rotate_x_axis_compose ( port_angle , trans ) ;
trans.translate_compose ( port_radiator_axis , trans ) ;

} ;
// ************************************************************
static void interp ( P p1 , P p2 , double alpha , P result )
// ************************************************************
{ // faster in place
result.x = p1.x * (1.0-alpha) + p2.x * alpha ;
result.y = p1.y * (1.0-alpha) + p2.y * alpha ;
result.z = p1.z * (1.0-alpha) + p2.z * alpha ;
}
// ************************************************************
public static void set_time_step ( int step , double beta ) 
// ************************************************************
{
double alpha = 360.0 * step / 92 ;
sunTransform = makeSunProjection ( beta , alpha /*orbit*/ ) ;
inverse = makeInverseSunProjection ( beta , alpha /*orbit*/ ) ;
toSun = new V ( ) ;
sunTransform.transform ( new V ( 0 , -1 , 0 ) , toSun ) ;
}
// ************************************************************
// ************************************************************
// ************************************************************
// ************************************************************
// ************************************************************
static double binary_search ( P lowin , P highin , 
                              P lowout , P highout ,
                              Prim prim )
// ************************************************************
{
double hx , hy , hz ;
double lx , ly , lz ;
double mx , my , mz ;

hx = highin.x ;
hy = highin.y ;
hz = highin.z ;

lx = lowin.x ;
ly = lowin.y ;
lz = lowin.z ;

int search_level = 0 ;
double v = 0.0 ;
double s = 1.0 ;
while ( search_level < max_binary_search_level )
   {
   mx = (hx + lx) * 0.5 ;
   my = (hy + ly) * 0.5 ;
   mz = (hz + lz) * 0.5 ;

   sray.o.x = mx ;
   sray.o.y = my ;
   sray.o.z = mz ;
   Prim q = prim.anyIntersection ( sray ) ;
   ++ binary_search_rays ;
   
   s *= 0.5 ;
   if ( q != null )
      {
      lx = mx ;
      ly = my ;
      lz = mz ;
      v += s ;
      }
   else
      {
      hx = mx ;
      hy = my ;
      hz = mz ;
      }
   ++ search_level ;
   }
lowout.x = lx ;
lowout.y = ly ;
lowout.z = lz ;
highout.x = hx ;
highout.y = hy ;
highout.z = hz ;
return v ;
}
// ************************************************************
static int primCount ( int level ,
                       int p1 , int i1 , 
                       int p2 , int i2 ,
                       int p3 , int i3 ) 
// ************************************************************
{
PrimList [] p = prim_stack[level] ;

if ( i1 >= p[p1].s || ! p[p1].valid[i1] ) 
   return 0 ;
int c = 1 ;
if ( i2 < p[p2].s && p[p2].valid[i2] )
   {
   if ( p[p2].prim[i2].order < 
        p[p1].prim[i1].order )
      return 0 ;
   else if ( p[p2].prim[i2].order == 
             p[p1].prim[i1].order )
      c = 2 ;
   }
if ( i3 < p[p3].s && p[p3].valid[i3] )
   {
   if ( p[p3].prim[i3].order < 
        p[p1].prim[i1].order )
      return 0 ;
   else if ( p[p3].prim[i3].order == 
             p[p1].prim[i1].order )
      c ++  ;
   }
return c ;
}
// ************************************************************
static double subdivide_triangle ( int level , P v1 , P v2 , P v3 ,
                                   double area )
// ************************************************************
{
// We accumulate an error of about 2^(-max_binary_search_level)
// for every level of recursion

if ( level >= temp_max_level )
   return 0 ;

Triangle t = triangle_stack [level] ;
PrimList [] p = prim_stack [level] ;

t.v1.x = v1.x ;
t.v1.y = v1.y ;
t.v1.z = v1.z ;
sray.o.x = v1.x ;
sray.o.y = v1.y ;
sray.o.z = v1.z ;
p[0].reset() ;
m.allIntersections ( sray , p[0] ) ;

t.v2.x = v2.x ;
t.v2.y = v2.y ;
t.v2.z = v2.z ;
sray.o.x = v2.x ;
sray.o.y = v2.y ;
sray.o.z = v2.z ;
p[1].reset() ;
m.allIntersections ( sray , p[1] ) ;

t.v3.x = v3.x ;
t.v3.y = v3.y ;
t.v3.z = v3.z ;
sray.o.x = v3.x ;
sray.o.y = v3.y ;
sray.o.z = v3.z ;
p[2].reset() ;
m.allIntersections ( sray , p[2] ) ;

//System.out.println ( p[0].s + " " + p[1].s + " " + p[2].s ) ;
return subdivide_triangle ( level , area ) ;
}
// ************************************************************
static double subdivide_triangle ( int level , double area )
// ************************************************************
{
// at this point triangle_stack[level] and prim_stack[level] are
// set to valid values for the current triangle to subdivide
//

if ( area < 1e-3 )
   return 0 ;

if ( --subdivide_countdown <= 0 )
   return 0 ;

if ( deepest_t_level < level ) 
   deepest_t_level = level ;
++ triangle_count ;

final double d_thresh_2 = 0.00001 ;
Triangle t = null ;
PrimList [] p = null ;
if ( level < temp_max_level )
   {
   t = triangle_stack [level] ;
   p = prim_stack [level] ;
   }
else
   {
   ++ depth_cutoff ;
   return 0 ;
   }
if ( area < 0.0000001 )
   {
   ++ size_cutoff ;
   return 0 ;
   }
                             

if ( p[0].s == 0 && p[1].s == 0 && p[2].s == 0 )
   {
   ++ zero_vertices ;
   return 0.0 ;
   }


int s0 = 0 ; 
int s1 = 0 ;
int s2 = 0 ;
while ( p[0].s > s0 && p[1].s > s1 && p[2].s > s2 )
   {
   if ( primCount ( level , 1 , s1 , 2 , s2 , 0, s0 ) == 3 )
      {
      ++ three_vertices ;
      return area ;
      }
   if ( primCount ( level , 1 , s1 , 2 , s2 , 0 , s0 ) > 0 )
      ++ s1 ;
   else if ( primCount ( level , 2 , s2 , 1 , s1 , 0 , s0 ) > 0 )
      ++ s2 ;
   else
      ++ s0 ;
   }

s1 = 0 ;
s2 = 0 ;
s0 = 0 ; 
double a ;
while ( p[1].s > s1 || p[2].s > s2 || p[0].s > s0 )
   {
   if ( primCount ( level , 1 , s1 , 2 , s2 , 0 , s0 ) == 2 &&
        primCount ( level , 2 , s2 , 1 , s1 , 0 , s0 ) == 2 ) 
      {
      a = subdivide_triangle2( level , t.v2 , 1 , t.v3 , 2 , t.v1 , 0 , 
                               p[1].prim[s1] , area ) ;
      if ( a > 0 ) 
         return a ;
      p[1].valid[s1] = false ;
      p[2].valid[s2] = false ;
      }
   else if ( primCount ( level , 2 , s2 , 1 , s1 , 0 , s0 ) == 2 &&
             primCount ( level , 0 , s0 , 2 , s2 , 1 , s1 ) == 2 ) 
      {
      a = subdivide_triangle2 ( level , t.v1 , 0 , t.v3 , 2 , t.v2 , 1 ,
                                p[0].prim[s0] , area ) ;
      if ( a > 0 )
         return a ;
      p[0].valid[s0] = false ;
      p[2].valid[s2] = false ;
      }
   else if ( primCount ( level , 1 , s1 , 2 , s2 , 0 , s0 ) == 2 &&
             primCount ( level , 0 , s0 , 2 , s2 , 1 , s1 ) == 2 ) 
      {
      a = subdivide_triangle2 ( level , t.v1 , 0 , t.v2 , 1 , t.v3 , 2 ,
                                p[1].prim[s1] , area ) ;
      if ( a > 0 ) 
         return a ;
      p[1].valid[s1] = false ;
      p[0].valid[s0] = false ;
      }

   if ( ( s1 < p[1].s && ! p[1].valid[s1] ) ||
        ( s2 < p[2].s && ! p[2].valid[s2] ) ||
        ( s0 < p[0].s && ! p[0].valid[s0] ) )
      {
      while ( s1 < p[1].s && ! p[1].valid[s1] ) 
         ++s1 ;
      while ( s2 < p[2].s && ! p[2].valid[s2] ) 
         ++s2 ;
      while ( s0 < p[0].s && ! p[0].valid[s0] ) 
         ++s0 ;
      }
   else
      {
      if (  primCount ( level , 1 , s1 , 2 , s2 , 0 , s0 ) > 0 )
         ++ s1 ;
      else if ( primCount ( level , 2 , s2 , 1 , s1 , 0 , s0 ) > 0 )
         ++ s2 ;
      else
         ++ s0 ;
      }
   }

s0 = 0 ; 
s1 = 0 ;
s2 = 0 ;
while ( s0 < p[0].s && ! p[0].valid[s0] ) 
   ++s0 ;
while ( s1 < p[1].s && ! p[1].valid[s1] ) 
   ++s1 ;
while ( s2 < p[2].s && ! p[2].valid[s2] ) 
   ++s2 ;
while ( p[1].s > s1 || p[2].s > s2 || p[0].s > s0 )
   {
   if ( primCount ( level , 1 , s1 , 2 , s2 , 0 , s0 ) > 0 )
      {
      a = subdivide_triangle1 ( level , t.v2 , 1 , t.v1 , 0 , t.v3 , 2 ,
                                p[1].prim[s1] , area ) ;
      if ( a > 0 )
         return a ;
      }
   else if ( primCount ( level , 2 , s2 , 1 , s1 , 0 , s0 ) > 0 )
      {
      a = subdivide_triangle1 ( level , t.v3 , 2 , t.v2 , 1 , t.v1 , 0 ,
                                p[2].prim[s2] , area ) ;
      if ( a > 0 )
         return a ;
      }
   else if ( primCount ( level , 0 , s0 , 2 , s2 , 1 , s1 ) > 0 ) 
      {
      a = subdivide_triangle1 ( level , t.v1 , 0 , t.v2 , 1 , t.v3 , 2 ,
                                p[0].prim[s0] , area ) ;
      if ( a > 0 )
         return a ;
      }
   
   if (  primCount ( level , 1 , s1 , 2 , s2 , 0 , s0 ) > 0 )
      ++ s1 ;
   else if ( primCount ( level , 2 , s2 , 1 , s1 , 0 , s0 ) > 0 )
      ++ s2 ;
   else
      ++ s0 ;
   while ( s1 < p[1].s && ! p[1].valid[s1] ) 
      ++s1 ;
   while ( s2 < p[2].s && ! p[2].valid[s2] ) 
      ++s2 ;
   while ( s0 < p[0].s && ! p[0].valid[s0] ) 
      ++s0 ;
   }

return 0.0 ;
}
// ************************************************************
static double subdivide_triangle2 ( int level , 
                                    P v1 , int s1 ,
                                    P v2 , int s2 ,
                                    P v3 , int s3 ,
                                    Prim prim , double area )
// ************************************************************
{
// v1 and v2 are covered by prim, v3 is not
// pointers to vertices are used instead of level so which vertices
// are covered is expresssed. Similarly which prim does the covering
// is directly passed in

++two_vertices ;

Triangle t1 = triangle_stack [level] ; // only so we can use other
// vertices as preallocated temporary variables

double a23 = binary_search ( v2 , v3 , t1.v23l , t1.v23h , prim ) ;
double a13 = binary_search ( v1 , v3 , t1.v31l , t1.v31h , prim ) ;

// case 1
// non-zero area trapeziod covered
// recurse on one 2-vertex case triangle
// case 2 
// non-zero area triangle covered, one of a32 or a13 > 0
// recurse on one 2-vertex case
// case 3
// non-zero area something covered with a32 and a13 == 0
// and bisector > 0
// recurse on two 2-vertex cases (this is the most problematic case)
//
// return interpolation results so vertices are IN the covering polyon 
// (low variable in interpolation method) so recursion works.


if ( a13 > 1e-3 || a23 > 1e-3 ) // case 1 and case 2
   {
   double covered_area = ( a13 + (1.0-a13) * a23 ) * area ;
   
   if ( level < temp_max_level - 1 ) 
      {
      Triangle t2 = triangle_stack [level+1] ;
      PrimList [] p1 = prim_stack[level] ;
      PrimList [] p2 = prim_stack[level+1] ;
      
      t2.v1.x = t1.v31h.x ;
      t2.v1.y = t1.v31h.y ;
      t2.v1.z = t1.v31h.z ;
      sray.o.x = t2.v1.x ;
      sray.o.y = t2.v1.y ;
      sray.o.z = t2.v1.z ;
      p2[0].reset () ;
      m.allIntersections ( sray , p2[0] ) ;
      
      t2.v2.x = t1.v23h.x ;
      t2.v2.y = t1.v23h.y ;
      t2.v2.z = t1.v23h.z ;
      sray.o.x = t2.v2.x ;
      sray.o.y = t2.v2.y ;
      sray.o.z = t2.v2.z ;
      p2[1].reset () ;
      m.allIntersections ( sray , p2[1] ) ;
      
      t2.v3.x = v3.x ;
      t2.v3.y = v3.y ;
      t2.v3.z = v3.z ;
      p2[2].copy ( p1[s3] ) ;

      t2.value = a13*a23 ;
      t2.area = area - covered_area;
      t2.caseno = 21 ;
      double ans = covered_area + 
         subdivide_triangle ( level+1 , area - covered_area ) ;
      two_vertex_area += covered_area ;
      return ans ;
      }
   else
      {
      double ans = covered_area ;
      two_vertex_area += covered_area ;
      return ans ;
      }
   }
//System.out.println ( "Case 3" ) ;
// 
// possibly case 3
//
//System.out.println ( "zero" ) ;
t1.mid.x = (v1.x + v2.x) * 0.5 ;
t1.mid.y = (v1.y + v2.y) * 0.5 ;
t1.mid.z = (v1.z + v2.z) * 0.5 ;

double bisector = binary_search ( t1.mid , v3 , t1.v12l , t1.v12h , prim ) ;

if ( level >= temp_max_level - 1 ) 
   {
   double ans = area * bisector ;
   two_vertex_area += ans ;
   return ans ;
   }
Triangle t2 = triangle_stack [level+1] ;
PrimList [] p1 = prim_stack[level] ;
PrimList [] p2 = prim_stack[level+1] ;

// zero length edges on v1-v3 and v2-v3
// this is either zero points in the interior or one or more
// points in the interior

if ( bisector < 1e-2 )
   {
   double ans = 0 ;
   t2.v1.x = t1.v31h.x ;
   t2.v1.y = t1.v31h.y ;
   t2.v1.z = t1.v31h.z ;
   sray.o.x = t2.v1.x ;
   sray.o.y = t2.v1.y ;
   sray.o.z = t2.v1.z ;
   p2[0].reset ( ) ;
   m.allIntersections ( sray , p2[0] ) ;

   t2.v2.x = t1.v23h.x ;
   t2.v2.y = t1.v23h.y ;
   t2.v2.z = t1.v23h.z ;
   sray.o.x = t2.v2.x ;
   sray.o.y = t2.v2.y ;
   sray.o.z = t2.v2.z ;
   p2[1].reset ( ) ;
   m.allIntersections ( sray , p2[1] ) ;

   t2.v3.x = v3.x ;
   t2.v3.y = v3.y ;
   t2.v3.z = v3.z ;
   p2[2].copy ( p1[s3] ) ;
   
   t2.value = bisector ;
   t2.area = area ;
   t2.caseno = 22 ;
   ans = subdivide_triangle ( level+1 , area ) ;

   return ans ;
   }
double covered_area = bisector * area ;

// case 3
// OK, now we have a point (t1.v12l) in the covering polygon and in
// the interior of the triangle. Finding a vertex of the covering
// polygon would allow the recursion to quickly give the "exact" answer
// but for now we will just use this point and the convergence will be 
// something like exponential wrt number of sub-triangles (which is 
// much better than fixed ratio subdivision schemes).

t2.v1.x = v1.x ;
t2.v1.y = v1.y ;
t2.v1.z = v1.z ;
p2[0].copy ( p1[s1] ) ;

t2.v2.x = v3.x ;
t2.v2.y = v3.y ;
t2.v2.z = v3.z ;
p2[1].copy ( p1[s3] ) ;

t2.v3.x = t1.v12l.x ;
t2.v3.y = t1.v12l.y ;
t2.v3.z = t1.v12l.z ;
sray.o.x = t2.v3.x ;
sray.o.y = t2.v3.y ;
sray.o.z = t2.v3.z ;
p2[2].reset () ;
m.allIntersections ( sray , p2[2] ) ;
      
t2.area = (area - covered_area ) * 0.5 ;
t2.caseno = 23 ;
t2.value = bisector ;
double c1 = subdivide_triangle ( level + 1 , (area - covered_area) * 0.5 ) ;

t2.v1.x = v2.x ;
t2.v1.y = v2.y ;
t2.v1.z = v2.z ;
p2[0].copy ( p1[s2] ) ;

t2.caseno = 24 ;
t2.value = bisector ;
t2.area = (area - covered_area ) * 0.5 ;
double c2 = subdivide_triangle ( level + 1 , (area - covered_area) * 0.5 ) ;

double ans = covered_area + c1 + c2 ;
two_vertex_area += covered_area ;
return ans ;
}
// ************************************************************
static double subdivide_triangle1 ( int level , 
                                    P v1 , int p1 , 
                                    P v2 , int p2 ,
                                    P v3 , int p3 ,
                                    Prim prim , double area )
// ************************************************************
{
// There are two cases, 
// case 1 when the covered part of both v1-v2 and v1-v3 
// edges are are non-zero
// this will recurse on a a trapezoid (a 2-vertex covered triangle
// and a 1-vertex covered triangle)
//
// case 2 when only one of the edges is non-zero.
// this will recurse on one 2-vertex covered triangle
//
// case 1 always has positive amount of coverage
// case 2 only has a positive amount of coverage when the
// bisector is non-zero
//
// It is hard to describe all these cases without diagrams

++one_vertex ;
PrimList [] ps = prim_stack [level] ;
Triangle t1 = triangle_stack [level] ;
Triangle t2 = null ;

//System.out.println ( "t1  " + ps[0].s + " " + ps[1].s + " " + ps[2].s ) ;

if ( level < temp_max_level - 1 )
   t2 = triangle_stack [level+1] ;

double a12 = binary_search ( v1 , v2 , t1.v12l , t1.v12h , prim ) ;
double a31 = binary_search ( v1 , v3 , t1.v31l , t1.v31h , prim ) ;

// ***********************************************************
// empty case (no recursion)
// ***********************************************************
if ( a12 < 1e-5 && a31 < 1e-5 )
   { 
   ++null_one_vertex ;
   return 0 ;  
   }
   // if only one is "zero" then the resulting two point
   // sub-triangle will be tested for bisection on the recursion
   // the above case is both zero...
// ***********************************************************
// full coverage case (no recursion)
// ***********************************************************
if ( a12 > 0.99999 && a31 > 0.99999 )
   {
   ++one_vertex_return_1 ;
   one_vertex_area += area ;
   return area ;
   }
if ( a12 > 1e-3 && a31 > 1e-3 ) // 
   {
// ***********************************************************
// case 1 (a triangle covered, recurse on trapezoid, two way)
// ***********************************************************
   double covered_area  = ( a12 * a31 ) * area ;
   ++one_vertex_two_edges ;
   if ( level < temp_max_level - 1 )
      {
      t2.v1.x = t1.v12l.x ;
      t2.v1.y = t1.v12l.y ;
      t2.v1.z = t1.v12l.z ;
      sray.o.x = t1.v12l.x ;
      sray.o.y = t1.v12l.y ;
      sray.o.z = t1.v12l.z ;
      prim_stack[level+1][0].reset () ;
      m.allIntersections ( sray , prim_stack[level+1][0] ) ;
      
      t2.v2.x = t1.v31l.x ;
      t2.v2.y = t1.v31l.y ;
      t2.v2.z = t1.v31l.z ;
      sray.o.x = t1.v31l.x ;
      sray.o.y = t1.v31l.y ;
      sray.o.z = t1.v31l.z ;
      prim_stack[level+1][1].reset () ;
      m.allIntersections ( sray , prim_stack[level+1][1] ) ;
      
      t2.v3.x = v2.x ;
      t2.v3.y = v2.y ;
      t2.v3.z = v2.z ;
      prim_stack[level+1][2].copy ( prim_stack[level][p2] ) ;
      
      double a1 = area * a31 - covered_area ;
      t2.value = a12*a31 ;
      t2.area = a1 ;
      t2.caseno = 11 ;
      double c1 = subdivide_triangle ( level + 1 , a1 ) ;
      
      t2.v1.x = v3.x ;
      t2.v1.y = v3.y ;
      t2.v1.z = v3.z ;
      prim_stack[level+1][0].copy ( prim_stack[level][p3] ) ;

      double a2 = area - covered_area - a1 ;
      t2.value = a12*a31 ;
      t2.area = a2 ;
      t2.caseno = 12 ;
      double c2 = subdivide_triangle ( level + 1 , a2 ) ;
      
      double ans = covered_area + c1 + c2 ;
      one_vertex_area += covered_area ;
      return ans ;
      }
   else
      {
      ++depth_cutoff ;
      one_vertex_area += covered_area ;
      return covered_area ;
      }
   }
// ***********************************************************
// case 2 (v1-v2 edge with two recursions, let two point code handle
// bisection recursion
// ***********************************************************

if ( a12 > 1e-3 )
   {
   ++one_vertex_one_edge ;
   
   t1.mid.x = ( v1.x + t1.v12l.x ) * 0.5 ;
   t1.mid.y = ( v1.y + t1.v12l.y ) * 0.5 ;
   t1.mid.z = ( v1.z + t1.v12l.z ) * 0.5 ;

   double bisector = binary_search( t1.mid , v3 , t1.v23l , t1.v23h , prim) ;

   if ( bisector < 1e-2 )
      {
      ++one_edge_bisector_zero ;
      //return 0 ;
      }
   else
      {
      ++one_edge_bisector_non_zero ;
      }
   double covered_area = a12*bisector ;
   if ( level >= temp_max_level - 1 )
      {
      ++depth_cutoff ;
      one_vertex_area += covered_area ;
      return covered_area ;
      }
   else
      {
      if ( bisector < 1e-2 ) 
         {
         t2.v1.x = t1.v12h.x ;
         t2.v1.y = t1.v12h.y ;
         t2.v1.z = t1.v12h.z ;
         t2.value = bisector ;
         t2.caseno = 131 ;
         }
      else
         {
         t2.v1.x = t1.v12l.x ;
         t2.v1.y = t1.v12l.y ;
         t2.v1.z = t1.v12l.z ;
         t2.value = a12 ;
         t2.caseno = 132 ;
         }
      sray.o.x = t2.v1.x ;
      sray.o.y = t2.v1.y ;
      sray.o.z = t2.v1.z ;
      prim_stack[level+1][0].reset () ;
      m.allIntersections ( sray , prim_stack[level+1][0] ) ;

      t2.v2.x = v2.x ;
      t2.v2.y = v2.y ;
      t2.v2.z = v2.z ;
      prim_stack[level+1][1].copy ( prim_stack[level][p2] ) ;
      
      t2.v3.x = v3.x ;
      t2.v3.y = v3.y ;
      t2.v3.z = v3.z ;
      prim_stack[level+1][2].copy ( prim_stack[level][p3] ) ;

      double a1 = (1.0 - a12 ) * area ;
      t2.area = a1 ;
      double c1 = subdivide_triangle ( level+1 , a1 ) ;
      
      if ( bisector < 1e-2 ) 
         {
         t2.v2.x = t1.v31h.x ;
         t2.v2.y = t1.v31h.y ;
         t2.v2.z = t1.v31h.z ;
         sray.o.x = t2.v2.x ;
         sray.o.y = t2.v2.y ;
         sray.o.z = t2.v2.z ;
         prim_stack[level+1][1].reset ( ) ;
         m.allIntersections ( sray , prim_stack[level+1][1] ) ;
         t2.value = bisector ;
         t2.caseno = 141 ;
         }
      else
         {
         t2.v2.x = v1.x ;
         t2.v2.y = v1.y ;
         t2.v2.z = v1.z ;
         prim_stack[level+1][1].copy ( prim_stack[level][p1] ) ;
         t2.value = bisector ;
         t2.caseno = 142 ;
         }
      
      double a2 = a12 * area ;
      t2.area = a2 ;
      double c2 = subdivide_triangle ( level+1 , a2 ) ;

      return c1+c2 ;
      }
   }
if ( a31 > 1e-3 )
   {
   ++one_vertex_one_edge ;
   
   t1.mid.x = ( v1.x + t1.v31l.x ) * 0.5 ;
   t1.mid.y = ( v1.y + t1.v31l.y ) * 0.5 ;
   t1.mid.z = ( v1.z + t1.v31l.z ) * 0.5 ;

   double bisector = binary_search ( t1.mid , v2 , t1.v23l , t1.v23h , prim) ;

   double covered_area = a31*bisector ;
   if ( bisector < 1e-2 )
      {
      ++one_edge_bisector_zero ;
      //return covered_area ;
      }
   else
      {
      ++one_edge_bisector_non_zero ;
      }
   if ( level >= temp_max_level - 1 )
      {
      ++depth_cutoff ;
      one_vertex_area += covered_area ;
      return covered_area ;
      }
   else
      {
      if ( bisector < 1e-2 )
         {
         t2.v1.x = t1.v31h.x ;
         t2.v1.y = t1.v31h.y ;
         t2.v1.z = t1.v31h.z ;
         t2.value = bisector ;
         t2.caseno = 151 ;
         }
      else
         {
         t2.v1.x = t1.v31l.x ;
         t2.v1.y = t1.v31l.y ;
         t2.v1.z = t1.v31l.z ;
         t2.value = bisector ;
         t2.caseno = 152 ;
         }
      sray.o.x = t2.v1.x ;
      sray.o.y = t2.v1.y ;
      sray.o.z = t2.v1.z ;
      prim_stack[level+1][0].reset () ;
      m.allIntersections ( sray , prim_stack[level+1][0] ) ;

      t2.v2.x = v2.x ;
      t2.v2.y = v2.y ;
      t2.v2.z = v2.z ;
      prim_stack[level+1][1].copy ( prim_stack[level][p2] ) ;

      t2.v3.x = v3.x ;
      t2.v3.y = v3.y ;
      t2.v3.z = v3.z ;
      prim_stack[level+1][2].copy ( prim_stack[level][p3] ) ;

      double a1 = (1.0 - a31) * area ;
      t2.area = a1 ;
      double c1 = subdivide_triangle ( level+1 , a1 ) ;
      
      if ( bisector < 1e-2 )
         {
         t2.v3.x = t1.v12h.x ;
         t2.v3.y = t1.v12h.y ;
         t2.v3.z = t1.v12h.z ;
         sray.o.x = t2.v3.x ;
         sray.o.y = t2.v3.y ;
         sray.o.z = t2.v3.z ;
         prim_stack[level+1][2].reset () ;
         m.allIntersections ( sray , prim_stack[level+1][2] ) ;
         t2.value = bisector ;
         t2.caseno = 161 ;
         }
      else
         {
         t2.v3.x = v1.x ;
         t2.v3.y = v1.y ;
         t2.v3.z = v1.z ;
         prim_stack[level+1][2].copy ( prim_stack[level][p1] ) ;
         t2.value = bisector ;
         t2.caseno = 162 ;
         }
      
      double a2 = a31 * area ;
      t2.area = a2 ;
      double c2 = subdivide_triangle ( level+1 , a2 ) ;

      return c1+c2 ;
      }
   }
return 0 ;
}
// ************************************************************
static int least ( PrimList p1 , int i1 ,
                   PrimList p2 , int i2 ,
                   PrimList p3 , int i3 ,
                   PrimList p4 , int i4 )
// ************************************************************
{
if ( i1 >= p1.s ) return 0 ;
int cnt = 1 ;
int od = p1.prim[i1].order ;

if ( i2 < p2.s && p2.prim[i2].order < od ) return 0 ;
if ( i3 < p3.s && p3.prim[i3].order < od ) return 0 ;
if ( i4 < p4.s && p4.prim[i4].order < od ) return 0 ;

if ( i2 < p2.s && p2.prim[i2].order == od ) cnt ++ ;
if ( i3 < p3.s && p3.prim[i3].order == od ) cnt ++ ;
if ( i4 < p4.s && p4.prim[i4].order == od ) cnt ++ ;
return cnt ;
}
static P str = new P ( ) ; // string top right
static P stl = new P ( ) ; // string top left
static P sbr = new P ( ) ; // string bottom right
static P sbl = new P ( ) ;
static P sstr = new P ( ) ; // substring of blocks top right
static P sstl = new P ( ) ; 
static P ssbr = new P ( ) ; 
static P ssbl = new P ( ) ;
static P bktr = new P ( ) ; // block of cells top right
static P bktl = new P ( ) ; 
static P bkbr = new P ( ) ; 
static P bkbl = new P ( ) ;
static double [] shadow_fraction = new double [41] ;

// ************************************************************
static double [] calculateOneBlanketT ( Polygon panel , 
                                        Structure m , 
                                        V toSun , M proj , 
                                        boolean is_panel_1 )
// ************************************************************
{
// The length of the area containing strings is
// 31 meters
// 0.75617 meters per string
// 0.378049 meters per half string
// cell size = 0.32 meters for 4 x 8cm
// gap width = 0.378049 - 0.32 = 0.058049 meters

final double gapl = 58.049 ; // mm
final double block_length = 320.0 + gapl ;
final double block_l_range = 1.0 ;
final double block_l_step = block_l_range / 2.0 ;
final double block_l_gap = gapl / ( block_length * 2 ) ;
final double half_l_gap = block_l_gap * 0.5 ;

// width of string is 4.470 meters
// width of block of cells is 10x8cm = .8 meters
// width of all cells = 5 x .8meters = 4 meters
// total gap width - 4.470 - 4 meters = 0.47m
// gap = 6 (2 gaps outside and 4 gaps between)
// gap width = 0.470meters / 6 = 0.078333meters

final double gapw = 78.3333 ; // mm
final double block_width = 4000.0 + gapw * 6 ;
final double block_w_range = (block_width-gapw) / (block_width) ;
final double block_w_step = block_w_range / 5.0 ;
final double block_w_gap = gapw / ( block_width ) ;

sray.d = toSun ;

panel.visible = false ; // avoid self intersections
for ( int string = 0 ; string < 41 ; string ++ )
   {
   interp ( panel.v[0] , panel.v[3] , (string)/41.0 , sbl ) ;
   interp ( panel.v[1] , panel.v[2] , (string)/41.0 , sbr ) ;
   interp ( panel.v[0] , panel.v[3] , (string+1.0)/41.0 , stl ) ;
   interp ( panel.v[1] , panel.v[2] , (string+1.0)/41.0 , str ) ;

   shadow_fraction[string] = 0 ;

   for ( int bw = 0 ; bw < 5 ; bw ++ ) // width, blocks per string
      {
      interp ( sbl , sbr , (bw*block_w_step + block_w_gap) , ssbl ) ;
      interp ( stl , str , (bw*block_w_step + block_w_gap) , sstl ) ;
      interp ( sbl , sbr , ((bw+1)*block_w_step ), ssbr ) ;
      interp ( stl , str , ((bw+1)*block_w_step ) , sstr ) ;
      
      for ( int bl = 0 ; bl < 2 ; ++ bl )
         {
         interp ( ssbl , sstl , (bl*block_l_step + half_l_gap) , bkbl ) ;
         interp ( ssbr , sstr , (bl*block_l_step + half_l_gap) , bkbr ) ;
         interp ( ssbl , sstl , ((bl+1)*block_l_step - half_l_gap), bktl ) ;
         interp ( ssbr , sstr , ((bl+1)*block_l_step - half_l_gap), bktr ) ;
      

         

         triangle_stack[0].v1.x = bkbl.x ;
         triangle_stack[0].v1.y = bkbl.y ;
         triangle_stack[0].v1.z = bkbl.z ;
         
         triangle_stack[0].v2.x = bktl.x ;
         triangle_stack[0].v2.y = bktl.y ;
         triangle_stack[0].v2.z = bktl.z ;
         
         triangle_stack[0].v3.x = bkbr.x ;
         triangle_stack[0].v3.y = bkbr.y ;
         triangle_stack[0].v3.z = bkbr.z ;
         
         subdivide_countdown = SUBDIVISION_LIMIT ;
         temp_max_level = max_t_level ;
         cutoff_threshold = 80 ;
         
         double t1 = subdivide_triangle ( 0 , bkbl , bktl , bkbr , 1.0 ) ;
         if ( t1 > 1.0 ) t1 = 1.0 ;
         if ( t1 < 0.0 ) t1 = 0.0 ; // bug insurance
         
         
         triangle_stack[0].v1.x = bktr.x ;
         triangle_stack[0].v1.y = bktr.y ;
         triangle_stack[0].v1.z = bktr.z ;
         
         triangle_stack[0].v2.x = bktl.x ;
         triangle_stack[0].v2.y = bktl.y ;
         triangle_stack[0].v2.z = bktl.z ;
         
         triangle_stack[0].v3.x = bkbr.x ;
         triangle_stack[0].v3.y = bkbr.y ;
         triangle_stack[0].v3.z = bkbr.z ;

         subdivide_countdown = SUBDIVISION_LIMIT ;
         temp_max_level = max_t_level ;
         cutoff_threshold = 80 ;
         double t2 = subdivide_triangle ( 0 , bktr , bktl , bkbr , 1.0 ) ;
         if ( t2 > 1.0 ) t2 = 1.0 ;
         if ( t2 < 0.0 ) t2 = 0.0 ;
         
         
         shadow_fraction[string] += (t1+t2) ; // factor of two handled below
         }
      }
   shadow_fraction[string] *= 1.0/20.0 ; // 10 blocks/string x 2 tri/block
   }
panel.visible = true ; // avoid self intersections
return shadow_fraction ;
}
// ************************************************************
static double [] calculateFourLongerons ( int side , int saw ,
                                          Structure m , V toSun , M proj )
// ************************************************************
{
double [] answer = new double [4] ;
Cylinder longeron ;

int casts = 0 ;

for ( int lon = 0 ; lon < 4 ; lon ++ )
   {
   longeron = (Cylinder) m.child[side].child[saw].solid[4].obj[lon] ;
   P v1 = longeron.a1 ;
   P v2 = longeron.a2 ;
   
   double delta = 200 / v1.dist(v2) ; // space initial samples 20cm
   boolean shadow = false , prev_shadow = false ;
   double pos = 0 , prev_pos = 0 ;
   
   double shadow_fraction = 0 ;
   for ( double alpha = 0 ; alpha < 1.000001 ; alpha += delta ) 
      {
      if ( alpha == 0 )
         {
         sray.o.x = v1.x ;
         sray.o.y = v1.y ;
         sray.o.z = v1.z ;
         prev_shadow = ( null != m.anyIntersection ( sray ) ) ;
         ++casts ;
         prev_pos = 0 ;
         }
      else
         {
         pos = alpha ;
         sray.o.x = v1.x * (1-alpha) + v2.x * alpha ;
         sray.o.y = v1.y * (1-alpha) + v2.y * alpha ;
         sray.o.z = v1.z * (1-alpha) + v2.z * alpha ;
         shadow = ( m.anyIntersection ( sray ) != null ) ;
         ++casts ;

         if ( prev_shadow && shadow )
            shadow_fraction += pos - prev_pos ;

         if ( prev_shadow != shadow )
            {
            double start = prev_pos ;
            double finish = pos ;
            while ( finish - start > 0.00001 )
               {
               double beta = (finish+start)/2 ;
               sray.o.x = v1.x * (1-beta) + v2.x * beta ;
               sray.o.y = v1.y * (1-beta) + v2.y * beta ;
               sray.o.z = v1.z * (1-beta) + v2.z * beta ;
               if ( ( null != m.anyIntersection ( sray ) ) == prev_shadow )
                  {
                  start = beta ;
                  }
               else
                  {
                  finish = beta ;
                  }
               ++casts ;
               }
            if ( prev_shadow )
               {
               shadow_fraction += (finish+start)/2 - prev_pos  ;
               }
            else
               {
               shadow_fraction += pos - (finish+start)/2 ;
               }
            }
         prev_pos = pos ;
         prev_shadow = shadow ;
         }
      }
   answer[lon] = shadow_fraction ;
   }

return answer ;
}
static double [][] stringShadowFraction = new double [2][41] ;
// ************************************************************
static void calculateOneSAW ( int control_index , int side , int saw ,
                              Structure m , V toSun , M proj ,
                              double [] answer ) 
// ************************************************************
{
double cosine = 0 ;
double [] longeronShadowFraction ;
double [] shadows ;

Polygon blanket ;

blanket = (Polygon) m.child[side].child[saw].solid[2].obj[0];
// always calculate string shadow fractions
// and return cosine without setting it to zero if negative
//
//cosine = Math.max ( blanket.normal.dot(toSun) , 0.0 ) ;

cosine = blanket.normal.dot(toSun) ;
// if ( cosine > 0 )
   {
   shadows = calculateOneBlanketT ( blanket , m , toSun , proj , true ) ;
   for ( int s = 0 ; s < 41 ; s ++ )
      stringShadowFraction [0][s] = shadows[s] ;

   blanket = (Polygon) m.child[side].child[saw].solid[3].obj[0];
   //borderOneBlanket ( blanket , m , toSun , proj , true ) ;
   shadows = calculateOneBlanketT ( blanket , m , toSun , proj , false ) ;
   for ( int s = 0 ; s < 41 ; s ++ )
      stringShadowFraction [1][s] = shadows[s] ;
   }
if ( control_index == -1 )
   {
   System.out.println ( step_number ) ;
   for ( int b = 0 ; b < 2 ; b++ )
      {
      for ( int s = 0 ; s < 41 ; s ++ )
         {
         System.out.print ( fw(stringShadowFraction[b][s]) + " " ) ;
         }
      System.out.println ( ) ;
      }
   }

answer[control_index] = cosine ;

for ( int b = 0 ; b < 2 ; b ++ )
   for ( int s = 0 ; s < 41 ; s ++ )
      answer[8 + control_index*82 + b*41 + s] = 
         stringShadowFraction[b][s] ;

longeronShadowFraction = 
   calculateFourLongerons ( side , saw , m , toSun, proj ) ;

for ( int L = 0 ; L < 4 ; L ++ )
   {
   answer[8 + 8*82 + control_index * 4 + L] = 
      longeronShadowFraction[L] ;
   }

}
static double [] opanswer = new double [8 + 8*2*41 + 8*4] ;
// ************************************************************
static double [] calculateOnePosition ( Structure m , V toSun , M proj )
// ************************************************************
{
// output is in 1A 2A order like almost all places

deepest_t_level = 0 ;
if ( triangle_stack[0] == null )
   {
   for ( int d = 0 ; d < MAX_ALEVEL ; ++d )
      {
      triangle_stack[d] = new Triangle ( ) ;
      prim_stack[d] = new PrimList[3] ;
      prim_stack[d][0] = new PrimList ( ) ;
      prim_stack[d][1] = new PrimList ( ) ;
      prim_stack[d][2] = new PrimList ( ) ;
      }
   }

//toSun.println ( ) ;
calculateOneSAW ( 0 , 0 , 0 , m , toSun , proj , opanswer ) ; // 1A
calculateOneSAW ( 1 , 1 , 0 , m , toSun , proj , opanswer ) ; // 2A
calculateOneSAW ( 2 , 0 , 1 , m , toSun , proj , opanswer ) ; // 3A
calculateOneSAW ( 3 , 1 , 1 , m , toSun , proj , opanswer ) ; // 4A
calculateOneSAW ( 4 , 0 , 2 , m , toSun , proj , opanswer ) ; // 1B
calculateOneSAW ( 5 , 1 , 2 , m , toSun , proj , opanswer ) ; // 2B
calculateOneSAW ( 6 , 0 , 3 , m , toSun , proj , opanswer ) ; // 3B
calculateOneSAW ( 7 , 1 , 3 , m , toSun , proj , opanswer ) ; // 4B
step_number ++ ;
return opanswer ;
} ;
// ************************************************************
// ************************************************************
static String pad ( int i )
// ************************************************************
{
String s = "" + i ;
while ( s.length() < 4 )
   s = "0"+s ;
return s ;
} ;
// ************************************************************
static void checkForErrors( ConstraintsChecker checker)
// ************************************************************
{
if (!check_constraints || checker.errors.size() == 0)
   {
   return;
   }
for (String s : checker.errors)
   {
   System.out.println("ERROR: " + s);
   }
System.exit(0);
}
// ************************************************************
static void parse ( String line, int minute, ConstraintsChecker checker ) 
// ************************************************************
{
String [] col = line.split("[ \t]*,[ \t]*") ;
if ( minute == 0 )
   {
   csv_cols = col.length ;
   if ( csv_cols == 22 )
      {
      System.out.println("CSV mode: strict " +
                         "(complete validation of your solution)");
      check_constraints = true ;
      }
   else if ( csv_cols == 13 )
      {
      System.out.println("CSV mode: relaxed (only very basic checks)");
      check_constraints = false ;
      }
   else if ( minute == 91 )
      {
      System.err.println("ERROR: your CSV file needs to have 93 lines and have" +
                         "22 or 13 columns \non each line, except possibly " +
                         "the first (did you forget that first line?)" );
      System.exit(0);
      }
   else
      {
      System.err.println("ERROR: your CSV file needs to have " +
                         "22 or 13 columns");
      System.exit(0);
      }
   }

if ( col.length != csv_cols )
   {
   System.err.println("ERROR: each line of CSV file must have the " + 
                      "same number of columns."+col.length+" "+csv_cols);
   System.exit(0);
   }

alpha = Double.parseDouble(col[1]) ;
if (check_constraints)
   {
   alpha = (minute * 360.0 / ConstraintsChecker.NUM_STATES) ;
   }
if (minute == 0 && check_constraints)
   {
   beta = Double.parseDouble(col[0]) ;
   yaw = Double.parseDouble(col[1]) ;
   checker.setBeta(beta);
   checkForErrors(checker);
   checker.setYaw(yaw);
   checkForErrors(checker);
   }
else if ( check_constraints )
   {
   if (beta != Double.parseDouble(col[0]))
      {
      System.err.println("ERROR: in strict mode, beta values in all rows" +
                         " must be the same. Minute = " + minute + ".");
      System.exit(0);
      }
   if (yaw != Double.parseDouble(col[1]))
      {
      System.err.println("ERROR: in strict mode, yaw values in all rows" +
                         " must be the same. Minute = " + minute + ".");
	  System.exit(0);
      }
   }
else
   {
   alpha = Double.parseDouble(col[0]) ;
   beta = Double.parseDouble(col[1]) ;
   yaw = Double.parseDouble(col[2]) ;
   }

if ( check_constraints )
   {
   for ( int c = 0 ; c < 10 ; c++ )
      {
      // order 1A, 2A, 
      position_history [minute][c] = Double.parseDouble ( col[c*2+2] ) ;
      velocity_history [minute][c] = Double.parseDouble ( col[c*2+3] ) ;
      }
   double[] data = new double[20];
   for (int i=0; i < 10; i++)
      {
      data[2*i] = position_history [minute][i];
      data[2*i+1] = velocity_history [minute][i];
      }
   checker.setDataForFrame(minute, data);
   checkForErrors(checker);
   }
else
   for ( int c = 0 ; c < 10 ; c++ )
      {
      // order 1A, 2A, 
      position_history [step_number][c] = Double.parseDouble ( col[c+3] ) ;
      velocity_history [step_number][c] = 0 ;
      }

for ( int c = 0 ; c < 10 ; c++ )
   {
   control [c] = position_history[step_number][c] ;
   }
if ( view_alpha > 3600 )
   view_alpha = (beta > 0) ? 20 : -20 ;
if ( view_beta > 3600 )
   view_beta = (beta > 0) ? 40 : -40 ;
} ;
// ************************************************************
static boolean match ( String s , String pat ) 
// ************************************************************
{
if ( pat.startsWith(s) ) return true ;
if ( ("-"+pat).startsWith(s) ) return true ;
if ( ("--"+pat).startsWith(s) ) return true ;
return false ;
} ;
// ************************************************************
public static String fw ( double x )
// ************************************************************
{
if ( Math.abs ( x ) < 0.005 )
   x = 0 ;
if ( x > 0 ) x += 0.0001 ;
if ( x < 0 ) x -= 0.0001 ;

String s = ""+x ;
if ( s.length() > 5 )
   s = s.substring(0,5) ;
while ( s.length() < 5 )
   s = s + "0" ;
return s ;
} ;
// ************************************************************
static void colorit ( Structure m , int sarj , int bga , 
                      int r , int g , int b ) 
// ************************************************************
{
m.child[sarj].child[bga].solid[2].obj[0].basered   = (short) r ;
m.child[sarj].child[bga].solid[2].obj[0].basegreen = (short) g ;
m.child[sarj].child[bga].solid[2].obj[0].baseblue  = (short) b;

m.child[sarj].child[bga].solid[3].obj[0].basered   = (short) r ;
m.child[sarj].child[bga].solid[3].obj[0].basegreen = (short) g ;
m.child[sarj].child[bga].solid[3].obj[0].baseblue  = (short) b;
} ;
// ************************************************************
// ************************************************************
ISSVis ( )
// ************************************************************
{
}
// ************************************************************
ISSVis ( 
        String [] tokens , boolean render1 , 
         boolean render92 , int rez )  
// ************************************************************
{

rendering = render1 || render92 ;
res = rez ;

// assume order is 1A 2A 3A 4A 1B 2B 3B 4B everywhere
// except locally in rotate_sarg_bja and evaluateOnePosition
// local control order: 1A 3A 1B 3B 2A 4A 2B 4B

if ( rendering )
   {
   allocate ( res ) ;
   }

ISS_Reader r = new ISS_Reader ( tokens );
m = r.readModel ( ) ;

colorit ( m , 0 , 0 , 250 , 0 , 0 ) ;
colorit ( m , 0 , 1 , 250 , 100 , 100 ) ;
colorit ( m , 0 , 2 , 230 , 230 , 150 ) ;
colorit ( m , 0 , 3 , 230 , 230 , 0 ) ;
colorit ( m , 1 , 0 , 150 , 250 , 250 ) ;
colorit ( m , 1 , 1 , 0 , 250 , 250 ) ;
colorit ( m , 1 , 2 , 0 , 0 , 250 ) ;
colorit ( m , 1 , 3 , 100 , 100 , 250  ) ;

longeron_visibility ( m , false ) ;
}
// ************************************************************
static void longeron_visibility ( Structure m , boolean can_see )
// ************************************************************
{
// make longerons visible 
m.child[0].child[0].solid[4].visible = can_see ;
m.child[0].child[1].solid[4].visible = can_see ;
m.child[0].child[2].solid[4].visible = can_see ;
m.child[0].child[3].solid[4].visible = can_see ;
m.child[1].child[0].solid[4].visible = can_see ;
m.child[1].child[1].solid[4].visible = can_see ;
m.child[1].child[2].solid[4].visible = can_see ;
m.child[1].child[3].solid[4].visible = can_see ;
}
// ************************************************************
static String[] initTitles()
// ************************************************************
{
String [] titles = new String [8] ;
titles [0] = "SAW:            1A       2A       3A       4A" +
                      "       1B       2B       3B       4B" ;

titles [1] = "Power:     " ;
titles [2] = "Cosine:    " ;
titles [3] = "String<1: " ;
titles [4] = "String=0: " ;
titles [5] = "Long10%:" ;
titles [6] = "Bad min:  " ;
titles [7] = "Power: " ;

return titles;
}
// ************************************************************
static void loadModel()
// ************************************************************
{
ISS_Reader r = new ISS_Reader ( mfile );
m = r.readModel ( ) ;

colorit ( m , 0 , 0 , 250 , 0 , 0 ) ;
colorit ( m , 0 , 1 , 250 , 100 , 100 ) ;
colorit ( m , 0 , 2 , 230 , 230 , 150 ) ;
colorit ( m , 0 , 3 , 230 , 230 , 0 ) ;
colorit ( m , 1 , 0 , 150 , 250 , 250 ) ;
colorit ( m , 1 , 1 , 0 , 250 , 250 ) ;
colorit ( m , 1 , 2 , 0 , 0 , 250 ) ;
colorit ( m , 1 , 3 , 100 , 100 , 250  ) ;

if ( rendering )
   {
   allocate ( res ) ;
   }
}
// ************************************************************
/*
 * ans -- return value from calculateOnePosition (or evaluateSingleState)
 * average_powers -- contains current average powers
 * cold_minutes -- contains current longeron failure counters
 * isLibraryFrame -- if true, average_powers and cold_minutes are not updated, as well as title[6] (bad min)
 */
static String[] calculateTitles(int frameNumber, double[] ans, double[] average_powers, int[] coldMinutes, boolean isLibraryFrame)
{
String[] titles = initTitles();

double stringPowerScale = solar * eff * stringArea ;
double [] darkStrings = new double [8] ;
double [] partialStrings = new double [8] ;
int [] darkLongeron = new int [8] ;
double [] SAWpower = new double [8]  ; // one time step

for ( int saw = 0 ; saw < 8 ; ++ saw )
   {
   SAWpower[saw] = 0 ;
   darkStrings[saw] = 0 ;
   for ( int blanket = 0 ; blanket < 2 ; ++ blanket )
      for ( int string = 0 ; string < 41 ; ++ string ) 
         {
         double f = ans[ 8 + saw*41*2 + 
                         blanket*41 + string ] ;
         double shadowFactor = Math.max ( 0.0 , 1 - f * 5 ) ;
                  
         double cosine_factor = Math.max ( 0.0 , ans[saw] ) ;
         SAWpower [saw] += cosine_factor * 
            shadowFactor * stringPowerScale ;
         if ( shadowFactor == 0 )
            darkStrings[saw] ++ ;
         if ( shadowFactor < 1.0 )
            partialStrings[saw] ++ ;
         }
   if (!isLibraryFrame) average_powers[saw] += SAWpower[saw] ;

   darkLongeron[saw] = 0 ;
   for ( int longeron = 0 ; longeron < 4 ; ++longeron )
      {
      double f = ans[8 + 41*2*8 + saw*4 + longeron];
      if ( f > 0.1 )
         darkLongeron[saw] ++ ;
      }

   if (isLibraryFrame) continue;

   if ( darkLongeron[saw] == 1 || darkLongeron[saw] == 3 )
      coldMinutes[saw] ++ ;
   else
      coldMinutes[saw] = Math.max ( 0 , coldMinutes[saw]-1) ;
            
   }
            
for ( int saw = 0 ; saw < 8 ; ++ saw ) 
   {
   titles[1] += ( "  " + fw(SAWpower[saw]/1000) ) ;
   titles[2] += ( "  " + fw(ans[saw] ) ) ;
   titles[3] += ( "  " + fw(partialStrings[saw] ) ) ;
   titles[4] += ( "  " + fw(darkStrings[saw] ) ) ;
   titles[5] += ( "  " + fw(darkLongeron[saw] ) ) ;
   titles[6] += ( "  " + fw(coldMinutes[saw] ) ) ;
   }
if ( show_power_by_step )
      System.out.println ( frameNumber + " " + titles[1] ) ;
                     
double total_power = 0 ;
for ( int pwr = 0 ; pwr < 8 ; ++pwr )
   total_power += SAWpower[pwr] ;
         
titles [7] += dfmt.format(total_power) + " watts" ;

return titles;
}
// ************************************************************
static NumberFormat ifmt, dfmt;
static void initFormatters()
// ************************************************************
{
ifmt = NumberFormat.getIntegerInstance ( Locale.US ) ;
ifmt.setGroupingUsed(true) ;
dfmt = NumberFormat.getInstance ( Locale.US ) ;
dfmt.setGroupingUsed(true) ;
dfmt.setMinimumFractionDigits(11) ;
}
// ************************************************************
/*
 * checker to be passed only if it knows the score.
 */
static String[] calculateFooters(double[] average_powers, ConstraintsChecker checker)
// ************************************************************
{
for ( int pwr = 0 ; pwr < 8 ; ++pwr )
   average_powers[pwr] /= max_frame ;

double average_total_power = 0 ;
for ( int pwr = 0 ; pwr < 8 ; ++pwr )
   average_total_power += average_powers[pwr] ;

String[] foot = new String [4] ;
foot[0] = "Excess Powers:   " ;
for ( int pwr = 0 ; pwr < 8 ; pwr ++ )
   foot[0] += fw((average_powers[pwr] -
                 checker.MIN_POWER_REQ[pwr]) / 1000.0 ) + " " ;
            
foot[1] = "Average Powers: " ;
for ( int pwr = 0 ; pwr < 8 ; pwr ++ )
   foot[1] += fw(average_powers[pwr] / 1000.0 ) + " " ;
            
foot[2] = "Total Average Power: " + 
           dfmt.format(average_total_power) +
          " watts" ;

foot[3] = ""  ;

if (checker != null) foot[3] = "Score: " + dfmt.format(checker.rawScore());

return foot;
}
// ************************************************************
static void do_render ( boolean setupFlag )
// ************************************************************
{
if ( setupFlag )
   {
   // set up main transformation for rendering
   sunTransform = makeSunProjection ( beta , alpha /*orbit*/ ) ;
   // inverse transform used if sample dots are drawn
   inverse = makeInverseSunProjection ( beta , alpha /*orbit*/ ) ;
   // toSun vector is used for power calculations
   //System.out.println ( alpha + " " + beta ) ;
   sunTransform.transform ( new V ( 0 , -1 , 0 ) , toSun ) ;
   sray.d.x = toSun.x ;
   sray.d.y = toSun.y ;
   sray.d.z = toSun.z ;
         
   //places model transformations (rotations) into data structure
   rotate_SARJ_BGA ( m , control ) ;
   // applies transformations
   m.transform ( ) ;
   }

if ( rendering )
   {
   M view_transform = makeViewProjection ( view_beta , view_alpha ) ;
   M inverse_view = makeInverseSunProjection ( view_beta , view_alpha ) ;
   V to_sun = new V ( ) ;
   V from_sun = new V ( ) ;
   sunTransform.transform ( new V ( 0 , -1 , 0 ) , to_sun ) ;
   sunTransform.transform ( new V ( 0 , 1 , 0 ) , from_sun ) ;

   if ( longerons_visible )
      longeron_visibility ( m ,true ) ;

   render_from_view ( res , m , view_transform , 
                     inverse_view , to_sun ) ;
   for ( int y = 0 ; y < res ; ++y )
      for ( int x = 0 ; x < res*3 ; ++x )
         big_raster[y][x] = (byte) raster[y][x] ;

   M sun_view = makeViewProjection ( beta , alpha ) ;
      render_from_sun ( res , m , sun_view ) ; 
                                         
   for ( int y = 0 ; y < res ; ++y )
      for ( int x = 0 ; x < res*3 ; ++x )
         big_raster[y][x+res+res+res] = (byte)raster[y][x] ;

   longeron_visibility ( m , false ) ;
   }
}
// ************************************************************
static void do_final_prints ( ConstraintsChecker checker, 
                              int [] maxHeat )
// ************************************************************
{
double avetotal = 0.0;
for ( int id = 0 ; id <  8; id++ )
   {
   System.out.println( "Average power by solar array " + 
                       checker.SOLAR_ARRAY_NAME[id] + " = " + 
                       dfmt.format(checker.totalPower[id]) + 
                       " watts.");
   avetotal += checker.totalPower[id];
   }
System.out.println("Total power = " + dfmt.format(avetotal));
System.out.println("Number of underpowered solar arrays (M) = " + 
                   checker.lowPowerBGAs.size());
for ( int id = 0 ; id < 8 ; id++ )
   {
   System.out.println("Total rotation by BGA " + 
                      checker.SOLAR_ARRAY_NAME[id] + " = " + 
                      dfmt.format(checker.totalRotation(id)) +
                      " degrees.");
            }
for ( int id = 0 ; id < 8 ; id++ )
   {
   System.out.println("Maximum longeron danger count by BGA " + 
                      checker.SOLAR_ARRAY_NAME[id] + " = " + 
                      maxHeat[id]);
   }
checkForErrors(checker);
System.out.println("All checks passed. Your solution is valid.");
System.out.println("Score = " + checker.rawScore());
}
// ************************************************************
static void do_csv ( String csvname )
// ************************************************************
{
double[] average_powers = new double[8];
int [] coldMinutes = new int [8] ;

ConstraintsChecker checker = new ConstraintsChecker();

if ( csvname != null )
   {
   try
      {
      FileInputStream csv = new FileInputStream ( csvname ) ;
      String line = ISS_Reader.readLine ( csv ) ; // skip first line
      step_number = 0 ;
      longeron_visibility ( m , false ) ;
      String[] titles = null;
      for ( int step = 0 ; step < max_frame ; step ++ )
         {
         line = ISS_Reader.readLine ( csv ) ;
         parse ( line, step, checker ) ; 
                              // sets beta, alpha, time and command
         
         do_render ( true ) ; // sets up transforms

         // main evaluation function
         double[] ans = calculateOnePosition ( m , toSun , inverse ) ;

         for ( int saw = 0 ; saw < 8 ; ++ saw )
            {
            checker.cosTheta[step][saw] = ans[saw];
            for ( int blanket = 0 ; blanket < 2 ; ++ blanket )
               for ( int string = 0 ; string < 41 ; ++ string ) 
                  {
                  checker.stringShadow[step][saw][blanket*41 + string] =
                     ans[ 8 + saw*41*2 + blanket*41 + string ];
                  }
            for ( int longeron = 0 ; longeron < 4 ; ++longeron )
               {
               checker.longeronShadow[step][saw][longeron] = 
                  ans[8 + 41*2*8 + saw*4 + longeron];
               }
            }

         titles = calculateTitles(step, ans, average_powers, coldMinutes, false); 
            
         if ( rendering )
            {
            Drawer.getDrawer().setFrame(step, big_raster , titles , null , "Minute " + step + " beta = " + beta ) ;
            
            }
         } // loop over csv lines
      
      int[] maxHeat = null;
      if (check_constraints)
         {
         checker.checkAnglesAndSpeeds();
         checkForErrors(checker);
         maxHeat = checker.checkLongerons();
         checkForErrors(checker);
         for (int t = 0; t < checker.NUM_STATES; t++)
            {
            checker.evaluatePower(t);
            }
         checker.aggregateAndCheckPower();
         }

      String[] foot = calculateFooters(average_powers, (check_constraints ? checker : null));
      if ( rendering ) 
         Drawer.getDrawer().setFrame(max_frame-1, big_raster, titles, foot, "Minute " + (max_frame-1) + " beta = " + beta);
      
      double tot_power = 0;
      for ( int p = 0 ; p < 8 ; ++ p ) 
         {
         if ( check_constraints )
            tot_power += checker.totalPower[p] ;
         else
            tot_power += average_powers [p] ;
         }
      
      
      System.out.println( "  Total average power: " + 
                          dfmt.format(tot_power) +
                          " watts." );
      int s = 0 ;
      
      if ( check_constraints )
         {
         do_final_prints(checker, maxHeat);
         }
      else
         {
         s = 0 ;
         System.out.println ( "Channel: 1A Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         System.out.println ( "Channel: 2A Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         System.out.println ( "Channel: 3A Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         System.out.println ( "Channel: 4A Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         System.out.println ( "Channel: 1B Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         System.out.println ( "Channel: 2B Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         System.out.println ( "Channel: 3B Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         System.out.println ( "Channel: 4B Average Power " + 
                              (average_powers[s++]/(1000))+
                              " kilowatts" ) ;
         }
      if ( rendering )
         play_movie ( "ISS: the movie" ) ;
      }
   catch ( Exception e )
      {
      e.printStackTrace ( ) ;
      }
   }
}
// ************************************************************
private static class ErrorStreamRedirector extends Thread
// ************************************************************
{
BufferedReader reader;

public ErrorStreamRedirector(InputStream is)
{
reader = new BufferedReader(new InputStreamReader(is));
}

public void run()
{
while (true)
   {
   String s;
   try
      {
      s = reader.readLine();
      }
   catch (Exception e)
      {
      e.printStackTrace();
      return;
      }
   if (s == null)
      {
      break;
      }
   System.err.println(s);
   }
}
}

private static final int SOLUTION_CALL = 1;
private static final int LIBRARY_CALL = 2;

// ************************************************************
private static void runTest( String cmd )
// ************************************************************
{
System.out.println("Execute mode: running \"" + cmd + "\".");
double[] average_powers = new double[8];
int [] coldMinutes = new int [8] ;

Process solution = null;
try
   {
   solution = Runtime.getRuntime().exec(cmd);
   }
catch (Exception e)
   {
   System.out.println("ERROR: unable to execute your solution.");
   System.exit(0);
   }

Scanner scanner = new Scanner(new InputStreamReader(solution.getInputStream()));
PrintWriter writer = new PrintWriter(solution.getOutputStream());
new ErrorStreamRedirector(solution.getErrorStream()).start();

writer.println(beta);
writer.flush();

yaw = Double.parseDouble(scanner.next());

ConstraintsChecker checker = new ConstraintsChecker();
check_constraints = true;
checker.setBeta(beta);
checker.setYaw(yaw);
checkForErrors(checker);

String[] titles = null;

for (int t = 0; t < checker.NUM_STATES; t++)
   {
   while (true)
      {
      int what = scanner.nextInt();
      if (what == SOLUTION_CALL)
         {
         double[] ret = new double[20];
         for (int i = 0; i < 20; i++) ret[i] = Double.parseDouble(scanner.next());
         checker.setDataForFrame(t, ret);
         checkForErrors(checker);
         checker.evaluateFrame(t);

         do_render(false);

         titles = calculateTitles(t, checker.lastAns, average_powers, coldMinutes, false); 
            
         if ( rendering )
            {
            Drawer.getDrawer().setFrame(t, big_raster , titles , null , "Minute " + t + " beta = " + beta) ;
            }

         break;
         }
      else if (what == LIBRARY_CALL)
         {
         int render_flag = scanner.nextInt();
         double[] input = new double[14];        
         for (int i = 1; i < 14; i++) 
            input[i] = Double.parseDouble(scanner.next());         
         double[] output = checker.evaluateSingleState(input);         
         for (int i = 0; i < output.length; i++) 
            writer.println(output[i]);         
         writer.flush();       
         if (render_flag == 1 && rendering)            
            {           
            do_render(false);            
            titles = calculateTitles(t, checker.lastAns, average_powers, coldMinutes, true);            
            Drawer.getDrawer().setFrame(t, big_raster , titles , null , "Library call frame") ;          
            }        
         }
      else
         {
         System.out.println("ERROR: expected " + SOLUTION_CALL + " (indicating getStateAtMinute call) or " +
                            LIBRARY_CALL + " (indicating \"library\" method call). Received " + what + " instead.");
         System.exit(0);
         }
      }
   }

checker.checkAnglesAndSpeeds();
checkForErrors(checker);

int [] longeronCounts = checker.checkLongerons();
checkForErrors(checker);

for (int t = 0; t < checker.NUM_STATES; t++)
   {
   checker.evaluatePower(t);
   }
checker.aggregateAndCheckPower();

String[] foot = calculateFooters(average_powers, 
                                 (check_constraints ? checker : null));
if (rendering)
   Drawer.getDrawer().setFrame(checker.NUM_STATES-1, big_raster,
                               titles, foot, "Minute " + 
                               (checker.NUM_STATES-1) +
                               " beta = " + beta);

/* System.out.println("Triangle count = " + ISSVis.triangle_count); */
/* System.out.println("Prim count = " + Prim.next_order) ; */

do_final_prints(checker, longeronCounts);

if ( rendering )
   play_movie ( "ISS: the movie" ) ;
}

// ************************************************************
public static void main ( String [] arg )
// ************************************************************
{
beta = 75 ;
alpha = 0 ;
view_beta = 3700 ;
view_alpha = 3700 ;
rendering = false ;
String csvname = null ;
String cmd = null ;

velocity_history = new double [92][10] ;
position_history = new double [92][10] ;

for ( int a = 0 ; a < arg.length ; a++ )
   {
   if ( match ( arg[a] , "exec") )
      {
      cmd = arg[++a] ;
      }
   else if ( match ( arg[a] , "beta") )
      {
      beta = Double.parseDouble ( arg[++a] ) ;
      }
   else if ( match ( arg[a] , "view_alpha") )
      {
      view_alpha = Double.parseDouble ( arg[++a] ) ;
      }
   else if ( match ( arg[a] , "view_beta") )
      {
      view_beta = Double.parseDouble ( arg[++a] ) ;
      }
   else if ( match ( arg[a] , "level") )
      {
      max_t_level = Integer.parseInt ( arg[++a] ) ;
      }
   else if ( match ( arg[a] , "frames") )
      {
      max_frame = Integer.parseInt ( arg[++a] ) ;
      }
   else if ( match ( arg[a] , "resolution") )
      {
      res = Integer.parseInt ( arg[++a] ) ;
      }
   else if ( match ( arg[a] , "subsamples") )
      {
      ss = Integer.parseInt ( arg[++a] ) ;
      }
   else if ( match ( arg[a] , "show_power") )
      {
      show_power_by_step = true ;
      }
   else if ( match ( arg[a] , "show_longerons") )
      {
      longerons_visible = true ;
      }
   else if ( match ( arg[a] , "rendering") )
      {
      rendering = true ;
      }
   else if ( match ( arg[a] , "norendering") )
      {
      rendering = false ;
      }
   else if ( match ( arg[a] , "csv") )
      {
      csvname = arg[++a] ;
      }
   else if ( match ( arg[a] , "model") )
      {
      mfile = arg[++a] ;
      }
   else
      {
      System.err.println ( "Error: unknown command option " 
                           + arg[a] ) ;
      System.exit(77) ;
      }
   }

loadModel();
initFormatters();

if ( cmd != null )
   {
   if ( view_alpha > 3600 )
      view_alpha = (beta > 0) ? 20 : -20 ;
   if ( view_beta > 3600 )
      view_beta = (beta > 0) ? 40 : -40 ;
   runTest ( cmd ) ;
   }

else if ( csvname != null )
   do_csv ( csvname ) ;
}
// ************************************************************
} // end class ISSVis
