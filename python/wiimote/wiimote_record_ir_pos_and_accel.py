import cwiid, time
from numpy import * #Import matrix libraries
from pylab import *
print "Press 1 & 2 on the Wiimote simultaneously to find it"
# To find remote, type the following in a Shell:
#   hcitool scan
# Then enter the address in the following command
wiimote = cwiid.Wiimote()

cwiid.LED1_ON

print "Wiimote activiated..."
print "Press + to start recording and - to stop recording"

# Set Wii Parameters to get
wiimote.enable(cwiid.FLAG_MESG_IFC)
wiimote.rpt_mode = cwiid.RPT_IR | cwiid.RPT_BTN

# Setup Initial Constants
t=[]
ir_x=[]
ir_y=[]
ir_s=[]
rec_flg =0
flg=True
t_i = time.time()   # Initial Time

# -----------------------------------------------------------------------------
# Check for Wii Commands until stop (- button) is pressed
while flg==True:
    time.sleep(0.01)    # Wait 1 hundredth of a second between Wiimote messages
    messages = wiimote.get_mesg()   # Get Wiimote messages

    for mesg in messages:   # Loop through Wiimote Messages
        if mesg[0] == cwiid.MESG_IR: # If message is IR data
            if rec_flg == 1:    # If recording
                for s in mesg[1]:   # Loop through IR LED sources
                    if s:   # If a source exists
                        ir_x.append(s['pos'][0])
                        ir_y.append(s['pos'][1])
                        ir_s.append(s['size'])

                        t.append(time.time()-t_i)                        
                        
        elif mesg[0] == cwiid.MESG_BTN:  # If Message is Button data
            if mesg[1] & cwiid.BTN_PLUS:    # Start Recording
                rec_flg = 1
                print "Recording..."
            elif mesg[1] & cwiid.BTN_MINUS: # Stop Recording
                flg=False
                break

# Calculate Velocity and Acceleration -----------------------------------------
print "Calculating velocity and acceleration..."
ir_v=[]
ir_a = []
for i in range(0,len(ir_y)-1):
    # Calculate Velocity
    v = (ir_y[i+1]-ir_y[i])/(t[i+1]-t[i])
    ir_v.append(v)
    
    # Calculate Acceleration
    if i < len(ir_y)-2:
        a = (ir_y[i+2]-2*ir_y[i+1]+ir_y[i])/((t[i+2]-t[i+1])*(t[i+1]-t[i]))
        ir_a.append(a)

#Create Plots ----------------------------------------------------------------

subplot(3,1,1)
plot(t,ir_y,'r')   #Acceleration Plot
title('IR LED Movement')
#xlabel('Time')
ylabel('Position')

subplot(3,1,2)
plot(t[1:],ir_v)
#title('Velocity of IR')
#xlabel('time')
ylabel('Velocity')

subplot(3,1,3)
plot(t[2:],ir_a)
#title('Acceleration of IR')
xlabel('time')
ylabel('Acceleration')

show()
