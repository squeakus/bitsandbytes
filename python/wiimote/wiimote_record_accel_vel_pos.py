import cwiid, time
from numpy import * #Import matrix libraries
from pylab import *
print "Press 1 & 2 on the Wiimote simultaneously to find it"
# To find remote, type the following in a Shell:
#   hcitool scan
# Then enter the address in the following command
wiimote = cwiid.Wiimote("00:19:1D:C1:6D:A1")

print "Wiimote activiated..."
print "Press + to start recording and - to stop recording"

wiimote.enable(cwiid.FLAG_MESG_IFC)

wiimote.rpt_mode = cwiid.RPT_ACC | cwiid.RPT_BTN

# Setup Constants
i = 0
t=[0]
ax=[0]
ay=[0]
az=[0]
rec_flg =0
flg=True
tprev=0
cal_flg = 0
cal_cnt = 0
ax_cal=[0]
ay_cal=[0]
az_cal=[0]
axc = 0
ayc = 0
azc = 0
vy=[0]
y=[0]

while flg==True:    # While Wiimote set to record
    time.sleep(0.01)
    messages = wiimote.get_mesg()   # Get Wiimote Messages

    for mesg in messages:   # Loop through Wiimote Messages
        if mesg[0] == cwiid.MESG_ACC: # If a accelerometer message
            if rec_flg == 1:    # Check if set to record
                
                # Record Time
                i=i+1
                tm = time.time()
                if i > 1:
                    t.append(tm-tprev+t[i-1])
                    tprev = tm
                else:
                    tprev = tm
                    t.append(0.01)
                
                # Record Acceleration Data
                ax.append(mesg[1][cwiid.X]-axc)
                ay.append(mesg[1][cwiid.Y]-ayc)
                az.append(mesg[1][cwiid.Z]-azc)
            
            # Record Calibration Data if Calibration button (home) pressed
            elif cal_flg ==1:
                cal_cnt = cal_cnt+1
                ax_cal.append(mesg[1][cwiid.X])
                ay_cal.append(mesg[1][cwiid.Y])
                az_cal.append(mesg[1][cwiid.Z])
                if cal_cnt > 200:
                    cal_flg=0
                    axc = mean(ax_cal)
                    ayc = mean(ay_cal)
                    azc = mean(az_cal)
                    print "Wiimote calibrated"
        elif mesg[0] == cwiid.MESG_BTN:  # If a Wiimote button message
            if mesg[1] & cwiid.BTN_PLUS:    # Start recording
                rec_flg = 1
            elif mesg[1] & cwiid.BTN_MINUS: # Stop recording
                flg=False
                break
            elif mesg[1] & cwiid.BTN_HOME:  # Start Calibration
                cal_flg = 1
                

axm=[0]
aym=[0]
azm=[0]
tm=[0]
avm=[0]
v=3
mcnt=0

# Smooth the data
for i in arange(1,len(ax),v):   # Take the average of "v" points
    mcnt=mcnt+1
    if i+v > len(ax): break
    axm.append(mean(ax[i:i+v]))
    aym.append(mean(ay[i:i+v]))
    azm.append(mean(az[i:i+v]))
    tm.append(mean(t[i:i+v]))
    if azm[mcnt] <=1 and azm[mcnt]>=-1:
        azm[mcnt]=0
    daz=azm[mcnt]-azm[mcnt-1]
    
    # Calculate Velocity
    if azm[mcnt]<=daz+1 and azm[mcnt]>=daz-1:   # Must have a given change in
                                                # acceleration to record vel
        vy.append(vy[mcnt-1])
    else:
        vy.append(azm[mcnt]*(tm[mcnt]-tm[mcnt-1])+vy[mcnt-1])
    # Calculate Position
    y.append(vy[mcnt]*(tm[mcnt]-tm[mcnt-1])+y[mcnt-1])    
    

#Create Plots
subplot(3,1,1)
plot(tm,y,'r')   #Acceleration Plot
title('Wiimote Raw Acceleration Data')
xlabel('time (s)')
ylabel('Position')
subplot(3,1,2)
plot(tm,vy,'b')   #Acceleration Plot
xlabel('time (s)')
ylabel('Velocity')
subplot(3,1,3)
plot(tm,azm,'k')   #Acceleration Plot
xlabel('time (s)')
ylabel('Acceleration')
show()
