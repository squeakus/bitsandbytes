"""
A gross oversimplification of the the kalman filter 

For each timestep:
current estimation = Gain * Measured + (1 - Gain) * previousestimation 

The only unknown is the gain, this is continually revised using new
info and something from the previous state Kalman filter finds the
most optimum averaging factor for each consequent state. Also somehow
remembers a little bit about the past states
a very good dummy intro:
http://bilgin.esme.org/BitsBytes/KalmanFilterforDummies.aspx
"""
import pylab

def main():
    measurements = [0.39, 0.5, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45, 0.67, 0.73, 0.8, 0.72, 0.64, 0.6, 0.7, 0.75, 0.75, 0.83, 0.68]
    std_dev = 0.1
    estimate = [0]
    noise = [1]

    for val in measurements:
        prev_est, prev_noise = time_update(estimate, noise)
        new_est, new_noise = measurement_update(val, prev_est,prev_noise,std_dev)
        estimate.append(new_est)
        noise.append(new_noise)

    pylab.plot(estimate)
    pylab.plot(noise)
    pylab.show()
        
def time_update(estimate, noise):
    return estimate[-1], noise[-1]

def measurement_update(measurement, prev_est, prev_noise, std_dev):
    gain = prev_noise / (prev_noise + std_dev)
    new_est = prev_est + (gain * (measurement - prev_est))
    new_noise = (1 - gain) * prev_noise
    print "est", round(new_est,3), "noise", round(new_noise,3)
    return new_est, new_noise
    

if __name__ == '__main__':
    main()
