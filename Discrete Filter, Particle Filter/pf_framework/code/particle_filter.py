'''

Exercise 2: Particle Filter
In the following you will implement a complete particle filter. A code skeleton with the particle
filter work flow is provided for you. A visualization of the particle filter state is also provided by
the framework.
The following folders are contained in the pf framework.tar.gz tarball:
data This folder contains files representing the world definition and sensor readings used by the
filter.
code This folder contains the particle filter framework with stubs for you to complete.
You can run the particle filter in the terminal: python particle filter.py. It will only work
properly once you filled in the blanks in the code.
(a) Complete the code blank in the sample motion model function by implementing the odom-
etry motion model and sampling from it. The function samples new particle positions based
on the old positions, the odometry measurements δ rot1 , δ trans and δ rot2 and the motion noise.
The motion noise parameters are:
[α 1 , α 2 , α 3 , α 4 ] = [0.1, 0.1, 0.05, 0.05]
The function returns the new set of parameters, after the motion update.
(b) Complete the function eval sensor model. This function implements the measurement up-
date step of a particle filter, using a range-only sensor. It takes as input landmarks positions
and landmark observations. It returns a list of weights for the particle set. See slide 15 of the
particle filter lecture for the definition of the weight w. Instead of computing a probability,
it is sufficient to compute the likelihood p(z|x, l). The standard deviation of the Gaussian
zero-mean measurement noise is σ r = 0.2.
(c) Complete the function resample particles by implementing stochastic universal sampling.
The function takes as an input a set of particles and the corresponding weights, and returns
a sampled set of particles.

'''

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
import math
#add random seed for generating comparable pseudo random numbers
np.random.seed(123)

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.
    
    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles

def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def normal_twelve(mu, sigma):
    x = 0.5 * np.sum(np.random.uniform(-sigma, sigma, 12))
    return mu + x


def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 
    # (probabilistic motion models slide 27)

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # generate new particle set after motion update
    new_particles = []
    
    '''your code here'''
    '''***        ***'''
    for particle in particles:

        #particle =  {'x': 0.0, 'y': 0.0, 'theta': 0.0}

        new_particle = dict()

        delta_rot1 = particle['x']
        delta_trans = particle['y']
        delta_rot2 = particle['theta']

        delta_hat_rot1 = delta_rot1 + normal_twelve(0, noise[0] * abs(delta_rot1) + noise[1] * delta_rot2)
        delta_hat_trans = delta_rot2 + normal_twelve(0, noise[2]* delta_rot2 + noise[3]*(abs(delta_rot1)+ abs(delta_trans)))
        delta_hat_rot2 = delta_trans + normal_twelve(0, noise[0]*abs(delta_trans) + noise[1] * delta_rot2)

        x_prime = delta_rot1 + delta_hat_trans * math.cos(delta_rot2+delta_hat_rot1)
        y_prime = delta_trans + delta_hat_trans * math.sin(delta_rot2+delta_hat_rot1)
        theta_prime = delta_rot2+delta_hat_rot1+delta_hat_rot2

        new_particle['x'], new_particle['y'], new_particle['theta'] = x_prime, y_prime, theta_prime
        new_particles.append(new_particle)
    
    return new_particles

def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # (probabilistic sensor models slide 33)
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    # print(landmarks)

    weights = []
    
    '''your code here'''
    '''***        ***'''

    for particle in particles:
        all_measurement_likelihood = 1.0
        for i in range(len(ids)):
            lm_id = ids[i]
            measurement_range = ranges[i]
            lx = landmarks[lm_id][0]
            ly = landmarks[lm_id][1]
            px = particle['x']
            py = particle['y']
   
            measurement_range_expected = np.sqrt( (lx - px)**2 + (ly - py)**2 )
     
            measurement_likelihood = scipy.stats.norm.pdf(measurement_range, measurement_range_expected, sigma_r)

            all_measurement_likelihood = all_measurement_likelihood * measurement_likelihood
        
        weights.append(all_measurement_likelihood)



    #normalize weights
    normalizer = sum(weights)
    weights = weights / normalizer

    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    '''your code here'''
    '''***        ***'''
    resolution = 1.0/len(particles)
    start = np.random.uniform(0,resolution)
    first_weight = weights[0] 
    new_particles = []
    i = 0
    for particle in particles:
        while start > first_weight:
            i = i + 1
            first_weight = first_weight + weights[i]
        new_particles.append(particles[i])
        start = start + resolution

    return new_particles

def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")
    print(len(sensor_readings))

    #initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    #run particle filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        plot_state(particles, landmarks, map_limits)

        #predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        #resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show('hold')

if __name__ == "__main__":
    main()