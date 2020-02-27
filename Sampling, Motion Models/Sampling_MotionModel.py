'''
Exercise 1 : Implement three functions in Python which generate samples of a normal distri-
bution N (μ, σ 2 ). The input parameters of these functions should be the mean μ
and the variance σ 2 of the normal distribution. As only source of randomness, use
samples of a uniform distribution.
• In the first function, generate the normal distributed samples by summing up
12 uniform distributed samples, as explained in the lecture.
• In the second function, use rejection sampling.
• In the third function, use the Box-Muller transformation method. The Box-
Muller method allows to generate samples from a standard normal distribution
using two uniformly distributed samples u 1 , u 2 ∈ [0, 1].

Exercise 2: Odometry-based Motion Model

'''
import numpy as np
import math
from scipy.stats import norm
from timeit import default_timer
import matplotlib.pyplot as plt

def sample(mu, sigma, n_samples,type):

	samples = []
	start_time = default_timer()
	for i in range(n_samples):		
		if type == "normal_twelve":
			x = 0.5 + np.sum(np.random.uniform(-sigma, sigma, 12))
			n_bins = 100
			
			samples.append(mu + x)

			out = mu + x

		if type =="normal_rejection":

			max_value_pdf = norm.pdf(mu)

			interval = sigma + 10

			while True:
				x = np.random.uniform(mu - interval, mu + interval, 1)[0]
				y = np.random.uniform(0, max_value_pdf, 1)

				if y <= norm.pdf(x):
					break

			n_bins = 100
			
			samples.append(x)
			
			out =  x

		if type =="normal_boxmuller":
			u = np.random.uniform(0,1,2)
			x = math.cos(2*np.pi*u[0]) * math.sqrt(-2 * math.log(u[1]))

			n_bins = 100
			
			samples.append(mu + sigma * x)
	
			out =  mu + sigma * x

		if type == "numpy_standard":
			n_bins = 100
			
			samples.append(np.random.normal(mu,sigma))
			out = np.random.normal(mu,sigma)


	end_time = default_timer()
	print("The function %s took %f " %(type, end_time - start_time))

			
	plt.figure()
	count, bins, ignored = plt.hist(samples, n_bins, normed=True)
	plt.plot(bins, norm(mu,sigma).pdf(bins), linewidth=2, color='r')
	plt.xlim([mu - 5*sigma, mu + 5*sigma])
	plt.title(str(type))

	return out

def normal_twelve(mu, sigma):
	x = 0.5 * np.sum(np.random.uniform(-sigma, sigma, 12))
	return mu + x

# Exercise 2A solution
def sample_motion_model(x_t, u_t, alpha):
	delta_hat_rot1 = u_t[0] + normal_twelve(0, alpha[0] * abs(u_t[0]) + alpha[1] * u_t[2])
	delta_hat_trans = u_t[2] + normal_twelve(0, alpha[2]* u_t[2] + alpha[3]*(abs(u_t[0])+ abs(u_t[1])))
	delta_hat_rot2 = u_t[1] + normal_twelve(0, alpha[0]*abs(u_t[1]) + alpha[1] *u_t[2])

	x_prime = x_t[0] + delta_hat_trans * math.cos(x_t[2]+delta_hat_rot1)
	y_prime = x_t[1] + delta_hat_trans * math.sin(x_t[2]+delta_hat_rot1)
	theta_prime = x_t[2]+delta_hat_rot1+delta_hat_rot2

	x_t_plus_one = [x_prime, y_prime, theta_prime]
	return np.array(x_t_plus_one)


def main():
	mu, sigma = 0, 1
	sample_function_types = ["normal_twelve","normal_rejection", "normal_boxmuller","numpy_standard"]

	# Exercise 1 solution	
	for i in range(len(sample_function_types)):
		sample(mu, sigma,int(10000),sample_function_types[i])
		plt.savefig(sample_function_types[i]+ "_plot.png")
		plt.show()


	# Exercise 2C solution	
	x_t = [2, 4, 0]
	u_t = [np.pi/2, 0, 1]
	alpha = [0.1, 0.1, 0.01, 0.01]
	n_samples = 5000
	x_prime = np.zeros([n_samples, 3])
	print("Sample Motion Model using Odometry method ")
	for i in range(0, n_samples):
		x_prime[i,:] = sample_motion_model(x_t,u_t,alpha)
	plt.plot(x_prime[:,0], x_prime[:,1], "r.")
	plt.plot(x_t[0], x_t[1], "g*")
	plt.xlim([1, 3])
	plt.axes().set_aspect('equal', adjustable ='box')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Sampling Motion Model using Odometry method")
	plt.savefig("motion_model_samples.jpg")
	plt.show()


if __name__ == '__main__':
	main()
