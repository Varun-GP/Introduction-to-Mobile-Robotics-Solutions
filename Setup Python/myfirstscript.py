''' Sheet 0
Topic: Setup'''

'''
Exercise 1: Defining functions
Functions in Python are usually defined inside a file. Create a file named
1myfirstscript.py and implement the following function:
f (x) = cos(x) exp(x)
Next, launch your script as python myfirstscript.py in the command line. In
Python multiple functions can be defined in the same file and the filename is inde-
pendent of the function names used in the file

Exercise 2: Plotting data
Every python file is a script which can be evaluated later. It can contain multiple
functions and other numerical computations all in one file. The matplotlib.pyplot
module can be used for plotting.
a) In the same python script write commands which plot the graph of the function
f in the interval [−2π, 2π]. Hint: python’s numpy module has as a special
variable for π: numpy.pi
b) Save the resulting plot as a PNG-file to your hard disk.

Exercise 3: Generating random numbers
Random numbers are important in probabilistic robotics so it is preferable to know
what kind of random variables are provided by Python and how to use them. Hint:
use numpy.
a) Create a vector with 100000 random variables which are normally distributed
with a mean of 5.0 and a standard deviation of 2.0.
b) Create a vector with 100000 uniformly distributed random variables between
0 and 10.
c) Compute the mean and standard deviation of the two vectors with random
variables. Are the results what you would expect?
d ) Plot histograms of the random variables you generated. The hist command
can be used to plot histograms. Take a look at help(matplotlib.pyplot.hist)
for more information about how to use it.
e) Modify your script so that the generated distributions are exactly the same
each time you call it.

'''
import numpy as np
import matplotlib.pyplot as plt

def simple_function(x):
	return np.cos(x) * np.exp(x)
	
def simple_plot(x):
	
	plt.figure()
	plt.plot(x,simple_function(x),'r-*')
	plt.title('Simple function: cos(x)exp(x)')
	plt.savefig('Function_curve.png')
	plt.show()


if __name__ == '__main__':
	y = simple_function(np.pi)
	print(y)
	
	x  = np.arange(-10.0,10.0,0.1)
	simple_plot(x)
	np.random.seed(1)
	mu, sigma = 5.0, 2.0
	normal = np.random.normal(mu, sigma, 100000)
	uniform =  np.random.uniform(0, 10, 100000)
	std = np.std([normal,uniform])
	mean = np.mean([normal,uniform])
	print(std, mean)
	
	fig, axs = plt.subplots(1, 2)

	axs[0].hist(normal, bins=50)
	axs[1].hist(uniform, bins=50)
	plt.show()
