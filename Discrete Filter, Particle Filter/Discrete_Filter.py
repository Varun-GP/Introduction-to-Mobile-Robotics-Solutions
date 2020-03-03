'''
Exercise 1: Discrete Filter
In this exercise you will be implementing a discrete Bayes filter accounting for the motion of a
robot on a 1-D constrained world.
Assume that the robot lives in a world with 20 cells and is positioned on the 10th cell. The
world is bounded, so the robot cannot move to outside of the specified area. Assume further that
at each time step the robot can execute either a move forward or a move backward command.
Unfortunately, the motion of the robot is subject to error, so if the robot executes an action it will
sometimes fail. When the robot moves forward we know that the following might happen:
1. With a 25% chance the robot will not move
2. With a 50% chance the robot will move to the next cell
3. With a 25% chance the robot will move two cells forward
4. There is a 0% chance of the robot either moving in the wrong direction or more than two
cells forwards

Assume the same model also when moving backward, just in the opposite direction.
Since the robot is living on a bounded world it is constrained by its limits, this changes the motion
probabilities on the boundary cells, namely:
1. If the robot is located at the last cell and tries to move forward, it will stay at the same cell
with a chance of 100%
2. If the robot is located at the second to last cell and tries to move forward, it will stay at the
same cell with a chance of 25%, while it will move to the next cell with a chance of 75%

Again, assume the same model when moving backward, just in the opposite direction.

Implement in Python a discrete Bayes filter and estimate the final belief on the position of the robot
after having executed 9 consecutive move forward commands and 3 consecutive move backward
commands. Plot the resulting belief on the position of the robot.

Hints: Start from an initial belief of:
bel = numpy.hstack ((numpy.zeros(10), 1, numpy.zeros(9)))
You can check your implementation by noting that the belief needs to sum to one (within a very
small error, due to the limited precision of the computer). Be careful about the bounds in the
world, those need to be handled ad-hoc
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def discrete_filter(bel,str):
	bel = np.hstack((np.zeros(10), 1, np.zeros(9)))
	print(bel)


	bel_prime = np.zeros(bel.shape[0])
	if str=="forward":
		for x in range(bel.shape[0]):
			if x >= 2:
				bel_two_cells = bel[x-2]
			else:
				bel_two_cells = 0
			if x >= 1:
				bel_next_cell = bel[x-1]
			else:
				bel_next_cell = 0
			bel_not_move = bel[x]
			
			if x < bel.shape[0]-1:
				bel_prime[x] = 0.25 * bel_two_cells+ 0.50 * bel_next_cell+ 0.25 * bel_not_move
	
			elif x == bel.shape[0]-1: 
				bel_prime[x] = 0.25*bel_two_cells+0.75 * bel_next_cell+1.00 * bel_not_move
	
	if str=="backward":
		for x in range(bel.shape[0]):
			if x < bel.shape[0]-2:
				bel_two_cells = bel[x+2]
			else:
				bel_two_cells = 0
			
			if x < bel.shape[0]-1:
				bel_next_cell = bel[x+1]
			else:
				bel_next_cell = 0
			bel_not_move = bel[x]
			if x > 0:
				bel_prime[x] = 0.25*bel_two_cells+0.50*bel_next_cell+0.25*bel_not_move
			elif x == 0:
			
				bel_prime[x] = 0.25*bel_two_cells+0.75*bel_next_cell+1.00*bel_not_move

	return bel_prime


def plot(bel, str):
	if str == "forward":
		plt.cla()
		plt.bar(range(0,bel.shape[0]),bel,width=0.8, color=(0.7, 0.4, 0.6, 0.6))
		plt.vlines(range(0,bel.shape[0]), 0, 10, colors=(0.2, 0.4, 0.6, 0.6), linestyles='dotted')
		plt.xlabel('Position in meters')
		plt.ylabel('Likelihood')
		plt.title('Robot Position Likelihood Plot (Forward) ---->', fontsize=10)
		plt.axis([0,bel.shape[0]-1,0,1])
		plt.draw()
		plt.pause(1)
	if str =="backward":

		plt.cla()
		plt.bar(range(0,bel.shape[0]),bel,width=0.8, color=(0.2, 0.7, 0.6, 0.6))
		plt.vlines(range(0,bel.shape[0]), 0, 10, colors=(0.2, 0.4, 0.6, 0.6), linestyles='dotted')
		plt.xlabel('Position in meters')
		plt.title('Robot Position Likelihood Plot (Backward)  <----', fontsize=10)
		plt.ylabel('Likelihood')
		plt.axis([0,bel.shape[0]-1,0,1])
		plt.draw()
		plt.pause(1)



def main():
	bel = np.hstack((np.zeros(10),1,np.zeros(9)))
	plt.figure()
	plt.ion()
	plt.show()

	user_input = [int(x) for x in input("Enter number of forward and backward moves (less than 20):\n").split()]
	print(user_input)
	for i in range(0,user_input[0]):
		plot(bel,"forward")
		bel = discrete_filter(bel,"forward")
	
	plt.savefig('Robot-Position-Likelihood-Plot-(Forward).jpg')

	for i in range(0,user_input[1]):
		plot(bel,"backward")
		bel = discrete_filter(bel,"backward")
	plt.savefig('Robot-Position-Likelihood-Plot-(Backward).jpg')


	plt.ioff()
	plt.show()

if __name__ == "__main__":
	main()
