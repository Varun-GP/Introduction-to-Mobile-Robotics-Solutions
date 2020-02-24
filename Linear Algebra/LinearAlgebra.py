import numpy as np
import math 
import matplotlib.pyplot as plt

'''
Exercise 1: Linear Algebra

Exercise 2: 2D Transformations as Affine Matrices

Exercise 3: Sensing

'''

def symmetric_positive_definite(M):
	MT = np.transpose(M)
	if len(M) ==2:
		if (MT[0][1] == M[0][1]) and (MT[1][0] == M[1][0]):
			return np.all(np.linalg.eigvals(M) > 0)
		else:
			return False
	else:
		print("Not a 2x2")

def Is_Orthogonal(M):
	dim = len(M)
	I = np.eye(dim)
	prod  = np.dot(np.transpose(M), M)

	if np.array_equal(I,prod):
		return True
	else:
		return False

def Sensing():


	scan = np.loadtxt('laserscan.dat')
	#print(scan)
	angle = np.linspace(-math.pi/2, math.pi/2,np.shape(scan)[0])
	#print(angle)

	x = scan * np.cos(angle)
	y = scan * np.sin(angle)
	plt.plot(x, y, 'r.', markersize=5)
	plt.title('Laser Scan data')
	plt.gca().set_aspect('equal',adjustable= 'box')
	plt.savefig('scan.jpg') 
	plt.show()


	T_global_robot = np.array(
	[[np.cos(math.pi/4), -np.sin(math.pi/4), 1],
	[np.sin(math.pi/4), np.cos(math.pi/4), 0.5],
	[0, 0, 1]])

	T_robot_laser = np.array(
	[[np.cos(math.pi), -np.sin(math.pi), 0.2],
	[np.sin(math.pi), np.cos(math.pi), 0.0],
	[0, 0, 1]])

	T_global_laser = np.dot(T_global_robot, T_robot_laser)

	print(T_global_laser)
	t = np.ones(len(x))
	#print(t)
	laser_scan = np.array([x, y, t])
	#print(laser_scan)
	laser_scan_global = np.dot(T_global_laser, laser_scan)
	print(laser_scan_global)
	plt.plot(laser_scan_global[0,:], laser_scan_global[1,:], 'r.', markersize=5)
	plt.plot(T_global_robot[0,2], T_global_robot[1,2], 'b*');
	plt.plot(T_global_laser[0,2], T_global_laser[1,2], 'g+');
	plt.gca().set_aspect('equal',adjustable = 'box')
	plt.savefig('scan_with_poses.jpg')
	plt.show()


if __name__ == '__main__':
	A = np.array([[0.25,0.1],[0.2,0.5]])
	B = np.array([[0.25,-0.3],[-0.3,0.5]])
	print(symmetric_positive_definite(A))
	print(symmetric_positive_definite(B))
	D =[1/3]*np.array([[2.0,2.0,-1.0],[2.0,-1.0,2.0],[-1.0,2.0,2.0]])

	print(Is_Orthogonal(D))

	Sensing()
