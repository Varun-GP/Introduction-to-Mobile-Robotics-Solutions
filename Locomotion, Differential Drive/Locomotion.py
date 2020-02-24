import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
'''
Exercise 1: Locomotion

Exercise 2: Differential Drive Implementation
'''

def diffdrive(x,y,theta,v_l,v_r,t,l):

	if (v_l == v_r):
		theta_n = theta
		x_n = x + v_l * t * np.cos(theta)
		y_n = y + v_l * t * np.sin(theta)
		print("Executed straight line")

	else:
		omega = (v_r - v_l)/l
		R = (l/2.0)*((v_l+v_r)/(v_r - v_l))
		ICCx = x - (R*np.sin(theta))
		ICCy = y + (R*np.cos(theta))
		#print(ICCx)
		#print(np.shape(np.array([[ICCx],[ICCy],[omega*t]])))
		DD = np.dot(np.array([[np.cos(omega*t), -1*np.sin(omega*t), 0],[np.sin(omega*t), np.cos(omega*t),0],[0,0,1]]),np.array([[x - ICCx],[y - ICCy],[theta]]))+ np.array([[ICCx],[ICCy],[omega*t]])
		#print(DD)
		x_n = DD[0][0]
		y_n = DD[1][0]
		theta_n = DD[2][0]
		print("Executed circular arc")



	return x_n , y_n , theta_n

if __name__ == '__main__':

	x , y , theta, l = 1.5 , 2.0 , (np.pi)/2.0, 0.5
	state_c0 = [x , y , theta]

	v_l, v_r, t = 0.3, 0.3, 3

	x,y,theta = diffdrive(x,y,theta,v_l,v_r,t,l)
	state_c1  = [x,y,theta]

	v_l, v_r, t = 0.1, -0.1, 1
	x,y,theta = diffdrive(x,y,theta,v_l,v_r,t,l)
	state_c2 = [x,y,theta]

	v_l, v_r, t = 0.2, 0.0, 2
	x,y,theta = diffdrive(x,y,theta,v_l,v_r,t,l)
	state_c3 = [x,y,theta]

	print("States:		", state_c1,state_c2,state_c3)
	fig, ax = plt.subplots()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.xlim([0,5])
	plt.ylim([0,5])
	

	R0 = Ellipse((state_c0[0],state_c0[1]), 0.2, 0.1, angle= state_c0[2]*180*1/np.pi,edgecolor='r')
	R1 = Ellipse((state_c1[0],state_c1[1]), 0.2, 0.1, angle=state_c1[2]*180*1/np.pi,edgecolor='r')
	R2 = Ellipse((state_c2[0],state_c2[1]), 0.2, 0.1, angle= state_c2[2]*180*1/np.pi,edgecolor='r')
	R3 = Ellipse((state_c3[0],state_c3[1]), 0.2, 0.1, angle=state_c3[2]*180*1/np.pi,edgecolor='r')

	ax.set_clip_box(ax.bbox)
	R0.set_facecolor(np.random.rand(3))
	ax.add_artist(R0)
	ax.plot(2,3,3*np.sin(np.pi),3*np.cos(np.pi))
	R1.set_facecolor(np.random.rand(3))
	ax.add_artist(R1)
	R2.set_facecolor(np.random.rand(3))
	ax.add_artist(R2)
	R3.set_facecolor(np.random.rand(3))
	ax.add_artist(R3)
	plt.text(state_c0[0], state_c0[1], 'R0' , color='k', size=8)
	plt.text(state_c1[0], state_c1[1], 'R1' , color='k', size=8)
	plt.text(state_c2[0], state_c2[1], 'R2' , color='k', size=8)
	plt.text(state_c3[0], state_c3[1], 'R3' , color='k', size=8)
	plt.savefig('robot_pose.jpg')
	plt.show()


