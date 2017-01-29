# Name: Fida Mohammad Thoker
#       Kunwar Abhinav Aditya
# 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


output_file = open("Output_file",'w')


# Solution for Assignment 71
# Data for assignment 71 as given in the sheet
X = np.asarray([(0, 3),(1, 3),(1, 2),(2, 2),(2, 4),(5, 8),(0, -3),(1, -3),(1, -2),(2, -4),(3, 1),(3, 0),(3, -2),(4, -1),(5, 1)])
Y = [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# figure number
fignum = 0
i = 1
fig = plt.gcf()
fig.canvas.set_window_title('Linear kernel Different C values')
for c in (0.05, 0.1,0.5,3):

	clf = svm.SVC(kernel='linear', C=c)
	clf.fit(X, Y)
	
	# get the hyperplane
	w = clf.coef_[0]
	a = -w[0] / w[1]
	xx = np.linspace(-20, 20)
	yy = a * xx - (clf.intercept_[0]) / w[1]
	
	# plot the parallel line that pass through the support vectors
	margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
	yy_down = yy + a * margin
	yy_up = yy - a * margin
	
	# plot the line, the points, and the nearest vectors to the plane
	plt.subplot(2,2,i)
	plt.title(" Linear_kernel_with_C="+str(c))
	plt.plot(xx, yy, 'k-')
	plt.plot(xx, yy_down, 'k--')
	plt.plot(xx, yy_up, 'k--')
	
	plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10)
	output_file.write("Support vectors for Assignment 71 with Parameter C="+str(c)+'\n')
	output_file.write(str(clf.support_vectors_)+'\n')
	plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=20, cmap=plt.cm.Paired)
	
	plt.axis('tight')
	x_min = -20
	x_max = 20
	y_min = -20
	y_max = 20
	i += 1
plt.savefig("Linear_kernel_Assignment_71")
plt.show()

# Svm for training files PA_H_t2 ----- PA_H_t7
for i in xrange(2,8):
	filename ='PA-H_t'+str(i)+'.dat' 
	file_data =  open(filename)
	X=[]
	Y=[]
	#ignore first line
	data = file_data.readlines()
	for f in range(1,len(data)):
		line =  data[f]
		line = line.strip()
		line = line.split()
		Y.append(float(line[0]))
		X.append(( float(line[1].split(':')[1]),float(line[2].split(':')[1])))
	X = np.asarray(X)
	fig = plt.gcf()
	j =1;
	fig.canvas.set_window_title('Rbf kernel under Different C values')
	for c in (0.02,0.5,1,100,1000):
	
		clf = svm.SVC(kernel='rbf', C=c)
		clf.fit(X, Y)
		xx, yy = np.meshgrid(np.linspace(-20, 20, 500), np.linspace(-20, 20, 500))
	# plot the decision function for each datapoint on the grid
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		plt.subplot(3,3,j)
		plt.title(" RBF_Kernal_C = "+str(c))
		plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
	 	contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
	
		plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
		output_file.write("Support vectors for "+filename+"+with C= "+str(c)+'\n')
		output_file.write(str(clf.support_vectors_)+'\n')
		plt.xticks(())
		plt.yticks(())
		plt.axis([-20, 20, -20, 20])
		j+=1
	plt.savefig("Rbf_kernel_PA_H_t"+str(i))
	plt.show()


# Checking and ploting the test data now
# Prediction for PA_H_test.dat
plt.show()
	
file_data =  open('PA-H_test.dat')
X=[]
Y=[]	
data = file_data.readlines()
#ignore first line
for i in range(1,len(data)):
	line =  data[i]
	line = line.strip()
	line = line.split()
	Y.append(float(line[0]))
	X.append(( float(line[1].split(':')[1]),float(line[2].split(':')[1])))
output = clf.predict(X)
X=np.asarray(X)
print output
output = np.asarray(output)
positive = np.where(output>0)
negative = np.where(output<0)

print "Positive Example indicies = "
print positive
output_file.write( " Now running the test data on svm\n")
output_file.write("Positive Example indicies = ")
output_file.write(str(positive)+'\n')

print "Negative Example indicies= "
print negative
output_file.write("Negative Example indicies = ")
output_file.write(str(negative )+'\n')
# print the examples predicted as positive by our svm
output_file.write("Positive Examples = ")
output_file.write(str(X[positive])+'\n')
print "Positive examples= "
print X[positive]


