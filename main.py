import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR

# Read all images
atletico_g = cv2.imread('images/atletico.jpg')
corinthians_g = cv2.imread('images/corinthians.jpg')
flamengo_g = cv2.imread('images/flamengo.jpg')
palmeiras_g = cv2.imread('images/palmeiras.jpg')
ceara_g = cv2.imread('images/ceara.jpg')
test_ceara_g = cv2.imread('images/ceara_2.jpg')

# Resize images to 10px x 10px
atletico = cv2.resize(atletico_g, (10,10))
corinthians = cv2.resize(corinthians_g, (10,10))
flamengo = cv2.resize(flamengo_g, (10,10))
palmeiras = cv2.resize(palmeiras_g, (10,10))
ceara = cv2.resize(ceara_g, (10,10))
test_ceara = cv2.resize(test_ceara_g, (10,10))

# Concat all arrays to one
X = np.concatenate((atletico, corinthians, flamengo, palmeiras, ceara), axis=0)

# Create index to arrays
y = [1,2,3,4,5]

# Set y as a array
y = np.array(y)

# Reshape y
Y = y.reshape(-1)

# Reshape X with length of y
X = X.reshape(len(y), -1)

# Create the classifier 
classifier_linear = SVC(kernel='linear')

print(40 * '-')
print('Started train of SVC model')

# Train the classifier with images and indexes
classifier_linear.fit(X,Y)

print('Finished train')
print(40 * '-')

# Predict the category of image 
prediction = classifier_linear.predict(test_ceara.reshape(1,-1))

# Score of predict 
score = classifier_linear.score(X,Y)

# Show prediction
print('Result: {}'.format(prediction))

# Show prediction score
print('Score of precision: {:.1f}%'.format(score * 100))

# Set result as image of prediction
if prediction == 1:
	result = atletico_g
elif prediction == 2:
	result = corinthians_g
elif prediction == 3:
	result = flamengo_g
elif prediction == 4:
	result = palmeiras_g
elif prediction == 5:
	result = ceara_g

# Show image based on prediction
cv2.imshow("Result", result)
# Show the image tested
cv2.imshow("Test", test_ceara_g)
# Wait for key
cv2.waitKey(0)

print('---------------------------------------')


# Create the classifier 
classifier_linear_regression = SVR(kernel='linear')

print('Start SVR Train')

# Train the classifier with images and indexes
classifier_linear_regression.fit(X,Y)

print('Finished train')
print(40 * '-')

# Predict the category of image 
prediction = classifier_linear_regression.predict(test_ceara.reshape(1,-1))

# Score of predict 
score = classifier_linear_regression.score(X,Y)

# Show prediction
print('Result: {}'.format(prediction))

# Show prediction score
print('Score of precision: {:.1f}%'.format(score * 100))

# Set result as image of prediction
if prediction == 1:
	result = atletico_g
elif prediction == 2:
	result = corinthians_g
elif prediction == 3:
	result = flamengo_g
elif prediction == 4:
	result = palmeiras_g
elif prediction == 5:
	result = ceara_g

# Show image based on prediction
cv2.imshow("Result", result)
# Show the image tested
cv2.imshow("Test", test_ceara_g)
# Wait for key
cv2.waitKey(0)