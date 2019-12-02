from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils.np_utils import to_categorical


class NeuralNetwork(object):

	def __init__(self, input_of_network, channels, output_of_network):
		# inputs for every pixel in image
		self.model = Sequential()
		self.input_numbers = input_of_network
		self.output_numbers = output_of_network
		self.channels = channels

	def create_layers(self):
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(self.output_numbers, activation='softmax'))

	def fit_and_evaluate(self, trainingSet, labels, epochs, batch_size):
		pass