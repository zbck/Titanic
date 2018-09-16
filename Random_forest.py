import csv
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Random_forest:


	def __init__(self, data_filepath, label_filepath):
		self.DATA, self.LABELS = self._load_data(data_filepath,
												label_filepath)

	def _load_data(self, data_filepath, label_filepath):
		'''Load the data from .npy files
		'''
		data = np.load(data_filepath)
		label = np.load(label_filepath)		
		return data, label

	def	rdm_forest_classifier(self):
		''' Train the random forest classifier
		''' 
		X_train, X_test, y_train, y_test = train_test_split(self.DATA,
															self.LABELS,
															test_size=0.33)
		forest = RandomForestClassifier(n_estimators=100, random_state=0)
		forest.fit(X_train, y_train)

		print(forest.score(X_train, y_train))
		print(forest.score(X_test, y_test))

if __name__=='__main__':
	
	data_filepath = 'data/train_data_clean.npy'
	label_filepath = 'data/train_label_clean.npy'
	random_forest = Random_forest(data_filepath, label_filepath)
	random_forest.rdm_forest_classifier()
