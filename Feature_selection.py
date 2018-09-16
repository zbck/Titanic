import csv
import numpy as np
from pathlib import Path


class Feature_selection:
	'''This class is used to select only some features
		of the samples written in a csv file.
	'''

	EXTENTION = '.csv'
	TITANIC_FEATURES = ['id', 'pclass', 'survived',
				'name', 'sex', 'age', 'sibsp',
				'parch', 'ticket', 'fare',
				'cabin', 'embarked', 'boat',
				'body', 'home.dest', 'has_cabin_number']
	# Array with clean features
	NEW_ARRAY = []
	
	def __init__(self, filepath, features_wanted):
		self.FILEPATH = filepath
		if self._check_file():
			self.FEATURES = features_wanted
			self._open_read_file()


	def _check_file(self):
		'''Check if the file is a .csv
		'''
		if Path(self.FILEPATH).suffix == self.EXTENTION:
			return True
		else:
			return False			


	def _open_read_file(self):
		'''Open and read csv files
			the spamreader will be used to read the rows of
			the csv file
		'''
		self.FILE = open(self.FILEPATH,"r")
		self.SPAMREADER = csv.reader(self.FILE)


	def _feature_identification(self):
		''' In order to access easily to which column corespond what
			create two dictionnaies:
			(key = feature label, value = column index)
				- dict_old : Old array index of the selected features
				- dict_new : New array index of the selected features 
		'''
		columns_index_old = [(feature, index) for index, feature in enumerate(self.TITANIC_FEATURES) if feature in self.FEATURES]
		self.dict_old = dict(columns_index_old)
		columns_index_new = [(feature, index) for index, feature in enumerate(self.dict_old)]
		self.dict_new = dict(columns_index_new)

	def feature_select(self):
		'''Keep in a numpy array only the selected columns
		'''
		# Take take the column number of the given
		# features labeles
		self._feature_identification()
		for row in self.SPAMREADER:
			new_row = []
			for index in self.dict_old.values():
					new_row.append(np.array(row[index]))
			self.NEW_ARRAY.append(np.array(new_row))
		del self.NEW_ARRAY[-1]
		# Be careful row containt NA either 
		#put a number the mean of the other value
		# or del self.NEW_ARRAY[1226]
		del self.NEW_ARRAY[0]

	def sex2int(self):
		''' Change to male = 0 and female = 1
		'''
		for i in range(len(self.NEW_ARRAY)):
			if self.NEW_ARRAY[i][self.dict_new['sex']] == 'male':	
				self.NEW_ARRAY[i][self.dict_new['sex']] = 0
			else:
				self.NEW_ARRAY[i][self.dict_new['sex']] = 1
	
	def array2file(self, output_file):
		''' Write an array into a csv file
		'''
		np.save(output_file, np.reshape(np.array(self.NEW_ARRAY),
							(-1,len(self.FEATURES))))
			
					

if __name__=='__main__':

	filepath = 'Cleaning-Titanic-Data/titanic_clean.csv'
	output_file = 'data/train_data_clean'
	#output_file = 'data/train_label_clean'
	features_wanted = ['pclass', 'sex',
						'age', 'sibsp', 'fare']
	#features_wanted = ['survived']
	param_selec = Feature_selection(filepath, features_wanted)
	param_selec.feature_select()
	param_selec.sex2int()
	param_selec.array2file(output_file)
