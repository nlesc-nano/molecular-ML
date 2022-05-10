import deepchem as dc
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import wandb
import IPython

def load_data(path_to_data):
	"""
	Data load, normalization and split using deepchem builted-in functions
	- Splitter: 
		frac_train (float, optional (default 0.8)) – The fraction of data to be used for the training split.
		frac_valid (float, optional (default 0.1)) – The fraction of data to be used for the validation split.
		frac_test (float, optional (default 0.1)) – The fraction of data to be used for the test split.
	"""

	#The data loader automatically drops the NaN's
	data_loader = dc.data.CSVLoader(['cone_angle'], feature_field='smiles', featurizer=dc.feat.CircularFingerprint())
	dataset_cone_angle = data_loader.create_dataset(path_to_data)
	
	#Normalizing the data to have zero mean and unit standard deviation.
	normalization_transformers = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_cone_angle)
	dataset_cone_angle=normalization_transformers.transform(dataset_cone_angle)

	#Splitting the data, defaults frac_train=0.8, frac_val=0.1, frac_test=0.1
	splitter = dc.splits.ScaffoldSplitter()
	data_train, data_val, data_test = splitter.train_valid_test_split(dataset_cone_angle,seed=82)

	return {'train': data_train,
			'val': data_val,
			'test': data_test}

def model_fingerprints(dataset):
	"""
	Creates and train a keras model that works in the DeepChem framework.
	Params: 
		dataset: Dictionary containing 3 DiskDataset (Deepchem), of the train, validation and testing datasets
	Returns the trained model
	"""
	n_neurons=1000
	model_keras = keras.Sequential([
		keras.layers.Dense(n_neurons,activation='relu'),
		keras.layers.Dense(n_neurons*2,activation='relu'),
		keras.layers.Dense(n_neurons*2,activation='relu'),
		keras.layers.Dense(n_neurons,activation='relu'),
		keras.layers.Dense(1,name="output_cone_angle")
		])
	#model=dc.models.KerasModel(model_keras, dc.models.losses.L2Loss(), optimizer=dc.models.optimizers.Adam())
	model=dc.models.KerasModel(model_keras,dc.models.losses.L2Loss(),batch_size=300)
	#vc = dc.models.ValidationCallback(…)
	
	#training, fit returns: The average loss over the most recent checkpoint interval
	loss_history=[] #list of the loss during training, size= number of steps
	last_loss=model.fit(dataset['train'],nb_epoch=20,all_losses=loss_history)
	print(f'Last loss during trainning: {last_loss}, and the whole history:')
	print(loss_history)
	return model

def evaluate_model(model,dataset):
	"""
	Model evaluation Post-training:
	 - calculation of the error on the 3 datasets
	 - plot figure prediction vs truth
	"""
	# Error evaluation
	rms = dc.metrics.Metric(dc.metrics.rms_score)
	train_score = model.evaluate(dataset['train'], [rms])
	val_score = model.evaluate(dataset['val'], [rms])
	test_score = model.evaluate(dataset['test'], [rms])
	print(f'train score: {train_score}, val score: {val_score}, test score: {test_score}')

	#Plot the predicted and truth test data
	test_predictions=model.predict(dataset['test'])
	test_labels=dataset['test'].y
	plt.figure(200)
	plt.plot(test_labels,test_predictions,'k.')
	plt.ylabel("Predictied cone_angle (normalized)") 
	plt.xlabel("Truth cone_angle (normalized)")
	plt.savefig("EvalFingerprintsModel.pdf")

if __name__ == '__main__':
	print("++++Data preparation")
	dataset_cone_angle = load_data('../data/cone_angle_carbox_11K.csv')
	print("++++Model definition and training")
	model = model_fingerprints(dataset_cone_angle)
	print("++++Model evaluation")
	evaluate_model(model,dataset_cone_angle)

