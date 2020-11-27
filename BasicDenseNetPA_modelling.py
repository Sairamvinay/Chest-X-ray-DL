import pickle
import keras
import pandas as pd
import numpy as np
#Download the train/test/val files from the drive
#Also the PA_images pickle file

IMAGE_SIZE = (128,128)

classes = ['No Finding','Infiltration',
           'Effusion','Atelectasis','Nodule', 'Mass',
           'Pleural_Thickening','Fibrosis','Pneumothorax',
           'Cardiomegaly','Consolidation','Emphysema',
           'Pneumonia','Hernia','Edema']

mapper = dict(zip(classes, range(0,len(classes))))

def one_hot_encode(y,num_classes= 15):
  num_samples = len(y)
  y_final = np.zeros((num_samples,num_classes))
  for i,label in enumerate(y):
    y_final[i][mapper[label]] = 1
  
  return y_final


def load_images():
	with open("model_files/PA_images.pkl",'rb') as f:
		PA_imgs = pickle.load(f)

	print("Loaded images!")
	return PA_imgs


def create_Xy(df,PA_imgs):
	#TRAIN SET X Creation
	all_train_samples = df.groupby(by = 'Sample #',axis = 0)
	X_1st = []
	X_2nd = []
	X_3rd = []
	y = []
	for sample_num,sample_df in all_train_samples:
	  label = sample_df["Finding Labels"].values[-1]
	  image_vectors = sample_df['Image Index'].map(lambda x: PA_imgs[x]).values
	  X_1st.append(image_vectors[0])
	  X_2nd.append(image_vectors[1])
	  X_3rd.append(image_vectors[2])
	  y.append(label)


	

	X_1st = np.array(X_1st) / 255.0
	X_2nd = np.array(X_2nd) / 255.0
	X_3rd = np.array(X_3rd) / 255.0

	X_1st = X_1st.reshape((X_1st.shape[0],X_1st.shape[1],X_1st.shape[2],1))
	X_2nd = X_2nd.reshape((X_2nd.shape[0],X_2nd.shape[1],X_2nd.shape[2],1))
	X_3rd = X_3rd.reshape((X_3rd.shape[0],X_3rd.shape[1],X_3rd.shape[2],1))

	y_ohe = one_hot_encode(y)

	return X_1st,X_2nd,X_3rd,y_ohe

def DenseNET():
	input1 = keras.Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],1))
	input2 = keras.Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],1))
	input3 = keras.Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],1))

	#Use DenseNET 169 for feature extraction
	feature_extract = keras.applications.DenseNet169(include_top=False,weights=None,input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],1))

	#get the CNN based output for each of the 3 images followups
	feature_extract.trainable = False
	cnn_out1 = feature_extract(input1)
	cnn_out2 = feature_extract(input2)
	cnn_out3 = feature_extract(input3)

	#flatten each of the outputs
	cout1 = keras.layers.Flatten()(cnn_out1)
	cout2 = keras.layers.Flatten()(cnn_out2)
	cout3 = keras.layers.Flatten()(cnn_out3)

	concatted = keras.layers.Concatenate(axis=-1)([cout1, cout2, cout3])
	#merge all the three 
	#lstm = keras.layers.LSTM(50,activation='tanh',recurrent_activation='sigmoid')
	#time_out = lstm([cout1,cout2,cout3])
	dropout = keras.layers.Dropout(0.2)(concatted)
	output_layer = keras.layers.Dense(15,activation='sigmoid')(dropout)
	model = keras.models.Model(inputs= [input1,input2,input3],outputs = output_layer)
	print(model.summary())
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics = keras.metrics.BinaryAccuracy())
	return model

def main():
	train_pa = pd.read_csv("data_csv_files/PA_train.csv")
	test_pa = pd.read_csv("data_csv_files/PA_test.csv")
	val_pa = pd.read_csv("data_csv_files/PA_val.csv")

	PA_images = load_images()
	
	print("Creating the variables for the model")
	X_train_1st,X_train_2nd,X_train_3rd,y_train_ohe = create_Xy(train_pa,PA_images)
	X_val_1st,X_val_2nd,X_val_3rd,y_val_ohe = create_Xy(val_pa,PA_images)
	X_test_1st,X_test_2nd,X_test_3rd,y_test_ohe = create_Xy(test_pa,PA_images)

	print("TRAIN SIZE X: ",X_train_1st.shape," Y: ",y_train_ohe.shape)
	print("VAL SIZE X: ",X_val_1st.shape," Y: ",y_val_ohe.shape)
	print("TEST SIZE X: ",X_test_1st.shape," Y: ",y_test_ohe.shape)

	model = DenseNET()
	history = model.fit([X_train_1st,X_train_2nd,X_train_3rd],y=y_train_ohe,epochs=5,batch_size=100,validation_data= ([X_val_1st,X_val_2nd,X_val_3rd],y_val_ohe))
	with open('model_files/bin_files/trainHistoryDict', 'wb') as file_pi:
		pickle.dump(history.history, file_pi)# serialize model to YAML

	model_yaml = model.to_yaml()
	with open("model_files/bin_files/model.yaml", "w") as yaml_file:
	    yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights("model_files/bin_files/model.h5")
	print("Saved model to disk")

if __name__ == '__main__':
	main()
	
