import numpy as np
import pickle
import keras

def CNN_model(num_cnn_layers = 1, num_layers = 5, num_units = 1000, pool_size = (2,2),filters = 128,kernel_size = 7,num_outputs=15):
  model = keras.models.Sequential()
  model.add(keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,activation = "relu",input_shape = (256,256,1)))
  model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
  for _ in range(num_cnn_layers-1):
    model.add(keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
  
  for _ in range(num_layers):
    model.add(keras.layers.Dense(num_units,activation="relu"))
    model.add(keras.layers.Dropout(0.4))
  
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(num_outputs,activation="softmax"))
  model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.01),loss = "categorical_crossentropy",metrics=["accuracy"])
  return model


with open("bin_files/train.pkl",'rb') as f:
	X_train_scale,y_train = pickle.load(f)

with open("bin_files/val.pkl",'rb') as f:
	X_val_scale,y_val = pickle.load(f)

with open("bin_files/test.pkl",'rb') as f:
	X_test_scale,y_test = pickle.load(f)


model = CNN_model(num_cnn_layers=3,num_layers=2,num_units=50)
print(model.summary())
history = model.fit(X_train_scale,y_train,batch_size=100,epochs= 10,verbose=1,validation_data=(X_val_scale,y_val))

with open('bin_files/trainHistoryDict', 'wb') as file_pi:
	pickle.dump(history.history, file_pi)# serialize model to YAML

model_yaml = model.to_yaml()
with open("bin_files/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("bin_files/model.h5")
print("Saved model to disk")
