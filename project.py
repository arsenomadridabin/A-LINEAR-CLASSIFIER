import pandas as pd
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 



def plot_table(table_data):
	fig = plt.figure(dpi=80)
	ax = fig.add_subplot(1,1,1)

	table = ax.table(cellText=table_data, loc='center')
	table.set_fontsize(14)
	table.scale(1,4)
	ax.axis('off')

	plt.show()

def export_validation_result_to_json(X_train,y_train,X_val,y_val):
	table_data = []
	# batch_sizes = [5,10]
	# number_of_epoches = [100,150]
	batch_sizes = [5]
	number_of_epoches = [100,150]
	table_data = [["Batch Size","Number of Epoches","Accuracy"]]
	validation_result = {}

	for batch in batch_sizes:
		for epoch in number_of_epoches:
			try:
				print("Running validation for batch size : {} and epoch: {}".format(batch,epoch))
				print("--------------------------------------------------------")
				input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=batch,num_epochs=epoch,shuffle=True)
				model.train(input_fn=input_func)

				validation_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_val,y=y_val,batch_size=10,num_epochs=100,shuffle=True)
				results = model.evaluate(validation_input_func)
				table_data.append([batch,epoch,round(results.get('accuracy')*100,2)])

				validation_result['Batch :{}, Epoch: {}'.format(batch,epoch)] = "{} %".format(round(results.get('accuracy')*100,2))
			except Exception as e:
				continue
	with open('validation_result.json', 'w') as outfile:
		json.dump(validation_result, outfile)
		print(validation_result)

	plot_table(table_data)

if __name__ == "__main__":


	df1 = pd.read_csv('Madison_Irrigated_2.csv') #class 1
	# Giving unreadable features a proper name , Feature1, Feature2 ,... and so..on.
	df1 = df1.rename(
	    columns = {
	        column_name : "Feature {}".format(index+1) for index,column_name in enumerate(df1.columns)
	    }
	)

	df2 = pd.read_csv('Madison_Rainfed_2.csv')#class 0
	df2 = df2.rename(
	    columns = {
	        column_name : "Feature {}".format(index+1) for index,column_name in enumerate(df2.columns)
	    }
	)

	#Assigning irrigated fields class 1
	df1['class'] = 1

	#Assigning rainfed fields class 0
	df2['class'] = 0

	frames = [df1,df2]
	combined_df = pd.concat(frames)


	feature_columns_with_label = [tf.feature_column.numeric_column(column_name) for column_name in combined_df.columns] 

	#Removing class label
	feature_columns = feature_columns_with_label[:-1]


	#Shuffling the data
	random_combined_df = combined_df.sample(frac=1)

	#Removing rows containing null values
	filtered_df = random_combined_df.dropna()


	#feature data
	x_data = filtered_df.drop(['class'],axis=1)
	#label
	label = filtered_df['class']


	#Splitting into trainging, validation and test data in the ratio (60:20:20)
	# 2 step method:
	#First convert to 80:20 using test size 0.2
	#Second convert size 80 in the ration 75:25 which eventualy become 60:20
	#Thus the overall train:valididation:test data set becomes : 60% : 20% : 20%

	X_temp_train, X_test, y_temp_train, y_test = train_test_split(x_data, label, test_size=0.2, random_state=1)
	X_train, X_val, y_train, y_val = train_test_split(X_temp_train,y_temp_train,test_size=0.25,random_state=1)

	#Defining our Linear Classifier Model
	model=tf.estimator.LinearClassifier(feature_columns=feature_columns,n_classes=2)

	# Validating model
	# result = export_validation_result_to_json(X_train,y_train,X_val,y_val)
	
	input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=100,shuffle=True)
	model.train(input_fn=input_func)
	


#training
# input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=100,shuffle=True)
# model.train(input_fn=input_func)

# #validation
# validation_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_val,y=y_val,batch_size=10,num_epochs=100,shuffle=True)
# results = model.evaluate(validation_input_func)

# #test
# eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=100,shuffle=True)

# results = model.evaluate(eval_inpu_func)


#DNN Classifier

# classifier = tf.estimator.DNNClassifier(
#     feature_columns=feature_columns,
#     # Two hidden layers of 30 and 10 nodes respectively.
#     hidden_units=[30, 10],
#     # The model must choose between 2 classes.
#     n_classes=2)


# classifier.train(
#     input_fn=lambda: input_fn(X_train, y_train),
#     steps=5000)



# RESULTS
# {
#   "accuracy": 0.8298887,
#   "accuracy_baseline": 0.7972973,
#   "auc": 0.6157332,
#   "auc_precision_recall": 0.4334341,
#   "average_loss": 5.877492,
#   "label/mean": 0.2027027,
#   "loss": 5.873911,
#   "precision": 0.8867925,
#   "prediction/mean": 0.0422275,
#   "recall": 0.18431373,
#   "global_step": 88060
# }




