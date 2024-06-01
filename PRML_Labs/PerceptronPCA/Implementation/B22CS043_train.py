import numpy as np
import pandas as pd
import sys
# Taking the Command line Arguments.
train_file = sys.argv[1]

# Function to find the dot product.
def function(arr,weights):
  ans = weights[0]
  for i in range(len(arr)):
    ans += weights[i+1] * arr[i]
  return ans

# Function to carry out Normalization of the dataset.
def MinMaxNormalize(train_df):
  for i in range(len(train_df.columns) - 1):
    col = train_df.columns[i]
    train_df[col] = (train_df[col] -train_df[col].min())/(train_df[col].max() - train_df[col].min())

  return train_df

# Function to find the Accuracy.
def accuracy_score(test_file,predicted_weights):
  df = pd.read_csv(test_file,sep = ' ',header = None)
  normalized_df = MinMaxNormalize(df)
  data = normalized_df.values
  data_points = len(df)
  correct = 0
  for i in range(data.shape[0]):
    coords = np.array(data[i][:-1])
    actual = int(data[i][-1])
    predicted = 0
    if function(coords,predicted_weights) >= 0:
      predicted = 1
    if(actual == predicted):
      correct += 1

  return correct/data_points

# Function to train the dataset.
def train_data(train_file,learning_rate,epochs):
  df = pd.read_csv(train_file,sep = ' ')
  normalized_df = MinMaxNormalize(df)
  data = normalized_df.values
  weights = np.random.random(5)
  # Python code to perform Perceptron Learning
  for j in range(epochs):
    # Scan the Dataset and Predict on the basis of the current weights
    for i in range(data.shape[0]):
      coords = np.array(data[i][:-1])
    #   print(coords)
      prediction = 0
      if function(coords,weights) >= 0:
        prediction = 1
      error = int(data[i][-1] - prediction)
      weights[0] += (learning_rate*error)
      for i in range(len(coords)):
        weights[i+1] += learning_rate*error*coords[i]
    # if j%10 == 0:
    #   print(accuracy_score("test.txt",weights))
  return weights

predicted_weights = train_data(train_file,0.01,1000)
print(predicted_weights)

# storing the weights in a separate file.
with open("weights.txt","w+") as file:
  for s in predicted_weights:
    file.write(str(s) + " ")

