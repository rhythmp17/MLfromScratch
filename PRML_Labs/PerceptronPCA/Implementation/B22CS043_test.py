import numpy as np
import pandas as pd
import sys

# function to find the dot product.
def function(arr,weights):
    ans = weights[0]
    for i in range(len(arr)):
        ans += weights[i+1] * arr[i]
    return ans

# Getting the command line argumments.
test_file = sys.argv[1]
with open('weights.txt', 'r') as file:
    contents = file.read().strip().split()  # Read file, strip extra spaces and split by whitespace
    weights = np.array(contents, dtype=float)  # Convert contents to a numpy array of floats

df = pd.read_csv(test_file,sep = ' ',header = None)
# print(df)
def MinMaxNormalize(train_df):
  for i in range(len(train_df.columns) - 1):
    col = train_df.columns[i]
    train_df[col] = (train_df[col] -train_df[col].min())/(train_df[col].max() - train_df[col].min())

  return train_df


predicted_labels = np.array([])

normalized_df = MinMaxNormalize(df)
# print(normalized_df.shape)
# print(df)
data = normalized_df.values
for i  in range(data.shape[0]):
    coords = np.array(data[i],dtype = float)
    # coords = coords[0:-1]
    # print(coords)
    if function(coords,weights) >= 0:
        predicted_labels = np.append(predicted_labels,int(1))
    else:
        predicted_labels = np.append(predicted_labels,int(0))

print(list(predicted_labels))