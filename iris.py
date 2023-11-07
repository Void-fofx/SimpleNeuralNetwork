import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# create a model class that inherits nn.Module
class Model(nn.Module):
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = f.relu(self.fc1(x))
    x = f.relu(self.fc2(x))
    x = self.out(x)
    return x

torch.manual_seed(41)

model = Model()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# change output type "species" to be numeric rather than text
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('virginica', 1.0)
my_df['species'] = my_df['species'].replace('versicolor', 2.0)
my_df

# train test and split
x = my_df.drop('species', axis=1)
y = my_df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# convert from numpy arrays to float tensors (float because the data is of the type float. i.e. sepal_length: 5.1)
x_train = torch.FloatTensor(x_train.to_numpy())
x_test = torch.FloatTensor(x_test.to_numpy())

y_train = torch.LongTensor(y_train.to_numpy())
y_test = torch.LongTensor(y_test.to_numpy())

# set criterion of model to measure error
# how far off are the predictions from the correct answer
criterion = nn.CrossEntropyLoss()
# choose optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train model
# how many epochs (epoch is one trip through all three layers)
epochs = 100
# loss should decrease
losses = []
for i in range(epochs):
  # get a prediction
  y_pred = model.forward(x_train)

  # measure loss
  loss = criterion(y_pred, y_train) # predicted value vs the y_train

  # keep track of losses
  losses.append(loss.detach().numpy())

  # print every 10 epochs
  if i % 10 == 0:
    print(f"Epoch {i} and loss: {loss}")

  # back propagation (feed error back in to fine tune weights)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel('loss/error')
plt.xlabel('Epoch')

# evaulate model using test data
with torch.no_grad(): # turn off back propagation
  y_eval = model.forward(x_test) # x_test is features of the dataset. y_eval will be predictions
  loss = criterion(y_eval, y_test)


correct = 0
with torch.no_grad():
  for i, data in enumerate(x_test):
    y_val = model.forward(data)

    print(f"{i+1}.) {str(y_val)} \t {y_test[i]}") # what type of iris does network think it is

    # correct
    if y_val.argmax().item() == y_test[i]:
      correct += 1
rows, _ = x_test.shape
print(f"\n We got {correct} of {rows} correct.")


# Test against unseen/new data (this should predict 'virginica')
# new_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])
# with torch.no_grad():
#   print(model(new_iris))

# save our model
torch.save(model.state_dict(), 'iris_model.pt')


# # load our model
# new_model = Model()
# new_model.load_state_dict(torch.load('iris_model.pt'))

# # check that it loaded correctly
# new_model.eval()