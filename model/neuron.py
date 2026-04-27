import torch, re
from torch import nn
import pandas as pd

from helper import normalized_data

data = pd.read_csv('../data/used_cars.csv')

# Features
data['age_of_car'] = data['model_year'].max() - data['model_year']
data['milage_of_car'] = data.apply(lambda x: re.sub(r'[,mi.]', '', x['milage']), axis=1).astype(int)
data['price_of_car'] = data.apply(lambda x: re.sub(r'[,$]', '', x['price']), axis=1).astype(int)

# normalization of feature and target
# data['normalized_age'] = normalized_data(data['age_of_car'], data['age_of_car'].mean(), data['age_of_car'].std())
# data['normalized_milage'] = normalized_data(data['milage_of_car'], data['milage_of_car'].mean(), data['milage_of_car'].std())
# data['normalized_price'] = normalized_data(data['price_of_car'], data['price_of_car'].mean(), data['price_of_car'].std())

# Features to torch tensor using tensor_stack
X = torch.column_stack([
    torch.tensor(data['age_of_car'], dtype=torch.float32),
    torch.tensor(data['milage_of_car'], dtype=torch.float32)
    ])

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
# print(X)

# Target feature
y = torch.tensor([data['price_of_car']], dtype=torch.float32).reshape((-1, 1))
y_mean = y.mean(axis=0)
y_std = y.std(axis=0)
y = (y - y_mean) / y_std

# Define neural network
model = nn.Linear(2, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


for i in range(0, 10000):
    # Training loop
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    # print(loss)
    if i % 100 == 0:
        print(f"The loss in loop{i} is {loss}")
        # print(model.weight)


X_test = torch.tensor([
    [5, 10000],
    [2, 10000],
    [5, 20000],
    [3, 30000]
], dtype=torch.float32)

prediction = model((X_test - X_mean)/X_std)
print(f"The normalaized predication is {prediction} and actual predicted price is {((prediction * y_std) + y_mean)}")


