
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt


## Creating a Writer, which writes data on the board
writer = SummaryWriter("runs/mnist2")

## Checking for GPU Support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Hyper Parameters
input_size = 784 # 28*28
hidden_size = 100
batch_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 2


## Dataset Downloading
## MNIST DATASET MEAN: 0.1307, STANDARD DEVIATION: 0.3081
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])		

train_dataset = torchvision.datasets.MNIST(train=True, download=True, transform=transform, root='./data')
test_dataset = torchvision.datasets.MNIST(train=False, transform=transform, root='./data')

## Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)    # [x, y, a, b] ---> x = batch_size, y = channel(1: GrayScale), a,b = dimensions


# Checking Samples
# for i in range(9):
#   plt.subplot(3, 3, i+1)
#   plt.imshow(samples[i][0], cmap='gray')
#plt.show()

img_grid = torchvision.utils.make_grid(samples)
writer.add_image('MNIST_IMAGE_LABEL', img_grid)
writer.close()


## Creating the Neural Network
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.layer1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.layer2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		output = self.layer1(x)
		output = self.relu(output)
		output = self.layer2(output)
		return output


model = NeuralNet(input_size, hidden_size, num_classes)

## Loss & Optimizer
criterion = nn.CrossEntropyLoss()		# In this we don't need to add softmax layer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()

## Training Loop
total_samples = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
	for i, (images, label) in enumerate(train_loader):
		# [100, 1, 28, 28] --->>> [100, 784]
		images = images.reshape(-1, 28*28).to(device)
		label = label.to(device)

		# forward pass
		outputs = model(images)
		loss = criterion(outputs, label)

		# backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		_, prediction = torch.max(outputs.data, 1)
		running_correct += (prediction == label).sum().item()


		if (i+1)%100 == 0:
			print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{total_samples}, loss: {loss.item():.4f}')
			writer.add_scalar('Training Loss',
							running_loss / 100,
							epoch*total_samples+i
				)

			writer.add_scalar('Accuracy',
							running_correct / 100,
							epoch*total_samples+i
				)

			running_loss = 0.0
			running_correct = 0


## Testing
label = []
preds = []
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for images, labels in test_loader:
		images = images.reshape(-1, 28*28).to(device)
		labels = labels.to(device)

		outputs = model(images)

		# value, index
		_, prediction = torch.max(outputs, 1)
		n_samples += labels.shape[0]
		n_correct += (prediction == labels).sum().item()

		class_prediction = [F.softmax(output, dim=0) for output in outputs]

		preds.append(class_prediction)
		label.append(prediction)

	preds = torch.cat([torch.stack(batch) for batch in preds])		# Shape: 10000/10
	label = torch.cat(label)			# concatenating the labels. Shape: 10000/1


	accuracy = 100 * n_correct / n_samples
	print(f'accuracy: {accuracy:.4f}')


	classes = range(10)
	for i in classes:
		label_i = label==i
		pred_i = preds[:, i]
		writer.add_pr_curve(str(i), label_i, pred_i, global_step=0)
		writer.close()