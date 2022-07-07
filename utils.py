
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# load Model
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

input_size = 784
hidden_size = 100
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

PATH = 'mnist_ffn.pth'
model.load_state_dict(torch.load(PATH))
model.eval()


# Image to Tensor
def transform_image(image_bytes):
	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((28, 28)),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))            # the mean = 0.1307 & standard deviation = 0.3081
	])

	image = Image.open(io.BytesIO(image_bytes))
    # torch.unsqueeze returns a new tensor with a dimension of size one inserted at the specified position.
	return transform(image).unsqueeze(0)                      # 0 means first dimension


# predict
def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    _, prediction = torch.max(outputs, 1)

    return prediction
