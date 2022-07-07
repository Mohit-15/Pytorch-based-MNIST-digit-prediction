# Pytorch based Digit prediction using MNIST Dataset

The model is made using Pytorch and the dataset used is MNIST dataset. 
The MNIST dataset is an acronym that stands for the **Modified National Institute of Standards and Technology dataset**. 
It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

<img src="https://machinelearningmastery.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset-1024x768.png" width='700px' height='300px' />

The mean value and deviation in **MNIST dataset** are 0.1307 and 0.3081.

### The Flask API
I have used flask framework to make the API, which takes the image and predict the digit class.

    POST: http://127.0.0.1:5000/predict/
    body: 
    {
	    "file": <image_file>
    }
    
I have added validation on the format of the input images, only (.png, .jpg, .jpeg) format are accepted.
The response of the request will look like this:

    {
      "class_name": "7",
      "prediction": 7
    }
    
I have created the two layer neural network for training the model. Two Linear layers, and one activation function used here us **ReLU**. As this problem is the multiclass classification problem, I've used **CrossEntropyLoss()** to find out the loss, then we don't need to add one more layer at the end i.e. Softmax Layer.

The transformers used in this project are, **Grayscale(num_output_channels=1), Normalize((0.1307,), (0.3081,)), ToTensor(), Resize((28, 28))**.

We first convert the request image into Image Bytes using BytesIO, then we pass the image_bytes to all the transformers mentioned above. Then we first resize the image_tensor to (-1, 28\*28), then pass the transformed Tensor to the pre-trained model, which will return the digit class, and the prediction in JSON format through the API response.

Then for visualizing the data, I've used tensorboard to show the metrics, graphs, images etc.

<img src="https://user-images.githubusercontent.com/52665879/177826662-0127d021-fcf9-4726-800a-8fc72f01e6cb.png" width="700px" height="300"/>
<img src="https://user-images.githubusercontent.com/52665879/177826851-2d8c89fc-c4c8-4b3d-a393-afa8017b5766.png" width="700px" height="300"/>
<img src="https://user-images.githubusercontent.com/52665879/177827231-fe99deef-6a5b-40a0-841a-d52608d55667.png" />
