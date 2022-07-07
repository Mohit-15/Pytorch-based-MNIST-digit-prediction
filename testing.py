import requests

response = requests.post('http://localhost:5000/predict',
		files={"file": open('eight.png', 'rb')}
	)
print(response.text)