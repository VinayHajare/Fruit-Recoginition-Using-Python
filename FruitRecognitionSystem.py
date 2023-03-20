import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms

#List containing names of Fruit Classes
Fruits = ['Acerola','Apple','Apricot','Avocado','Banana','Black Berry','Blue Berry','Cantaloupe','Cherry','Coconut','Fig','Grapefruit','Grape','Guava','Kiwi Fruit','Lemon','Lime','Mango','Olive','Orange','Passion Fruit','Peach','Pear','Pineapple','Plum','Pomegranate','Raspberry','Strawberry','Tomato','Watermelon']

#Load the Pytorch model
model = torch.load('./fruit_resnet18(88%).pt')

#Set model to evalution mode
model.eval()

#Define the transformation to apply to the input image
transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
	])

#Define a function to make a prediction on the input image
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output).item()
    return predicted


#Define function to handle "Browse" button click
def browse_file():
	file_path = filedialog.askopenfilename()
	if file_path:
		image = Image.open(file_path)
		image = image.resize((300,300),Image.LANCZOS)
		photo = ImageTk.PhotoImage(image)
		image_label.configure(image=photo)
		image_label.image = photo
		prediction_label.configure(text="")
		predict_button.configure(state="normal")
		global image_path
		image_path = file_path

#Define function to handle the "Predict" button click
def predict():
	if image_path:
		predicted_class = predict_image(image_path)
		predicted_label = Fruits[predicted_class]
		prediction_label.configure(text=f"Predicted Fruit: {predicted_label}")
	else:
		prediction_label.configure("Please select an image first")
	predict_button.configure(state="disabled")

#Create the main window
window = tk.Tk()
window.title("Fruit Regonition")
window.geometry("440x470")

#Create the "Browse" label
browse_button = tk.Button(window, text="Browse", command=browse_file)
browse_button.pack(pady=10)

#Create the image label
image_label = tk.Label(window)
image_label.pack(pady=10)

#Create the "Predict" button
predict_button = tk.Button(window, text="Predict", command=predict, state="disabled")
predict_button.pack(pady=10)

#Create a prediction label
prediction_label = tk.Label(window,font=("Arial",16))
prediction_label.pack(pady=10)

#Run the main loop
window.mainloop()