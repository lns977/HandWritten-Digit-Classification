import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import pickle

def preprocess_image (image):
    image = image.resize((28,28)).convert('L')

    image = np.array(image)

    image = image/255.0

    image = 1-image

    image = image.flatten().reshape(28,28,1)
    return image

def predict_digit():
    if 'model' in globals():
        image = preprocess_image(canvas_image)

        prediction = model.predict(image)

        predict_digit=np.argmax(prediction)
        result_label.config(text=f"Predicted Digit: {predict_digit}")
    else:
        result_label.config(text="Please Load a model first")

def clear_canvas():
    canvas_image.delete('all')
    result_label.config(text='')

def draw(event):
    x,y = event.x, event.y
    r=8
    canvas_image.create_oval(x-r,y-r,x+r,y+r, fill='black')
    draw_image.ellipse([x-r,y-r,x+r,y+r],fill='white')

# def save_image():
#     canvas_image.save("capture.png", "PNG")

root = tk.Tk()
root.title("Handwritten Digit Recognisation")

with open('D:\B-tech\Machine-Learning\Digit_Recognisation\model.pkl','rb') as f:
    model = pickle.load(f)

canvas_image = Canvas(root,width=280,height=280,bg='white')
canvas_image.pack()

draw_image = Image.new('RGB',(280,280),'black')
draw = ImageDraw.Draw(draw_image)

canvas_image.blind("<B1-Motion>",draw)

predict_button = Button(root,text="Predict",command=predict_digit)
predict_button.pack(side=tk.LEFT,padx=10,pady=10)

clear_button = Button(root,text='Clear',command=clear_canvas)
clear_button.pack(side=tk.RIGHT,padx=10,pady=10)

result_label = Label(root,text='')
result_label.pack()

root.mainloop()