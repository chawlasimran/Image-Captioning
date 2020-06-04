from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
#import pyttsx3


root = Tk()
root.geometry("800x500")
root.title('Image_Captioning')


# extract features from each photo in the directory
def extract_features(file):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(file, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature


def func():
	global my_image
	root.filename = filedialog.askopenfilename(initialdir = "/", title = "Select a file", filetype =(("png files","*.png"),("jpg files","*.jpg"),("jpeg files","*.jpeg")))
	my_label = Label(root,text=root.filename)
	my_image = Image.open(root.filename)
	my_image = my_image.resize((250, 250), Image.ANTIALIAS)
	my_image = ImageTk.PhotoImage(my_image)
	my_image_display = Label(image = my_image).pack()
	# load the tokenizer
	tokenizer = load(open('C:/Users/INSPIRON/Desktop/Image_Captioning/tokenizer.pkl', 'rb'))
	# pre-define the max sequence length (from training)
	max_length = 34
	# load the model
	model = load_model('C:/Users/INSPIRON/Desktop/Image_Captioning/model_18.h5')
	# load and prepare the photograph
	photo = extract_features(root.filename)
	# generate description
	description = generate_desc(model, tokenizer, photo, max_length)
	#print(description)

	#Remove startseq and endseq
	query = description
	stopwords = ['startseq','endseq']
	querywords = query.split()
	resultwords  = [word for word in querywords if word.lower() not in stopwords]
	result = ' '.join(resultwords)
	#engine = pyttsx3.init()
    

	#print(result)
	Label(root,text='Caption : '+result).pack()
	#engine.say(result)
	#engine.setProperty('rate',120)
	#engine.setProperty('volume',0.9)
	#engine.runAndWait()

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

my_btn = Button(root,text = "Upload an Image",command=func).pack()

root.mainloop()