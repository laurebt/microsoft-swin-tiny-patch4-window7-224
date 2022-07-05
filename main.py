# https://huggingface.co/microsoft/swin-tiny-patch4-window7-224

from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests
import streamlit as st

#######################################################
def __get_image__(image_file):

	file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
	img = Image.open(image_file)

	return img, file_details


#######################################################
# Image classification
def compute(img):

	feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
	model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

	inputs = feature_extractor(images=img, return_tensors="pt")
	outputs = model(**inputs)
	logits = outputs.logits

	# model predicts one of the 1000 ImageNet classes
	predicted_class_idx = logits.argmax(-1).item()

	return model.config.id2label[predicted_class_idx]

#######################################################
def st_ui():

	st.title("Image classification")

	st.subheader("Image (png or jpg)")
	image_file = st.file_uploader(label="Upload an image")

	if image_file is not None:
		img, file_details = __get_image__(image_file)

		st.image(img,caption='uploaded image')

		st.subheader("Prediction")
		prediction = compute(img)
		st.text(prediction)

#######################################################
if __name__ == '__main__':

	st_ui()
