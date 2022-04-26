# https://huggingface.co/microsoft/swin-tiny-patch4-window7-224

from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests
import streamlit as st

#######################################################
def __get_image__():
  
  st.subgeader("Image")
  
  image_file = st.file_uploader(label="Upload an image", type=["jpg", "jpeg"])
  
  if image_file is not None:
    
    # To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
		st.write(file_details)

    # To View Uploaded Image
		image = Image.open(image_file)
    
   return image
  
#######################################################
# Image classification
def compute(image):

  feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
  model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

  inputs = feature_extractor(images=image, return_tensors="pt")
  outputs = model(**inputs)
  logits = outputs.logits
  
  # model predicts one of the 1000 ImageNet classes
  predicted_class_idx = logits.argmax(-1).item()
  print("Predicted class:", model.config.id2label[predicted_class_idx])

#######################################################
if __name__ == '__main__':
  
  st.title("Sentiment analysis on Breaking Bad quotes")
  image = __get_image__()
  compute(image):
    
    
