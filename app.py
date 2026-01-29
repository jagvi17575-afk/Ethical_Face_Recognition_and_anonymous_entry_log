import streamlit as st
import tf_keras as keras
import numpy as np
from PIL import Image
from google import genai

model=keras.models.load_model("keras_model.h5")
client = genai.Client(api_key="AIzaSyDZdC71ATSNMWw-mHEavcNwEP6gdo6_YM0")

st.title("Ethical Face Recognition & Anonymous Entry Log")
st.warning("No data is stored and shared with anyone, application uses camera with your consent")
checkbox=st.checkbox("I agree to all terms and conditions.")

label=["Student 1","Student 2","Student 3","Student 4","Student 5","Student 6","Teacher 1","Teacher 2","Teacher 3","Principal","Authorised person","Clerk 1","Intruder","Intruder","Intruder","Intruder"]

if checkbox:
    image=st.camera_input("see here")   #Data Collection via camera
    #image processing
    if image:
        img=Image.open(image).resize((224,224))
        image_array=np.array(img)/255.0
        image_array=np.expand_dims(image_array,axis=0)

        #Prediction layer
        with st.spinner("Processing image ethically..."):
            prediction=model.predict(image_array)
        
        #Getting the highest probability number+names from label 0= Authorized person, 1 = Student and so on
        class_index=np.argmax(prediction)
        index=label[class_index]
        
        st.success(index)

        if index=="Intruder":
            st.error("intruder alert")
            prompt = ("Explain in one calm, ethical, and non-judgmental sentence that the AI system could not match this image with authorized categories and therefore access was restricted. ")
        else:
            st.info("access allowed")
            prompt = ("Explain in one calm, ethical, and non-judgmental sentence that the AI system identified this person as authorized based on learned patterns.")

        with st.spinner("AI generating a response..."):
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents = prompt  ,
                )
        
            st.write(response.text)