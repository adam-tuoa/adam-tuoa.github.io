---
title: "ChatGPT & DALL-E 2 with Python"
---

## ChatGPT & DALL-E 2

Basic code comes from Sophia Yang's article:

https://towardsdatascience.com/chatgpt-and-dall-e-2-in-a-panel-app-1c921d7d9021

I added the query loop, image and conversation saving

Nothing fancy - just in Jupyter Notebook.

For saving - create two folders in the directory where script is filed

- `chats`
- `images`

When chatting:

- ChaptGPT: default - just enter prompt
- DALL-E 2: type `image:` plus text 
- Save image: after image is generated, type `save`
- To finish: type `exit`

After you finish a chat, you will be asked if you wish to save the conversation


```python
from chatgpt import Conversation
import urllib.request
import sys
import os
import openai
from IPython.display import display, HTML
from IPython.display import Image


import pandas as pd

working_directory = os.getcwd()
working_directory = "/Users/adam/data/ChatGPT/"

openai.api_key = "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"
```

# A list of models 

If you wish to try other models, a list can be obtained with use of this code:

`models = pd.DataFrame(openai.Model.list()["data"])`  



```python
# Functions

def openai_completion(prompt):
    # Send prompt/Get ChatGPT reponse
    response = openai.Completion.create(
      model="text-davinci-003",  # this can be changed to access different models
      prompt=prompt,
      max_tokens=2000,           # you can change this to different values for longer shorter prompts/replies
      temperature=0.5            # the "straightness" of the replies 
    )
    return response['choices'][0]['text']


def openai_image(prompt):
    #Send prompt/Get DALL-E 2 response
    response = openai.Image.create(
      prompt=prompt,
      n=1,
      size="256x256"             # this can be different values "256x256", "512x512", "1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url

# 
def append_strings_to_df(question, response):
    # Conversation text as dataframe - ready for saving upon request (enter "save" as prompt)
    global conversation
    dict = {"Question": question, "Response": response}
    data = pd.DataFrame(dict, index=[0])
    conversation = pd.concat([conversation, data])
    
```


```python
# Set up empty conversation df
conversation = pd.DataFrame(columns=["Question", "Response"])

# Main prompt loop
while True:
    question = input(">>")
    if question.lower() == "exit":
        append_strings_to_df(question, "")
        break
    elif question[0:7].lower() == "image: ":            #create image based on prompt after "image: "
        image_link = openai_image(question[7:])
        image_obj = Image(url=image_link)
        display(image_obj)
        file_name = question[7:18] + ".png"             #extracts text after "image: " to create filename
        append_strings_to_df(question, image_link)
    elif question.lower() == "save":                    #to save image
        file_path = working_directory + "images/" + file_name
        urllib.request.urlretrieve(image_link, file_path)
    else:    
        response = openai_completion(question)          #chatGPT response
        print(response)
        print()
        append_strings_to_df(question, response)

# After exit - option to save conversation        
while True:
    save_convo = input("Save the conversation (Y/N)? :").lower()
    if save_convo == "y":
        file_name = input("Name? :")
        conversation.to_csv(working_directory + "chats/" + file_name + ".csv", index=False)
        
        break
    elif save_convo == "n":
        print("No probs - thanks!")
        break   
```
