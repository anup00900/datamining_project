from tkinter import *
import tkinter.font as font
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle
import webbrowser
from tkinter import ttk
import preprocess_query

sample_data= pd.read_csv('prepared_data.csv')
sample_duplicates= pd.read_csv('prepared_duplicates.csv')
sample_limit= len(sample_data)
# Taking the first sample_limit sample questions
sample_data= sample_data[:sample_limit]


# loading the pre-trained model
path= 'https://tfhub.dev/google/universal-sentence-encoder/4'
use_model= hub.load(path)

# importing the use_embedding
with open('use_embedding.data','rb') as filehandle:
    use_embedding= pickle.load(filehandle)

# importing the sbert model
filename= 'sbert_model.sav'
sbert_model= pickle.load(open(filename, 'rb'))

# importing the sbert_embedding
with open('sbert_embedding.data','rb') as filehandle:
    sbert_embedding= pickle.load(filehandle)

def cosine(u, v):
    ''' this function returns the cosine similarity b/w two vectors'''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# creating a GUI window
root = Tk()
root.title("Question Similarities")

frame1 = LabelFrame(root,text="Search Query")
frame1.grid(row=0,column=0,padx = 10,pady = 10,sticky = 'nswe')
frame2 = LabelFrame(root,text="Similar Questions")
frame2.grid(row=1,column=0,padx = 10,pady = 10,sticky = 'nswe')


def oneClick():

    global query
    query= InputQuestion.get()
    query= preprocess_query.preProcessing(query)
    use_query_vec = use_model([query])[0]

    # Finding the similiarity scores with all the questions in our sample_data (sample of sample_limit questions)
    use_cosine_simscore=[]
    for i in range(len(use_embedding)):
        current_question_use_embedding= use_embedding[i]
        use_cosine_sim = cosine(use_query_vec, current_question_use_embedding)
        use_cosine_simscore.append(use_cosine_sim)

    use_cosine_simscore= np.array(use_cosine_simscore)

    # Finding the indices of the top N most similar questions
    N= 7
    indices= np.argpartition(use_cosine_simscore, -N)[-N:]

    similar_questions=[]
    title = []
    links = []
    for i in indices:
        similar_questions.append(sample_data.iloc[i]['title'])
        title.append(sample_data.iloc[i]['title'])
        links.append(sample_data.iloc[i]['link'])
    # displaying the query question and the top N similar questions in the GUI
    in_frame1 = LabelFrame(frame2,text="Matching "+ str(1))
    in_frame1.grid(row=0,column=0,padx = 8,pady = 8,sticky = 'nswe')

    Label(in_frame1,text="Title : "+title[0]).grid(row=0,sticky = 'w',padx = 7,pady = 7)
    in_btn1 = ttk.Button(in_frame1,width=25,text="Question Link",command=lambda: webbrowser.open(links[0]))
    in_btn1.grid(row=2,padx = 7,pady = 7,sticky = 'W')


    in_frame2 = LabelFrame(frame2,text="Matching "+ str(2))
    in_frame2.grid(row=1,column=0,padx = 8,pady = 8,sticky = 'nswe')

    Label(in_frame2,text="Title : "+title[1]).grid(row=0,sticky = 'w',padx = 7,pady = 7)
    in_btn2 = ttk.Button(in_frame2,width=25,text="Question Link",command=lambda: webbrowser.open(links[1]))
    in_btn2.grid(row=2,padx = 7,pady = 7,sticky = 'W')


    in_frame3 = LabelFrame(frame2,text="Matching "+ str(3))
    in_frame3.grid(row=2,column=0,padx = 8,pady = 8,sticky = 'nswe')

    Label(in_frame3,text="Title : "+title[2]).grid(row=0,sticky = 'w',padx = 7,pady = 7)
    in_btn3 = ttk.Button(in_frame3,width=25,text="Question Link",command=lambda: webbrowser.open(links[2]))
    in_btn3.grid(row=2,padx = 7,pady = 7,sticky = 'W')


    in_frame4 = LabelFrame(frame2,text="Matching "+ str(4))
    in_frame4.grid(row=3,column=0,padx = 8,pady = 8,sticky = 'nswe')

    Label(in_frame4,text="Title : "+title[3]).grid(row=0,sticky = 'w',padx = 7,pady = 7)
    in_btn4 = ttk.Button(in_frame4,width=25,text="Question Link",command=lambda: webbrowser.open(links[3]))
    in_btn4.grid(row=2,padx = 7,pady = 7,sticky = 'W')


    in_frame5 = LabelFrame(frame2,text="Matching "+ str(5))
    in_frame5.grid(row=4,column=0,padx = 8,pady = 8,sticky = 'nswe')

    Label(in_frame5,text="Title : "+title[4]).grid(row=0,sticky = 'w',padx = 7,pady = 7)
    in_btn5 = ttk.Button(in_frame5,width=25,text="Question Link",command=lambda: webbrowser.open(links[4]))
    in_btn5.grid(row=2,padx = 7,pady = 7,sticky = 'W')


    in_frame6 = LabelFrame(frame2,text="Matching "+ str(6))
    in_frame6.grid(row=5,column=0,padx = 8,pady = 8,sticky = 'nswe')

    Label(in_frame6,text="Title : "+title[5]).grid(row=0,sticky = 'w',padx = 7,pady = 7)
    in_btn6 = ttk.Button(in_frame6,width=25,text="Question Link",command=lambda: webbrowser.open(links[5]))
    in_btn6.grid(row=2,padx = 7,pady = 7,sticky = 'W')


    in_frame7 = LabelFrame(frame2,text="Matching "+ str(7))
    in_frame7.grid(row=6,column=0,padx = 8,pady = 8,sticky = 'nswe')

    Label(in_frame7,text="Title : "+title[6]).grid(row=0,sticky = 'w',padx = 7,pady = 7)
    in_btn7 = ttk.Button(in_frame7,width=25,text="Question Link",command=lambda: webbrowser.open(links[6]))
    in_btn7.grid(row=2,padx = 7,pady = 7,sticky = 'W')



btn1 = Button(frame1,width=25,text="Search",command=oneClick)
btn1.grid(row=2,column=0,padx = 10,pady = 10,sticky = 'nswe')

Label(frame1,text="Please enter the query question in the box and hit search!").grid(row=0,sticky = 'w',padx = 10,pady = 10)
InputQuestion = Entry(frame1, width=75, borderwidth= 2)
InputQuestion.grid(row=1,column=0,sticky='e',padx=10,pady=10)

root.mainloop()
