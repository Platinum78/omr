
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import OMR_base_class as omr
import os


# In[2]:


BASE_DIR = '/home/susung/Desktop/OMR_Response/'
answersheet = omr.OMR_Sheet(BASE_DIR+"SCAN0001.JPG")
answersheet.find_markers(800,1000)
vertical, horizontal = answersheet.sort_points()
latticepoint = answersheet.lattice()
answer = omr.AnswerSheet(answersheet.read_response(answersheet=True))


# In[3]:


filenames = os.listdir(BASE_DIR)
sheet = omr.OMR_Sheet()
for filename in filenames:
    sheet.load_sheet(BASE_DIR+filename)

    sheet.find_markers(800,1000)
    vertical, horizontal = sheet.sort_points()
    latticepoint = sheet.lattice()

    id_str, ans_str = sheet.read_response(threshold=220)
    score, ans = answer.mark_score(ans_str)
    print("correct: %s | id:%12s | score: %d"%(sheet.is_correct, id_str, score))

print('Finished!')


# In[4]:


sheet.img.shape

