from tkinter import *
from tkinter import messagebox
import json

fields = 'Episodes', 'Learning Rate', 'Gamma', 'Batch Size'

def fetch(entries):
   for entry in entries:
      field = entry[0]
      text  = entry[1].get()
      print('%s: "%s"' % (field, text)) 

def makeform(root, fields):
   entries = []
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=15, text=field, anchor='w')
      ent = Entry(row)
      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries.append((field, ent))
   return entries

def save_parameters_to_file(entries, root):
   params_dict = dict()

   for entry in entries:
      field = entry[0]
      text  = entry[1].get()
      if field == "Episodes":
      	if int(text) < 5:
      		messagebox.showinfo("Error", "Too few episodes. Please enter reasonable number of episodes")
      		return
      	elif int(text) > 100000:
      		messagebox.showinfo("Error", "Too many episodes. Please enter reasonable number of episodes")
      		return
      if text == "":
      	messagebox.showinfo("Error", "Please enter a value for " + str(field))
      	return
      params_dict[field] = float(text)

   with open("./params.json", "w") as outfile:
      json.dump(params_dict, outfile)

   messagebox.showinfo("Yayyy!", "The parameter values were successfully saved. Beginning training!")
   root.destroy()
   return

root = Tk()
ents = makeform(root, fields)
root.title("Reinforcement Learning")
root.bind('<Return>', (lambda event, e = ents: fetch(e)))

b1 = Button(root, text = 'SUBMIT', command = (lambda e=ents: save_parameters_to_file(e, root)), bg = "blue")
b1.pack(side = LEFT, padx = 5, pady = 5)
root.mainloop()