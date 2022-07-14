import random
import discord
import logging
import numpy as np

#It is strongly recommended that the logging module is configured, as no errors or warnings will be output if it is not set up.
logging.basicConfig(level=logging.INFO)

client = discord.Client()

@client.event
async def on_ready():
  print("We have logged in as " + str(client.user))

@client.event
async def on_message(message):
  if message.author == client.user:
    return

  # greetings
  list_of_greetings = ["hi","hello","hey"]
  greet = False
  words = message.content.split(" ")

  if words[0][0:2].lower() == "hi":
    greet = True
  elif words[0][0:5].lower() == "hello":
    greet = True
  elif words[0][0:3].lower() == "hey":
    greet = True
  
  if greet:
    await message.channel.send(random.choice(list_of_greetings).capitalize() + " " + str(message.author.name) + "!")
    
  # !help option
  if message.content.startswith("!help"):
    await message.reply("""I'm KLIYO, the bot version of Sharbel! and this server is a testing site for Sharbel's Machine Learning projects.

    - If you want information about the Machine Learning Models, send in the chat:

      !info

    - If you want to test the Machine Learning Models, send in the chat:

      !test
    
    - If you want to check the accuracy of the latest model, send in the chat:

      !accuracy

Enjoy :)""", mention_author = True)

  # !accuracy option
  if message.content.startswith("!accuracy"):
    await message.channel.send("Accuracy: 93.40/100")
  
  # !info option
  if message.content.startswith("!info"):
    await message.channel.send("""
Model #1
    Regression Model using Univariate Linear Regression
    This regression model determines the profit a new restaurant would make if they open their business in a new area in town.
    To determine the potential profit, we need the population of the new area. 
    You will be providing a population number of an area of your choice!

Model #2
    Classification Model using Logistic Regression
    This classification model estimates a university applicant's probability of admission based on the scores of the 2 final exams
    they did in high school.
    Let's pretend you are a new student and you already have your 2 final grades and you want to see if you have a chance of
    getting admitted!

Model #3
    Regression Model using Multivariate Linear Regression
    This regression model predicts the insurance medical cost of treatment of a person based on their age, BMI, number of
    children they have, and whether they are smokers or not.
    You will notice that the cost of treatment will increase when the person is older, if their bmi is high, if they have more children,
    and most importantly, if they smoke!

Have fun testing!
    """)

  # !test option
  if message.content.startswith("!test"):
    channel = message.channel
    await channel.send("""
    
Which Machine Learning Model do you want to test?

    1- Regression Model using Univariate Linear Regression
    2- Classification Model using Logistic Regression
    3- Regression Model using Multivariate Linear Regression
    
Enter the number of your choice model in the chat!
    """)

    def check(m):
      return m.channel == channel
    msg = await client.wait_for("message", check = check)
    
    model_choice = int(msg.content)
    
    if model_choice == 1:
      await channel.send("To start, enter a population number that is between 0.0 and 10.0 (for 10.0 being 100,000 people)")
      msg1 = await client.wait_for("message", check = check)
      x_test = float(msg1.content)
      w,b = get_wb0()
      prediction = x_test * w + b
      await channel.send("For a population " + str(x_test) + ", we predict a profit of $" + str(round((prediction*10000), 4)))
    
    elif model_choice == 2:
      await channel.send("To start, enter the 2 final exam grades (over 100) next to each other as such: 78 95")
      msg2 = await client.wait_for("message", check = check)
      ans = msg2.content.split()
      exam1 = float(ans[0])
      exam2 = float(ans[1])
      model_input = np.array([[exam1, exam2]])
      w,b = get_wb1()
      res = predict(model_input, w, b)
      if res[0] == 1:
        res = "Admitted :)"
      else:
        res = "Not Admitted :("
      await channel.send("Admission Decision: " + res)
    
    elif model_choice == 3:
      await channel.send("""
To start, you have to enter the age of the person, their bmi, the number of children they have, and whether they are smokers or not.
Write them in the chat in this form:

age bmi children smokers

for example: 19 27 0 1

(hint: put 1 if they are smokers, 0 if they are not)
      """)
      msg3 = await client.wait_for("message", check = check)
      ans = msg3.content.split()
      age = float(ans[0])
      bmi = float(ans[1])
      num_of_children = float(ans[2])
      smokers_or_not = float(ans[3])
      model_input = np.array([[age/64.0, bmi/53.13, num_of_children/5, smokers_or_not/1]])
      w,b = get_wb2()
      prediction = np.dot(model_input, w) + b
      prediction *= 63770.428010
      await channel.send("The Insurance Medical Cost of Treatment: $" + str(round(prediction[0], 4)))

def get_token():
  file = open("token.txt", "r")
  token = file.readline()
  file.close()
  return token

def get_wb0():
  file = open("wb0.txt", "r")
  wb = file.readlines()
  w = float(wb[0].strip())
  b = float(wb[1].strip())
  file.close()
  return w,b

def get_wb1():
  file = open("wb1.txt", "r")
  wb = file.readlines()
  num1 = float(wb[0].strip())
  num2 = float(wb[1].strip())
  w = np.array([[num1], [num2]])
  num3 = float(wb[2].strip())
  b = np.array([num3])
  file.close()
  return w,b

def get_wb2():
  file = open("wb2.txt", "r")
  wb = file.readlines()
  w_arr = wb[0].strip().split()
  w = np.array([float(x) for x in w_arr])
  b = float(wb[1].strip())
  file.close()
  return w, b

def predict(X, w, b):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):   
        z_i = np.dot(X[i], w) + b
        f_wb = 1 / (1 + np.exp(-z_i))
        p[i] = f_wb >= 0.5
    return p

client.run(get_token())