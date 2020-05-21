import tkinter as tk
import time
from tkinter import *
import requests
import webbrowser
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import simpleaudio as sa
import warnings
import nltk
import numpy as np
import random
import re, string, unicodedata
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import wikipedia as wk
from collections import defaultdict
warnings.filterwarnings("ignore", category=DeprecationWarning)

HEIGHT = 700
WEIGHT = 800

root = tk.Tk()
root.title(" TEAM CURE")

canvas = tk.Canvas(root, height=HEIGHT, width=WEIGHT)
canvas.pack()

frame = tk.Frame(root, bg='#80c1ff')
frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)


def epharm1():
    def ep11():
        webbrowser.open("www.1mg.com/labs?utm_source=ASM&utm_medium=Internal_Labs&utm_campaign=May-Asm")

    def ep12():
        webbrowser.open("https://pharmeasy.in/")

    def ep13():
        webbrowser.open("https://www.netmeds.com/")

    def ep14():
        webbrowser.open("https://generico.in/")

    top = Toplevel()
    top.title("E-pharm")
    canvas1 = tk.Canvas(top, height=HEIGHT, width=WEIGHT)
    canvas1.pack()
    frame1 = tk.Frame(top, bg='#80c1ff')
    frame1.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    button21 = tk.Button(frame1, text="1MG", bg='white', fg='red', font=40, command=lambda: ep11())
    button21.place(relx=0.1, rely=0.25, relwidth=0.8, relheight=0.1)

    button22 = tk.Button(frame1, text="Pharmacy", bg='white', fg='red', font=40, command=lambda: ep12())
    button22.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.1)

    button23 = tk.Button(frame1, text="Netmeds", bg='white', fg='red', font=40, command=lambda: ep13())
    button23.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.1)

    button24 = tk.Button(frame1, text="Generico", bg='white', fg='red', font=40, command=lambda: ep14())
    button24.place(relx=0.1, rely=0.7, relwidth=0.8, relheight=0.1)


def docter():
    def ep11():
        webbrowser.open("https://www.esanjeevaniopd.in/")

    def ep12():
        webbrowser.open("https://tatabridgital.com/#/")

    def ep13():
        webbrowser.open("https://connectsense.techmahindra.com/")

    def ep14():
        webbrowser.open("https://www.cure.fit/care/consult")

    top = Toplevel()
    top.title("Consult Doctor")
    canvas1 = tk.Canvas(top, height=HEIGHT, width=WEIGHT)
    canvas1.pack()
    frame1 = tk.Frame(top, bg='#80c1ff')
    frame1.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    button21 = tk.Button(frame1, text="Esanjeevaniopd", bg='white', fg='red', font=40, command=lambda: ep11())
    button21.place(relx=0.1, rely=0.25, relwidth=0.8, relheight=0.1)

    button22 = tk.Button(frame1, text="Tata Bridgital", bg='white', fg='red', font=40, command=lambda: ep12())
    button22.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.1)

    button23 = tk.Button(frame1, text="Connect Sense", bg='white', fg='red', font=40, command=lambda: ep13())
    button23.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.1)

    button24 = tk.Button(frame1, text="Cure Fit", bg='white', fg='red', font=40, command=lambda: ep14())
    button24.place(relx=0.1, rely=0.7, relwidth=0.8, relheight=0.1)


def donation():
    def ep11():
        webbrowser.open("https://www.pmcares.gov.in/en/#")

    def ep12():
        webbrowser.open("https://www.who.int/emergencies/diseases/novel-coronavirus-2019/donate")

    top = Toplevel()
    top.title("Donation")
    canvas1 = tk.Canvas(top, height=HEIGHT, width=WEIGHT)
    canvas1.pack()
    frame1 = tk.Frame(top, bg='#80c1ff')
    frame1.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    button21 = tk.Button(frame1, text="PM Care", bg='white', fg='red', font=40, command=lambda: ep11())
    button21.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.1)

    button22 = tk.Button(frame1, text="WHO", bg='white', fg='red', font=40, command=lambda: ep12())
    button22.place(relx=0.1, rely=0.6, relwidth=0.8, relheight=0.1)


def grocery():
    def ep11():
        webbrowser.open(
            "https://www.amazon.in/s?k=vegetables+in+fresh&rh=p_n_alm_brand_id%3Actnow&dc&crid=F53PV4JPJ05A&qid=1589516338&rnid=17107034031&sprefix=vegeta%2Caps%2C284&ref=sr_hi_1")

    def ep12():
        webbrowser.open("https://www.bigbasket.com/")

    top = Toplevel()
    top.title("Grocery")
    canvas1 = tk.Canvas(top, height=HEIGHT, width=WEIGHT)
    canvas1.pack()
    frame1 = tk.Frame(top, bg='#80c1ff')
    frame1.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    button21 = tk.Button(frame1, text="Amazon", bg='white', fg='red', font=40, command=lambda: ep11())
    button21.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.1)

    button22 = tk.Button(frame1, text="Big Basket", bg='white', fg='red', font=40, command=lambda: ep12())
    button22.place(relx=0.1, rely=0.6, relwidth=0.8, relheight=0.1)


def mobile():
    webbrowser.open("https://zipzup.in/")


def onclick1():
    top = Toplevel()
    top.title("CovidBot")
    canvas1 = tk.Canvas(top, height=HEIGHT, width=WEIGHT)
    canvas1.pack()
    frame1 = tk.Frame(top, bg='#80c1ff')
    frame1.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    button21 = tk.Button(frame1, text="ePharm", bg='white', fg='red', font=40, command=lambda: epharm1())
    button21.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.1)

    button22 = tk.Button(frame1, text="Consult doctor", bg='white', fg='red', font=40, command=lambda: docter())
    button22.place(relx=0.1, rely=0.25, relwidth=0.8, relheight=0.1)

    button23 = tk.Button(frame1, text="Donation", bg='white', fg='red', font=40, command=lambda: donation())
    button23.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.1)

    button22 = tk.Button(frame1, text="Groceries", bg='white', fg='red', font=40, command=lambda: grocery())
    button22.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.1)

    button23 = tk.Button(frame1, text="Mobile service", bg='white', fg='red', font=40, command=lambda: mobile())
    button23.place(relx=0.1, rely=0.7, relwidth=0.8, relheight=0.1)

    authenticator = IAMAuthenticator('17c4MX25HKWu1wVVfEr0I4Tjyg1D5U8inYrwGbUwk5Kl')
    text_to_speech = TextToSpeechV1(
        authenticator=authenticator
    )

    text_to_speech.set_service_url(
        'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/538ffb13-4219-4216-9ece-66e72e92f5f9')

    with open('covidbot.wav', 'wb') as audio_file:
        audio_file.write(
            text_to_speech.synthesize(
                'Hello! I am covid bot, I am specially developed to find solutions for the different covid crisis!',
                voice='en-US_LisaVoice',
                accept='audio/wav'
            ).get_result().content)
    filename = 'covidbot.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def onclick2():
    root.destroy()
    # Getting current Location
    res = requests.get('https://ipinfo.io/')
    data = res.json()

    authenticator = IAMAuthenticator('17c4MX25HKWu1wVVfEr0I4Tjyg1D5U8inYrwGbUwk5Kl')
    text_to_speech = TextToSpeechV1(
        authenticator=authenticator
    )

    text_to_speech.set_service_url(
        'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/538ffb13-4219-4216-9ece-66e72e92f5f9')

    with open('healthcarebot.wav', 'wb') as audio_file:
        audio_file.write(
            text_to_speech.synthesize(
                "I am like your friend , Please don't hide anything from me!! I can Help you to diagnose the disease!",
                voice='en-US_AllisonVoice',
                accept='audio/wav'
            ).get_result().content)
    filename = 'healthcarebot.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    city = data['city']
    country = data['country']

    location = data['loc'].split(',')
    latitude = location[0]
    longitude = location[1]

    pls = ""
    covid = ['You may have Corona Virus']
    training_data = pd.read_csv('Training.csv')
    testing_data = pd.read_csv('Testing.csv')
    cols = training_data.columns
    cols = cols[:-1]
    x = training_data[cols]
    y = training_data['prognosis']
    y1 = y

    reduced_data = training_data.groupby(training_data['prognosis']).max()

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx = testing_data[cols]
    testy = testing_data['prognosis']
    testy = le.transform(testy)

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)

    # print(clf.score(x_train,y_train))
    # scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=3)

    # print(clf.score(testx,testy))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols


    print("Please only reply in yes or no for each symptoms!\n\n")
    def print_disease(node):

        node = node[0]

        val = node.nonzero()

        disease = le.inverse_transform(val[0])
        return disease

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        # print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        symptoms_present = []

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]

                threshold = tree_.threshold[node]

                print(name + " ?")
                ans = input()
                ans = ans.lower()
                if ans == 'yes':
                    val = 1
                    recurse(tree_.children_left[node], depth + 1)

                else:
                    val = 0
                if val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])
                disease_name = ("You may have " + present_disease)
                print(disease_name)
                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                print("symptoms present  " + str(list(symptoms_present)))
                print("symptoms given " + str(list(symptoms_given)))
                confidence_level = (1.0 * len(symptoms_present)) / len(symptoms_given)
                print("My confidence level is " + str(confidence_level))
                if disease_name == covid and country == 'IN':
                    print("Your current location is : \nCountry: " + country + "\nCity: " + city)
                    print(
                        "Dont worry I will help you with helpline numbers which is officially released by Ministry of Health and Family Welfare,Government of India")
                    print("Please consult at an earliest!!")
                    webbrowser.open("https://www.mohfw.gov.in/pdf/coronvavirushelplinenumber.pdf")

                elif disease_name == covid and country != 'IN':
                    print("\nAs per your current location, I can help you with the following for Corona Virus")
                    webbrowser.open("https://www.who.int/")
                else:
                    print("\n")

        recurse(0, 1)

    tree_to_code(clf, cols)
    time.sleep(5)


def onclick3():
    root.destroy()
    authenticator = IAMAuthenticator('17c4MX25HKWu1wVVfEr0I4Tjyg1D5U8inYrwGbUwk5Kl')
    text_to_speech = TextToSpeechV1(
        authenticator=authenticator
    )

    text_to_speech.set_service_url(
        'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/538ffb13-4219-4216-9ece-66e72e92f5f9')

    with open('wikibot.wav', 'wb') as audio_file:
        audio_file.write(
            text_to_speech.synthesize(
                "My name is wiki bot and I'm blessed with knowledge!  aha thanks to wikipedia!",
                voice='en-US_MichaelVoice',
                accept='audio/wav'
            ).get_result().content)
    filename = 'wikibot.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    data = open('HR.txt', 'r', errors='ignore')
    raw_data = data.read()
    raw_data = raw_data.lower()

    sent_tokens = nltk.sent_tokenize(raw_data)


    def Normalize(text):
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

        word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))


        new_words = []
        for word in word_token:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)


        rmv = []
        for w in new_words:
            text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", w)
            rmv.append(text)


        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        lmtzr = WordNetLemmatizer()
        lemma_list = []
        rmv = [i for i in rmv if i]
        for token, tag in nltk.pos_tag(rmv):
            lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
            lemma_list.append(lemma)
        return lemma_list

    # Greetings
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

    def greeting(user_response):
        for word in user_response.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

    def response(user_response):
        robo_response = ''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)

        vals = linear_kernel(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if (req_tfidf == 0) or "tell me about" in user_response:

            print("Checking Wikipedia")
            if user_response:
                robo_response = wikipedia_data(user_response)
                return robo_response
        else:
            robo_response = robo_response + sent_tokens[idx]
            return robo_response


    def wikipedia_data(input):
        reg_ex = re.search('(.*)', input)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                ny = wk.summary(topic, sentences=3)
                return ny
        except Exception as e:
            print(e)

    flag = True
    print("Ask me anything: ")
    while (flag == True):
        user_response = input()
        user_response = user_response.lower()
        if (user_response != 'bye'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                flag = False
                print("Wikibot : You are welcome..")
            else:
                if (greeting(user_response) != None):
                    print("Wikibot : " + greeting(user_response))
                else:
                    print("Wikibot : ", end="")
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag = False
            authenticator = IAMAuthenticator('17c4MX25HKWu1wVVfEr0I4Tjyg1D5U8inYrwGbUwk5Kl')
            text_to_speech = TextToSpeechV1(
                authenticator=authenticator
            )

            text_to_speech.set_service_url(
                'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/538ffb13-4219-4216-9ece-66e72e92f5f9')

            with open('wikiexit.wav', 'wb') as audio_file:
                audio_file.write(
                    text_to_speech.synthesize(
                        "Bye! take care!!",
                        voice='en-US_MichaelVoice',
                        accept='audio/wav'
                    ).get_result().content)
            filename = 'wikiexit.wav'
            wave_obj = sa.WaveObject.from_wave_file(filename)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            time.sleep(5)




def onclick4():
    authenticator = IAMAuthenticator('17c4MX25HKWu1wVVfEr0I4Tjyg1D5U8inYrwGbUwk5Kl')
    text_to_speech = TextToSpeechV1(
        authenticator=authenticator
    )

    text_to_speech.set_service_url(
        'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/538ffb13-4219-4216-9ece-66e72e92f5f9')

    with open('edubot.wav', 'wb') as audio_file:
        audio_file.write(
            text_to_speech.synthesize(
                "Knowledge is power!  and the process of gaining knowledge never stops!  So I am here to help you to find any book you want so as to gain knowledge!",
                voice='en-US_LisaVoice',
                accept='audio/wav'
            ).get_result().content)
    filename = 'edubot.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    def nextclic():
        bookcap = text12.get()
        webbrowser.open("https://b-ok.cc/s/" + bookcap)

    top = Toplevel()
    top.title("Edubot")
    canvas1 = tk.Canvas(top, height=HEIGHT, width=WEIGHT)
    canvas1.pack()
    frame1 = tk.Frame(top, bg='#80c1ff')
    frame1.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    label = tk.Label(frame1, text="ELIBRARY")
    label.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.1)
    label = tk.Label(frame1, text="BOOK NAME/GENRE")
    label.place(relx=0.1, rely=0.3, relwidth=0.35, relheight=0.1)
    text12 = tk.Entry(frame1)
    text12.place(relx=0.5, rely=0.3, relwidth=0.35, relheight=0.1)

    button21 = tk.Button(frame1, text="Search", bg='white', fg='red', font=40, command=lambda: nextclic())
    button21.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.1)


button1 = tk.Button(frame, text="CovidBot", bg='yellow', fg='red', command=lambda: onclick1(), font=40)
button1.place(relx=0.1, rely=0.25, relwidth=0.8, relheight=0.1)

button = tk.Button(frame, text="HealthCare Bot", bg='yellow', fg='red', command=lambda: onclick2(), font=40)
button.place(relx=0.1, rely=0.4, relwidth=0.8, relheight=0.1)

button = tk.Button(frame, text="Wikibot", bg='yellow', fg='red', command=lambda: onclick3(), font=40)
button.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.1)

button = tk.Button(frame, text="Edubot", bg='yellow', fg='red', command=lambda: onclick4(), font=40)
button.place(relx=0.1, rely=0.7, relwidth=0.8, relheight=0.1)

authenticator = IAMAuthenticator('17c4MX25HKWu1wVVfEr0I4Tjyg1D5U8inYrwGbUwk5Kl')
text_to_speech = TextToSpeechV1(
    authenticator=authenticator
)

text_to_speech.set_service_url('https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/538ffb13-4219-4216-9ece-66e72e92f5f9')

with open('welcome.wav', 'wb') as audio_file:
    audio_file.write(
        text_to_speech.synthesize(
            'Hello, welcome to the world of bots which has been developed to ease your life!',
            voice='en-GB_KateVoice',
            accept='audio/wav'
        ).get_result().content)
filename = 'welcome.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()




root.mainloop()

authenticator = IAMAuthenticator('17c4MX25HKWu1wVVfEr0I4Tjyg1D5U8inYrwGbUwk5Kl')
text_to_speech = TextToSpeechV1(
    authenticator=authenticator
)

text_to_speech.set_service_url(
    'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/538ffb13-4219-4216-9ece-66e72e92f5f9')

with open('exit.wav', 'wb') as audio_file:
    audio_file.write(
        text_to_speech.synthesize(
            "Hope you liked our service, bye! take care!",
            voice='en-GB_KateVoice',
            accept='audio/wav'
        ).get_result().content)
filename = 'exit.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()