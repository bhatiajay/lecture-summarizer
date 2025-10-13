import speech_recognition as sr
#import sys
import spacy

nlp = spacy.load("en_core_web_sm")

#duration = int(sys.argv[1])
duration = int(input('What is your duration?\n'))


r = sr.Recognizer()
print("Please talk")
with sr.Microphone() as source:
    audio_data = r.record(source, duration=duration)
    print("Recognizing...")
    text = r.recognize_google(audio_data)

#testText = "For students to succeed at finding the derivatives and antiderivatives with calculus, they will need facility with algebraic expressions, particularly in modification and transformation of such expressions. Leonhard Euler wrote the first precalculus book in 1748 called Introductio in analysin infinitorum (Latin: Introduction to the Analysis of the Infinite), which was meant as a survey of concepts and methods in analysis and analytic geometry preliminary to the study of differential and integral calculus.[2] He began with the fundamental concepts of variables and functions. His innovation is noted for its use of exponentiation to introduce the transcendental functions. The general logarithm, to an arbitrary positive base, Euler presents as the inverse of an exponential function."    

doc = nlp (text=text)

wordCollection = {}

for word in doc:    
    word = word.text.lower()
    if word in wordCollection:
        wordCollection[word] += 1
    else:
        wordCollection[word] = 1

sentences = []

sentenceScore = 0

for i, sentence in enumerate(doc.sents):   
    for word in sentence:
        word = word.text.lower()
        sentenceScore += wordCollection[word]
    sentences.append((i, sentence.text.replace("\n", ""), sentenceScore/len(sentence)))

sortedSentence = sorted(sentences, key=lambda x: -x[2])

threeTop = sorted(sortedSentence[:3], key=lambda x: x[0])

summary = ""

for sentence in threeTop:
    summary += sentence[1] + ""

print(summary)

