import speech_recognition as sr
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


# duration = int(input('What is your duration?\n'))

# r = sr.Recognizer()
# print("Please talk")
# with sr.Microphone() as source:
#     audio_data = r.record(source, duration=duration)
#     print("Recognizing...")
#     text = r.recognize_google(audio_data)


text = "Holy Roman Empire, the varying complex of lands in western and central Europe ruled by the Holy Roman emperor, a title held first by Frankish and then by German kings for 10 centuries. The Holy Roman Empire existed from 800 to 1806. For histories of the territories governed at various times by the empire, see France; Germany; Italy. The precise term Sacrum Romanum Imperium dates only from 1254, though the term Holy Empire reaches back to 1157, and the term Roman Empire was used from 1034 to denote the lands under Conrad IIs rule. The term Roman emperor is older, dating from Otto II (died 983). This title, however, was not used by Otto IIs predecessors, from Charlemagne (or Charles I) to Otto I, who simply employed the phrase imperator augustus (“august emperor”) without any territorial adjunct. The first title that Charlemagne is known to have used, immediately after his coronation in 800, is “Charles, most serene Augustus, crowned by God, great and pacific emperor, governing the Roman empire.” This clumsy formula, however, was soon discarded. These questions about terms reveal some of the problems involved in the nature and early history of the empire. It can be regarded as a political institution, or approached from the point of view of political theory, or treated in the context of the history of Christendom as the secular counterpart of a world religion. The history of the empire is also not to be confused or identified with the history of its constituent kingdoms, Germany and Italy, though clearly they are interrelated. The constituent territories retained their identity; the emperors, in addition to the imperial crown, also wore the crowns of their kingdoms. Finally, whereas none of the earlier emperors from Otto I had assumed the imperial title until actually crowned by the pope in Rome, after Charles V none was emperor in this sense, though all laid claim to the imperial dignity as if they had been duly crowned as well as elected. Despite these anomalies and others, the empire, at least in the Middle Ages, was by common assent, along with the papacy, the most important institution of western Europe. Theologians, lawyers, popes, ecclesiastics, rulers, rebels like Arnold of Brescia and Cola di Rienzo, literary figures like Dante and Petrarch, and the practical men, members of the high nobility, on whom the emperors relied for support, all saw the empire in a different light and had their own ideas of its origin, function, and justification. Among these heterogeneous and often incompatible views, three may be said to predominate: (1) the papal theory, according to which the empire was the secular arm of the church, set up by the papacy for its own purposes and therefore answerable to the pope and, in the last resort, to be disposed of by him; (2) the imperial, or Frankish, theory, which placed greater emphasis on conquest and hegemony as the source of the emperor’s power and authority and according to which he was responsible directly to God; and (3) the popular, or Roman, theory (the “people” at this stage being synonymous with the nobility and in this instance with the Roman nobility), according to which the empire, following the tradition of Roman law, was a delegation of powers by the Roman people. Of the three theories the last was the least important; it was evidently directed against the pope, whose constitutive role it implicitly denied, but it was also a specifically Italian reaction against the predominance in practice of Frankish and German elements. It is also important to distinguish between the universalist and localist conceptions of the empire, which have been the source of considerable controversy among historians. According to the former, the empire was a universal monarchy, a “commonwealth of the whole world, whose sublime unity transcended every minor distinction”; and the emperor “was entitled to the obedience of Christendom.” According to the latter, the emperor had no ambition for universal dominion; his policy was limited in the same way as that of every other ruler, and when he made more far-reaching claims his object was normally to ward off the attacks either of the pope or of the Byzantine emperor. According to this view, also, the origin of the empire is to be explained by specific local circumstances rather than by far-flung theories."

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    print(summary)


summarize(text, 0.10)