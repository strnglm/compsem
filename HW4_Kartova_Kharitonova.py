# Работу совместно выполнили Елена Картова и Марта Харитонова
# Вариант 1

#ШАГ 1: граф -> матрица

#Достаем список слов (из ДЗ1)
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
languages = ['als', 'arb', 'bul', 'cat', 'cmn', 'dan', 'ell', 'eng', 'eus',
'fin', 'fra', 'glg', 'heb', 'hrv', 'ind', 'isl', 'ita', 'ita_iwn','jpn', 'lit', 'nld', 'nno', 'nob', 'pol', 'por', 'ron', 'slk',
'slv', 'spa', 'swe', 'tha', 'zsm']
lemmas_by_lang = dict()
for lang in languages:
    lemmas_by_lang[lang] = list(wn.synset('search.v.01').lemma_names(lang))

#Превращаем словарь со словами просто в список со словами
list_of_words_repeat = []
for lang in lemmas_by_lang:
    for lemma in lemmas_by_lang[lang]:
        list_of_words_repeat.append(lemma)

#Убираем повторяющиеся слова
list_of_words = []
list_of_words = list(dict.fromkeys(list_of_words_repeat))

# Берем граф из ДЗ1 и достаем узлы (синсеты то есть)
import networkx as nx
g = nx.read_gexf('graph.gexf')
nodes = []
stroka = []
for synset in g.nodes():
    stroka.append(synset[8:])
for i in stroka:
    nodes.append(i[:-2])

# Создаем датасет и добавляем слова в качестве названий столбцов
import pandas as pd
df = pd.DataFrame(columns=list_of_words)
df.insert(0, 'синсет',[])

# Функция для проверки наличия слова в синсете
def word_in_synset(word, synset):
    for lang in languages:
        if word in synset.lemma_names(lang):
            return True
    return False

# Добавим в датасет строки (добавляем строку и расставляем 0/1)
i = 0
for synset in nodes:
    df.at[i, 'синсет'] = synset
    for word in list_of_words:
        if word_in_synset(word, wn.synset(synset)):
            df.at[i, word] = 1
        else:
            df.at[i, word] = 0
    i += 1

# Матрица готова!
#print(df)

#ШАГ 2: сокращаем размерность

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics import euclidean_distances

matrix = df.loc[:, 'gjurmim':'ค้นหา']
array = matrix.values

similarities = euclidean_distances(array)
mds = manifold.MDS(n_components=2)

array_mds = mds.fit_transform(array)

plt.scatter(array_mds[:, 0], array_mds[:, 1])
phrases = df['синсет'].values
x = []
y = []
for pair in array_mds:
    x.append(pair[0])
    y.append(pair[1])
for i in range(len(x)):
    plt.scatter(x[i], y[i], c = 'black', s=10)
    plt.annotate(phrases[i], (x[i], y[i]+0.1), rotation=-30)
#plt.show()

#ШАГ 3: анализ

'''
В целом заметны следующие закономерности:

1. Большинство слов кучкуется в форме овала/прямоугольника,
что показывает близость значений подавляющего большинства синсетов
2. Близко к друг другу находятся пары синсетов, где один представляет собой одиночный глагол,
а другой - фразовый с тем же самым основным глаголом. Например, "quest" и "quest_for"
3. Интересно, что среди "выбросов" матрицы находятся синсеты "search.v.1", "search.v.2" и "search.v.4".
Мы думаем, что это может быть связано с тем, что эти значения 'search' достаточно специфичны относительно других значений,
которые находятся в основной куче. Значит, эти значения не близки дрругим значениям синсетов
'''

# ШАГ 4: добавление русских слов

df_rus = pd.DataFrame(columns=['синсет', 'слово', 'вектор'])
df_rus['синсет'] = nodes
# Чтобы получить этот список, мы достали все узлы, а ещё определения к синсетам через syn.definition(),
# а потом вручную подобрали перевод к каждому синсету.
rus_words = ["добиваться", "разведывать", "обнаруживать", "пытаться", "приглашать", "гнаться", "преследовать",
             "запрашивать", "резвиться", "записывать", "экзаменовать", "разведать", "обыскать", "контролировать",
             "инспектировать", "лечить", "абордаж", "расследование", "исследовать", "прочесывать", "сдерживать",
             "оглядеть", "выслеживать", "подходить", "подвезти", "собирать", "попрошайничать", "поиск", "захватывать",
             "просить", "ощупывать", "искать", "узнать", "открывать", "осматривать", "узнавание", "зондировать",
             "узнавать", "оценивать", "стремиться", "ворошить", "разыскиваемый", "нападать", "выискивать", "изучить",
             "желать", "разобраться", "нащупывать", "вторгаться", "расследовать", "требовать", "разыскивать",
             "предвкушать", "розыск", "тестировать", "отыскивать", "распознавать", "обыскивать", "вопрошать",
             "поизучать", "попросить", "охотиться", "проверять", "изучать", "спрашивать", "хотеть", "найти",
             "определить", "находить", "анализировать", "учиться"]
df_rus['слово'] = rus_words

import scipy
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

# У нас не получилось разархивировать модель по инструкции, поэтому мы достали ее вручную (ID=220)
model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)

# Вытащим вектор каждого слова и записываем в матрицу
vectors = []
for word in rus_words:
    word_formatted = ''
    # неглаголов у нас не так много - только 1 причастие и 5 существительных, поэтому проще, кажется, сделать так
    if word == 'разыскиваемый':
        word_formatted = word + '_ADJ'
    elif word == 'абордаж' or 'расследование' or 'поиск' or 'узнавание' or 'розыск':
        word_formatted = word + '_NOUN'
    else:
        word_formatted = word + '_VERB'
    vectors.append(model[word_formatted])

df_rus['вектор'] = vectors

# Матрица готова!
#print(df_rus)

# Теперь сокращаем ее и выводим

matrix_rus = df_rus.loc[:, 'слово':'вектор']
array_rus = matrix_rus.values

similarities_rus = euclidean_distances(array_rus)
mds_rus = manifold.MDS(n_components=2)

array_mds_rus = mds.fit_transform(array_rus)

plt.scatter(array_mds_rus[:, 0], array_mds_rus[:, 1])
phrases_rus = df_rus['слово'].values
x_rus = []
y_rus = []
for pair in array_mds_rus:
    x_rus.append(pair[0])
    y_rus.append(pair[1])
for i in range(len(x_rus)):
    plt.scatter(x_rus[i], y_rus[i], c='black', s=10)
    plt.annotate(phrases_rus[i], (x_rus[i], y_rus[i]+0.1), rotation=-30)

#plt.show()






