# This is a sample Python script.
import os
import re

# ------------- Spark imports--------------------------
import sparknlp
from pyspark.sql.types import StringType
from sparknlp.base import *
from sparknlp.annotator import *
import pyspark.sql.functions as F
from sparknlp.pretrained import PretrainedPipeline


#-----------------------------------------------------
from pyspark.ml import Pipeline
import PyPDF2 as PyPDF2
import gensim
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import  *
import numpy as np
from gensim import corpora, models
from gensim.utils import tokenize
import nltk
from nltk.corpus import wordnet as wn
import fasttext as ft

nltk.download("wordnet")

stemmer = SnowballStemmer("english")

ft_model = ft.load_model("/home/nkosikhona/Downloads/lid.176.bin")
spark = sparknlp.start()



def get_af_stop_words(file_name="/home/nkosikhona/af_stopwords.text"):
    with open(file_name,'rb') as file_af_stop:
        af_stopwords = [line.strip() for line in file_af_stop]
    return af_stopwords

def language_id(text,model = ft_model):
    text = text.replace("\n"," ")
    prediction = model.predict([text])
    # returns language label as __label__en, and accuracy eg 0.998
    return prediction[0][0][0],[1][0][0]

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text,pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len (token)> 3:
            result.append(lemmatize_stemming(token.strip()))
    return result

def pdf_to_text_all_pages(file_path):
    current_text =""
    with open(file_path,'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = pdf_reader.numPages
        for i in range(num_pages):
            current_page =pdf_reader.getPage(i)
            current_text+=current_page.extractText()+"\n"
    return current_text
def text_remove_links(text):
    temp = re.sub(r"http\S+", " ", text)
    temp = re.sub(r"www.\S+", " ", temp)
    return temp

def text_remove_numbers(text):
    temp = re.sub("[^a-zA-Z]", " ",text)
    return temp

def display_text(text):
    print(text)

def clean_text(text):
    temp_text = text_remove_links(text)
    temp_text = text_remove_numbers(temp_text)

    display_text(temp_text)
    return temp_text

def go_through_files_get_text(directory,initial=0,total=50):
    all_documents =[]
    for file_name in os.listdir(directory):
        initial=initial+1
        file =os.path.join(directory,file_name)
        if(file_name.lower().endswith('.pdf')):
            if os.path.isfile(file):
                all_documents.append(pdf_to_text_all_pages(file))
        if initial>=total:
            break

    return all_documents
def get_dictionary(preporocessed_docs):
    dictionary =gensim.corpora.Dictionary(preporocessed_docs)
    dictionary.filter_extremes(no_below=15,no_above=0.1,keep_tokens=100000)
    return dictionary
def get_bag_of_words(dictionary,processed_documents):
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_documents]
    return bow_corpus

def view_dictionary(dictionary):
    count =0;
    for k,v in dictionary.iteritmes():
        print(k,v)
        count+=1
        if count>10:
            break
def prepare_text_for_lda_en(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token.strip())>4]
    tokens = [token for token in tokens if token not in gensim.parsing.preprocessing.STOPWORDS]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def prepare_text_for_lda_af(text_af_list):
    #https://sparknlp.org/2021/04/02/lemma_af.html spark syntax
    document_assembler = DocumentAssembler()
    document_assembler.setInputCol("text")
    document_assembler.setOutputCol("document")

    tokenizer = Tokenizer()
    tokenizer.setInputCols(["document"])
    tokenizer.setOutputCol("token")

    lemmatizer = LemmatizerModel.pretrained("lemma","af")
    lemmatizer.setInputCols(["token"])
    lemmatizer.setOutputCol("lemma")

    nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])

    df = spark.createDataFrame(text_af_list, StringType()).toDF("text")
    result = nlp_pipeline.fit(df).transform(df)
    #result.select('lemma.result').show(truncate=False)
    result_df = result.select(F.explode(F.arrays_zip(result.token.result,
                                                     result.lemma.result)).alias("cols")) \
        .select(F.expr("cols['0']").alias("token"),
                F.expr("cols['1']").alias("lemma")).toPandas()
    #result_pd = result.select(F.explode(F.arrays_zip(result.lemma.result)).alias("cols")) \
    #    .select(F.expr("cols['0']").alias("lemma")).toPandas()

    return result_df
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
if __name__ == '__main__':
    af_list = ["Ons het besliste teen-resessiebesteding deur die regering geïmplementeer , veral op infrastruktuur ."]
    df = prepare_text_for_lda_af(af_list)
    lemma_list = df['lemma']
    print(lemma_list[0])

    # array of documents
    #all_documents = go_through_files_get_text("/home/nkosikhona/all_articels")
    #clean_documents =[]
    # clean each document
    #for document in all_documents:
    #   clean_documents.append(clean_text(document))
    #tokenized_documents= []
    #for document in clean_documents:
    #    tokens = prepare_text_for_lda_en(document)
    #    tokenized_documents.append(tokens)
    #dictionary = corpora.Dictionary(tokenized_documents)

    #corpus = [dictionary.doc2bow(text) for text in tokenized_documents]





