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
import pyLDAvis.gensim
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
    print(prediction[0][0][0],prediction[1][0][0])
    return prediction[0][0][0],prediction[1][0][0]

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text,pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len (token)> 3:
            result.append(lemmatize_stemming(token.strip()))
    return result

'''Fuction creates two lists and put all the documents of each language in separate list'''
def get_en_af_lists(docu_list,af_docu_list =[],en_doc_list= []):
    print(" Start: language identification")
    count_en = 0
    count_af = 0
    for document in docu_list:
        lang,_ =language_id(document)
        if( lang =='__label__en'):
            en_doc_list.append(document)
            count_en= count_en+1
        else:
            af_docu_list.append(document)
            count_af=count_af+1
    # returns  two lists containing documents, separeted into englis and afrikaans documents
    print("Done langauge identification:","Total documents =",str((count_af+count_en)))
    print("Found",str(count_en),"English documents","And",str(count_af),"Afrikaans documents")
    return en_doc_list,af_docu_list



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
    print ("...Start preparing en text for lda removing stop words and lematizing")
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token.strip())>4]
    tokens = [token for token in tokens if token not in gensim.parsing.preprocessing.STOPWORDS]
    tokens = [get_lemma(token) for token in tokens]
    print("...Done preparing en text for lda removing stop words and lematizing")
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
    #af_list = ["Ons het besliste teen-resessiebesteding deur die regering geïmplementeer , veral op infrastruktuur ."]
    #df = prepare_text_for_lda_af(af_list)
    #lemma_list = df['lemma']
    #print(lemma_list[0])

    # array of documents
    all_documents = go_through_files_get_text("/home/nkosikhona/all_articels")
    clean_documents =[]
    # clean each document
    for document in all_documents:
       clean_documents.append(clean_text(document))


    clean_en_docu,clean_af_doc =get_en_af_lists(clean_documents)
    tokenized_documents_en = []
    tokenized_documents_af = []
    for document in clean_en_docu:
        tokens = prepare_text_for_lda_en(document)
        tokenized_documents_en.append(tokens)
    dictionary_en = corpora.Dictionary(tokenized_documents_en)
    corpus_en = [dictionary_en.doc2bow(text) for text in tokenized_documents_en]

    # build lda model
    lda_model = gensim.models.ldamodel.LdaModel(corpus_en,num_topics=10,id2word=dictionary_en,passes=15)

    # display topics


    lda_display = pyLDAvis.gensim.prepare(lda_model,corpus_en,dictionary_en)


    #pyLDAvis.enable_notebook()
    pyLDAvis.save_html(lda_display,'LDA_visualization.html')
    #pyLDAvis.show(lda_display)


