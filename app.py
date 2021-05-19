import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import re
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go
import plotly.express as px



@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/adepril/datasets/main/insurance-reviews-france-Comments.csv")
    df = df.drop(['Unnamed: 0'],axis=1)
    df = df.dropna()
    df["Comment"]= df["Comment"].str.lower()

    # Word Tokenization and deleting punctuation
    comments=[]
    for comment in df["Comment"].apply(str):
        WordTokenizer = []
        for word in  re.sub("\W"," ",comment ).split():
            WordTokenizer.append(word)
        comments.append(WordTokenizer)

    #Ajoute une nouvelle colonne
    df["Word_Tokenizer"]= comments

    # Set new Spacy's Stop Word list by deleting negation word 
    stop_words=set(STOP_WORDS)

    deselect_stop_words = ['n\'','ne','pas','plus','personne','aucun','ni','aucune','rien']
    for w in deselect_stop_words:
        if w in stop_words:
            stop_words.remove(w)
        else:
            continue

    # Add a new column for comments without StopWords
    AllfilteredComment=[]
    for comment in df["Word_Tokenizer"]:
        filteredComment = [w for w in comment if not ((w in stop_words) or (len(w) == 1))]
        AllfilteredComment.append(' '.join(filteredComment))
        
    df["CommentAferPreproc"]=AllfilteredComment

    # Sentiment Analysis with TextBlob
    sentiments = []
    for i in df["CommentAferPreproc"]:
        sentiment = tb(i).sentiment[0]
        if (sentiment > 0):
            sentiments.append('Positif')
        elif (sentiment < 0):
            sentiments.append('Negatif')
        else:
            sentiments.append('Neutre')   

    # Ajoute une colonne : Sentiment
    df["sentiment"]=sentiments

    return df

df = load_data()


def main():
    page = st.sidebar.selectbox(
        "Choisir un graphique",
        [
            "Dataset",
            "Total des commentaires",
            "Commentaires par assurance",
            "Commentaires par année",
            "Sentiments par assurance"
        ],
    )

    if page == "Dataset":
        #st.header("Analyse des commentaires des clients d'assurances")
        """
        # Analyse des commentaires des clients d'assurances
        Sélectionner une représentation graphique dans le menu de gauche
        """
        st.write(df)
    elif page == "Total des commentaires":
        st.header("Satisfaction des clients")
        bar_chart() 
        pie_chart()

    elif page == "Commentaires par assurance":
        st.header("Nombre de commentaires par assurance")
        histo_1()

    elif page == "Commentaires par année":
        st.header("Nombre de commentaires par année")
        histo_2()

    elif page == "Sentiments par assurance":
        dff = df.sort_values('Year',ascending=False)
        annees = dff["Year"].unique()
        new_list_annees = np.append("Choisir une année",annees)
        
        choix_annee = st.sidebar.selectbox("", new_list_annees)
        st.header("Sentiments par assurance")
        #st.write("Choix: ",choix_annee)
        if(choix_annee=="Choisir une année"):
            histo_3()
        else:
            histo_3bis(choix_annee)


def bar_chart():
    _x =  ["Positif","Negatif","Neutre"]
    _y = df["sentiment"].value_counts()
    data = [go.Bar(
                x = _x,
                y = _y
                )] 
    layout = go.Layout({
      "title":"Graphique en bâton"
    })
    fig = go.Figure(data=data, layout = layout)
    st.plotly_chart(fig)


def pie_chart():
    labels = ["Positif","Negatif","Neutre"]
    values = df["sentiment"].value_counts()
    colors = ["bleu", "red", "green"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hoverinfo="label+percent",
                textinfo="value",
            )
        ],
        layout={"title":"Graphique en camenbert" }   
    )
    fig.update_traces(marker=dict(colors=colors))
    st.plotly_chart(fig)

def histo_1():
    fig = px.histogram(
        df, 
        x="Name",
        color="Name",
        labels={"Name":"Assurances"}
        )
    fig.update_layout(
        title_text='Nombre de commentaires par assurance', 
        xaxis_title_text='Assurances', 
        yaxis_title_text='Nombre de commentaires', 
        width=900,
        bargap=0.2, 
        bargroupgap=0.1
    )
    st.plotly_chart(fig)

def histo_2():
    fig = px.histogram(
        df, 
        x="Year",
        color="Year",
        labels={"Year":"Années"}
        )
    fig.update_layout(
        title_text='Nombre de commentaires par année', 
        xaxis_title_text='Assurances', 
        yaxis_title_text='Nombre de commentaires', 
        width=900,
        bargap=0.2, 
        bargroupgap=0.1
    )
    st.plotly_chart(fig)

def histo_3():
    fig = px.histogram(
        df, 
        x="Name",
        color="sentiment",
        )
    fig.update_layout(
        title_text='Sentiments par assurance', 
        xaxis_title_text='Assurances', 
        yaxis_title_text='Nombre de commentaires', 
        width=900,
        bargap=0.2, 
        bargroupgap=0.1
    )
    st.plotly_chart(fig)

def histo_3bis(annee):
    df_annee = df.query("Year == {0}".format(annee))

    fig = px.histogram(
        df_annee, 
        x="Name",
        color="sentiment",
        )
    fig.update_layout(
        title_text='Sentiments par assurance pour '+annee, 
        xaxis_title_text='Assurances', 
        yaxis_title_text='Nombre de commentaires', 
        width=900,
        bargap=0.2, 
        bargroupgap=0.1
    )
    st.plotly_chart(fig)
 

if __name__ == "__main__":
    main()

