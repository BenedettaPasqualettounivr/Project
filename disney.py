import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

disney_plus_titles_df = ("disney_plus_titles.csv")

st.title("DISNEY+")
st.image("Image/logo.png")


st.header("DATA ANALYSIS AND VISUALITAZION")

st.subheader("Analysis of the structure of the dataset")
st.text("We start analyzing our dataset in order to understand better the data.")
st.text("We see the coherence between the beginning and the end of the dataset and if there") 
st.text("are some null cells with codes like:")
st.code('''disney_plus_titles_df.head(5)''')
st.code('''disney_plus_titles_df.tail(5)''')
st.code('''disney_plus_titles_df.isnull().sum()''')
st.code('''disney_plus_titles_df.describe()''')

st.subheader("ANALYSIS OF THE MOST RELEVANT VARIABLES")
st.subheader("Release year")
st.text("We start analyzing some relevant variables in order to make some interesting") 
st.text("conclusions for the company.")
st.text("It's important to understand if Disney+ has a growth. Rearrenging the release")  
st.text("year, we can see that there is a positive trend since 2021 is the year with most")
st.text("products released.")
st.image("Image/grafico1.png")

st.subheader("Type")
st.text("Now we analyze the percentage of the Movies and TV Shows to understand which type") 
st.text("is more widespread.")
st.image("Image/grafico2.png")

st.markdown("CORRELATION BETWEEN TYPE AND RELEASE YEAR")
st.text("Through a Boolean mask,")
st.code('''mask_relaese_year = disney_2021['release_year'] == 2021''')
st.text("we can see that there are 70 Movies and 55 TV Shows, so we can conclude")
st.text("that Movies are the most important products. Despite the number of Movies") 
st.text("is still higher than that of TV Shows, in the last years TV Shows are gaining")
st.text("in importance.")

st.subheader("Data added")
st.text("For the company it's important to know which is the month where there were")
st.text("released more products, in order to understand which is the most suitable")
st.text("period to add a new product.")
st.text("As we can see by the graph, November is the more preferable month.")
st.image("Image/grafico3.png")

st.subheader("Genre")
st.text("At the same time, it's important to know which is the most appreciated genre,")
st.text("in order to build a good market offer.")
st.text("We can see that the most productive genres are Animation, Comedy and Family.")
st.image("Image/grafico4.png")
st.text("This result is so importan because this three genres are the most preferred both")
st.text("in Movies and TV Shows.")
st.text("Even if we apply a value.counts in the general column listed_in, in Movies or in")
st.text("TV Shows the result doesn't change.")
st.code('''disney_kind = disney_plus_titles_df['listed_in'].value_counts()''')
st.code('''disney_kind_movies = disney_plus_titles_df['listed_in'].value_counts('Movie')''')
st.code('''disney_kind_Tvshows = disney_plus_titles_df['listed_in'].value_counts('TV shows')''')

st.subheader("Rating")
st.text("So far we have analyzed the most profitable genre and period, now we will focus on")
st.text("the rating. In this way we are able to understand which could be the best target for")
st.text("Disney+.")
st.code('''disney_rating = disney_plus_titles_df['rating'].value_counts()''')
st.code('''disney_rating.dropna(inplace = True)''')
st.image("Image/grafico5.png")
st.text("The most popular rating is TV-G.")
st.image("Image/grafico6.png")
st.text("The result is relevant since even if we analyze separatly Movies and TV shows")
st.text("the result doesn't change.")

st.subheader("Country")
st.text("Now we analyze the variable 'Country' to see where are released more products.")
st.text("We can see that United States are the most productive Country, since there were")
st.text("released 1005 products.")
st.image("Image/grafico7.png")

st.subheader("Directors")
st.text("Focusing on the directors, it is intresting to see which are the most productive")
st.text("and to do so we use this code:")
st.code('''disney_directors = disney_plus_titles_df['director']
disney_directors.dropna(inplace = True)
disney_10_directors = disney_directors.value_counts()
disney_10_directors[0:10]
names = [name for name in disney_10_directors.index[0:10]]''')
st.text("The result is that the 10 most productive directors are: Jack Hannah, John Lasseter,")
st.text("Paul Hoen, Robert Stevenson, Charles Nichols, Vincent McEveety, Bob Peterson, James")
st.text("Algar, Kenny Ortega, Wilfred Jackson.")
st.image("Image/grafico8.png")

st.header("CLUSTERING")
st.text("We find the distribution of the rating, analyzing genre and release year together.")
st.text("The data of this variables were strings so we changed them into numbers using")
st.text("label_encoder, in order to create the clusters.")
st.code('''label_encoder = preprocessing.LabelEncoder()
disney_plus_titles_df['genre_new']= label_encoder.fit_transform(disney_plus_titles_df['genre_new'])
disney_plus_titles_df['genre_new'].unique()''')
st.code('''label_encoder = preprocessing.LabelEncoder()
disney_plus_titles_df['rating_new']= label_encoder.fit_transform(disney_plus_titles_df['rating'])
disney_plus_titles_df['rating_new'].unique()''')
st.text("We used the Elbow method to find that 5 is the optimal number of clusters, as we")
st.text("can see by the graph.")
st.image("Image/grafico9.png")
st.text("So, we build the model with 5 clusters:")
st.code('''kmode = KModes(n_clusters=5, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(data)''')
st.text("Having this information we are able to represent it in a graphical way.")
st.text("The initial situation is:")
st.image("Image/grafico10.png")
st.text("Through the distance calculation it's assigned the wright cluster.")
st.code('''square_distances = []
x = disney_plus_titles_df[['genre_new','release_year']]
for i in range(1, 20):    
    km = KMeans(n_clusters=5, random_state=42)    
    km.fit(x)    
    square_distances.append(km.inertia_)''')
st.text("What we obtain is this graphical visualization")
st.image("Image/grafico11.png")
st.text("The centers of the clusters are defined by this code:")
st.code('''km.cluster_centers_''')
