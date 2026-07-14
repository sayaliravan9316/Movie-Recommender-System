import pandas as pd
import streamlit as st
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("clean-data_nlp.csv")

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

similarities = cosine_similarity(vectors)

#file = open("similarity.pkl","rb")
#similarities = pickle.load(file)
#file.close()

st.markdown("""
<div class="glass">
<h1>🎬 Movie Recommender</h1>
<p>Select a movie and discover similar movies.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Welcome Celebration ----------------

if "welcome" not in st.session_state:
    st.balloons()
    st.session_state.welcome = True

st.success("🎉 Welcome to Movie Recommendation System!")

st.info("🍿 Select your favourite movie and click on 'Recommend' to discover similar movies.")


df = pd.read_csv("clean-data_nlp.csv")
movies = df['title'].tolist()
name = st.selectbox("🎥 Select a Movie", movies)
st.title("Movie Recommender System")


def get_name_by_index(i):
    if i<len(df) and i>0:
        return df.loc[i,'title']
    else:
        return""
    
def get_index_from_name(name):
    clear_user_name= name.strip().lower().replace(' ','').replace('-','')
    match = df[df['title'].str.lower().str.replace(' ', '').str.replace('-', '') == clear_user_name]

    if not match.empty:
        return match.index[0] 
    else:
        return -1

if st.button("Recommend"):
    index = get_index_from_name(name)
    if index == -1:
        st.write("Movie not found. Please check the spelling and try again.")
    else:
        st.write(f"Recommendations for '{name}' will be displayed here.")
        st.write(f"Movie index is { index }")
        similarity_indexes = list(enumerate(similarities[index]))
        similarity_indexes = sorted(similarity_indexes, key=lambda x: x[1], reverse=True)
        for i in range(1, 6):
            st.write(i, ":", get_name_by_index(similarity_indexes[i][0]))


st.markdown("---")

st.markdown(
    "<center>🎬 Developed by Sayali Ravan ❤️</center>",
    unsafe_allow_html=True
)