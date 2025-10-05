import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer,AutoConfig
import torch
import torch.nn.functional as F

@st.cache_resource
def nltk_libs():

    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

nltk_libs()

st.title("📱Sentiment Analysis - Tweets")
data= pd.read_csv("C:\\Users\\wincn\\streamlit projects\\test.csv",encoding="cp1252")


st.subheader("View Dataset")
with st.expander("Dataset preview"):
    st.dataframe(data)

# st.subheader("Dataset Shape before preprocessing:")
# st.code(data.shape)

#check for missing values 
st.markdown("### Checking for missing values:")
st.code(data.isnull().sum())

st.markdown("### After removing missing values")
data=data.dropna()
st.code(data.isnull().sum())

# st.markdown("### Checking for duplicates in 'text' column:")
# st.code(data['text'].duplicated().sum())


sentiment_counts=data['sentiment'].value_counts().reset_index()
sentiment_counts.columns=['Sentiment','Count']
st.markdown("### Count graph")
st.bar_chart(sentiment_counts,x='Sentiment',y='Count',color='Sentiment')


data['text']=data['text'].astype("string")


def clean(text):

    if pd.isna(text):
        return ""
    
    text=text.lower()

    text=re.sub(r'https?://\S+|www\.\S+','',text)
    text=re.sub(r'#\w+','',text)
    text=re.sub(r'@\w+','',text)
    text=re.sub(r'[^\w\s]|_','',text)
    text=re.sub(r'@\w+','',text)
    text=re.sub(r'\d+','',text)
    text=re.sub(r'\s+',' ',text)

    return text

def preprocess(text):

    text=clean(text)
    lemmatizer=WordNetLemmatizer()
    tokens=word_tokenize(text)
    stop_words= set(stopwords.words('english'))
    tokens=[token for token in tokens if token not in stop_words]
    lemmas=[lemmatizer.lemmatize(token) for token in tokens]
    processed=' '.join(lemmas)
    return processed



st.subheader("Cleaned and Processed Text (Before vs. After)")
data['processed_text']=data['text'].apply(preprocess)


col1,col2=st.columns(2)
with col1:
    with st.expander("Before"):
        for text in data['text']:
            st.write('-',text)
with col2:
    with st.expander("After"):
        for p_text in data['processed_text']:
            st.write('-',p_text)


le=LabelEncoder()  
data['sentiment_encoded']=le.fit_transform(data['sentiment'])
data['sentiment_encoded'].dropna()

X=data['processed_text']
y=data['sentiment_encoded']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

def train_model(model_name,X_train, y_train, X_test, y_test):
    vectorizer=TfidfVectorizer()
    X_train_tfidf=vectorizer.fit_transform(X_train)
    X_test_tfidf=vectorizer.transform(X_test)

    model=model_name.fit(X_train_tfidf,y_train)
    y_pred=model.predict(X_test_tfidf)

    accuracy= accuracy_score(y_test,y_pred)
    st.write(f"Accuracy score of {model_name}: **{accuracy:.4f}**")

    st.code(f"Classification Report:\n{classification_report(y_test,y_pred)}")

    cm=confusion_matrix(y_test,y_pred)
    #st.dataframe(pd.DataFrame(cm))

    with st.expander("Display Confusion Matrix and Comparison Chart"):
        col3,col4=st.columns(2)
        with col3:
            fig,ax=plt.subplots()
            sns.heatmap(cm,cmap='Blues',annot=True,fmt='d',
                xticklabels=['Negative','Neutral','Positive'],
                yticklabels=['Negative','Neutral','Positive'],
                ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {model_name}")
            st.pyplot(fig)
        with col4:
            actual=pd.Series(y_test).value_counts().sort_index()
            predicted=pd.Series(y_pred).value_counts().sort_index()
            pred_actual=pd.DataFrame({
                "Actual":actual,
                "Predicted":predicted
                }).reindex(range(len(le.classes_)),fill_value=0)
            pred_actual.index=le.classes_
            st.bar_chart(pred_actual,stack=False,y_label="Count")

    return accuracy

st.subheader("Models Results")
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs(["Logistic Regression","Random Forest","SVC","KNN","MultinomialNB","ComplementNB","Passive Aggressive","RoBERTa"])
with tab1:
    train_model(LogisticRegression(max_iter=1000),X_train,y_train,X_test,y_test)
with tab2:
    train_model(RandomForestClassifier(n_estimators=100),X_train,y_train,X_test,y_test)
with tab3:
    train_model(LinearSVC(max_iter=1000),X_train,y_train,X_test,y_test)
with tab4:
    train_model(KNeighborsClassifier(n_neighbors=5),X_train,y_train,X_test,y_test)
with tab5:
    train_model(MultinomialNB(alpha=0.5),X_train,y_train,X_test,y_test)
with tab6:
    train_model(ComplementNB(alpha=0.5),X_train,y_train,X_test,y_test)
with tab7:
    train_model(PassiveAggressiveClassifier(max_iter=1000),X_train,y_train,X_test,y_test)


    
    

# def train_model_cvs(model_name,X,y):
#     pipe=Pipeline([
#         ("tfidf",TfidfVectorizer()),
#         ("model",model_name)
#     ])
#     kf=KFold(n_splits=6,random_state=42,shuffle=True)
#     cv_results=cross_val_score(pipe,X,y,cv=kf)
#     mean_score=f"Mean Score for {model_name}:{np.mean(cv_results)}"
#     st.code(mean_score)
#     st.code(cv_results)
#     #return cv_results,mean_score

# train_model_cvs(LogisticRegression(max_iter=1000),X,y)
# train_model_cvs(RandomForestClassifier(n_estimators=100),X,y)
# train_model_cvs(LinearSVC(max_iter=1000),X,y)
# train_model_cvs(KNeighborsClassifier(n_neighbors=5),X,y)





MODEL="cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer=AutoTokenizer.from_pretrained("roberta-base")
hf_config=AutoConfig.from_pretrained(MODEL)
hf_model=AutoModelForSequenceClassification.from_pretrained(MODEL,output_attentions=True,output_hidden_states=True)

X_raw_test=data.loc[X_test.index,"text"].astype(str).tolist()
encode=tokenizer(
    X_raw_test, padding=True, truncation=True, max_length=128, return_tensors='pt'
)

with torch.no_grad():
    logits=hf_model(**encode).logits
    probability=F.softmax(logits, dim=1).cpu().numpy()

y_pred_ids=probability.argmax(axis=1)

name_to_id={}
for class_name in le.classes_:
    lower=class_name.lower()
    class_id=le.transform([class_name])[0]
    name_to_id[lower]=class_id

y_pred_tr=[]
for id in y_pred_ids:
    name=hf_config.id2label[int(id)]
    le_id=name_to_id[name.lower()]
    y_pred_tr.append(le_id)

with tab8:
    acc_tr=accuracy_score(y_test,y_pred_tr)
    st.write(f"Accuracy score of Transformer: {acc_tr:.4f}")
    st.code(f"Classification Report:\n{classification_report(y_test,y_pred_tr,target_names=le.classes_)}")

    with st.expander("Display Confusion Matrix and Comparison Chart"):
        col4,col5=st.columns(2)
        with col4:
            cm_tr=confusion_matrix(y_test,y_pred_tr)
            fig,ax=plt.subplots()
            sns.heatmap(cm_tr,cmap="Blues",annot=True,fmt='d',
                        xticklabels=le.classes_,
                        yticklabels=le.classes_,
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix - Transformer")
            st.pyplot(fig)
        with col5:
            actual_tr=pd.Series(y_test).value_counts().sort_index()
            pred_tr=pd.Series(y_pred_tr).value_counts().sort_index()
            pred_actual_tr=pd.DataFrame({"Actual":actual_tr,"Predicted":pred_tr}).reindex(range(len(le.classes_)),fill_value=0)
            pred_actual_tr.index=le.classes_
            st.bar_chart(pred_actual_tr,y_label="Count",stack=False)
    
    inputs=tokenizer(clean("This is the worst product ever, I hate it so much."),return_tensors="pt")
    inputs['input_ids']=inputs['input_ids'][:,1:-1]
    inputs['attention_mask']=inputs['attention_mask'][:,1:-1]
    embeddings=hf_model.roberta.embeddings.word_embeddings(inputs['input_ids'])
    embeddings.retain_grad()
    outputs=hf_model(inputs_embeds=embeddings)
    logits=outputs.logits
    sentiment_score=logits[0,2] # 2 for positive
    sentiment_score.backward()
    gradients=embeddings.grad
    avg_gradients=gradients[0].mean(dim=1).detach().numpy()

    fig,ax=plt.subplots()
    ax.plot(avg_gradients,marker='o')
    ax.set_title("Averaged Gradients for Tokens")
    ax.set_xlabel("Token")
    ax.set_ylabel("Average Gradient Value")
    tokens=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    tokens=[t.replace("Ġ","") for t in tokens]
    ax.set_xticks(range(len(avg_gradients)))
    ax.set_xticklabels(tokens,rotation=45)
    ax.grid(True)
    st.pyplot(fig)


    attention=outputs.attentions
    attention_matrix=attention[0][0][0].detach().numpy()
    fig,ax=plt.subplots()
    sns.heatmap(attention_matrix,xticklabels=tokens,yticklabels=tokens,cmap="viridis",ax=ax)
    ax.set_title("Attention Weights")
    st.pyplot(fig)

    fig,ax=plt.subplots(3,4,figsize=(15,10))
    for i, ax in enumerate(ax.flat):
        sns.heatmap(attention[0][0][i].detach().numpy(),xticklabels=tokens,yticklabels=tokens, ax=ax,cmap="viridis")
        ax.set_title(f"Head {i+1}")
    plt.tight_layout()
    st.pyplot(fig)  

    hidden_states=outputs.hidden_states
    target_token="worst"
    target_index=tokens.index(target_token)
    token_hidden_states=[
        state[:,target_index,:].detach().numpy() for state in hidden_states
    ]

    mean_states=[state.mean() for state in token_hidden_states]
    fig,ax=plt.subplots()
    ax.plot(mean_states)
    ax.set_title("Mean Hidden State of 'worst' Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Activation")
    st.pyplot(fig)



