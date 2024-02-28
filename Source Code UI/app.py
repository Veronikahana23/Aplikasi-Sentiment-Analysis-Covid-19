from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL, MySQLdb
import pymysql

from os import remove
import csv
import pandas as pd
import re
import demoji
import emoji
from emoji import demojize
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# splitting data latih 80 : uji 20
import collections, numpy

# perform algoritma KNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import sys
import json
import base64

app = Flask(__name__)
app.debug = True
app.secret_key = "skripsiku"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'skripsi'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)
db = pymysql.connect(host="localhost", port=3306, user="root", password="", db="skripsi")

@app.route("/", methods=['GET', 'POST'])
def index():
    # Check if user is loggedin
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route("/page_login")
def pageLogin():
    return render_template('login.html')

@app.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
 # connect
    curl = mysql.connection.cursor()
    curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
  
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        curl.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = curl.fetchone()
   
    # If account exists in admin table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            #return 'Logged in successfully!'
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'username/password SALAH!'
    
    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   flash('You were logged out')
   # Redirect to login page
   return redirect(url_for('login'))

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    curl = mysql.connection.cursor()
    curl.execute("SELECT * FROM dataset")
    data = curl.fetchall()
    curl.close()
    return render_template("dataset.html", dataset=data)

@app.route("/preprocessing", methods=['GET', 'POST'])
def preprocessing():   
    curl = mysql.connection.cursor()
    curl.execute("SELECT * FROM preprocessing")
    preprocess = curl.fetchall()
    curl.close()
    return render_template('preprocessing.html', preprocessing=preprocess)

@app.route("/hasil_klasifikasi", methods=['GET', 'POST'])
def hasil_klasifikasi():   
    curl = mysql.connection.cursor()
    curl.execute("SELECT * FROM preprocessing")
    hasil = curl.fetchall()
    curl.close()
    return render_template('hasil.html', hasil_klasifikasi=hasil)

@app.route('/testing')
def show_test():
    cursor = db.cursor()
    sql = "SELECT * FROM data_test"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('testing.html', rows=results)

@app.route('/add_test')
def add_rec_test():
    return render_template('tambah_test.html')

@app.route('/edit_test/<int:id>')
def edit_rec_test(id):
     cursor = db.cursor()
     sql = "select *  from data_test where id = %d" % id
     cursor.execute(sql)
     results = cursor.fetchone()
     return render_template('edit_test.html', rows=results)


@app.route('/delete_test/<int:id>')
def delete_rec_test(id):
        try:
            cursor = db.cursor()
            sql="delete from data_test where id = %d" %id
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
        finally:
            return redirect(url_for('show_test'))
            #return render_template('success.html', sql)
            db.close()


@app.route('/submit_add_test',methods=['POST','GET'])
def submit_save_add_test():
    if request.method == "POST":
        try:
            vtext = request.form['text']
            vsentimen = request.form['sentimen']
            vlabel = request.form['label']

            cursor = db.cursor()
            sql = " insert into data_test (text, sentimen, label )"
            sql = sql + " values ('" + vtext + "', '" + vsentimen + "', '" + vlabel + "')"

            cursor.execute(sql)
            db.commit()
            msg = "Komentar BERHASIL Di Tambahkan"
        except:
            db.rollback()
            msg = "Komentar GAGAL Di Tambahkan"

        finally:
            return redirect(url_for('show_test'))
            # return print(sql)
            db.close()

@app.route('/submit_edit_test',methods=['POST','GET'])
def submit_save_edit_test():
    if request.method == "POST":
        try:
            vid = request.form['id']
            vtext = request.form['text']
            vsentimen = request.form['sentimen']
            vlabel = request.form['label']

            cursor = db.cursor()

            sql = " update data_test set text='" + vtext + "',"
            sql = sql + " sentimen ='" + vsentimen + "',"
            sql = sql + " label ='" + vlabel + "'"
            sql = sql + " where id = " + vid
            cursor.execute(sql)
            db.commit()
            msg = "Record successfully update"
        except:
            db.rollback()
            msg = "error in update operation"

        finally:
            return redirect(url_for('show_test'))
            # return print(sql)
            db.close()

@app.route('/training')
def show_train():
    cursor = db.cursor()
    sql = "SELECT * FROM data_train"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('training.html', rows=results)

@app.route('/add_train')
def add_rec_train():
    return render_template('tambah_train.html')


@app.route('/edit_train/<int:id>')
def edit_rec_train(id):
     cursor = db.cursor()
     sql = "select *  from data_train where id = %d" % id
     cursor.execute(sql)
     results = cursor.fetchone()
     return render_template('edit_train.html', rows=results)


@app.route('/delete_train/<int:id>')
def delete_rec_train(id):
        try:
            cursor = db.cursor()
            sql="delete from data_train where id = %d" %id
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
        finally:
            return redirect(url_for('show_train'))
            #return render_template('success.html', sql)
            db.close()


@app.route('/submit_add_train',methods=['POST','GET'])
def submit_save_add_train():
    if request.method == "POST":
        try:
            vtext = request.form['text']
            vsentimen = request.form['sentimen']
            vlabel = request.form['label']

            cursor = db.cursor()
            sql = " insert into data_train (text, sentimen, label )"
            sql = sql + " values ('" + vtext + "', '" + vsentimen + "', '" + vlabel + "')"

            cursor.execute(sql)
            db.commit()
            msg = "Komentar BERHASIL Di Tambahkan"
        except:
            db.rollback()
            msg = "Komentar GAGAL Di Tambahkan"

        finally:
            return redirect(url_for('show_train'))
            # return print(sql)
            db.close()

@app.route('/submit_edit_train',methods=['POST','GET'])
def submit_save_edit_train():
    if request.method == "POST":
        try:
            vid = request.form['id']
            vtext = request.form['text']
            vsentimen = request.form['sentimen']
            vlabel = request.form['label']

            cursor = db.cursor()

            sql = " update data_train set text='" + vtext + "',"
            sql = sql + " sentimen ='" + vsentimen + "',"
            sql = sql + " label ='" + vlabel + "'"
            sql = sql + " where id = " + vid
            cursor.execute(sql)
            db.commit()
            msg = "Record successfully update"
        except:
            db.rollback()
            msg = "error in update operation"

        finally:
            return redirect(url_for('show_train'))
            # return print(sql)
            db.close()

@app.route('/score_dt')
def score_dt():
    cursor = db.cursor()
    sql = "SELECT * FROM score_dt"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('score_dt.html', rows=results)

@app.route('/delete_score_dt/<int:id>')
def delete_rec_score_dt(id):
        try:
            cursor = db.cursor()
            sql="delete from score_dt where id = %d" %id
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
        finally:
            return redirect(url_for('score_dt'))
            #return render_template('success.html', sql)
            db.close()

@app.route('/score_knn')
def score_knn():
    cursor = db.cursor()
    sql = "SELECT * FROM score_knn"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('score_knn.html', rows=results)

@app.route('/delete_score_knn/<int:id>')
def delete_rec_score_knn(id):
        try:
            cursor = db.cursor()
            sql="delete from score_knn where id = %d" %id
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
        finally:
            return redirect(url_for('score_knn'))
            #return render_template('success.html', sql)
            db.close()

@app.route('/hasil_dt')
def hasil_dt():
    cursor = db.cursor()
    sql = "SELECT * FROM hasil_dt"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('hasil_dt.html', rows=results)

@app.route('/delete_hasil_dt/<int:id>')
def delete_rec_hasil_dt(id):
        try:
            cursor = db.cursor()
            sql="delete from hasil_dt where id = %d" %id
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
        finally:
            return redirect(url_for('hasil_dt'))
            #return render_template('success.html', sql)
            db.close()

@app.route('/hasil_knn')
def hasil_knn():
    cursor = db.cursor()
    sql = "SELECT * FROM hasil_knn"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('hasil_knn.html', rows=results)

@app.route('/delete_hasil_knn/<int:id>')
def delete_rec_hasil_knn(id):
        try:
            cursor = db.cursor()
            sql="delete from hasil_knn where id = %d" %id
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
        finally:
            return redirect(url_for('hasil_knn'))
            #return render_template('success.html', sql)
            db.close()

@app.route("/ujicoba")
def ujicoba():
    return render_template('ujicoba.html')   

@app.route('/hasiluji', methods=["GET"])
def hasiluji(): 

    subject = request.args.get("sub")
    subject = [subject]

    result = {}

    def case_folding(tokens): 
        return tokens.lower()

    test_casefolding=[]
    for i in range(0, len(subject)):
        test_casefolding.append(case_folding(subject[i]))
        
    result['casefolding'] = ' '.join(list(map(lambda x: str(x), test_casefolding)))
    casefolding = result['casefolding']

    def remove_num(text):  
        text = re.sub(r'@[\w|a-z0-9]*', ' ', text) #menghapus @username      
        text = re.sub(r'#[\w|a-z0-9]*', ' ', text) #menghapus #hashtags    
        text = re.sub(r'https://*[\r\n]*', ' ', text) #menghapus link http 
        text = re.sub(r'www.[^ ]+', ' ', text) #menhilangkan situs website 
        text = re.sub(r"\d+", "", text) # menghapus bilangan
        text = re.sub(r'\r', ' ', text) #menghapus \r    
        text = re.sub(r'\n', ' ', text) #menghapus \n (newline)   
        text = demoji.replace(text, ' ')
        
        return text
    
    test_removenum=[]
    for i in range(0,len(test_casefolding)):
        test_removenum.append(remove_num(test_casefolding[i]))

    result ['remove_num'] = ' '.join(list(map(lambda x: str(x), test_removenum)))
    removenum = result ['remove_num']

    def remove_punct(text):  
        text_nopunct = ''
        text_nopunct = re.sub('['+string.punctuation+']', ' ',text)
    
        return text_nopunct
    
    test_removepunct=[]
    for i in range(0,len(test_removenum)):
        test_removepunct.append(remove_punct(test_removenum[i]))

    result ['removepunct'] = ' '.join(list(map(lambda x: str(x), test_removepunct)))
    removepunct = result ['removepunct']  
   
    def open_kamus_prepro(x):
        kamus={}
        with open(x,'r') as file :
            for line in file :
                slang=line.replace("'","").split(':')
                kamus[slang[0].strip()]=slang[1].rstrip('\n').lstrip()
        return kamus

    kamus_slang = open_kamus_prepro('slangwords.txt')

    def slangword(text):
        sentence_list = text.split()
        new_sentence = []
        
        for word in sentence_list:
            for candidate_replacement in kamus_slang:
                if candidate_replacement == word:
                    word = word.replace(candidate_replacement, kamus_slang[candidate_replacement])
            new_sentence.append(word)
        return " ".join(new_sentence)

    test_slangword=[]
    for i in range(0,len(test_removepunct)):
        test_slangword.append(slangword(test_removepunct[i]))

    slangword_ = test_slangword

    result['hasil_token'] = [word_tokenize(sen) for sen in test_slangword]
    hasil_token = result['hasil_token']

    kamus_stopword=[]
    with open('stopwords.txt','r') as file :
        for line in file :
            slang=line.replace("'","").strip()
            kamus_stopword.append(slang)

    def remove_stop_words(tokens):
        return [word for word in tokens if word not in kamus_stopword]

    stopword= [remove_stop_words(sen) for sen in hasil_token] 

    result['remove_stop_words'] = ' '.join(list(map(lambda x: str(x), stopword)))
    remove_stop_words = result['remove_stop_words']

    text_final = [' '.join(sen) for sen in stopword]

    result['text_final'] = [' '.join(sen) for sen in stopword]
    text_final_ = result['text_final']

    # KNN (K=1)
    vectorizer_k1 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k1 = pickle.load(open("knn1.pickle", 'rb'))

    vect_k1 = vectorizer_k1.transform(text_final)[0]
    prediksiknn_k1 = loaded_model_k1.predict(vect_k1)[0]

    result['probabilitasknn_k1']= loaded_model_k1.predict_proba(vect_k1)[0]
    probabilitasknn_k1_ = result['probabilitasknn_k1']

    result['probabilitasknn_new_k1'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k1_[0], probabilitasknn_k1_[-1], probabilitasknn_k1_[1])
    probabilitas_knn_k1 = result['probabilitasknn_new_k1']

    # KNN (K=2)
    vectorizer_k2 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k2 = pickle.load(open("knn2.pickle", 'rb'))

    vect_k2 = vectorizer_k2.transform(text_final)[0]
    prediksiknn_k2 = loaded_model_k2.predict(vect_k2)[0]

    result['probabilitasknn_k2']= loaded_model_k2.predict_proba(vect_k2)[0]
    probabilitasknn_k2_ = result['probabilitasknn_k2']

    result['probabilitasknn_new_k2'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k2_[0], probabilitasknn_k2_[-1], probabilitasknn_k2_[1])
    probabilitas_knn_k2 = result['probabilitasknn_new_k2']

    # KNN (K=3)
    vectorizer_k3 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k3 = pickle.load(open("knn3.pickle", 'rb'))

    vect_k3 = vectorizer_k3.transform(text_final)[0]
    prediksiknn_k3 = loaded_model_k3.predict(vect_k3)[0]

    result['probabilitasknn_k3']= loaded_model_k3.predict_proba(vect_k3)[0]
    probabilitasknn_k3_ = result['probabilitasknn_k3']

    result['probabilitasknn_new_k3'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k3_[0], probabilitasknn_k3_[-1], probabilitasknn_k3_[1])
    probabilitas_knn_k3 = result['probabilitasknn_new_k3']

    # KNN (K=4)
    vectorizer_k4 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k4 = pickle.load(open("knn4.pickle", 'rb'))

    vect_k4 = vectorizer_k4.transform(text_final)[0]
    prediksiknn_k4 = loaded_model_k4.predict(vect_k4)[0]

    result['probabilitasknn_k4']= loaded_model_k4.predict_proba(vect_k4)[0]
    probabilitasknn_k4_ = result['probabilitasknn_k4']

    result['probabilitasknn_new_k4'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k4_[0], probabilitasknn_k4_[-1], probabilitasknn_k4_[1])
    probabilitas_knn_k4 = result['probabilitasknn_new_k4']

    # KNN (K=5)
    vectorizer_k5 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k5 = pickle.load(open("knn5.pickle", 'rb'))

    vect_k5 = vectorizer_k5.transform(text_final)[0]
    prediksiknn_k5 = loaded_model_k5.predict(vect_k5)[0]

    result['probabilitasknn_k5']= loaded_model_k5.predict_proba(vect_k5)[0]
    probabilitasknn_k5_ = result['probabilitasknn_k5']

    result['probabilitasknn_new_k5'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k5_[0], probabilitasknn_k5_[-1], probabilitasknn_k5_[1])
    probabilitas_knn_k5 = result['probabilitasknn_new_k5']

    # KNN (K=6)
    vectorizer_k6 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k6 = pickle.load(open("knn6.pickle", 'rb'))

    vect_k6 = vectorizer_k6.transform(text_final)[0]
    prediksiknn_k6 = loaded_model_k6.predict(vect_k6)[0]

    result['probabilitasknn_k6']= loaded_model_k6.predict_proba(vect_k6)[0]
    probabilitasknn_k6_ = result['probabilitasknn_k6']

    result['probabilitasknn_new_k6'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k6_[0], probabilitasknn_k6_[-1], probabilitasknn_k6_[1])
    probabilitas_knn_k6 = result['probabilitasknn_new_k6']

     # KNN (K=7)
    vectorizer_k7 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k7 = pickle.load(open("knn7.pickle", 'rb'))

    vect_k7 = vectorizer_k7.transform(text_final)[0]
    prediksiknn_k7 = loaded_model_k7.predict(vect_k7)[0]

    result['probabilitasknn_k7']= loaded_model_k7.predict_proba(vect_k7)[0]
    probabilitasknn_k7_ = result['probabilitasknn_k7']

    result['probabilitasknn_new_k7'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k7_[0], probabilitasknn_k7_[-1], probabilitasknn_k7_[1])
    probabilitas_knn_k7 = result['probabilitasknn_new_k7']

    # KNN (K=8)
    vectorizer_k8 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k8 = pickle.load(open("knn8.pickle", 'rb'))

    vect_k8 = vectorizer_k8.transform(text_final)[0]
    prediksiknn_k8 = loaded_model_k8.predict(vect_k8)[0]

    result['probabilitasknn_k8']= loaded_model_k8.predict_proba(vect_k8)[0]
    probabilitasknn_k8_ = result['probabilitasknn_k8']

    result['probabilitasknn_new_k8'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k8_[0], probabilitasknn_k8_[-1], probabilitasknn_k8_[1])
    probabilitas_knn_k8 = result['probabilitasknn_new_k8']
    
    # KNN (K=9)
    vectorizer = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model = pickle.load(open("knn9.pickle", 'rb'))

    vect = vectorizer.transform(text_final)[0]
    prediksiknn = loaded_model.predict(vect)[0]

    result['probabilitasknn']= loaded_model.predict_proba(vect)[0]
    probabilitasknn_ = result['probabilitasknn']

    result['probabilitasknn_new'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_[0], probabilitasknn_[-1], probabilitasknn_[1])
    probabilitas_knn = result['probabilitasknn_new']
    
    # KNN (K=10)
    vectorizer_k10 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k10 = pickle.load(open("knn10.pickle", 'rb'))

    vect_k10 = vectorizer_k10.transform(text_final)[0]
    prediksiknn_k10 = loaded_model_k10.predict(vect_k10)[0]

    result['probabilitasknn_k10']= loaded_model_k10.predict_proba(vect_k10)[0]
    probabilitasknn_k10_ = result['probabilitasknn_k10']

    result['probabilitasknn_new_k10'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k10_[0], probabilitasknn_k10_[-1], probabilitasknn_k10_[1])
    probabilitas_knn_k10 = result['probabilitasknn_new_k10']
    
    #Decision Tree
    vectorizer_dt = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_dt = pickle.load(open("DT.pickle", 'rb'))

    vect_dt = vectorizer_dt.transform(text_final)[0]
    prediksidt = loaded_model_dt.predict(vect_dt)[0]

    result['probabilitasdt']= loaded_model_dt.predict_proba(vect_dt)[0]
    probabilitasdt_ = result['probabilitasdt']

    result['probabilitasdt_new'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasdt_[0], probabilitasdt_[-1], probabilitasdt_[1])
    probabilitas_dt = result['probabilitasdt_new']


    return render_template("hasiluji.html", 
                                subject = subject,
                                casefolding = casefolding, 
                                removepunct = removepunct,
                                removenum = removenum,
                                slangword_ = slangword_,
                                hasil_token = hasil_token,
                                remove_stop_words = remove_stop_words,
                                text_final_ = text_final_,
                                probabilitas_knn = probabilitas_knn,
                                probabilitas_knn_k1 = probabilitas_knn_k1,
                                probabilitas_knn_k2 = probabilitas_knn_k2,
                                probabilitas_knn_k3 = probabilitas_knn_k3,
                                probabilitas_knn_k4 = probabilitas_knn_k4,
                                probabilitas_knn_k5 = probabilitas_knn_k5,
                                probabilitas_knn_k6 = probabilitas_knn_k6,
                                probabilitas_knn_k7 = probabilitas_knn_k7,
                                probabilitas_knn_k8 = probabilitas_knn_k8,
                                probabilitas_knn_k10 = probabilitas_knn_k10,
                                probabilitas_dt = probabilitas_dt
                                ) 
@app.route("/web")
def web():
    return render_template('sistem.html')

@app.route('/sistem_dataset', methods=['GET', 'POST'])
def sistem_dataset():
    curl = mysql.connection.cursor()
    curl.execute("SELECT * FROM dataset")
    data = curl.fetchall()
    curl.close()
    return render_template("sistem_dataset.html", sistem_dataset=data)

@app.route("/sistem_preprocessing", methods=['GET', 'POST'])
def sistem_preprocessing():   
    curl = mysql.connection.cursor()
    curl.execute("SELECT * FROM preprocessing")
    preprocess = curl.fetchall()
    curl.close()
    return render_template('sistem_preprocessing.html', sistem_preprocessing=preprocess)

@app.route('/sistem_test', methods=['GET', 'POST'])
def sistem_test():
    curl = mysql.connection.cursor()
    curl.execute("SELECT * FROM data_test")
    data = curl.fetchall()
    curl.close()
    return render_template("sistem_test.html", sistem_test=data)

@app.route('/sistem_train', methods=['GET', 'POST'])
def sistem_train():
    curl = mysql.connection.cursor()
    curl.execute("SELECT * FROM data_train")
    data = curl.fetchall()
    curl.close()
    return render_template("sistem_train.html", sistem_train=data)

@app.route("/sistem_ujicoba")
def sistem_ujicoba():
    return render_template('sistem_ujicoba.html') 

@app.route('/sistem_hasiluji', methods=["GET"])
def sistem_hasiluji(): 

    subject = request.args.get("sub")
    subject = [subject]

    result = {}

    def case_folding(tokens): 
        return tokens.lower()

    test_casefolding=[]
    for i in range(0, len(subject)):
        test_casefolding.append(case_folding(subject[i]))
        
    result['casefolding'] = ' '.join(list(map(lambda x: str(x), test_casefolding)))
    casefolding = result['casefolding']

    def remove_num(text):  
        text = re.sub(r'@[\w|a-z0-9]*', ' ', text) #menghapus @username      
        text = re.sub(r'#[\w|a-z0-9]*', ' ', text) #menghapus #hashtags    
        text = re.sub(r'https://*[\r\n]*', ' ', text) #menghapus link http 
        text = re.sub(r'www.[^ ]+', ' ', text) #menhilangkan situs website 
        text = re.sub(r"\d+", "", text) # menghapus bilangan
        text = re.sub(r'\r', ' ', text) #menghapus \r    
        text = re.sub(r'\n', ' ', text) #menghapus \n (newline)   
        text = demoji.replace(text, ' ')
        
        return text
    
    test_removenum=[]
    for i in range(0,len(test_casefolding)):
        test_removenum.append(remove_num(test_casefolding[i]))

    result ['remove_num'] = ' '.join(list(map(lambda x: str(x), test_removenum)))
    removenum = result ['remove_num']

    def remove_punct(text):  
        text_nopunct = ''
        text_nopunct = re.sub('['+string.punctuation+']', ' ',text)
        return text_nopunct
    
    test_removepunct=[]
    for i in range(0,len(test_removenum)):
        test_removepunct.append(remove_punct(test_removenum[i]))

    result ['removepunct'] = ' '.join(list(map(lambda x: str(x), test_removepunct)))
    removepunct = result ['removepunct']  
   
    def open_kamus_prepro(x):
        kamus={}
        with open(x,'r') as file :
            for line in file :
                slang=line.replace("'","").split(':')
                kamus[slang[0].strip()]=slang[1].rstrip('\n').lstrip()
        return kamus

    kamus_slang = open_kamus_prepro('slangwords.txt')

    def slangword(text):
        sentence_list = text.split()
        new_sentence = []
        
        for word in sentence_list:
            for candidate_replacement in kamus_slang:
                if candidate_replacement == word:
                    word = word.replace(candidate_replacement, kamus_slang[candidate_replacement])
            new_sentence.append(word)
        return " ".join(new_sentence)

    test_slangword=[]
    for i in range(0,len(test_removepunct)):
        test_slangword.append(slangword(test_removepunct[i]))

    slangword_ = test_slangword

    result['hasil_token'] = [word_tokenize(sen) for sen in test_slangword]
    hasil_token = result['hasil_token']

    kamus_stopword=[]
    with open('stopwords.txt','r') as file :
        for line in file :
            slang=line.replace("'","").strip()
            kamus_stopword.append(slang)

    def remove_stop_words(tokens):
        return [word for word in tokens if word not in kamus_stopword]

    stopword= [remove_stop_words(sen) for sen in hasil_token] 

    result['remove_stop_words'] = ' '.join(list(map(lambda x: str(x), stopword)))
    remove_stop_words = result['remove_stop_words']

    text_final = [' '.join(sen) for sen in stopword]

    result['text_final'] = [' '.join(sen) for sen in stopword]
    text_final_ = result['text_final']

    # KNN (K=1)
    vectorizer_k1 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k1 = pickle.load(open("knn1.pickle", 'rb'))

    vect_k1 = vectorizer_k1.transform(text_final)[0]
    prediksiknn_k1 = loaded_model_k1.predict(vect_k1)[0]

    result['probabilitasknn_k1']= loaded_model_k1.predict_proba(vect_k1)[0]
    probabilitasknn_k1_ = result['probabilitasknn_k1']

    result['probabilitasknn_new_k1'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k1_[0], probabilitasknn_k1_[-1], probabilitasknn_k1_[1])
    probabilitas_knn_k1 = result['probabilitasknn_new_k1']

    # KNN (K=2)
    vectorizer_k2 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k2 = pickle.load(open("knn2.pickle", 'rb'))

    vect_k2 = vectorizer_k2.transform(text_final)[0]
    prediksiknn_k2 = loaded_model_k2.predict(vect_k2)[0]

    result['probabilitasknn_k2']= loaded_model_k2.predict_proba(vect_k2)[0]
    probabilitasknn_k2_ = result['probabilitasknn_k2']

    result['probabilitasknn_new_k2'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k2_[0], probabilitasknn_k2_[-1], probabilitasknn_k2_[1])
    probabilitas_knn_k2 = result['probabilitasknn_new_k2']

    # KNN (K=3)
    vectorizer_k3 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k3 = pickle.load(open("knn3.pickle", 'rb'))

    vect_k3 = vectorizer_k3.transform(text_final)[0]
    prediksiknn_k3 = loaded_model_k3.predict(vect_k3)[0]

    result['probabilitasknn_k3']= loaded_model_k3.predict_proba(vect_k3)[0]
    probabilitasknn_k3_ = result['probabilitasknn_k3']

    result['probabilitasknn_new_k3'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k3_[0], probabilitasknn_k3_[-1], probabilitasknn_k3_[1])
    probabilitas_knn_k3 = result['probabilitasknn_new_k3']

    # KNN (K=4)
    vectorizer_k4 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k4 = pickle.load(open("knn4.pickle", 'rb'))

    vect_k4 = vectorizer_k4.transform(text_final)[0]
    prediksiknn_k4 = loaded_model_k4.predict(vect_k4)[0]

    result['probabilitasknn_k4']= loaded_model_k4.predict_proba(vect_k4)[0]
    probabilitasknn_k4_ = result['probabilitasknn_k4']

    result['probabilitasknn_new_k4'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k4_[0], probabilitasknn_k4_[-1], probabilitasknn_k4_[1])
    probabilitas_knn_k4 = result['probabilitasknn_new_k4']

    # KNN (K=5)
    vectorizer_k5 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k5 = pickle.load(open("knn5.pickle", 'rb'))

    vect_k5 = vectorizer_k5.transform(text_final)[0]
    prediksiknn_k5 = loaded_model_k5.predict(vect_k5)[0]

    result['probabilitasknn_k5']= loaded_model_k5.predict_proba(vect_k5)[0]
    probabilitasknn_k5_ = result['probabilitasknn_k5']

    result['probabilitasknn_new_k5'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k5_[0], probabilitasknn_k5_[-1], probabilitasknn_k5_[1])
    probabilitas_knn_k5 = result['probabilitasknn_new_k5']

    # KNN (K=6)
    vectorizer_k6 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k6 = pickle.load(open("knn6.pickle", 'rb'))

    vect_k6 = vectorizer_k6.transform(text_final)[0]
    prediksiknn_k6 = loaded_model_k6.predict(vect_k6)[0]

    result['probabilitasknn_k6']= loaded_model_k6.predict_proba(vect_k6)[0]
    probabilitasknn_k6_ = result['probabilitasknn_k6']

    result['probabilitasknn_new_k6'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k6_[0], probabilitasknn_k6_[-1], probabilitasknn_k6_[1])
    probabilitas_knn_k6 = result['probabilitasknn_new_k6']

     # KNN (K=7)
    vectorizer_k7 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k7 = pickle.load(open("knn7.pickle", 'rb'))

    vect_k7 = vectorizer_k7.transform(text_final)[0]
    prediksiknn_k7 = loaded_model_k7.predict(vect_k7)[0]

    result['probabilitasknn_k7']= loaded_model_k7.predict_proba(vect_k7)[0]
    probabilitasknn_k7_ = result['probabilitasknn_k7']

    result['probabilitasknn_new_k7'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k7_[0], probabilitasknn_k7_[-1], probabilitasknn_k7_[1])
    probabilitas_knn_k7 = result['probabilitasknn_new_k7']

    # KNN (K=8)
    vectorizer_k8 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k8 = pickle.load(open("knn8.pickle", 'rb'))

    vect_k8 = vectorizer_k8.transform(text_final)[0]
    prediksiknn_k8 = loaded_model_k8.predict(vect_k8)[0]

    result['probabilitasknn_k8']= loaded_model_k8.predict_proba(vect_k8)[0]
    probabilitasknn_k8_ = result['probabilitasknn_k8']

    result['probabilitasknn_new_k8'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k8_[0], probabilitasknn_k8_[-1], probabilitasknn_k8_[1])
    probabilitas_knn_k8 = result['probabilitasknn_new_k8']
    
    # KNN (K=9)
    vectorizer = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model = pickle.load(open("knn9.pickle", 'rb'))

    vect = vectorizer.transform(text_final)[0]
    prediksiknn = loaded_model.predict(vect)[0]

    result['probabilitasknn']= loaded_model.predict_proba(vect)[0]
    probabilitasknn_ = result['probabilitasknn']

    result['probabilitasknn_new'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_[0], probabilitasknn_[-1], probabilitasknn_[1])
    probabilitas_knn = result['probabilitasknn_new']
    
    # KNN (K=10)
    vectorizer_k10 = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_k10 = pickle.load(open("knn10.pickle", 'rb'))

    vect_k10 = vectorizer_k10.transform(text_final)[0]
    prediksiknn_k10 = loaded_model_k10.predict(vect_k10)[0]

    result['probabilitasknn_k10']= loaded_model_k10.predict_proba(vect_k10)[0]
    probabilitasknn_k10_ = result['probabilitasknn_k10']

    result['probabilitasknn_new_k10'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasknn_k10_[0], probabilitasknn_k10_[-1], probabilitasknn_k10_[1])
    probabilitas_knn_k10 = result['probabilitasknn_new_k10']
    
    #Decision Tree
    vectorizer_dt = pickle.load(open("tfidf.pickle",'rb'))
    loaded_model_dt = pickle.load(open("DT.pickle", 'rb'))

    vect_dt = vectorizer_dt.transform(text_final)[0]
    prediksidt = loaded_model_dt.predict(vect_dt)[0]

    result['probabilitasdt']= loaded_model_dt.predict_proba(vect_dt)[0]
    probabilitasdt_ = result['probabilitasdt']

    result['probabilitasdt_new'] = "[Negatif : {:.2%}] -- \n[Positif : {:.2%}] -- \n[Netral : {:.2%}]".format(probabilitasdt_[0], probabilitasdt_[-1], probabilitasdt_[1])
    probabilitas_dt = result['probabilitasdt_new']


    return render_template("sistem_hasiluji.html", 
                                subject = subject,
                                casefolding = casefolding, 
                                removepunct = removepunct,
                                removenum = removenum,
                                slangword_ = slangword_,
                                hasil_token = hasil_token,
                                remove_stop_words = remove_stop_words,
                                text_final_ = text_final_,
                                probabilitas_knn = probabilitas_knn,
                                probabilitas_knn_k1 = probabilitas_knn_k1,
                                probabilitas_knn_k2 = probabilitas_knn_k2,
                                probabilitas_knn_k3 = probabilitas_knn_k3,
                                probabilitas_knn_k4 = probabilitas_knn_k4,
                                probabilitas_knn_k5 = probabilitas_knn_k5,
                                probabilitas_knn_k6 = probabilitas_knn_k6,
                                probabilitas_knn_k7 = probabilitas_knn_k7,
                                probabilitas_knn_k8 = probabilitas_knn_k8,
                                probabilitas_knn_k10 = probabilitas_knn_k10,
                                probabilitas_dt = probabilitas_dt
                                ) 

if __name__ == '__main__':
    app.run()

    