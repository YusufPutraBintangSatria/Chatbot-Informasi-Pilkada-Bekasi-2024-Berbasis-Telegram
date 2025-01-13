import os
import re
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from difflib import get_close_matches  # Menambahkan difflib untuk koreksi typo lebih lanjut

# Inisialisasi NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

# ======== Koreksi Typo dengan difflib ========
def correct_typo(text):
    # Menggunakan difflib untuk mencari kata yang paling mendekati
    word_list = list(stop_words)  # Menggunakan daftar stopwords untuk mencari kata yang paling mirip
    corrected = get_close_matches(text, word_list, n=1)
    if corrected:
        return corrected[0]  # Kembalikan kata yang paling mirip
    return text

# ======== Preprocessing NLP ========
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Cleaning: Hapus karakter non-alphabetic
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenisasi teks menjadi kata-kata
    words = text.split()

    # Koreksi typo dengan difflib
    words = [correct_typo(word) for word in words]

    # Stopword removal
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# ======== TF-IDF Vectorizer ========
class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.idf_values = None

    def fit(self, documents):
        documents = [preprocess_text(doc) for doc in documents]
        word_set = set()
        for doc in documents:
            words = doc.split()
            word_set.update(words)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(word_set))}
        N = len(documents)
        df = {word: sum(1 for doc in documents if word in doc.split()) for word in self.vocabulary}
        self.idf_values = {word: np.log(N / df[word]) for word in df}
        return self

    def transform(self, documents):
        documents = [preprocess_text(doc) for doc in documents]
        X = np.zeros((len(documents), len(self.vocabulary)))
        for i, doc in enumerate(documents):
            word_counts = Counter(doc.split())
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    j = self.vocabulary[word]
                    X[i, j] = count * self.idf_values[word]
        return X

# ======== Dataset dan Model ========
def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['Answer'] = df['Answer'].apply(lambda x: x.replace("\\n", "\n") if isinstance(x, str) else x)
    return df['Question'].tolist(), df['Answer'].tolist()

csv_path = "pilkada_bekasi_dataset.csv"
all_documents, all_answers = load_dataset_from_csv(csv_path)

vectorizer = TFIDFVectorizer()
vectorizer.fit(all_documents)
X = vectorizer.transform(all_documents)

# ======== Jawaban Pertanyaan ========
def jawab_pertanyaan(input_query, threshold=0.3):
    input_query = preprocess_text(input_query)  # Preprocess query
    
    # Menambahkan pengecekan kata kunci seperti "kapan" dan "pilkada"
    if 'kapan' in input_query and 'pilkada' in input_query:
        return "Pilkada Bekasi 2024 dijadwalkan pada 27 November 2024."

    # Model TF-IDF untuk pertanyaan lainnya
    query_vector = vectorizer.transform([input_query])
    cosine_similarities = cosine_similarity(query_vector, X)
    index_max = cosine_similarities.argmax()
    max_score = cosine_similarities[0, index_max]

    if max_score < threshold:
        return "Maaf, saya kurang mengerti. Bisa ulangi lagi pertanyaannya?"
    return all_answers[index_max]

# ======== Telegram Bot ========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "Selamat datang di Bot Pilkada Bekasi 2024! 🎉\n"
        "Ajukan pertanyaan terkait Pilkada Bekasi 2024, dan saya akan menjawab sebaik mungkin.\n\n"
    )
    await update.message.reply_text(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    response = jawab_pertanyaan(user_message)
    await update.message.reply_text(response)

def main():
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        raise ValueError("Token Telegram tidak ditemukan. Pastikan file .env berisi TELEGRAM_BOT_TOKEN.")

    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    main()
