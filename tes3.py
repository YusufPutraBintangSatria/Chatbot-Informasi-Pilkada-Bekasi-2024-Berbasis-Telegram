import os
import re
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

# Inisialisasi NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

# ======== Preprocessing NLP ========
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Cleaning: Hapus karakter non-alphabetic
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenisasi teks menjadi kata-kata
    words = text.split()

    # Stopword removal
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# ======== Dataset dan Model ========
def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['Answer'] = df['Answer'].apply(lambda x: x.replace("\\n", "\n") if isinstance(x, str) else x)
    return df['Question'].tolist(), df['Answer'].tolist()

csv_path = "pilkada_bekasi_dataset.csv"
all_documents, all_answers = load_dataset_from_csv(csv_path)

# ======== TF-IDF Vectorizer from Sklearn ========
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
X = vectorizer.fit_transform(all_documents).toarray()

# ======== Jawaban Pertanyaan ========
def jawab_pertanyaan(input_query, threshold=0.3):
    input_query = preprocess_text(input_query)  # Preprocess query
    query_vector = vectorizer.transform([input_query]).toarray()

    # Menghitung cosine similarity antara query dan dokumen
    cosine_similarities = cosine_similarity(query_vector, X)
    index_max = cosine_similarities.argmax()  # Menemukan indeks dengan kesamaan tertinggi
    max_score = cosine_similarities[0, index_max]

    if max_score < threshold:
        return "Maaf, saya kurang mengerti. Bisa ulangi lagi pertanyaannya?"
    
    # Mengembalikan jawaban berdasarkan kesamaan tertinggi
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
    logging.info(f"Processing message: {user_message}")
    
    try:
        response = jawab_pertanyaan(user_message)
        logging.info(f"Response generated: {response}")
        await update.message.reply_text(response)
    except TimedOut:
        logging.error("Timed out while responding.")
        await update.message.reply_text("Maaf, permintaan Anda melebihi waktu yang diizinkan. Silakan coba lagi.")

def main():
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        raise ValueError("Token Telegram tidak ditemukan. Pastikan file .env berisi TELEGRAM_BOT_TOKEN.")

    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run polling with timeout optimizations
    application.run_polling(timeout=10, read_timeout=20)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
