from sklearn.metrics import accuracy_score

# Dataset Tes
test_data = [
    ("Kapan Pilkada Bekasi 2024 dilaksanakan?", "Pilkada Bekasi 2024 dijadwalkan pada 27 November 2024."),
    ("Siapa saja kandidat dalam Pilkada Bekasi 2024?", "Saat ini, daftar kandidat resmi belum diumumkan. Nantikan informasi lebih lanjut dari KPU."),
    ("Apa syarat untuk memilih di Pilkada Bekasi 2024?", "Pemilih harus berusia minimal 17 tahun, memiliki KTP elektronik, dan terdaftar dalam daftar pemilih tetap (DPT)."),
    # Tambahkan lebih banyak pertanyaan dan jawaban yang benar di sini
]

# Fungsi untuk menghitung akurasi
def hitung_akurasi(test_data):
    true_answers = [answer for _, answer in test_data]
    predicted_answers = [jawab_pertanyaan(query) for query, _ in test_data]

    # Bandingkan hasil prediksi dengan jawaban yang benar
    correct_predictions = [pred == true for pred, true in zip(predicted_answers, true_answers)]
    accuracy = sum(correct_predictions) / len(test_data)
    return accuracy

# Menghitung akurasi
akurasi = hitung_akurasi(test_data)
print(f"Akurasi: {akurasi * 100:.2f}%")
