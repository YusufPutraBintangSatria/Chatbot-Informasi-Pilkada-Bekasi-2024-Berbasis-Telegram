from symspellpy.symspellpy import SymSpell, Verbosity

# Inisialisasi SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Muat kamus dari file "kbbi.txt"
dictionary_path = "kbbi.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Fungsi untuk mengetes koreksi typo
def test_typo_correction(word):
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        print(f"Input: {word} -> Koreksi: {suggestions[0].term}")
    else:
        print(f"Input: {word} -> Tidak ditemukan koreksi")

# Loop untuk menerima input dari terminal
while True:
    user_input = input("Masukkan kata (atau ketik 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        print("Program selesai.")
        break
    test_typo_correction(user_input)
