import argparse
import re
import string
import time
from pathlib import Path

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===========================================================================
# 1. KAMUS NORMALISASI SLANG INDONESIA
# ===========================================================================
SLANG_DICT = {
    # Negasi (PENTING untuk sentimen)
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "ngga": "tidak",
    "nggak": "tidak", "enggak": "tidak", "ngak": "tidak", "tdk": "tidak",
    "tak": "tidak", "kaga": "tidak", "engga": "tidak",
    "blm": "belum", "blom": "belum", "lom": "belum",
    "jgn": "jangan",

    # Intensifier
    "bgt": "banget", "bngt": "banget", "bner": "benar", "bnr": "benar",
    "skali": "sekali", "skl": "sekali",

    # Konjungsi / kata umum
    "krn": "karena", "krna": "karena", "kr": "karena",
    "udh": "sudah", "udah": "sudah", "dah": "sudah", "uda": "sudah",
    "blh": "boleh", "bs": "bisa", "bsa": "bisa", "ga bisa": "tidak bisa",
    "kpn": "kapan", "knp": "kenapa", "knapa": "kenapa",
    "yg": "yang", "yng": "yang", "dg": "dengan", "dgn": "dengan",
    "utk": "untuk", "u": "untuk", "buat": "untuk",
    "tp": "tapi", "tpi": "tapi", "ttp": "tetap",
    "jd": "jadi", "jdi": "jadi", "jga": "juga", "jg": "juga",
    "sm": "sama", "sma": "sama", "klo": "kalau", "klau": "kalau", "kl": "kalau",
    "trs": "terus", "trus": "terus", "lg": "lagi", "lgi": "lagi",
    "aja": "saja", "doang": "saja", "ja": "saja",
    "bgmn": "bagaimana", "gmn": "bagaimana", "gimana": "bagaimana",
    "dpt": "dapat", "dpat": "dapat", "ngedapet": "dapat", "dapet": "dapat",
    "pake": "pakai", "make": "pakai", "dipake": "dipakai",
    "kasi": "kasih", "ngasih": "kasih",
    "ngerti": "mengerti",
    "pdhl": "padahal", "pdhal": "padahal",

    # Ekspresi review
    "kesel": "kesal", "ksel": "kesal",
    "mantul": "mantap", "mantab": "mantap", "mantaaap": "mantap",
    "joss": "bagus", "kece": "bagus", "keren": "bagus", "ok": "oke",
    "okeh": "oke", "okay": "oke", "okey": "oke", "good": "bagus",
    "nice": "bagus", "best": "terbaik", "great": "bagus",
    "bad": "buruk", "jelek": "buruk", "parah": "buruk",
    "lemot": "lambat", "lelet": "lambat", "lola": "lambat",
    "ribet": "rumit", "njlimet": "rumit",

    # Domain Gojek
    "drv": "driver", "dr": "driver", "abang": "driver", "bang": "driver",
    "ojol": "ojek", "apk": "aplikasi", "app": "aplikasi", "aplikasinya": "aplikasi",
    "vocer": "voucher", "vouchernya": "voucher",
    "cs": "customer service",
    "ongkir": "ongkos",

    # Kata ganti
    "sy": "saya", "ak": "aku", "aq": "aku", "gw": "saya", "gue": "saya",
    "lu": "kamu", "lo": "kamu", "kmu": "kamu",
    "thx": "terima kasih", "tq": "terima kasih", "makasih": "terima kasih",
    "mksh": "terima kasih", "trims": "terima kasih",
    "min": "admin", "mimin": "admin",
}

# ===========================================================================
# 2. STOPWORD SADAR-NEGASI
# ===========================================================================
KEEP_WORDS = {
    "tidak", "bukan", "jangan", "belum", "tanpa", "kurang",
    "sangat", "sekali", "banget", "agak", "lebih", "paling",
    "tetapi", "tapi", "namun",
}

_stopword_factory = StopWordRemoverFactory()
_default_stopwords = set(_stopword_factory.get_stop_words())
CUSTOM_STOPWORDS = _default_stopwords - KEEP_WORDS

_stemmer = StemmerFactory().create_stemmer()
_stem_cache: dict = {}


# ===========================================================================
# 3. FUNGSI-FUNGSI PREPROCESSING
# ===========================================================================
def case_folding(text: str) -> str:
    """Ubah teks menjadi huruf kecil semua."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Hapus URL dari teks."""
    return re.sub(r"http\S+|www\S+", " ", text)


def remove_numbers(text: str) -> str:
    """Hapus semua angka dari teks."""
    return re.sub(r"\d+", " ", text)


def remove_punctuation(text: str) -> str:
    """Hapus tanda baca dari teks."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_special_characters(text: str) -> str:
    """Hapus karakter non-alfabet (emoji, simbol, dll.) — sisakan a-z dan spasi."""
    return re.sub(r"[^a-z\s]", " ", text)


def remove_whitespace(text: str) -> str:
    """Normalkan spasi berlebih dan trim tepi."""
    return re.sub(r"\s+", " ", text).strip()


def repeat_char_norm(text: str) -> str:
    """Normalisasi huruf berulang: 'baguuus' -> 'baguss', 'mantaaap' -> 'mantaap'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_slang(text: str) -> str:
    """Ganti kata slang dengan bentuk bakunya, token per token."""
    tokens = text.split()
    return " ".join(SLANG_DICT.get(tok, tok) for tok in tokens)


def remove_stopwords_custom(text: str) -> str:
    """Hapus stopword umum, pertahankan kata negasi dan intensifier."""
    return " ".join(tok for tok in text.split() if tok not in CUSTOM_STOPWORDS)


def _stem_cached(token: str) -> str:
    """Stemming dengan cache untuk mempercepat proses."""
    result = _stem_cache.get(token)
    if result is None:
        result = _stemmer.stem(token)
        _stem_cache[token] = result
    return result


def stem_text(text: str) -> str:
    """Stem setiap token dalam teks ke bentuk dasar."""
    return " ".join(_stem_cached(tok) for tok in text.split())


def clean_text(text: str) -> str:
    """
    Pipeline preprocessing lengkap untuk satu review Gojek.

    Urutan langkah:
    1. Case folding
    2. Hapus URL
    3. Hapus angka
    4. Hapus tanda baca
    5. Hapus karakter khusus
    6. Normalisasi whitespace
    7. Normalisasi huruf berulang
    8. Normalisasi slang
    9. Hapus stopword (sadar-negasi)
    10. Stemming (Sastrawi + cache)
    11. Normalisasi whitespace akhir
    """
    text = case_folding(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = remove_whitespace(text)
    text = repeat_char_norm(text)
    text = normalize_slang(text)
    text = remove_stopwords_custom(text)
    text = stem_text(text)
    text = remove_whitespace(text)
    return text


# ===========================================================================
# 4. PIPELINE UTAMA
# ===========================================================================
def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    
    # --- 4.1 Load data ---
    print(f"[1/5] Memuat dataset dari: {input_path}")
    df = pd.read_csv(input_path)
    print(f"      Shape awal : {df.shape}")

    # --- 4.2 Validasi kolom wajib ---
    required_cols = {"content", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    # --- 4.3 Bersihkan missing values ---
    print("[2/5] Menghapus baris dengan missing values pada kolom 'content' / 'label'...")
    df = df.dropna(subset=["content", "label"]).copy()
    df["content"] = df["content"].astype(str)

    # --- 4.4 Hapus duplikat ---
    print("[3/5] Menghapus duplikat...")
    before = len(df)
    df = df.drop_duplicates(subset=["content"]).copy()
    print(f"      Duplikat dihapus: {before - len(df)} baris")

    # --- 4.5 Preprocessing teks ---
    print("[4/5] Menjalankan preprocessing teks (case folding → stemming)...")
    t0 = time.time()
    df["clean_text"] = df["content"].apply(clean_text)
    elapsed = time.time() - t0
    print(f"      Selesai dalam {elapsed:.1f}s | Token di-cache: {len(_stem_cache):,}")

    # Buang baris yang menjadi kosong setelah cleaning
    df = df[df["clean_text"].str.len() > 0].copy()
    print(f"      Shape akhir : {df.shape}")

    # --- 4.6 Simpan hasil ---
    print(f"[5/5] Menyimpan hasil ke: {output_path}")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("      Selesai!")

    return df


# ===========================================================================
# 5. ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing dataset sentimen review Gojek."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../dataset_sentimen_raw/dataset_sentimen.csv",
        help="Path ke file CSV input (default: ../dataset_sentimen_raw/dataset_sentimen.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_sentimen_preprocessing/dataset_sentimen_preprocessing.csv",
        help="Path file CSV output (default: dataset_sentimen_preprocessing/dataset_sentimen_preprocessing.csv)",
    )
    args = parser.parse_args()

    result_df = run_preprocessing(args.input, args.output)

    print("\n=== RINGKASAN ===")
    print(f"Total baris   : {len(result_df):,}")
    print("Distribusi label:")
    print(result_df["label"].value_counts().to_string())
    print("\nSampel hasil cleaning:")
    sample = result_df[["content", "clean_text"]].head(5)
    for _, row in sample.iterrows():
        print(f"  IN : {str(row['content'])[:70]}")
        print(f"  OUT: {str(row['clean_text'])[:70]}")
        print()
