Ini adalah sebuah sistem untuk menghilangkan background pada gambar dengan gaya seni cat air.

SETUP:
Pastikan semua library dalam file `requirements.txt` telah diinstal.

Pemasangan bisa dilakukan secara global maupun dengan virtual environment.

Untuk melakukannya dengan virtual environment, lakukan langkah-langkah berikut.

1. Jalankan `python -m venv .venv`

2. Jalankan `.venv/Scripts/activate`

3. Install semua library dengan menjalankan `pip install -r requirements.txt`

QUICK GUIDE:

Mode Trimap Otomatis

1. Pastikan `SAKLAR_METODE` pada bagian `PENGATURAN EKSEKUSI UTAMA` di bagian paling bawah kode bernilai `'auto'`

2. Masukkan gambar input ke dalam folder `input_folder/`

3. Jalankan `python RemoveBackgroundClassic.py`

4. Tunggu proses selesai dan image output akan muncul di folder `output_folder_klasik/`

Mode Trimap Eksternal (manual)

1. Pastikan `SAKLAR_METODE` pada bagian `PENGATURAN EKSEKUSI UTAMA` di bagian paling bawah kode bernilai `'external'`

2. Masukkan gambar input ke dalam folder `input_folder`

3. Jalankan `python generate_draft_trimap.py`, tunggu proses selesai, dan draft awal trimap setiap citra input akan muncul di folder `output_drafts/`

Notes: File trimap setiap citra input bisa juga dibuat sendiri, dengan catatan nama filenya sama dengan nama citra aslinya ditambah `_mask` sebelum format gambar.

4. Modifikasi trimap-trimap awal menggunakan aplikasi editing seperti MS Paint, Photoshop, atau GIMP. Masukkan gambar hasil modifikasi ke dalam folder `mask_folder/`

Notes: Jika anda membuat file trimapnya dari awal, anda juga perlu memindahkannya ke folder tersebut.

5. Pastikan pasangan citra input dan masknya sudah ada di folder `input_folder/` dan `mask_folder/` lalu jalankan `python RemoveBackgroundClassic.py`

6. Tunggu proses selesai dan image output akan muncul di folder `output_folder_klasik/`