import os
import cv2
import numpy as np

# ==========================================
# VARIABEL PENGATURAN (PARAMETER TUNING)
# ==========================================
# Direktori input dan output
FOLDER_INPUT = "./input_folder"
FOLDER_OUTPUT = "./output_drafts"

# Pengaturan Morfologi untuk Sure Foreground (Bagian Inti Objek / Putih / 255)
# Semakin besar iterasi/kernel, semakin mengecil area solid (putih) pada trimap.
ERODE_KERNEL_SIZE = 1
ERODE_ITERATIONS = 1

# Pengaturan Morfologi untuk Sure Background (Bagian Kertas / Hitam / 0)
# Semakin besar iterasi/kernel, semakin luas area abu-abu (area transisi/batas yang dicari GrabCut).
DILATE_KERNEL_SIZE = 31
DILATE_ITERATIONS = 5

# Pengaturan Denoise awal (Gaussian Blur)
BLUR_KERNEL_SIZE = 5

def buat_draft_trimap(
    input_path: str, 
    output_path: str, 
    erode_ksize: int = 9, 
    erode_iters: int = 3, 
    dilate_ksize: int = 15, 
    dilate_iters: int = 3,
    blur_ksize: int = 5
) -> None:
    """
    Menghasilkan draft trimap untuk algoritma GrabCut menggunakan murni operasi
    Morfologi Matematika (Classical Computer Vision).
    
    Trimap terdiri dari 3 nilai:
    - 0   (Hitam) : Pasti Background (Sure Background)
    - 255 (Putih) : Pasti Foreground (Sure Foreground)
    - 128 (Abu)   : Probable/Unknown (Area transisi tempat GrabCut akan menebak)
    """
    
    # 1. LOAD & GRAYSCALE
    # Muat gambar; kita menggunakan IMREAD_COLOR lalu konversi agar stabil
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca, mungkin rusak atau format tidak didukung: {input_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. DENOISE
    # Menghaluskan tekstur kertas yang kasar agar tidak merusak threshold
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    # 3. OTSU'S THRESHOLDING
    # Menggunakan THRESH_BINARY_INV karena kertas berwarna cerah (putih) dan cat air lebih gelap.
    # Hasil: Cat air menjadi putih (255), kertas menjadi hitam (0).
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. MORPHOLOGICAL OPERATIONS (The Core Logic)
    # Siapkan kernel (matriks berisikan angka 1) untuk erosi dan dilasi
    kernel_erode = np.ones((erode_ksize, erode_ksize), np.uint8)
    kernel_dilate = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    
    # A. SURE FOREGROUND (Inti Padat Objek)
    # Mengikis (erode) mask Otsu secara agresif untuk memastikan bahwa yang tersisa
    # benar-benar adalah bagian dalam dari cat air.
    sure_fg = cv2.erode(otsu_mask, kernel_erode, iterations=erode_iters)
    
    # B. SURE BACKGROUND (Area Aman Kertas Latar)
    # Memperluas (dilate) mask Otsu secara masif. Hasil dilasi ini akan menutupi 
    # cipratan halus dan bayangan. Setelah diinversi (bitwise_not), bagian yang 
    # tersisa adalah area luar yang KITA YAKIN 100% adalah kertas murni.
    dilated_mask = cv2.dilate(otsu_mask, kernel_dilate, iterations=dilate_iters)
    sure_bg = cv2.bitwise_not(dilated_mask)
    
    # C. UNKNOWN AREA (Area Probable)
    # Secara implisit, area abu-abu adalah selisih antara hasil dilasi dan erosi.
    
    # 5. ASSEMBLE THE TRIMAP
    # Buat kanvas kosong berisi nilai 128 (Abu-abu / Unknown)
    trimap = np.full(gray.shape, 128, dtype=np.uint8)
    
    # Timpa dengan nilai Putih (255) untuk area yang pasti objek
    trimap[sure_fg == 255] = 255
    
    # Timpa dengan nilai Hitam (0) untuk area yang pasti latar (kertas)
    trimap[sure_bg == 255] = 0
    
    # 6. OUTPUT
    # Simpan mask ke disk sebagai PNG 1-channel (Grayscale)
    cv2.imwrite(output_path, trimap)

def eksekusi_batch_trimap(folder_in: str, folder_out: str) -> None:
    """
    Mengiterasi semua gambar pada folder input dan menghasilkan draft trimap ke folder output.
    """
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
        print(f"Folder output '{folder_out}' berhasil dibuat.\n")

    if not os.path.exists(folder_in):
        print(f"Error: Folder input '{folder_in}' tidak ditemukan.")
        return

    daftar_file = os.listdir(folder_in)
    file_gambar = [f for f in daftar_file if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_file = len(file_gambar)
    if total_file == 0:
        print(f"Tidak ada gambar di dalam folder '{folder_in}'.")
        return

    print(f"Memulai pembuatan Draft Trimap untuk {total_file} gambar...")
    print("-" * 50)

    for index, filename in enumerate(file_gambar, start=1):
        in_path = os.path.join(folder_in, filename)
        nama_file_tanpa_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(folder_out, f"{nama_file_tanpa_ext}_mask.png")
        
        print(f"[{index}/{total_file}] Menghasilkan trimap untuk: {filename}...")
        try:
            buat_draft_trimap(
                input_path=in_path, 
                output_path=out_path,
                erode_ksize=ERODE_KERNEL_SIZE,
                erode_iters=ERODE_ITERATIONS,
                dilate_ksize=DILATE_KERNEL_SIZE,
                dilate_iters=DILATE_ITERATIONS,
                blur_ksize=BLUR_KERNEL_SIZE
            )
            print(" -> Sukses!")
        except Exception as e:
            print(f" -> GAGAL memproses {filename}. Error: {e}")
            
    print("-" * 50)
    print(f"\nSelesai! Draft trimap disimpan di '{folder_out}'.")
    print("Silakan buka file-file tersebut di MS Paint / Photoshop untuk merapikannya,")
    print("lalu berikan path folder tersebut sebagai input 'mask_path' ke script GrabCut.")

if __name__ == "__main__":
    eksekusi_batch_trimap(FOLDER_INPUT, FOLDER_OUTPUT)
