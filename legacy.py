import os
import cv2
import numpy as np
from PIL import Image

def hapus_background_hybrid_klasik(
    input_path: str, 
    output_path: str, 
    white_threshold: float = 1.0, 
    alpha_boost: float = 1.0, 
    blur_radius: float = 3.0, 
    noise_threshold: float = 0.05
) -> None:
    """
    Menghapus background putih pada gambar cat air menggunakan kombinasi
    algoritma GrabCut dan matematika (Color-to-Alpha) murni tanpa Deep Learning.
    """
    
    # ====================================================
    # STEP 1 — LOAD & NORMALIZE
    # ====================================================
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
        
    # Membaca gambar RGB menggunakan Pillow (kestabilan profil warna)
    img_pil = Image.open(input_path).convert('RGB')
    img_rgb = np.array(img_pil)
    
    # Konversi ke float32 dan normalisasi rentang matriks [0, 1]
    img_np = img_rgb.astype(np.float32) / 255.0

    # ====================================================
    # STEP 2 — GRABCUT EXECUTION
    # ====================================================
    # Membuat citra grayscale uint8 untuk menebak struktur (trimap creation)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Inisiasi mask: Secara bawaan, anggap semua area berpotensi background (Probable Background)
    mask = np.full(gray.shape, cv2.GC_PR_BGD, dtype=np.uint8) 
    
    # Heuristik: Area gelap adalah probable foreground (GC_PR_FGD), area sangat gelap adalah mutlak foreground (GC_FGD)
    mask[gray < 230] = cv2.GC_PR_FGD  
    mask[gray < 50]  = cv2.GC_FGD      
    
    # Mengamankan tepian gambar (margin) dipastikan sebagai background (Kertas putih)
    margin = 5
    if gray.shape[0] > margin * 2 and gray.shape[1] > margin * 2:
        mask[:margin, :] = cv2.GC_BGD
        mask[-margin:, :] = cv2.GC_BGD
        mask[:, :margin] = cv2.GC_BGD
        mask[:, -margin:] = cv2.GC_BGD
        
    # Alokasi model dinamis yang diwajibkan oleh arsitektur GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Eksekusi GrabCut (Optimisasi Graph-Cut): 5 iterasi menggunakan masker trimap (GC_INIT_WITH_MASK)
    cv2.grabCut(img_rgb, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # ====================================================
    # STEP 3 — MASK SOFTENING
    # ====================================================
    # Menyatukan Probable Foreground dan Foreground mutlak menjadi mask Biner Solid [0.0 atau 1.0]
    M_binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    
    # Aplikasi GaussianBlur untuk menciptakan efek bulu (feathering), membunuh batas pikselasi bergerigi tajam
    if blur_radius > 0:
        # sigmaX menangani radius blur, (0,0) akan membiarkan OpenCV memilih ukuran kernel optimal
        M_blur = cv2.GaussianBlur(M_binary, (0, 0), sigmaX=blur_radius)
    else:
        M_blur = M_binary
        
    # Ekspansi sumbu untuk keselarasan aljabar pada operasi multi-saluran RGB
    M_blur_3d = np.expand_dims(M_blur, axis=-1)

    # ====================================================
    # STEP 4 — COLOR TO ALPHA EXTRACTION
    # ====================================================
    # Menerapkan white_threshold untuk mengonversi area abu-abu kusam menjadi putih bersih
    # Logika Tahan Banting: 
    # Jika threshold > 1.0, asumsikan user ingin "menerangkan" (kali silang agar nilainya mendekati/melebihi 1.0 lalu di-clip)
    # Jika threshold <= 1.0, asumsikan user memakai representasi persentase (dibagi agar nilai yang tadinya < 1.0 bisa mencapai 1.0)
    if white_threshold > 1.0:
        img_np_adj = np.clip(img_np * white_threshold, 0.0, 1.0)
    else:
        # Menghindari division by zero jika user iseng memasukkan 0.0
        safe_wt = max(white_threshold, 1e-5)
        img_np_adj = np.clip(img_np / safe_wt, 0.0, 1.0)
    
    # Formula Matematika Murni Color-to-Alpha: α = 1.0 - minimum(R, G, B)
    alpha_c2a = 1.0 - np.min(img_np_adj, axis=-1)
    
    # Memanipulasi tingkat kontras alpha (Alpha Boost)
    alpha_c2a = np.clip(alpha_c2a * alpha_boost, 0.0, 1.0)
    
    # Menghapus debu/noise pada background murni
    alpha_c2a[alpha_c2a < noise_threshold] = 0.0
    alpha_c2a_3d = np.expand_dims(alpha_c2a, axis=-1)
    
    # Memproteksi komputasi dari Division by Zero dengan menyisipkan konstanta minimal
    alpha_safe_3d = np.where(alpha_c2a_3d == 0, 1e-10, alpha_c2a_3d)
    
    # C_obj = (C_original - 1.0 + alpha) / alpha
    # Merekonstruksi warna pigmen asli tanpa adanya kontaminasi (kertas) warna putih
    C_obj = (img_np - 1.0 + alpha_c2a_3d) / alpha_safe_3d
    C_obj = np.clip(C_obj, 0.0, 1.0)
    
    # Sanitasi: Memastikan nilai RGB lenyap total bila alpha benar-benar kosong
    C_obj[alpha_c2a == 0] = 0.0

    # ====================================================
    # STEP 5 — ALPHA MERGING
    # ====================================================
    # alpha_final = M_blur + (1.0 - M_blur) * alpha_c2a
    # Menggabungkan area perlindungan solid (M_blur) dengan area sapuan transparan halus pinggiran (alpha_c2a)
    alpha_final = M_blur + (1.0 - M_blur) * alpha_c2a
    alpha_final = np.clip(alpha_final, 0.0, 1.0)
    
    # C_final = M_blur * C_original + (1.0 - M_blur) * C_obj
    C_final = M_blur_3d * img_np + (1.0 - M_blur_3d) * C_obj
    C_final = np.clip(C_final, 0.0, 1.0)

    # ====================================================
    # STEP 6 — OUTPUT
    # ====================================================
    # Kembalikan tipe matriks menjadi deret bita spasial uint8 (0-255)
    C_final_8u = (C_final * 255.0).astype(np.uint8)
    alpha_final_8u = (alpha_final * 255.0).astype(np.uint8)
    
    # Tumpuk (stack) RGB dan Alpha menjadi kanal RGBA utuh
    rgba_matrix = np.dstack((C_final_8u, alpha_final_8u))
    
    # Simpan dengan format asali PNG via Pillow
    rgba_image = Image.fromarray(rgba_matrix, 'RGBA')
    rgba_image.save(output_path, "PNG", optimize=True)

def proses_semua_di_folder(folder_input: str, folder_output: str, wt: float=1.0, ab: float=1.0, br: float=3.0, nt: float=0.05) -> None:
    """
    Eksekutor batch: Memproses iterasi untuk seluruha entitas PNG/JPG di folder yang ditentukan.
    """
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
        print(f"Folder '{folder_output}' berhasil dibuat.\n")

    if os.path.exists(folder_input):
        daftar_file = os.listdir(folder_input)
        file_gambar = [f for f in daftar_file if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_file = len(file_gambar)
        if total_file == 0:
            print(f"Tidak ada gambar (JPG/PNG) di dalam folder '{folder_input}'.")
            return

        print(f"Ditemukan {total_file} gambar. Memulai proses batch murni Computer Vision...\n")
        print(f"Parameter Terpakai -> WT:{wt} | AB:{ab} | Blur:{br} | Noise Threshold:{nt}")
        print("-" * 50)

        for index, filename in enumerate(file_gambar, start=1):
            in_path = os.path.join(folder_input, filename)
            nama_file_tanpa_ext = os.path.splitext(filename)[0]
            out_path = os.path.join(folder_output, f"{nama_file_tanpa_ext}_transparan_CV.png")
            
            print(f"[{index}/{total_file}] Memproses: {filename}...")
            try:
                hapus_background_hybrid_klasik(
                    in_path, 
                    out_path, 
                    white_threshold=wt, 
                    alpha_boost=ab, 
                    blur_radius=br, 
                    noise_threshold=nt
                )
                print(" -> Sukses!")
            except Exception as e:
                print(f" -> GAGAL memproses {filename}. Error: {e}")
        
        print("-" * 50)
        print(f"\nSelesai! Semua file disimpan di folder '{folder_output}'.")
    else:
        print(f"Error: Folder '{folder_input}' tidak ditemukan!")

# ==========================================
# PENGATURAN EKSEKUSI (PILIH MODE DI SINI)
# ==========================================
if __name__ == "__main__":
    
    # SAKELAR MODE (1=Satu File, 2=Satu Folder, 3=Eksperimen Tuning Parameter)
    MODE_PILIHAN = 2 

    if MODE_PILIHAN == 1:
        print("--- MENJALANKAN MODE 1: SATU GAMBAR ---")
        file_masuk = 'gambar_tes.jpg' 
        file_keluar = 'gambar_tes_bersih.png'
        
        TOLERANSI_PUTIH = 1.05 
        SOLIDITAS = 1.2        
        BLUR_RADIUS = 3.0
        BATAS_NOISE = 0.05
        
        try:
            print(f"Memproses {file_masuk}...")
            hapus_background_hybrid_klasik(
                file_masuk, 
                file_keluar, 
                white_threshold=TOLERANSI_PUTIH, 
                alpha_boost=SOLIDITAS, 
                blur_radius=BLUR_RADIUS,
                noise_threshold=BATAS_NOISE
            )
            print(f"Selesai! Disimpan sebagai: {file_keluar}")
        except FileNotFoundError as e:
            print(e)
            
    elif MODE_PILIHAN == 2:
        print("--- MENJALANKAN MODE 2: FOLDER BATCH ---")
        folder_asal = "./input_folder" 
        folder_tujuan = "./output_folder_klasik"
        
        TOLERANSI_PUTIH = 1.05 
        SOLIDITAS = 1.1 
        BLUR_RADIUS = 3.0 
        BATAS_NOISE = 0.04 
        
        proses_semua_di_folder(
            folder_asal, 
            folder_tujuan, 
            wt=TOLERANSI_PUTIH, 
            ab=SOLIDITAS, 
            br=BLUR_RADIUS, 
            nt=BATAS_NOISE
        )
        
    elif MODE_PILIHAN == 3:
        print("--- MENJALANKAN MODE 3: EKSPERIMEN PARAMETER ---")
        file_masuk = './input_folder/sample.jpg'
        
        tes_white_threshold = [1.0, 1.1] 
        tes_alpha_boost = [1.0, 1.3]
        tes_blur_radius = [2.0, 4.0]
        BATAS_NOISE = 0.05
        
        if not os.path.exists(file_masuk):
            print(f"Error: Gambar '{file_masuk}' tidak ditemukan untuk dieksperimen!")
        else:
            total_kombinasi = len(tes_white_threshold) * len(tes_alpha_boost) * len(tes_blur_radius)
            hitung = 1
            
            for wt in tes_white_threshold:
                for ab in tes_alpha_boost:
                    for br in tes_blur_radius:
                        nama_output = f"eksperimen_CV_WT{wt}_AB{ab}_BR{br}.png"
                        print(f"[{hitung}/{total_kombinasi}] Uji WT={wt}, AB={ab}, BR={br} -> {nama_output}")
                        try:
                            hapus_background_hybrid_klasik(
                                file_masuk, 
                                nama_output, 
                                white_threshold=wt, 
                                alpha_boost=ab, 
                                blur_radius=br, 
                                noise_threshold=BATAS_NOISE
                            )
                        except Exception as e:
                            print(f"Error saat komputasi: {e}")
                        hitung += 1
                        
            print("\nEksperimen Selesai! Pilih output terbaik Anda.")