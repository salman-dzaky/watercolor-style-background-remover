import numpy as np
from PIL import Image, ImageFilter
from rembg import remove, new_session
import os

def hybrid_watercolor_remover(input_path, output_path, session=None, white_threshold=1.0, alpha_boost=1.0, blur_radius=3, noise_threshold=0.05):
    """
    Menghapus background putih pada gambar cat air menggunakan kombinasi
    perhitungan matematika (Color-to-Alpha) dan AI (rembg).
    """
    # 1. BUKA GAMBAR ASLI
    img_asli = Image.open(input_path).convert('RGBA')
    img_np = np.array(img_asli).astype(np.float32) / 255.0
    
    # KOREKSI WHITE THRESHOLD (Toleransi Putih)
    # Memaksa warna yang mendekati putih (kusam) menjadi putih murni (1.0)
    img_np[:,:,:3] = np.clip(img_np[:,:,:3] / white_threshold, 0.0, 1.0)
    
    R, G, B = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    
    # 2. KALKULASI MATEMATIKA (Color-to-Alpha untuk menyelamatkan cipratan)
    alpha_c2a = 1.0 - np.min(img_np[:,:,:3], axis=2)
    
    # KOREKSI ALPHA BOOST (Kekuatan Soliditas)
    # Menebalkan warna yang pudar agar lebih pekat/solid
    alpha_c2a = np.clip(alpha_c2a * alpha_boost, 0.0, 1.0)
    
    # KOREKSI NOISE (Menghilangkan noise/warna bening di background)
    # Mengabaikan area yang sangat transparan (dianggap sebagai kotoran background)
    alpha_c2a[alpha_c2a < noise_threshold] = 0.0
    
    alpha_safe = np.where(alpha_c2a == 0, 1e-10, alpha_c2a)
    
    R_c2a = np.clip((R - 1.0 + alpha_c2a) / alpha_safe, 0.0, 1.0)
    G_c2a = np.clip((G - 1.0 + alpha_c2a) / alpha_safe, 0.0, 1.0)
    B_c2a = np.clip((B - 1.0 + alpha_c2a) / alpha_safe, 0.0, 1.0)
    
    # Bersihkan residu di area putih murni
    R_c2a[alpha_c2a == 0] = 0
    G_c2a[alpha_c2a == 0] = 0
    B_c2a[alpha_c2a == 0] = 0
    
    rgb_c2a = np.dstack((R_c2a, G_c2a, B_c2a))
    
    # 3. AI MASKING (Untuk Melindungi Subjek Utama)
    if session is None:
        img_ai = remove(img_asli)
    else:
        img_ai = remove(img_asli, session=session)
        
    mask_ai_img = img_ai.split()[3] # Ambil channel Alpha (Topeng/Mask) dari AI
    
    # KOREKSI BLUR RADIUS
    # Menghaluskan tepian potongan AI agar lebih menyatu dengan cipratan cat air
    if blur_radius > 0:
        mask_ai_img = mask_ai_img.filter(ImageFilter.GaussianBlur(blur_radius))
        
    mask_ai = np.array(mask_ai_img).astype(np.float32) / 255.0
    
    # 4. PENGGABUNGAN AJAIB (BLENDING)
    # Menggabungkan alpha dari AI (diutamakan) dengan alpha dari matematika
    alpha_final = mask_ai * 1.0 + (1.0 - mask_ai) * alpha_c2a
    
    mask_rgb = np.stack([mask_ai]*3, axis=-1)
    rgb_asli = img_np[:,:,:3]
    
    # Menggabungkan warna asli (dilindungi AI) dengan warna hasil kalkulasi (cipratan)
    rgb_final = mask_rgb * rgb_asli + (1.0 - mask_rgb) * rgb_c2a
    
    # 5. SIMPAN HASIL
    rgba_final = np.dstack((rgb_final, alpha_final))
    rgba_final = (rgba_final * 255.0).astype(np.uint8)
    
    hasil = Image.fromarray(rgba_final, 'RGBA')
    hasil.save(output_path, "PNG", optimize=True)

def proses_semua_di_folder(folder_input, folder_output, wt=1.0, ab=1.0, br=3, nt=0.05, ai_model="isnet-general-use"):
    """
    Memproses semua gambar di dalam sebuah folder.
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

        print(f"Ditemukan {total_file} gambar. Memulai proses batch...\n")
        print(f"Parameter Terpakai -> WT:{wt} | AB:{ab} | Blur:{br} | Noise Threshold:{nt} | Model:{ai_model}")
        print("-" * 50)

        # Inisialisasi model AI sekali saja di sini agar lebih cepat
        print(f"Memuat model AI ({ai_model})...")
        try:
            session = new_session(ai_model)
        except Exception as e:
            print(f"Gagal memuat model {ai_model}, menggunakan default. Error: {e}")
            session = None

        for index, filename in enumerate(file_gambar, start=1):
            in_path = os.path.join(folder_input, filename)
            nama_file_tanpa_ext = os.path.splitext(filename)[0]
            out_path = os.path.join(folder_output, f"{nama_file_tanpa_ext}_transparan.png")
            
            print(f"[{index}/{total_file}] Memproses: {filename}...")
            try:
                hybrid_watercolor_remover(in_path, out_path, session=session, white_threshold=wt, alpha_boost=ab, blur_radius=br, noise_threshold=nt)
                print(f" -> Sukses!")
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
    
    # SAKELAR MODE (Pilih salah satu dengan mengubah angka menjadi 1, 2, atau 3):
    # Mode 1: Proses SATU gambar biasa.
    # Mode 2: Proses BANYAK gambar di dalam folder.
    # Mode 3: EKSPERIMEN untuk mencari parameter terbaik (Banyak output untuk 1 gambar).
    MODE_PILIHAN = 2 

    if MODE_PILIHAN == 1:
        print("--- MENJALANKAN MODE 1: SATU GAMBAR ---")
        file_masuk = 'gambar_tes.jpg' 
        file_keluar = 'gambar_tes_bersih.png'
        
        # Atur parameter standar Anda di sini
        TOLERANSI_PUTIH = 1 
        SOLIDITAS = 1.2        
        BLUR_AI = 5
        BATAS_NOISE = 0.05
        MODEL_AI = "isnet-general-use"
        
        try:
            print(f"Memproses {file_masuk}...")
            session = new_session(MODEL_AI)
            hybrid_watercolor_remover(file_masuk, file_keluar, 
                                      session=session,
                                      white_threshold=TOLERANSI_PUTIH, 
                                      alpha_boost=SOLIDITAS, 
                                      blur_radius=BLUR_AI,
                                      noise_threshold=BATAS_NOISE)
            print(f"Selesai! Disimpan sebagai: {file_keluar}")
        except FileNotFoundError:
            print(f"Error: Gambar '{file_masuk}' tidak ditemukan!")
            
    elif MODE_PILIHAN == 2:
        print("--- MENJALANKAN MODE 2: FOLDER BATCH ---")
        folder_asal = "./input_folder" 
        folder_tujuan = "./output_folder2"
        
        # Parameter yang akan diterapkan ke SEMUA foto di folder
        TOLERANSI_PUTIH = 1 #untuk sinar gunakan 1
        SOLIDITAS = 1 #untuk sinar gunakan 1, reguler 1.3
        BLUR_AI = 0.5 #untuk sinar gunakan 3, reguler 5
        BATAS_NOISE = 1 #untuk sinar gunakan 0.03, reguler 0.05
        MODEL_AI = "isnet-general-use"
        
        proses_semua_di_folder(folder_asal, folder_tujuan, wt=TOLERANSI_PUTIH, ab=SOLIDITAS, br=BLUR_AI, nt=BATAS_NOISE, ai_model=MODEL_AI)
        
    elif MODE_PILIHAN == 3:
        print("--- MENJALANKAN MODE 3: EKSPERIMEN PARAMETER ---")
        file_masuk = './input_folder/Generated Image May 20, 2026 - 9_10AM.jpg'  # Ganti dengan nama file gambar tersulit/terkotor Anda
        
        # Program akan mencoba semua kombinasi dari angka-angka di bawah ini
        # Semakin banyak variasi, semakin lama prosesnya.
        tes_white_threshold = [1.25, 1.30, 1.35, 1.40, 1.45] 
        tes_alpha_boost = [1.3, 1.6]
        tes_blur_radius = [4, 5]
        BATAS_NOISE = 0.05
        MODEL_AI = "isnet-general-use"
        
        if not os.path.exists(file_masuk):
            print(f"Error: Gambar '{file_masuk}' tidak ditemukan untuk dieksperimen!")
        else:
            print(f"Memulai eksperimen pada '{file_masuk}'...")
            session = new_session(MODEL_AI)
            total_kombinasi = len(tes_white_threshold) * len(tes_alpha_boost) * len(tes_blur_radius)
            hitung = 1
            
            for wt in tes_white_threshold:
                for ab in tes_alpha_boost:
                    for br in tes_blur_radius:
                        # Buat nama file berdasarkan nilai parameter
                        nama_output = f"eksperimen_WT{wt}_AB{ab}_BR{br}.png"
                        print(f"[{hitung}/{total_kombinasi}] Mencoba WT={wt}, AB={ab}, BR={br} -> {nama_output}")
                        
                        try:
                            hybrid_watercolor_remover(file_masuk, nama_output, session=session, white_threshold=wt, alpha_boost=ab, blur_radius=br, noise_threshold=BATAS_NOISE)
                        except Exception as e:
                            print(f"Error saat memproses kombinasi ini: {e}")
                        hitung += 1
                        
            print("\nEksperimen Selesai! Silakan cek folder Anda dan pilih gambar yang hasilnya paling sempurna.")
