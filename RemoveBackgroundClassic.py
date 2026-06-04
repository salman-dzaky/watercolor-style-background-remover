import os
import cv2
import numpy as np
from PIL import Image

def hapus_background_hybrid_klasik(
    input_path: str, 
    output_path: str, 
    mask_path: str = None,
    mode: str = 'auto', # SAKLAR METODE: 'auto' atau 'external'
    white_threshold: float = 1.0, 
    alpha_boost: float = 1.0, 
    blur_radius: float = 3.0, 
    noise_threshold: float = 0.05
) -> None:
    """
    Menghapus background putih pada gambar cat air menggunakan kombinasi
    algoritma GrabCut dan matematika (Color-to-Alpha) murni tanpa Deep Learning.
    
    Parameter 'mode' berfungsi sebagai SAKLAR:
    - 'auto'     : Menggunakan metode heuristik threshold (legacy) untuk menebak mask.
    - 'external' : Wajib memberikan mask_path. Akan menggunakan mask eksternal yang di-load.
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
    # STEP 2 — PEMBUATAN MASK (SAKLAR MODE)
    # ====================================================
    if mode == 'external':
        # [METODE 1: MASK EKSTERNAL]
        if not mask_path or not os.path.exists(mask_path):
            raise FileNotFoundError(f"STRICT INSTRUCTION ERROR: Mode 'external' mewajibkan file mask, namun tidak ditemukan: {mask_path}")

        # Load mask dari disk
        mask_loaded = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_loaded is None:
            raise ValueError(f"Gagal memuat citra mask dari: {mask_path}")
            
        # Konversi nilai piksel standar (0-255) menjadi konstanta GrabCut (0, 1, 2, 3)
        if mask_loaded.max() > 3:
            mask = np.full(mask_loaded.shape, cv2.GC_PR_BGD, dtype=np.uint8)
            mask[mask_loaded == 0] = cv2.GC_BGD         # Hitam = Background mutlak
            mask[mask_loaded == 255] = cv2.GC_FGD       # Putih = Foreground mutlak
        else:
            mask = mask_loaded.copy()

    elif mode == 'auto':
        # [METODE 2: MASK OTOMATIS / HEURISTIK]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Inisiasi mask: Secara bawaan, anggap semua area berpotensi background (Probable Background)
        mask = np.full(gray.shape, cv2.GC_PR_BGD, dtype=np.uint8) 
        
        # Heuristik: Area gelap adalah probable foreground (GC_PR_FGD), area sangat gelap adalah mutlak foreground (GC_FGD)
        mask[gray < 230] = cv2.GC_PR_FGD  
        mask[gray < 50]  = cv2.GC_FGD      
        
        # Mengamankan tepian gambar (margin) dipastikan sebagai background (Kertas putih)
        margin = 3
        if gray.shape[0] > margin * 2 and gray.shape[1] > margin * 2:
            mask[:margin, :] = cv2.GC_BGD
            mask[-margin:, :] = cv2.GC_BGD
            mask[:, :margin] = cv2.GC_BGD
            mask[:, -margin:] = cv2.GC_BGD
            
    else:
        raise ValueError(f"Mode tidak dikenal: '{mode}'. Gunakan 'auto' atau 'external'.")

    # ====================================================
    # STEP 2.5 — GRABCUT EXECUTION
    # ====================================================
    # Alokasi model dinamis yang diwajibkan oleh arsitektur GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Eksekusi GrabCut (Optimisasi Graph-Cut): 5 iterasi menggunakan masker (GC_INIT_WITH_MASK)
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
    if white_threshold > 1.0:
        img_np_adj = np.clip(img_np * white_threshold, 0.0, 1.0)
    else:
        safe_wt = max(white_threshold, 1e-5) # Menghindari division by zero
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

def proses_semua_di_folder(
    folder_input: str, 
    folder_output: str, 
    mode: str = 'auto', 
    folder_mask: str = None, 
    wt: float=1.0, ab: float=1.0, br: float=3.0, nt: float=0.05
) -> None:
    """
    Eksekutor batch yang mendukung kedua mode (auto dan external).
    """
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
        print(f"Folder '{folder_output}' berhasil dibuat.\n")

    if mode == 'external' and not os.path.exists(folder_mask):
        print(f"Error: Mode 'external' diaktifkan tapi Folder mask '{folder_mask}' tidak ditemukan!")
        return

    if os.path.exists(folder_input):
        daftar_file = os.listdir(folder_input)
        
        # Abaikan file yang merupakan mask jika berada di folder yang sama
        file_gambar = [
            f for f in daftar_file 
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.lower().endswith(('_mask.png', '_mask.jpg'))
        ]
        
        total_file = len(file_gambar)
        if total_file == 0:
            print(f"Tidak ada gambar (JPG/PNG) di dalam folder '{folder_input}'.")
            return

        print(f"Ditemukan {total_file} gambar. Memulai proses batch dengan MODE: [{mode.upper()}]\n")
        print(f"Parameter Terpakai -> WT:{wt} | AB:{ab} | Blur:{br} | Noise Threshold:{nt}")
        print("-" * 50)

        for index, filename in enumerate(file_gambar, start=1):
            # Mengabaikan file tersembunyi sistem atau file gitkeep secara eksplisit
            if filename.startswith('.') or filename == '.gitkeep':
                continue

            in_path = os.path.join(folder_input, filename)
            nama_file_tanpa_ext = os.path.splitext(filename)[0]
            out_path = os.path.join(folder_output, f"{nama_file_tanpa_ext}_transparan_CV.png")
            
            mask_path = None
            if mode == 'external':
                # Mencari kandidat file mask di folder mask
                mask_kandidat = [
                    os.path.join(folder_mask, f"{nama_file_tanpa_ext}_mask.png"),
                    os.path.join(folder_mask, f"{nama_file_tanpa_ext}_mask.jpg"),
                    os.path.join(folder_mask, f"{nama_file_tanpa_ext}.png"),
                    os.path.join(folder_mask, f"{nama_file_tanpa_ext}.jpg")
                ]
                
                for kandidat in mask_kandidat:
                    if os.path.exists(kandidat):
                        mask_path = kandidat
                        break
                
                if mask_path is None:
                    print(f"[{index}/{total_file}] LEWATI: Mask untuk '{filename}' tidak ditemukan di folder '{folder_mask}'.")
                    continue

            print(f"[{index}/{total_file}] Memproses: {filename}" + (f" (Mask: {os.path.basename(mask_path)})" if mask_path else ""))
            try:
                hapus_background_hybrid_klasik(
                    input_path=in_path, 
                    output_path=out_path, 
                    mask_path=mask_path,
                    mode=mode,
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
# PENGATURAN EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # SAKLAR UTAMA (Pilih antara 'auto' atau 'external')
    # - 'auto'     : Menebak background secara otomatis (logika legacy).
    # - 'external' : Menggunakan mask dari luar (logika Custom).
    # ---------------------------------------------------------
    SAKLAR_METODE = 'auto' 
    
    # SAKELAR MODE EKSEKUSI (1=Satu File, 2=Satu Folder)
    MODE_EKSEKUSI = 2 

    # Parameter Global
    TOLERANSI_PUTIH = 1.05 
    SOLIDITAS = 1.2        
    BLUR_RADIUS = 3.0
    BATAS_NOISE = 0.05

    print(f"=== PROGRAM DIMULAI (Metode Masking: {SAKLAR_METODE.upper()}) ===")

    if MODE_EKSEKUSI == 1:
        print("--- MODE 1: SATU GAMBAR ---")
        file_masuk = 'gambar_tes.jpg' 
        file_keluar = 'gambar_tes_bersih.png'
        file_mask = 'gambar_tes_mask.png' # Hanya dibaca jika SAKLAR_METODE = 'external'
        
        try:
            hapus_background_hybrid_klasik(
                input_path=file_masuk, 
                output_path=file_keluar, 
                mask_path=file_mask if SAKLAR_METODE == 'external' else None,
                mode=SAKLAR_METODE,
                white_threshold=TOLERANSI_PUTIH, 
                alpha_boost=SOLIDITAS, 
                blur_radius=BLUR_RADIUS,
                noise_threshold=BATAS_NOISE
            )
            print(f"Selesai! Disimpan sebagai: {file_keluar}")
        except Exception as e:
            print(f"Error: {e}")
            
    elif MODE_EKSEKUSI == 2:
        print("--- MODE 2: FOLDER BATCH ---")
        folder_asal = "./input_folder" 
        folder_tujuan = "./output_folder_klasik"
        folder_mask = "./mask_folder" # Hanya dibaca jika SAKLAR_METODE = 'external'
        
        proses_semua_di_folder(
            folder_input=folder_asal, 
            folder_output=folder_tujuan, 
            mode=SAKLAR_METODE,
            folder_mask=folder_mask,
            wt=TOLERANSI_PUTIH, 
            ab=SOLIDITAS, 
            br=BLUR_RADIUS, 
            nt=BATAS_NOISE
        )
