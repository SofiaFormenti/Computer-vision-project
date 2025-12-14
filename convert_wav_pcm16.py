import os
import subprocess

# Dossier contenant tes wav
SAMPLES_DIR = "samples"

# Chemin vers ffmpeg (modifie si nécessaire)
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

print("=== CONVERSION WAV → PCM 16-bit ===")
print("Dossier scanné :", os.path.abspath(SAMPLES_DIR))
print("------------------------------------")

if not os.path.exists(FFMPEG):
    print(" ERREUR : ffmpeg introuvable :", FFMPEG)
    print("Corrige le chemin avant d exécuter.")
    exit()

for filename in os.listdir(SAMPLES_DIR):
    if filename.lower().endswith(".wav"):
        src = os.path.join(SAMPLES_DIR, filename)
<<<<<<< HEAD
        dst = os.path.join(SAMPLES_DIR, "fixed_" + filename)
=======
        dst = os.path.join(SAMPLES_DIR, "conv_" + filename)
>>>>>>> 8051cdaf0a755a63338fad55e7fb193cba19ed7a

        print(f"→ Conversion : {filename}")

        # Commande ffmpeg
        cmd = [
            FFMPEG,
            "-y",       # overwrite
            "-i", src,  # input
            "-acodec", "pcm_s16le",  # WAV 16-bit PCM
            "-ar", "44100",          # (optionnel) force 44.1 kHz
            dst
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("------------------------------------")
print("✔ Conversion terminée !")
print("Les fichiers 'fixed_xxx.wav' sont prêts.")
print("Révérifie qu'ils fonctionnent puis remplace les anciens fichiers.")
