# 1. Image de base
FROM python:3.10-slim

# 2. Installer bibliothèques système (libGL + audio + fonts si besoin)
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    portaudio19-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg && \
    apt-get clean

# 3. Définir le dossier de travail
WORKDIR /app

# 4. Copier tous les fichiers
COPY . .

# 5. Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Exposer le port de l'application
EXPOSE 5000

# 7. Lancer le serveur Flask via Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]


