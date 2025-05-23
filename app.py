import cv2
import base64
import io
import numpy as np
from PIL import Image, UnidentifiedImageError # Tambahkan UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback # Untuk logging error yang lebih detail

app = Flask(__name__)
CORS(app)

def convert_to_y_cr_cb(image_np_rgb):
    """Konversi gambar NumPy RGB ke YCrCb menggunakan OpenCV."""
    # OpenCV biasanya bekerja dengan BGR, jadi konversi RGB -> BGR dulu
    img_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
    # Kemudian konversi BGR -> YCrCb
    y_cr_cb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return y_cr_cb_img

def analyze_chili_attributes(y_cr_cb_img):
    """
    Menganalisis gambar YCrCb untuk menentukan kematangan, kualitas,
    dan menghitung rata-rata channel Y, Cr, Cb.
    """
    Y, Cr, Cb = cv2.split(y_cr_cb_img)
    
    mean_Y = np.mean(Y)
    mean_Cr = np.mean(Cr)
    mean_Cb = np.mean(Cb) # Hitung juga mean_Cb di sini

    # Logika penentuan kematangan dan kualitas (sesuaikan threshold jika perlu)
    if mean_Cr > 140: # Semakin tinggi Cr, cenderung semakin merah
        kematangan = "Matang"
    elif mean_Cr > 130: # Bisa jadi oranye atau merah muda
        kematangan = "Setengah Matang"
    else: # Cr rendah, mungkin hijau atau kuning
        kematangan = "Mentah / Belum Matang"

    if mean_Y > 120 and kematangan == "Matang": # Kecerahan baik dan sudah matang
        kualitas = "Sangat Baik"
    elif mean_Y > 100:
        kualitas = "Baik"
    elif mean_Y > 80:
        kualitas = "Cukup"
    else:
        kualitas = "Kurang Baik"
        
    # Jika belum matang, kualitasnya mungkin perlu disesuaikan lagi
    if kematangan != "Matang" and kualitas in ["Sangat Baik", "Baik"]:
        kualitas = "Sedang (Belum Optimal)"


    return kualitas, kematangan, float(mean_Y), float(mean_Cr), float(mean_Cb)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "Missing 'image' key in JSON payload"}), 400
        
        img_data_uri = data['image']

        # Memisahkan header dari data base64 murni
        # Contoh: "data:image/jpeg;base64,THE_ACTUAL_DATA"
        try:
            header, encoded_data = img_data_uri.split(",", 1)
            img_bytes = base64.b64decode(encoded_data)
        except (ValueError, IndexError, base64.binascii.Error) as e:
            app.logger.error(f"Base64 decoding error: {e} - Input: {img_data_uri[:100]}...") # Log sebagian input
            return jsonify({"error": f"Invalid base64 image data format: {e}"}), 400

        # Membuka gambar menggunakan PIL dan konversi ke RGB
        try:
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except UnidentifiedImageError:
            app.logger.error("Cannot identify image file (PIL UnidentifiedImageError)")
            return jsonify({"error": "Cannot identify image file. Invalid image format or corrupted."}), 400
        
        img_np_rgb = np.array(img_pil) # Konversi PIL Image ke NumPy array (format RGB)

        # Konversi ke YCrCb
        y_cr_cb_img = convert_to_y_cr_cb(img_np_rgb)
        
        # Analisis kualitas dan kematangan
        kualitas, kematangan, mean_y_val, mean_cr_val, mean_cb_val = analyze_chili_attributes(y_cr_cb_img)

        # Persiapkan data warna rata-rata untuk respons
        mean_colors_response = {
            'Y': round(mean_y_val, 2),
            'Cr': round(mean_cr_val, 2),
            'Cb': round(mean_cb_val, 2),
        }

        return jsonify({
            'kualitas_cabai': kualitas,
            'tingkat_kematangan': kematangan,
            'mean_ycbcr': mean_colors_response # Kunci tetap mean_ycbcr agar Flutter tidak perlu diubah
        })

    except Exception as e:
        # Log error yang tidak terduga ke konsol server
        app.logger.error(f"An unexpected error occurred: {e}")
        app.logger.error(traceback.format_exc()) 
        return jsonify({"error": "An unexpected error occurred on the server. Please check server logs."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)