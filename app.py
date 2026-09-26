import os
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.mkdtemp()

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

def procesar_video_mosaico(video_input_path, imagenes_teselas_paths, video_output_path, resolucion_ancho=120):
    teselas = []
    for path in imagenes_teselas_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color_promedio = img_rgb.mean(axis=(0, 1))
        teselas.append({'path': path, 'img_bgr': img, 'color': color_promedio})
    
    if not teselas:
        raise ValueError("No se pudieron cargar las imágenes de teselas.")
    
    cap = cv2.VideoCapture(video_input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    aspect_ratio = alto_orig / ancho_orig
    target_width = resolucion_ancho
    target_height = int(target_width * aspect_ratio)
    
    temp_video_no_audio = os.path.join(UPLOAD_FOLDER, "temp_output_no_audio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    pixel_size = max(1, 1080 // target_width)
    out_width = target_width * pixel_size
    out_height = target_height * pixel_size
    
    out_writer = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (out_width, out_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        mosaic_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        for y in range(target_height):
            for x in range(target_width):
                pixel_color = frame_rgb[y, x]
                
                best_tesela = None
                min_dist = float('inf')
                for t in teselas:
                    dist = np.linalg.norm(pixel_color - t['color'])
                    if dist < min_dist:
                        min_dist = dist
                        best_tesela = t
                
                tile_resized = cv2.resize(best_tesela['img_bgr'], (pixel_size, pixel_size))
                mosaic_frame[y*pixel_size:(y+1)*pixel_size, x*pixel_size:(x+1)*pixel_size] = tile_resized
        
        out_writer.write(mosaic_frame)
    
    cap.release()
    out_writer.release()
    
    cmd_ffmpeg = ['ffmpeg', '-y', '-i', temp_video_no_audio, '-i', video_input_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0?',
        '-shortest', video_output_path]
    subprocess.run(cmd_ffmpeg, check=True, capture_output=True)
    
    if os.path.exists(temp_video_no_audio):
        os.remove(temp_video_no_audio)

@app.route('/procesar', methods=['POST'])
def procesar():
    try:
        if 'video' not in request.files or 'teselas' not in request.files:
            return jsonify({'error': 'Missing video or tile images'}), 400
        
        video_file = request.files['video']
        tile_files = request.files.getlist('teselas')
        
        if not allowed_file(video_file.filename, {'mp4'}) or not tile_files:
            return jsonify({'error': 'Invalid files'}), 400
        
        process_temp_dir = tempfile.mkdtemp()
        
        try:
            video_path = os.path.join(process_temp_dir, secure_filename(video_file.filename))
            video_file.save(video_path)
            
            teselas_paths = []
            for idx, tile_file in enumerate(tile_files):
                if allowed_file(tile_file.filename, {'jpg', 'jpeg', 'png'}):
                    tile_path = os.path.join(process_temp_dir, secure_filename(f"tile_{idx}_{tile_file.filename}"))
                    tile_file.save(tile_path)
                    teselas_paths.append(tile_path)
            
            if not teselas_paths:
                return jsonify({'error': 'No valid tile images'}), 400
            
            output_path = os.path.join(process_temp_dir, "video_mosaico.mp4")
            procesar_video_mosaico(video_path, teselas_paths, output_path)
            
            return send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name='video_procesado.mp4')
        finally:
            shutil.rmtree(process_temp_dir, ignore_errors=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
