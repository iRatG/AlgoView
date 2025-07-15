from flask import Blueprint, render_template, request, jsonify
import os
import pandas as pd
import logging

bp = Blueprint('datasets', __name__)
logger = logging.getLogger(__name__)

@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    """Страница загрузки данных"""
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'Файл не найден'}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Файл не выбран'}), 400
                
            if file and file.filename.endswith('.csv'):
                # Сохраняем файл
                upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
                if not os.path.exists(upload_dir):
                    os.makedirs(upload_dir)
                    
                filepath = os.path.join(upload_dir, 'UCI_Credit_Card.csv')
                file.save(filepath)
                
                # Проверяем, что файл читается
                try:
                    df = pd.read_csv(filepath)
                    logger.info(f"Файл успешно загружен и прочитан. Размер: {df.shape}")
                    return jsonify({'success': True, 'message': 'Файл успешно загружен'})
                except Exception as e:
                    logger.error(f"Ошибка при чтении файла: {str(e)}")
                    return jsonify({'error': 'Ошибка при чтении файла'}), 400
            else:
                return jsonify({'error': 'Разрешены только CSV файлы'}), 400
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {str(e)}")
            return jsonify({'error': 'Ошибка при загрузке файла'}), 500
            
    return render_template('datasets/upload.html')

@bp.route('/list')
def list_datasets():
    """Страница со списком загруженных датасетов"""
    datasets = []
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    
    if os.path.exists(upload_folder):
        for filename in os.listdir(upload_folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(upload_folder, filename)
                file_stats = os.stat(filepath)
                
                # Получаем информацию о файле
                dataset_info = {
                    'filename': filename,
                    'size': file_stats.st_size,
                    'created': datetime.fromtimestamp(file_stats.st_ctime),
                    'preview': get_dataset_preview(filepath)
                }
                datasets.append(dataset_info)
    
    return render_template('datasets/list.html', datasets=datasets)

@bp.route('/download/<filename>')
def download_dataset(filename):
    """Скачивание датасета"""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename) 