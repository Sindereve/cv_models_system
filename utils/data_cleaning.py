import os
from PIL import Image, ImageFile, UnidentifiedImageError
from pathlib import Path
from tqdm import tqdm
import warnings

# Разрешаем загрузку частично поврежденных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Подавляем только предупреждения о truncated TIFF файлах
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

def auto_clean_dataset(data_dir: str):
    """
    """
    print(f"⚪[auto_clean_dataset] Start")
    
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'deleted_files': 0,
    }
    
    # Проходим по всем файлам рекурсивно
    for root, dirs, files in os.walk(data_dir):
        
        print(f"📁 Root:{root}")
        if not files:
            print(f" ➖ No files")
            continue

        for filename in tqdm(files):
            file_path = os.path.join(root, filename)
            stats['total_files'] += 1
            
            # Проверяем только изображения
            if Path(filename).suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Проверяем валидность изображения
            if is_valid_image(file_path):
                stats['valid_files'] += 1
            else:
                os.remove(file_path)
                stats['deleted_files'] += 1
    
    # Вывод результатов
    print("🟢[auto_clean_dataset] Finish")
    print(f" ➖ All file: {stats['total_files']}")
    print(f" ➖ Good file: {stats['valid_files']}")
    print(f" ➖ Count deleted: {stats['deleted_files']}")
    
    return stats

def is_valid_image(file_path: str) -> bool:
    """
    Проверяет, является ли файл валидным изображением
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False