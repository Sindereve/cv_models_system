from pathlib import Path
from shared.logging import get_logger

logger = get_logger(__name__)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

class ImageDataset():
    def __init__(
            self,
            base_path: str,
        ):
        self.base_path = Path("datasets/"+base_path)

        self.classes = self._load_info_dataset()

    def _is_image(self, file: str):
        return file.split('.')[-1].lower() in ALLOWED_EXTENSIONS    
    
    def _load_info_dataset(self):
        """
            Загрузка данных о классе
        """
        logger.debug(f'🔘 Start load info dataset')
        classes = {}
        for dir_path in self.base_path.iterdir():
            if dir_path.is_dir():
                class_name = str(dir_path).split('\\')[-1]
                logger.debug(f'| New class: {class_name}')

                class_files = []
                class_image = []
                for class_file in dir_path.iterdir():
                    if class_file.is_file():
                        if self._is_image(str(class_file)):
                            class_image.append(class_file) 
                        else:
                            class_files.append(class_file)
                        
                classes[class_name] = {
                    'count_file': class_files,
                    'count_image': class_image
                }
                logger.debug(f'| Class {class_name} ready')
        logger.info(f'Info for dataset is ready!')
        return classes
