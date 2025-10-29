import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

DATA_DIR = "./data"

def removes(files: list[str]) -> None:
    """Удаляем ненужные файлы из скаченных файлов"""
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🟡 Remove file: {file_path}")
        else:  
            print(f"🟡 File not found: {file_path}")

def download_PetImages_CatVsDog():
    try: 
        print('⚪ Start download dataset CatsVsDogs')
        name_dataset = 'shaunthesheep/microsoft-catsvsdogs-dataset'
        api.dataset_download_files(
            dataset=name_dataset, 
            path=DATA_DIR, 
            unzip=True,
            quiet=False
        )
        filename = ['MSR-LA - 3467.docx', 'readme[1].txt']
        removes(filename)
        print('🟢 End download')
    except Exception as e:
        print(f'🔴 Error download:{e}')

if __name__ == '__main__':
    download_PetImages_CatVsDog()