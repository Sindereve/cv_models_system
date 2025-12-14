// Добавь этот код в начало app.js после объявления переменных
let uploadedFile = null;
let uploadInProgress = false;

// Инициализация загрузки файлов
function initFileUpload() {
    const fileInput = document.getElementById('dataset_file');
    const uploadArea = document.getElementById('fileUploadArea');
    const fileInfo = document.getElementById('fileInfo');
    
    if (!fileInput || !uploadArea) return;
    
    // Клик по области загрузки
    uploadArea.addEventListener('click', () => {
        if (!uploadInProgress) {
            fileInput.click();
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        if (uploadInProgress) return;
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // Выбор файла через input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// Обработка выбора файла
function handleFileSelect(file) {
    // Проверка типа файла
    const allowedTypes = ['application/zip', 'application/x-zip-compressed', 'application/x-tar', 'application/gzip'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const allowedExtensions = ['zip', 'tar', 'gz'];
    
    if (!allowedExtensions.includes(fileExtension) && !allowedTypes.includes(file.type)) {
        addLog(`❌ Неподдерживаемый формат файла: ${fileExtension}`, 'error');
        showNotification('Поддерживаются только ZIP, TAR, GZ файлы');
        return;
    }
    
    // Проверка размера (максимум 2GB)
    const maxSize = 2 * 1024 * 1024 * 1024; // 2GB в байтах
    if (file.size > maxSize) {
        addLog(`❌ Файл слишком большой: ${formatFileSize(file.size)}`, 'error');
        showNotification('Максимальный размер файла: 2GB');
        return;
    }
    
    uploadedFile = file;
    updateFileInfo(file);
    addLog(`✅ Файл выбран: ${file.name} (${formatFileSize(file.size)})`, 'success');
}

// Обновление информации о файле
function updateFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const uploadProgress = document.getElementById('uploadProgress');
    
    if (fileInfo && fileName && fileSize) {
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        
        if (uploadProgress) {
            uploadProgress.style.display = 'none';
        }
    }
}

// Удаление файла
function removeFile() {
    uploadedFile = null;
    const fileInfo = document.getElementById('fileInfo');
    const fileInput = document.getElementById('dataset_file');
    
    if (fileInfo) {
        fileInfo.style.display = 'none';
    }
    
    if (fileInput) {
        fileInput.value = '';
    }
    
    addLog('🗑️ Файл удален', 'info');
}

// Форматирование размера файла
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Загрузка файла на сервер
async function uploadFile(file) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append('dataset', file);
        formData.append('filename', file.name);
        
        const xhr = new XMLHttpRequest();
        const uploadProgress = document.getElementById('uploadProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        // Показываем прогресс
        if (uploadProgress) {
            uploadProgress.style.display = 'block';
        }
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                
                if (progressFill) {
                    progressFill.style.width = percentComplete + '%';
                }
                
                if (progressText) {
                    progressText.textContent = Math.round(percentComplete) + '%';
                }
                
                addLog(`📤 Загрузка: ${Math.round(percentComplete)}%`, 'info', true);
            }
        });
        
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                addLog(`✅ Файл загружен: ${response.file_path}`, 'success');
                resolve(response.file_path);
            } else {
                reject(new Error(`Ошибка загрузки: ${xhr.statusText}`));
            }
            
            // Скрываем прогресс
            if (uploadProgress) {
                uploadProgress.style.display = 'none';
            }
        });
        
        xhr.addEventListener('error', () => {
            reject(new Error('Ошибка сети при загрузке файла'));
            
            if (uploadProgress) {
                uploadProgress.style.display = 'none';
            }
        });
        
        // Отправляем файл
        xhr.open('POST', `${API_BASE}/upload/dataset`);
        xhr.send(formData);
    });
}

// Обновленная функция startTraining для работы с файлами
async function startTraining(e) {
    e.preventDefault();
    
    const button = document.getElementById('startTrainingBtn');
    if (!button) return;
    
    const originalText = button.textContent;
    
    try {
        button.textContent = '⏳ Подготовка...';
        button.disabled = true;
        
        addLog('📤 Начинаем процесс обучения...', 'info');
        
        let datasetPath = '';
        
        // Если пользователь загрузил файл
        if (uploadedFile) {
            addLog('📦 Загрузка датасета на сервер...', 'info');
            uploadInProgress = true;
            
            try {
                datasetPath = await uploadFile(uploadedFile);
                addLog(`✅ Датасет загружен: ${datasetPath}`, 'success');
            } catch (uploadError) {
                addLog(`❌ Ошибка загрузки файла: ${uploadError.message}`, 'error');
                throw uploadError;
            } finally {
                uploadInProgress = false;
            }
        } else {
            // Используем путь из поля ввода (для совместимости)
            datasetPath = document.getElementById('path_data_dir').value;
            if (!datasetPath) {
                addLog('⚠️ Файл не загружен и путь не указан. Используется значение по умолчанию.', 'warning');
                datasetPath = 'data\\cars';
            }
        }
        
        // Собираем данные из формы
        const config = {
            data_loader_params: {
                path_data_dir: datasetPath, // Используем путь к загруженному файлу
                img_w_size: parseInt(getValue('img_w_size')),
                img_h_size: parseInt(getValue('img_h_size')),
                total_img: parseInt(getValue('total_img')),
                batch_size: parseInt(getValue('batch_size')),
                train_ratio: parseFloat(getValue('train_ratio')),
                val_ratio: parseFloat(getValue('val_ratio')),
                is_calculate_normalize_dataset: isChecked('is_calculate_normalize_dataset')
            },
            trainer_params: {
                loss_fn: getValue('loss_fn'),
                optimizer: getValue('optimizer'),
                scheduler: getValue('scheduler'),
                device: getValue('device'),
                log_mlflow: isChecked('log_mlflow'),
                mlflow_uri: getValue('mlflow_uri'),
                log_artifacts: isChecked('log_artifacts'),
                experiment_name: getValue('experiment_name'),
                run_name: getValue('run_name') || null,
                mlflow_tags: parseJsonField('mlflow_tags'),
                epochs: parseInt(getValue('epochs'))
            },
            model_params: {
                type: getValue('model_type'),
                name: getValue('model_name'),
                weights: isChecked('model_weights'),
                ...parseJsonField('extra_model_params')
            }
        };
        
        addLog('✅ Конфигурация собрана', 'success');
        
        // Валидация
        if (!validateConfig(config)) {
            button.textContent = originalText;
            button.disabled = false;
            return;
        }
        
        addLog('📤 Отправка конфигурации на сервер...', 'info');
        
        // Отправляем на BFF
        const response = await axios.post(`${API_BASE}/train/start`, config);
        
        currentTrainingId = response.data.trainingId;
        
        addLog(`✅ Обучение запущено! ID: ${currentTrainingId}`, 'success');
        
        if (response.data.queuePosition) {
            addLog(`📊 Позиция в очереди: ${response.data.queuePosition}`, 'info');
        }
        
        // Показываем статус
        displayTrainingStatus({
            trainingId: currentTrainingId,
            status: 'queued',
            message: 'Ожидание в очереди'
        });
        
        // Начинаем мониторинг
        startMonitoring(currentTrainingId);
        
    } catch (error) {
        console.error('Failed to start training:', error);
        addLog(`❌ Ошибка запуска обучения: ${error.message}`, 'error');
        showNotification(`Ошибка: ${error.message}`);
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}

// Обновленная функция validateConfig для файлов
function validateConfig(config) {
    const { data_loader_params, trainer_params } = config;
    
    // Если не загружен файл и путь пустой
    if (!uploadedFile && (!data_loader_params.path_data_dir || data_loader_params.path_data_dir.trim() === '')) {
        addLog('❌ Необходимо загрузить датасет или указать путь к данным', 'error');
        showNotification('Загрузите датасет или укажите путь к данным');
        return false;
    }
    
    // Проверка ratios
    const totalRatio = data_loader_params.train_ratio + data_loader_params.val_ratio;
    if (totalRatio > 1) {
        addLog(`❌ Сумма train_ratio (${data_loader_params.train_ratio}) и val_ratio (${data_loader_params.val_ratio}) превышает 1`, 'error');
        showNotification('Сумма Train Ratio и Validation Ratio не должна превышать 1');
        return false;
    }
    
    // Проверка размеров изображений
    if (data_loader_params.img_w_size < 32 || data_loader_params.img_h_size < 32) {
        addLog('❌ Размеры изображения должны быть не менее 32x32', 'error');
        showNotification('Минимальный размер изображения: 32x32');
        return false;
    }
    
    // Проверка MLflow URI
    if (trainer_params.log_mlflow && (!trainer_params.mlflow_uri || trainer_params.mlflow_uri.trim() === '')) {
        addLog('❌ Не указан MLflow URI', 'error');
        showNotification('Укажите адрес MLflow сервера');
        return false;
    }
    
    return true;
}

// Добавляем инициализацию загрузки файлов в DOMContentLoaded
document.addEventListener('DOMContentLoaded', function() {
    // ... существующий код ...
    
    // Инициализация загрузки файлов
    initFileUpload();
    
    // ... остальной код ...
});