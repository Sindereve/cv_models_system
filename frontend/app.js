// ==========================
// CONFIG
// ==========================
const API_BASE_URL = "/api";

let uploadedDatasetPath = null;
let uploadFile = null;
let allTasks = [];
let statusInterval = null;

// ==========================
// AXIOS INSTANCE
// ==========================
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000
});

// ==========================
// LOGGER
// ==========================
function addLog(message, type = "info") {
    const logs = document.getElementById("trainingLogs");
    const div = document.createElement("div");
    div.className = `log-entry ${type}`;
    div.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logs.appendChild(div);
    logs.scrollTop = logs.scrollHeight;
}

function clearLogs() {
    document.getElementById("trainingLogs").innerHTML = "";
    addLog("Логи очищены", "info");
}

// ==========================
// FILE UPLOAD (DRAG & DROP)
// ==========================
const uploadArea = document.getElementById("fileUploadArea");
const fileInput = document.getElementById("dataset_file");

uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file) return;

    uploadFile = file;

    document.getElementById("fileInfo").style.display = "block";
    document.getElementById("fileName").textContent = file.name;
    document.getElementById("fileSize").textContent =
        (file.size / 1024 / 1024).toFixed(2) + " MB";

    uploadDataset();
}

function removeFile() {
    uploadFile = null;
    uploadedDatasetPath = null;
    document.getElementById("fileInfo").style.display = "none";
    addLog("Файл удалён", "warning");
}

// ==========================
// UPLOAD DATASET
// ==========================
async function uploadDataset() {
    if (!uploadFile) return;

    const formData = new FormData();
    formData.append("dataset", uploadFile);

    const progressBar = document.getElementById("uploadProgress");
    const progressFill = document.getElementById("progressFill");
    const progressText = document.getElementById("progressText");

    progressBar.style.display = "block";
    addLog(`Загрузка датасета: ${uploadFile.name}`, "info");

    try {
        const response = await api.post("/upload_dataset", formData, {
            headers: { "Content-Type": "multipart/form-data" },
            onUploadProgress: (progressEvent) => {
                const percent = Math.round(
                    (progressEvent.loaded * 100) / progressEvent.total
                );
                progressFill.style.width = percent + "%";
                progressText.textContent = percent + "%";
            }
        });

        uploadedDatasetPath = response.data.path;
        addLog("Датасет успешно загружен", "success");
    } catch (err) {
        addLog("Ошибка загрузки датасета", "error");
        console.error(err);
    }
}

// ==========================
// FORM SUBMIT (CREATE TASK)
// ==========================
document.getElementById("trainingForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!uploadedDatasetPath) {
        addLog("Сначала загрузите датасет", "warning");
        return;
    }

    const config = collectTrainingConfig();

    addLog("Отправка задачи на обучение...", "info");

    try {
        const response = await api.post("/new_task", config);
        const newTask = response.data;
        addLog(`Задача создана: ${newTask.task_id}`, "success");

        fetchAllTasks();
    } catch (err) {
        addLog("Ошибка создания задачи", "error");
        console.error(err);
    }
});

// ==========================
// COLLECT CONFIG
// ==========================
function collectTrainingConfig() {
    const schedulerValue = document.getElementById("scheduler").value || null;
    const deviceValue = document.getElementById("device").value || null;

    return {
        data_loader_params: {
            path_data_dir: uploadedDatasetPath,
            img_w_size: Number(document.getElementById("img_w_size").value),
            img_h_size: Number(document.getElementById("img_h_size").value),
            total_img: Number(document.getElementById("total_img").value),
            batch_size: Number(document.getElementById("batch_size").value),
            train_ratio: Number(document.getElementById("train_ratio").value),
            val_ratio: Number(document.getElementById("val_ratio").value),
            is_calculate_normalize_dataset:
                document.getElementById("is_calculate_normalize_dataset").checked
        },
        model_params: {
            type: document.getElementById("model_type").value,
            name: document.getElementById("model_name").value,
            weights: document.getElementById("model_weights").checked
        },
        trainer_params: {
            loss_fn_config: {
                type: document.getElementById("loss_fn").value,
                params: {}
            },
            optimizer_config: {
                type: document.getElementById("optimizer").value,
                params: {}
            },
            ...(schedulerValue && { scheduler_config: { type: schedulerValue, params: {} } }),
            device: deviceValue,
            epochs: Number(document.getElementById("epochs").value),
            log_mlflow: document.getElementById("log_mlflow").checked,
            log_artifacts: document.getElementById("log_artifacts").checked,
            experiment_name: document.getElementById("experiment_name").value,
            run_name: document.getElementById("run_name").value || null,
            mlflow_uri: document.getElementById("mlflow_uri").value
        }
    };
}

// ==========================
// TASKS MANAGEMENT
// ==========================
async function fetchAllTasks() {
    try {
        const response = await api.get("/tasks");
        allTasks = response.data.tasks || [];
        renderTasks(allTasks);
    } catch (err) {
        console.error(err);
        addLog("Не удалось получить список задач", "error");
    }
}

function renderTasks(tasks) {
    const container = document.getElementById("trainingStatus");
    container.innerHTML = "";

    if (!tasks || tasks.length === 0) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "Нет активных обучений";
        container.appendChild(empty);
        return;
    }

    // Ограничиваем до последних 7 задач
    const lastTasks = tasks.slice(-7);

    lastTasks.forEach(task => {
        const taskCard = document.createElement("div");
        taskCard.className = "task-status-card";

        const statusClass = task.status.toLowerCase();

        // Достаем эпохи и модель из config
        const epochs = task.config?.trainer_params?.epochs ?? "-";
        const modelName = task.config?.model_params?.name ?? "-";
        const modelType = task.config?.model_params?.type ?? "-";

        taskCard.innerHTML = `
            <div class="task-header">
                <h3>${task.task_id}</h3>
                <span class="task-status-badge ${statusClass}">${task.status}</span>
            </div>
            <div class="task-details">
                <p><strong>Модель:</strong> ${modelType} (${modelName})</p>
                <p><strong>Epochs:</strong> ${epochs}</p>
                <p><strong>Device:</strong> ${task.config?.trainer_params?.device ?? "auto"}</p>
                <p><strong>Queue:</strong> ${task.queue ?? "-"}</p>
            </div>
        `;

        // Кнопка удаления
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn-small btn-danger";
        deleteBtn.textContent = "Удалить";
        deleteBtn.addEventListener("click", () => deleteTask(task.task_id));
        taskCard.appendChild(deleteBtn);

        container.appendChild(taskCard);
    });
}


async function deleteTask(taskId) {
    if (!confirm(`Вы уверены, что хотите удалить задачу ${taskId}?`)) return;

    try {
        const response = await api.delete(`/task/${taskId}`);
        addLog(response.data.message, "success");

        fetchAllTasks(); // обновляем список после удаления
    } catch (err) {
        console.error(err);
        addLog(`Ошибка при удалении задачи ${taskId}`, "error");
    }
}

// ==========================
// START POLLING
// ==========================
function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    fetchAllTasks();
    statusInterval = setInterval(fetchAllTasks, 5000);
}

// ==========================
// HEALTHCHECK
// ==========================
async function refreshStatus() {
    try {
        await api.get("/health");
        setStatus("apiStatus", true);
        setStatus("redisStatus", true);
        document.getElementById("systemStatus").textContent = "Система работает";
    } catch {
        setStatus("apiStatus", false);
        setStatus("redisStatus", false);
        document.getElementById("systemStatus").textContent = "Ошибка соединения";
    }

    document.getElementById("lastUpdate").textContent =
        "Обновлено: " + new Date().toLocaleTimeString();
}

function setStatus(id, ok) {
    const el = document.getElementById(id);
    el.textContent = ok ? "✓ Подключен" : "✗ Ошибка";
    el.className = "status-text " + (ok ? "healthy" : "unhealthy");
}

// ==========================
// MLflow CHECK
// ==========================
document.getElementById("checkMlflowBtn").addEventListener("click", async () => {
    const uri = document.getElementById("mlflow_uri").value;
    const statusEl = document.getElementById("mlflowStatus");
    statusEl.textContent = "Проверка...";
    statusEl.classList.remove("healthy", "unhealthy");

    try {
        await axios.get(`${uri}/api/2.0/preview/mlflow/experiments/list`);
        statusEl.textContent = "✓ Подключен";
        statusEl.classList.add("healthy");
    } catch (err) {
        statusEl.textContent = "✗ Не подключен";
        statusEl.classList.add("unhealthy");
    }
});

// ==========================
// INIT
// ==========================
addLog("Frontend инициализирован", "info");
refreshStatus();
startStatusPolling();
