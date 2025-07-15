// Функция для обновления метрик
function updateMetrics(metrics) {
    const metricsDiv = document.getElementById('metrics');
    const dataInfoDiv = document.getElementById('data-info');
    const dataTypeP = document.getElementById('data-type');
    const dataSizeP = document.getElementById('data-size');
    
    if (!metricsDiv) return;

    // Обновляем информацию о данных
    if (dataInfoDiv && dataTypeP && dataSizeP) {
        if (metrics.data_type === 'real') {
            dataTypeP.textContent = 'Тип данных: Реальные данные UCI Credit Card';
            dataSizeP.textContent = `Размер выборки: ${metrics.data_size.toLocaleString()} записей`;
            dataInfoDiv.style.display = 'block';
            delete metrics.data_type;
            delete metrics.data_size;
        } else {
            dataInfoDiv.style.display = 'none';
        }
    }

    const metricsHtml = Object.entries(metrics)
        .map(([key, value]) => {
            // Преобразуем ключ в более читаемый формат
            const label = key
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            
            // Форматируем значение как процент
            const formattedValue = (value * 100).toFixed(2);
            
            return `
                <div class="mb-3">
                    <div class="d-flex justify-content-between align-items-center mb-1">
                        <strong>${label}:</strong>
                        <span class="badge bg-primary">${formattedValue}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${formattedValue}%" 
                             aria-valuenow="${formattedValue}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>
            `;
        })
        .join('');

    metricsDiv.innerHTML = metricsHtml;
}

// Функция для обновления визуализации
async function updateVisualization() {
    const vizDiv = document.getElementById('visualization');
    const errorElement = document.getElementById('error-message');
    
    if (!vizDiv) return;
    
    try {
        // Сначала пробуем получить визуализацию
        const response = await fetch(`/api/visualize/${algorithmId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        const data = await response.json();
        
        if (data.error) {
            // Если получили ошибку о необученной модели, пробуем обучить
            if (data.error.includes('модель не обучена')) {
                await trainModel();
                return;
            }
            throw new Error(data.error);
        }

        if (!data.data || !data.layout) {
            throw new Error('Некорректный формат данных визуализации');
        }

        // Настраиваем отзывчивый размер графика
        const layout = {
            ...data.layout,
            autosize: true,
            margin: { l: 50, r: 20, t: 20, b: 50 },
            showlegend: true,
            hovermode: 'closest',
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: '#ffffff'
        };

        // Создаем график
        await Plotly.newPlot('visualization', data.data, layout, {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
                format: 'png',
                filename: 'visualization',
                height: 800,
                width: 1200,
                scale: 2
            }
        });

    } catch (error) {
        console.error('Ошибка визуализации:', error);
        if (errorElement) {
            errorElement.innerHTML = `
                <div class="alert alert-danger">
                    <h5 class="alert-heading">Ошибка визуализации</h5>
                    <p>${error.message}</p>
                    <hr>
                    <p class="mb-0">Попробуйте обновить страницу или выбрать другие параметры визуализации.</p>
                </div>
            `;
            errorElement.style.display = 'block';
        }
    }
}

// Функция для обработки ошибок API
function handleApiError(error, errorElement) {
    console.error('API Error:', error);
    let errorMessage = 'Произошла ошибка';
    
    if (error.response) {
        // Ошибка от сервера
        errorMessage = `Ошибка сервера: ${error.response.status}`;
        if (error.response.data && error.response.data.error) {
            errorMessage = error.response.data.error;
        }
    } else if (error.request) {
        // Ошибка сети
        errorMessage = 'Ошибка сети. Проверьте подключение к интернету.';
    } else {
        // Другие ошибки
        errorMessage = error.message || 'Неизвестная ошибка';
    }
    
    if (errorElement) {
        errorElement.innerHTML = `
            <div class="alert alert-danger">
                <h5 class="alert-heading">Ошибка</h5>
                <p>${errorMessage}</p>
                <hr>
                <p class="mb-0">Попробуйте повторить операцию позже.</p>
            </div>
        `;
        errorElement.style.display = 'block';
    } else {
        alert(errorMessage);
    }
}

// Функция для обучения модели
async function trainModel() {
    const button = document.getElementById('train-button');
    const errorElement = document.getElementById('error-message');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    if (!button) return;
    
    // Отключаем кнопку и показываем индикатор загрузки
    button.disabled = true;
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Обучение...';
    
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
    }
    
    // Скрываем предыдущие ошибки
    if (errorElement) {
        errorElement.style.display = 'none';
    }

    try {
        const response = await fetch(`/api/train/${algorithmId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP ошибка! статус: ${response.status}`);
        }

        if (!data || typeof data !== 'object') {
            throw new Error('Некорректный формат ответа от сервера');
        }

        if (data.error) {
            throw new Error(data.error);
        }

        // Обновляем метрики
        if (data.metrics) {
            updateMetrics(data.metrics);
        }

        // Обновляем визуализацию
        if (data.visualization) {
            await updateVisualization();
        }

    } catch (error) {
        console.error('Ошибка при обучении:', error);
        if (errorElement) {
            errorElement.innerHTML = `
                <div class="alert alert-danger">
                    <h5 class="alert-heading">Ошибка при обучении модели</h5>
                    <p>${error.message}</p>
                    <hr>
                    <p class="mb-0">Попробуйте обновить страницу или выбрать другой алгоритм.</p>
                </div>
            `;
            errorElement.style.display = 'block';
        }
    } finally {
        // Возвращаем кнопку в исходное состояние
        button.disabled = false;
        button.innerHTML = originalText;
        
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }
}

// Инициализация при загрузке страницы
window.addEventListener('load', updateVisualization); 