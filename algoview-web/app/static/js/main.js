// Функция для отображения индикатора загрузки
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('loading');
    }
}

// Функция для скрытия индикатора загрузки
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('loading');
    }
}

// Функция для отображения ошибки
function showError(message) {
    // Создаем элемент для отображения ошибки
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.role = 'alert';
    errorDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Добавляем сообщение об ошибке в начало контейнера
    const container = document.querySelector('.container');
    container.insertBefore(errorDiv, container.firstChild);
    
    // Автоматически скрываем сообщение через 5 секунд
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Функция для форматирования чисел
function formatNumber(number) {
    return new Intl.NumberFormat('ru-RU').format(number);
}

// Функция для форматирования процентов
function formatPercent(number) {
    return new Intl.NumberFormat('ru-RU', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(number);
}

// Обработчик ошибок fetch
function handleFetchError(error) {
    console.error('Error:', error);
    showError('Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже.');
}

// Функция для проверки ответа сервера
function checkResponse(response) {
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

// Функция для обновления URL с параметрами
function updateUrlParams(params) {
    const url = new URL(window.location.href);
    Object.keys(params).forEach(key => {
        url.searchParams.set(key, params[key]);
    });
    window.history.pushState({}, '', url);
}

// Функция для получения параметров из URL
function getUrlParams() {
    const params = {};
    new URLSearchParams(window.location.search).forEach((value, key) => {
        params[key] = value;
    });
    return params;
}

// Инициализация всплывающих подсказок Bootstrap
document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}); 