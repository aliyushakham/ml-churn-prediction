"""
Кастомная HTML страница для документации API
Минималистичный темный дизайн
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

def get_custom_docs_html() -> HTMLResponse:
    """Возвращает кастомную HTML страницу с документацией"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Churn Prediction API</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #b0b0b0;
                --accent: #4a9eff;
                --border: #404040;
                --success: #4caf50;
                --warning: #ff9800;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                line-height: 1.6;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 24px;
            }
            
            /* Header */
            .header {
                margin-bottom: 48px;
            }
            
            .header h1 {
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 8px;
                letter-spacing: -0.5px;
            }
            
            .header p {
                color: var(--text-secondary);
                font-size: 1.1rem;
            }
            
            /* Stats grid */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 48px;
            }
            
            .stat-card {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
            }
            
            .stat-label {
                color: var(--text-secondary);
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }
            
            .stat-value {
                font-size: 1.8rem;
                font-weight: 600;
            }
            
            /* Endpoints section */
            .section-title {
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 20px;
                color: var(--text-primary);
            }
            
            .endpoints-grid {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                overflow: hidden;
                margin-bottom: 48px;
            }
            
            .endpoint-row {
                display: flex;
                align-items: center;
                padding: 16px 20px;
                border-bottom: 1px solid var(--border);
            }
            
            .endpoint-row:last-child {
                border-bottom: none;
            }
            
            .method {
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 0.9rem;
                min-width: 60px;
                text-align: center;
                margin-right: 20px;
            }
            
            .method.get {
                background: rgba(74, 158, 255, 0.1);
                color: #4a9eff;
            }
            
            .method.post {
                background: rgba(76, 175, 80, 0.1);
                color: #4caf50;
            }
            
            .path {
                font-family: 'Menlo', 'Monaco', monospace;
                font-size: 1rem;
                flex: 1;
                color: var(--text-primary);
            }
            
            .description {
                color: var(--text-secondary);
                font-size: 0.95rem;
                margin-right: 20px;
            }
            
            .try-link {
                color: var(--accent);
                text-decoration: none;
                font-size: 0.9rem;
                padding: 6px 12px;
                border: 1px solid var(--border);
                border-radius: 6px;
                transition: all 0.2s;
            }
            
            .try-link:hover {
                background: var(--accent);
                color: white;
                border-color: var(--accent);
            }
            
            /* Features grid */
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 48px;
            }
            
            .feature-card {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 24px;
            }
            
            .feature-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 16px;
                color: var(--text-primary);
            }
            
            .feature-list {
                list-style: none;
            }
            
            .feature-list li {
                color: var(--text-secondary);
                margin-bottom: 8px;
                padding-left: 20px;
                position: relative;
            }
            
            .feature-list li::before {
                content: "•";
                color: var(--accent);
                position: absolute;
                left: 0;
            }
            
            /* Footer */
            .footer {
                padding-top: 32px;
                border-top: 1px solid var(--border);
                text-align: center;
                color: var(--text-secondary);
                font-size: 0.9rem;
            }
            
            .footer a {
                color: var(--text-primary);
                text-decoration: none;
                border-bottom: 1px dotted var(--border);
            }
            
            .footer a:hover {
                color: var(--accent);
                border-bottom-color: var(--accent);
            }
            
            /* Quick example */
            .example-box {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 48px;
            }
            
            .example-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 16px;
            }
            
            .code-block {
                background: #0d0d0d;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Menlo', 'Monaco', monospace;
                font-size: 0.9rem;
                color: #e6e6e6;
                overflow-x: auto;
            }
            
            .code-block pre {
                margin: 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>Churn Prediction API</h1>
                <p>ML сервис для предсказания оттока клиентов</p>
            </div>
            
            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Модель</div>
                    <div class="stat-value">XGBoost</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">ROC-AUC</div>
                    <div class="stat-value">0.85</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Признаков</div>
                    <div class="stat-value">25</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Статус</div>
                    <div class="stat-value">Active</div>
                </div>
            </div>
            
            <!-- Endpoints -->
            <h2 class="section-title">Эндпоинты</h2>
            <div class="endpoints-grid">
                <div class="endpoint-row">
                    <span class="method get">GET</span>
                    <span class="path">/</span>
                    <span class="description">Информация о сервисе</span>
                    <a href="/" class="try-link" target="_blank">Попробовать →</a>
                </div>
                
                <div class="endpoint-row">
                    <span class="method get">GET</span>
                    <span class="path">/health</span>
                    <span class="description">Проверка здоровья</span>
                    <a href="/health" class="try-link" target="_blank">Попробовать →</a>
                </div>
                
                <div class="endpoint-row">
                    <span class="method post">POST</span>
                    <span class="path">/predict</span>
                    <span class="description">Предсказание для одного клиента</span>
                    <a href="/docs#/default/predict_predict_post" class="try-link" target="_blank">Документация →</a>
                </div>
                
                <div class="endpoint-row">
                    <span class="method post">POST</span>
                    <span class="path">/predict_batch</span>
                    <span class="description">Пакетное предсказание</span>
                    <a href="/docs#/default/predict_batch_predict_batch_post" class="try-link" target="_blank">Документация →</a>
                </div>
            </div>
            
            <!-- Quick Example -->
            <h2 class="section-title">Пример запроса</h2>
            <div class="example-box">
                <div class="example-title">Python</div>
                <div class="code-block">
                    <pre>import requests

url = "http://localhost:8000/predict"
data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "Contract": "Month-to-month"
}

response = requests.post(url, json=data)
print(response.json())</pre>
                </div>
            </div>
            
            <!-- Features -->
            <h2 class="section-title">О проекте</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-title">📊 Входные данные</div>
                    <ul class="feature-list">
                        <li>Демография (пол, возраст)</li>
                        <li>Услуги (интернет, ТВ)</li>
                        <li>Платежи и контракты</li>
                        <li>Длительность обслуживания</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="feature-title">🎯 Что получаете</div>
                    <ul class="feature-list">
                        <li>Вероятность оттока (0-1)</li>
                        <li>Бинарное предсказание</li>
                        <li>Уровень уверенности</li>
                        <li>Факторы риска</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="feature-title">⚙️ Технологии</div>
                    <ul class="feature-list">
                        <li>FastAPI / Python 3.11</li>
                        <li>XGBoost / scikit-learn</li>
                        <li>Pandas / NumPy</li>
                        <li>Docker (опционально)</li>
                    </ul>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>👩‍💻 Автор: <a href="#">Алия</a> · 
                <a href="https://github.com/aliyushakham/ml-churn-prediction" target="_blank">GitHub</a> · 
                <a href="/docs" target="_blank">Swagger UI</a></p>
                <p style="margin-top: 8px;">© 2026 Churn Prediction API. Все права защищены.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content, status_code=200)