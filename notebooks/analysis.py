import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("АНАЛИЗ ДАННЫХ ОТТОКА КЛИЕНТОВ")
print("=" * 50)

# Загружаем данные
df = pd.read_csv('data/raw/telco_churn.csv')
print(f"\n📊 Размер данных: {df.shape}")
print(f"📋 Колонки: {list(df.columns)}")

# Смотрим на отток
print(f"\n🎯 Отток клиентов:")
print(df['Churn'].value_counts())
print(f"В процентах:")
print(df['Churn'].value_counts(normalize=True) * 100)

# Базовая статистика
print(f"\n📈 Статистика по числовым признакам:")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

# Сравнение ушедших и оставшихся
print(f"\n🔄 Сравнение по контрактам:")
print(pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100)

# Сохраняем простые графики
plt.figure(figsize=(10, 6))
df['Churn'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Распределение оттока клиентов')
plt.xlabel('Churn')
plt.ylabel('Количество')
plt.tight_layout()
plt.savefig('artifacts/metrics/churn_distribution.png')
print("\n✅ График сохранен в artifacts/metrics/churn_distribution.png")