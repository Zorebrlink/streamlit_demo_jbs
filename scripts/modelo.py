# ==============================================================================
# 1. SETUP
# ==============================================================================
import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Usaremos o CountEncoder da biblioteca category_encoders
# Se não a tiver instalada, use: pip install category_encoders
from category_encoders import CountEncoder

# Definição de caminhos para salvar os artefatos
GRAFICOS_PATH = '../../graficos/'
RESULTADOS_PATH = '../../resultados/'
MODELO_PATH = '../'

# Cria os diretórios se eles não existirem
os.makedirs(GRAFICOS_PATH, exist_ok=True)
os.makedirs(RESULTADOS_PATH, exist_ok=True)
os.makedirs(MODELO_PATH, exist_ok=True)

# ==============================================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
# Carrega o dataset
df = pd.read_csv('../../Bases/Gold/df382.csv', sep=';')

# Cria a variável alvo 'porcentagem'
df['porcentagem'] = df['Qtd_Flor_Solta'] / df['nPeca']

# Lista de colunas a serem removidas
drop_cols = [
    'nCdOrdemProducao', 'nPeca', 'Qtd_Flor_Solta', 'nCdProdutoFinal', 
    'nCdProdutoPedido', 'cCdRastreabilidade', 'nCdSeqProdutivaFormula',
    'nCdSeqProdutivaRoteiro', 'nCdLinhaProducao'
]
df = df.drop(columns=drop_cols)

# ==============================================================================
# 3. DEFINIÇÃO DE FEATURES E PRÉ-PROCESSAMENTO
# ==============================================================================
# Definição das colunas numéricas e categóricas
num_cols = [
    'cRetrabalho', 'iQtdeApontamentoAtividade', 'nMedidaOrdemProducao', 
    'nTempoMetodo', 'nTempoImprodutivo', 'nTempoRetrabalho', 'nTempoSemMetodo', 
    'nTempoOcioso', 'nQtdeMetodo', 'nPesoLiquidoEntrada_prod', 
    'nPesoLiquidoEntrada_cep', 'nMedidaEntrada', 'nValorArredondado',
]

cat_cols = [
    "nCdProdutoProcesso", "cLinhaProducao", "nCdTurno", 
    "nCdEquipamento", "nCdSetor"
]

# Seleciona apenas as colunas necessárias e remove duplicatas
df = df[num_cols + cat_cols + ['porcentagem']].drop_duplicates()

# Separação das variáveis preditoras (X) e da variável alvo (y)
X = df.drop(columns=['porcentagem'])
y = df['porcentagem']

# Pipeline de pré-processamento para variáveis categóricas
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value="Na")),
    ('freq_enc', CountEncoder(normalize=True))
])

# Pipeline de pré-processamento para variáveis numéricas
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
    ('scaler', StandardScaler())
])

# ColumnTransformer para aplicar os pipelines às colunas corretas
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols),
    ('num', num_transformer, num_cols)
])

# ==============================================================================
# 4. TREINAMENTO DO MODELO
# ==============================================================================
# Criação do pipeline final com pré-processamento e o regressor
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model.fit(X_train, y_train)

# ==============================================================================
# 5. AVALIAÇÃO DO MODELO
# ==============================================================================
# Previsões com os dados de teste
y_pred = model.predict(X_test)

# Cálculo das métricas de avaliação
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("--- Métricas de Avaliação ---")
print(f'R2 Score: {r2:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')

# ==============================================================================
# 6. ANÁLISE DE FEATURE IMPORTANCE
# ==============================================================================
# Extração das importâncias do modelo treinado
importances = model.named_steps['regressor'].feature_importances_

# Nomes das features na ordem correta após o pré-processamento
feature_names = cat_cols + num_cols

# Cria um DataFrame para facilitar a visualização
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Seleciona as 5 variáveis mais importantes
top5_features_df = feature_importance_df.head(5)

print("\n--- Top 5 Variáveis Mais Importantes ---")
print(top5_features_df)

# ==============================================================================
# 7. VISUALIZAÇÃO DE DADOS (PLOTLY)
# ==============================================================================
print("\nGerando e salvando gráficos para as top 5 variáveis...")

# Adiciona a coluna 'porcentagem' de volta para a plotagem
df_plot = X.join(y)

for index, row in top5_features_df.iterrows():
    feature_name = row['feature']
    
    if feature_name in cat_cols:
        # Gráfico de Boxplot para variáveis categóricas
        fig = px.box(
            df_plot, 
            x=feature_name, 
            y='porcentagem',
            title=f'Boxplot de Porcentagem vs {feature_name}',
            color=feature_name
        )
        file_name = f"{GRAFICOS_PATH}{feature_name}_boxplot.png"
        
    elif feature_name in num_cols:
        # Gráfico de Dispersão para variáveis numéricas
        fig = px.scatter(
            df_plot, 
            x=feature_name, 
            y='porcentagem',
            title=f'Dispersão de Porcentagem vs {feature_name}',
            trendline='ols' # Adiciona uma linha de tendência
        )
        file_name = f"{GRAFICOS_PATH}{feature_name}_scatter.png"
        
    # Salva o gráfico
    fig.write_image(file_name, width=800, height=600)
    print(f"Gráfico salvo em: {file_name}")

# ==============================================================================
# 8. SALVAMENTO DOS RESULTADOS
# ==============================================================================
# Salva o modelo treinado
model_file = f'{MODELO_PATH}modelo_flor_solta.joblib'
joblib.dump(model, model_file)
print(f"\nModelo salvo em: {model_file}")

# Salva as métricas e a importância das variáveis em um arquivo de texto
results_file = f'{RESULTADOS_PATH}resultados.txt'
with open(results_file, 'w') as f:
    f.write("--- Métricas de Avaliação do Modelo ---\n")
    f.write(f'R2 Score: {r2:.4f}\n')
    f.write(f'Mean Absolute Error (MAE): {mae:.4f}\n')
    f.write(f'Mean Squared Error (MSE): {mse:.4f}\n\n')
    
    f.write("--- Top 5 Variáveis Mais Importantes ---\n")
    for index, row in top5_features_df.iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")

print(f"Resultados salvos em: {results_file}")