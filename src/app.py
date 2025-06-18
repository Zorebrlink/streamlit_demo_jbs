import streamlit as st
import pandas as pd
import joblib
import os
import re
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. SETUP E DEFINIÇÃO DE CAMINHOS
# ==============================================================================
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_PATH, 'bases', 'df382.csv')
MODEL_PATH = os.path.join(ROOT_PATH, 'model', 'modelo_flor_solta.joblib')
RESULTS_PATH = os.path.join(ROOT_PATH, 'resultados', 'resultados.txt')
GRAFICOS_PATH = os.path.join(ROOT_PATH, 'graficos')

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Dashboard de Análise de 'Flor Solta' (Semi Acabado 382)</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #0B83F3;'>", unsafe_allow_html=True)

# ==============================================================================
# 2. FEATURES USADAS NO MODELO
# ==============================================================================
num_cols = [
    'cRetrabalho', 'iQtdeApontamentoAtividade', 'nMedidaOrdemProducao', 
    'nTempoMetodo', 'nTempoImprodutivo', 'nTempoRetrabalho', 'nTempoSemMetodo', 
    'nTempoOcioso', 'nQtdeMetodo', 'nPesoLiquidoEntrada_prod', 
    'nPesoLiquidoEntrada_cep', 'nMedidaEntrada', 'nValorArredondado'
]
cat_cols = [
    "nCdProdutoProcesso", "cLinhaProducao", "nCdTurno", "nCdEquipamento", "nCdSetor"
]

# ==============================================================================
# 3. FUNÇÕES DE CACHE E DE UTILIDADE
# ==============================================================================
@st.cache_data
def carregar_dados(caminho):
    try:
        df = pd.read_csv(caminho, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(caminho, sep=';', encoding='latin-1')
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de dados não encontrado em '{caminho}'.")
        return pd.DataFrame()
    # Garante coluna porcentagem para gráficos/simulador
    if 'Qtd_Flor_Solta' in df.columns and 'nPeca' in df.columns and 'porcentagem' not in df.columns:
        df['porcentagem'] = df['Qtd_Flor_Solta'] / df['nPeca']
    return df

@st.cache_resource
def carregar_modelo(caminho):
    try:
        return joblib.load(caminho)
    except FileNotFoundError:
        st.error(f"Erro: Modelo não encontrado em '{caminho}'. Execute o script de treinamento primeiro.")
        return None

@st.cache_data
def carregar_resultados(caminho):
    try:
        with open(caminho, 'r', encoding='latin-1') as f:
            content = f.read()
        metrics = {
            'R2 Score': re.search(r"R2 Score: ([\d.]+)", content).group(1),
            'MAE': re.search(r"Mean Absolute Error \(MAE\): ([\d.]+)", content).group(1),
            'MSE': re.search(r"Mean Squared Error \(MSE\): ([\d.]+)", content).group(1)
        }
        try:
            importances_section = content.split("--- Top 5 Variáveis Mais Importantes ---\n")[1]
            importances_raw = re.findall(r"([\w_]+): ([\d.]+)", importances_section)
        except IndexError:
            st.warning("Seção de importância de variáveis não encontrada no formato esperado em 'resultados.txt'.")
            importances_raw = []
        importances_df = pd.DataFrame(importances_raw, columns=['feature', 'importance'])
        if not importances_df.empty:
            importances_df['importance'] = importances_df['importance'].astype(float)
        return metrics, importances_df
    except (FileNotFoundError, AttributeError, TypeError):
        st.error(f"Erro ao carregar ou processar o arquivo de resultados em '{caminho}'. Execute o script de treinamento.")
        return {}, pd.DataFrame(columns=['feature', 'importance'])

def gerar_insights_dos_pesos(importances_df):
    if importances_df.empty:
        return ["Sem informações de importância das variáveis."]
    insights = []
    for _, row in importances_df.iterrows():
        var = row['feature']
        peso = row['importance']
        efeito = (
            f"A variável **{var}** contribui com {peso:.1%} do poder preditivo do modelo. "
            "Alterações nessa variável têm impacto relevante na previsão da porcentagem de flor solta."
        )
        insights.append(efeito)
    return insights

def remover_outliers(series):
    # Método IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    mask = (series >= (q1 - 1.5 * iqr)) & (series <= (q3 + 1.5 * iqr))
    return mask

# ==============================================================================
# 4. CARREGAMENTO DOS ARQUIVOS
# ==============================================================================
df_principal = carregar_dados(DATA_PATH)
modelo = carregar_modelo(MODEL_PATH)
metrics, importances_df = carregar_resultados(RESULTS_PATH)

# ==============================================================================
# 5. LAYOUT DO DASHBOARD
# ==============================================================================
col1, col2, col3 = st.columns([1.3, 2, 1.3], gap="large")

# ===== COLUNA DA ESQUERDA: INSIGHTS DINÂMICOS =====
with col1:
    with st.container(border=True):
        st.subheader("💡 Insights automáticos sobre a Produção", anchor=False)
        insights_gerados = gerar_insights_dos_pesos(importances_df)
        for ins in insights_gerados:
            st.markdown(f"- {ins}")

# ===== CENTRO: GRÁFICOS (EM ABAS) E SIMULADOR =====
with col2:
    tabs = st.tabs(["🌟 Importância das Variáveis", "🎨 Análise por Variável", "🧪 Simulador de Predição"])

    with tabs[0]:
        st.subheader("Variáveis Mais Relevantes para a Predição", anchor=False, divider='blue')
        if not importances_df.empty:
            fig_imp = px.bar(
                importances_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 5 Variáveis por Importância (do Modelo)',
                text_auto='.3f'
            )
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
            fig_imp.update_traces(marker_color='#0B83F3', textposition='outside')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Não foi possível carregar a importância das variáveis.")

    with tabs[1]:
        st.subheader("Análise Gráfica das Principais Variáveis", anchor=False, divider='blue')
        st.info("Gráficos gerados automaticamente (outliers removidos) para as variáveis mais importantes.")
        if not importances_df.empty and not df_principal.empty:
            for feature_name in importances_df['feature']:
                if feature_name not in df_principal.columns or 'porcentagem' not in df_principal.columns:
                    continue
                # Remove outliers da variável X para o gráfico
                mask = remover_outliers(df_principal[feature_name])
                df_graf = df_principal[mask].copy()
                # Gráfico com matplotlib e exibe direto (sem PNGs salvos)
                fig, ax = plt.subplots(figsize=(7, 4))
                if feature_name in cat_cols:
                    # Boxplot categórica
                    df_graf.boxplot(column='porcentagem', by=feature_name, ax=ax)
                    ax.set_title(f'Boxplot de Porcentagem vs {feature_name}')
                    ax.set_xlabel(feature_name)
                    ax.set_ylabel('porcentagem')
                    plt.suptitle('')
                elif feature_name in num_cols:
                    # Scatterplot numérica
                    ax.scatter(df_graf[feature_name], df_graf['porcentagem'], alpha=0.7)
                    ax.set_title(f'Dispersão de Porcentagem vs {feature_name}')
                    ax.set_xlabel(feature_name)
                    ax.set_ylabel('porcentagem')
                else:
                    plt.close(fig)
                    continue
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Não foi possível carregar os dados para análise gráfica.")

    with tabs[2]:
        st.subheader("Simulador de Predição de Flor Solta", anchor=False, divider='blue')
        with st.container(border=True):
            input_data = {}
            with st.expander("🔢 Preencher Variáveis Numéricas"):
                for col in num_cols:
                    if not df_principal.empty and col in df_principal.columns:
                        default_val = float(df_principal[col].median())
                    else:
                        default_val = 0
                    input_data[col] = st.number_input(f"{col}", value=default_val, format="%.4f", key=f"num_{col}")
            with st.expander("🎛️ Preencher Variáveis Categóricas"):
                for col in cat_cols:
                    if not df_principal.empty and col in df_principal.columns:
                        opcoes = [""] + sorted(df_principal[col].dropna().astype(str).unique().tolist())
                        default_val = df_principal[col].mode(dropna=True).iloc[0]
                        default_index = opcoes.index(str(default_val)) if str(default_val) in opcoes else 0
                    else:
                        opcoes = [""]
                        default_index = 0
                    input_data[col] = st.selectbox(f"{col}", options=opcoes, index=default_index, key=f"cat_{col}")
            if st.button("🚀 Prever Porcentagem de Flor Solta", use_container_width=True, type="primary"):
                if modelo:
                    df_input = pd.DataFrame([input_data])
                    try:
                        df_input = df_input[num_cols + cat_cols]
                        pred = modelo.predict(df_input)[0]
                        st.success("✅ Previsão realizada com sucesso!")
                        st.metric(label="Porcentagem de Flor Solta Prevista", value=f"{pred:.2%}")
                    except Exception as e:
                        st.error(f"❌ Erro ao tentar prever: {e}")
                else:
                    st.error("Modelo não carregado. Não é possível prever.")

# ===== COLUNA DA DIREITA: CARD DE DESEMPENHO DO MODELO =====
with col3:
    with st.container(border=True):
        st.subheader("Desempenho do Modelo", anchor=False)
        if metrics:
            st.metric(label="R² Score", value=f"{metrics.get('R2 Score', 'N/A')}")
            st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics.get('MAE', 'N/A')}")
            st.metric(label="Mean Squared Error (MSE)", value=f"{metrics.get('MSE', 'N/A')}")
            st.caption("Métricas calculadas sobre o conjunto de teste.")
        else:
            st.warning("Métricas não disponíveis.")
