## Importando Bibliotecas
import streamlit as st
import pandas as pd
import category_encoders
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

## Carregando os dados
df = pd.read_csv("Dataset_limpo.csv")

## Dados de entrada
X = df.drop(['Yield'], axis=1)
y = df['Yield']

## Divisão dos dados em treinamento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Separando em variáveis numéricas e categóricas
num_var = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_var = X_train.select_dtypes(include=['object', 'bool']).columns

## Título
st.title(""" 
Predizendo a produtividade da cana-de-açucar
""")

## Descrição
st.write("Os dados foram disponibilizados pelo professor Guilherme Sanchez, condutor do módulo de pandas no curso 'Data Science no Agronégócio', distribuido pela empresa Agroadvance.")
st.write("OBS: Este projeto não tem como objetivo retorno financeiro, apenas aplicar alguns conceitos obtidos durante os estudos em Ciência de Dados")

## Sidebar
### Título sidebar
st.sidebar.title("Valores da amostra")

### Nome do talhão
talhao_input = st.sidebar.text_input("Nome do talhão")

### Número da amostra
amostra_input = st.sidebar.text_input("Número da amostra")

image = Image.open("Cana-de-acucar.jpg")

st.image(image, use_column_width=True)
st.write("Fonte: Cana de açúcar com pendão (Foto: Evandro Marques – www.coisasdaroca.com)")

## Página
def get_data():
    season_options = st.sidebar.selectbox("Season", df['Season'].unique())
    elevation = st.sidebar.slider("Elevation (m)", 0.0, 1000.0, 0.0)
    silt = st.sidebar.slider("Silt", 0.0, 150.0, 0.0)
    p = st.sidebar.slider("P", 0.0, 600.0, 0.0)
    mg = st.sidebar.slider("Mg", 0.0, 25.0, 0.0)
    cec = st.sidebar.slider("CEC", 0.0, 170.0, 0.0)
    bs = st.sidebar.slider("BS", 0.0, 100.0, 0.0)
    b = st.sidebar.slider("B", 0.0, 2.0, 0.0)
    fe = st.sidebar.slider("Fe", 0.0, 150.0, 0.0)
    mn = st.sidebar.slider("Mn", 0.0, 35.0, 0.0)

    data = {
            "Elevation (m)": elevation,
            "Season": season_options,
            "Silt": silt,
            "P": p,
            "Mg": mg,
            "CEC": cec,
            "BS": bs,
            "B": b,
            "Fe": fe,
            "Mn": mn}
    
    features = pd.DataFrame(data, index=[0])
    return features

user_input_var = get_data()

## Gráfico
# graf = st.bar_chart(user_input_var)

st.subheader("Valores da amostra:")
st.write("Talhão:", talhao_input)
st.write("Amostra:", amostra_input)
st.write(user_input_var)

## Treinando o modelo 
### Os parâmetros foram selecionados usando o optuna
params = {
        'n_estimators': 900, 
        'max_features': 'auto', 
        'max_depth': 17, 
        'min_samples_split': 4, 
        'min_samples_leaf': 1, 
        'bootstrap': True,
        'random_state': 42
        }

## Pipeline
preprocessor_final = ColumnTransformer([('num_continuas', StandardScaler(), num_var), 
                                        ('str_categoricas', category_encoders.CatBoostEncoder(), cat_var)])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_final),
    ('model', RandomForestRegressor(**params))])

pipeline.fit(X_train, y_train)

## Predição dados de teste
y_pred = pipeline.predict(X_test)

## Métrica do modelo
r = round(r2_score(y_test, y_pred), 4) #coeficiente de determinação
mae = round(mean_absolute_error(y_test, y_pred), 4) #erro médio absoluto
mse = round(mean_squared_error(y_test, y_pred), 4) #erro médio quadrático
rmse = round(np.sqrt(mse), 4) #raiz do erro médio quadrático

st.subheader("Métricas do modelo nos dados de teste:")
st.write("Coeficiente de determinação - R²(%):", r*100)
st.write("Erro absoluto médio - MAE:", mae)
st.write("Erro quadrático médio - MSE:", mse)
st.write("Raiz do erro quadrático médio - RMSE:", rmse)

## Gráfico feature importance
st.subheader("Importância das características para o modelo:")
feature_importance = pd.DataFrame(pipeline['model'].feature_importances_, index=X_train.columns, columns=['importância']).sort_values('importância', ascending=False)

fig, ax = plt.subplots()
feature_importance.plot.bar(ax=ax)
ax.set_title("Importância das variáveis - até 100%")
ax.set_ylabel("Importância")
st.pyplot(fig)

## Predição dados de entrada
predicao_user = pipeline.predict(user_input_var)


st.subheader("Predição com base nos valores selecionados:")
st.write('Produtividade (t/ha)',predicao_user)

st.write("Produzido por Guilherme P. de O. Ribeiro")
st.write("Linkedin: https://www.linkedin.com/in/guilherme-portela-de-oliveira-ribeiro/")
st.write("Github: https://github.com/GuiPortela02?tab=repositories")