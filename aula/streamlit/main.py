import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
import streamlit as st
# import os

def gera_dados() -> pd.DataFrame:
    # st.write(os.listdir("./aula/streamlit/data/"))
    try:
        mensagem = "carregado."
        link = "./aula/streamlit/data/df2.csv"
        df = pd.read_csv(link, sep=';')
    except:
        mensagem = "gerado."
        mu, sigma, n = 10, 1, 100 # mean, standard deviation and number
        np.random.seed(25)
        gramas = np.random.normal(mu, sigma, n)
        cals = gramas + np.random.normal(mu, sigma+.3, n)
        df = pd.DataFrame(data = {
                                    'gramas': gramas,
                                    'cals': cals
                                }
                        )
    finally:
        st.sidebar.markdown(f"##### Banco de dados {mensagem}")

    return df

def reta(
        x: float,
        intercept: int = 5,
        slope: int = 5,
        bias: int = 0
    ) -> float:
    y = intercept + x * slope + bias
    return y


def grafico(df: pd.DataFrame) -> None:

    st.sidebar.markdown(
        "<h3>Selecione Valores da Equação</h3>",
        unsafe_allow_html=True
    )

    use_linha = st.sidebar.checkbox("Mostrar Linha")
    use_erro = st.sidebar.checkbox("Mostrar Erro")
    use_reg = st.sidebar.checkbox("Mostrar Regressão Linear")
    reg = smf.ols('cals ~ gramas', data = df).fit()
    
    intercep = round(df['cals'].mean(), 2)
    max_intercep = round(intercep + intercep*2, 2)
    min_intercep = round(intercep - intercep*2, 2)
    
    coef = round(reg.params.gramas, 2)
    max_coef = round(coef + coef*2, 2)
    min_coef = 0.0

    dados_reta = [intercep, coef,]

    if use_linha:
        intercep = st.sidebar.slider(
                                label='Interceptor',
                                min_value=min_intercep,
                                max_value=max_intercep,
                                value=intercep,
                                step=0.01
                            )

        coef = st.sidebar.slider(
                                label='Coeficiente',
                                min_value=min_coef,
                                max_value=max_coef,
                                value=0.0,
                                step=0.001
                            )

        dados_reta = [intercep, coef,]
    
    if use_linha:
        df['cals_curva'] = df['gramas'].apply(reta, args=dados_reta)
        st.html("<h1>Calorias = Interceptor + Coeficiente Angular * Peso + Variação</h1>")
    
    col1, col2 = st.columns([.7,.3])

    with col1:
        fig = plt.figure(figsize=(10, 10))
        x = 'gramas'
        y = 'cals'
        sns.scatterplot(
            x=x,
            y=y,
            legend=False,
            s=50,
            data=df
        )
        if use_linha:
            plt.plot(df['gramas'], df['cals_curva'], c='r', label='Manual')
        
        x_obs = df['gramas']

        if use_reg:
            y_pred = reg.predict(x_obs)
            plt.plot(x_obs, y_pred, 'g', label='Regressão')
            st.sidebar.write(reg.summary())

        plt.legend(loc="upper left")

        plt.xlabel(x.title())
        plt.ylabel(y.title())

        if use_erro:
            for n in range(10):
                index = df['gramas'].sample(1).index[0]
                x = df['gramas'][index]
                y = df['cals_curva'][index]
                d = df['cals'][index] - y

                plt.arrow(
                    x,
                    y,
                    0,
                    d,
                    head_width=.1,
                    head_length=.50,
                    length_includes_head=True,
                    color='grey',
                )

        rmse_reg = 0
        rmse_manual = 0
        if use_linha:
            rmse_manual = mean_squared_error(df['cals_curva'], df['cals'])
        if use_reg:
            rmse_reg = mean_squared_error(y_pred, df['cals'])

        plt.title(f"Distribuição de calorias por grama de Strogonof", loc='left')

        plt.xlim([
            df.gramas.min()-1,
            df.gramas.max()+1
        ])
        plt.ylim([
            df.cals.min()-1,
            df.cals.max()+1
        ])
        st.pyplot(fig)
    
    with col2:
        if use_linha:
            st.markdown(f"## y = {dados_reta[0]} + x*{dados_reta[1]}")
            st.markdown(f"## RMSE manual:    {rmse_manual:,.2f}")
        
        if use_reg:
            st.markdown(f"## RMSE regressão: {rmse_reg:,.2f}")

if __name__ == '__main__':

    st.set_page_config(layout="wide")
    df = gera_dados()
    grafico(df=df)