import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
import streamlit as st

def gera_dados() -> pd.DataFrame:
    mu, sigma, n = 10, 1, 100 # mean, standard deviation and number
    np.random.seed(25)
    gramas = np.random.normal(mu, sigma, n)
    cals = gramas + np.random.normal(mu, sigma+.3, n)
    df = pd.DataFrame(data = {
                                'gramas': gramas,
                                'cals': cals
                            }
                    )
    return df

def reta(
        x: float,
        intercept: int = 5,
        slope: int = 5,
        bias: int = 0
    ) -> float:
    y = intercept + x * slope + bias
    return y


def grafico(df: pd.DataFrame, dados_reta: list = [10.6198, 1,]) -> None:
    st.sidebar.markdown(
        "<h3>Selecione Valores da Equação</h3>",
        unsafe_allow_html=True
    )

    use_linha = st.sidebar.checkbox("Mostrar Linha")

    if use_linha:
        intercep = st.sidebar.slider(
                                label='Interceptor',
                                min_value=7.0,
                                max_value=20.0,
                                value=20.0,
                                step=0.01
                            )

        coef = st.sidebar.slider(
                                label='Coeficiente',
                                min_value=0.0,
                                max_value=2.0,
                                value=0.0,
                                step=0.001
                            )

        dados_reta = [intercep, coef,]

    use_erro = st.sidebar.checkbox("Mostrar Erro")
    use_reg = st.sidebar.checkbox("Mostrar Regressão Linear")
    
    if use_reg:
        reg = smf.ols('cals ~ gramas', data = df).fit()
    
    if use_linha:
        df['cals_curva'] = df['gramas'].apply(reta, args=dados_reta)
    
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

    plt.title(f"RMSE manual:    {rmse_manual:,.2f}\nRMSE regressão: {rmse_reg:,.2f}", loc='left')

    plt.xlim([
        df.gramas.min()-1,
        df.gramas.max()+1
    ])
    plt.ylim([
        df.cals.min()-1,
        df.cals.max()+1
    ])
    st.pyplot(fig)

if __name__ == '__main__':

    df = gera_dados()
    grafico(df=df)