import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder

def EDA():
    #Exploratory data analysis

    df = pd.read_csv('clientes-v2.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #Renaming columns
    df = df.rename(columns={'nome': 'NOME', 'cpf': 'CPF', 'idade': 'IDADE', 'data': 'DATA', 'endereco': 'ENDEREÇO',
                        'estado': 'ESTADO', 'pais': 'PAÍS', 'salario': 'SALÁRIO', 'nivel_educacao': 'NÍVEL EDUCACIONAL',
                       'numero_filhos': 'NÚMERO DE FILHOS', 'estado_civil': 'ESTADO CIVIL', 'anos_experiencia': 'ANOS DE EXPERIÊNCIA', 'area_atuacao': 'ÁREA DE ATUAÇÃO'})

    #Using head and tail to ensure the data is the same
    print(df.head())
    print(df.tail())

    print('\nNúmero de linhas: ', df.shape)
    print('\nDados nulos:', df.isnull().sum().sum())
    print('\nDados duplicados:', df.duplicated().sum())
    print('\nDados únicos:\n', df.nunique())
    print('\nEstatística dos dados:\n', df.describe())

    df.to_csv('clientes-v3.csv')

def CLEAN():
    #Data cleaning

    df = pd.read_csv('clientes-v3.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Removing null data
    df.dropna(inplace=True)
    print('Dados nulos: ', df.isnull().sum().sum())

    #Changing date format
    df['DATA'] = pd.to_datetime(df['DATA'], format='%d/%m%Y', errors='coerce')

    #Removing user identifying columns
    df = df[['IDADE', 'DATA', 'ESTADO', 'SALÁRIO', 'NÍVEL EDUCACIONAL', 'NÚMERO DE FILHOS', 'ESTADO CIVIL', 'ANOS DE EXPERIÊNCIA', 'ÁREA DE ATUAÇÃO']]

    print(df.head(5))

    df.to_csv('clientes-v4.csv')

def PREPARATION():
    df = pd.read_csv('clientes-v4.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #Normalizing data from IDADE and SALARIO using MixMaxScaler
    scaler = MinMaxScaler()
    df['IDADE'] = scaler.fit_transform(df[['IDADE']])

    print(df.head(5))

#EDA()
#CLEAN()
PREPARATION()



