import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
from scipy import stats

def aula1():
    df = pd.read_csv('clientes-v2.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors="coerce")

    print(df.head().to_string())
    print(df.tail().to_string())

    print(df.info())
    print('Dados nulos: \n', df.isnull().sum())
    print('Porcentagem de dados nulos: ', df.isnull().mean() * 100)

    df.dropna(inplace=True)
    print('Dados nulos após remoção: ', df.isnull().sum().sum())

    print('Dados duplicados: ', df.duplicated().sum())
    print('Dados únicos: \n', df.nunique())
    print('Estatística dos dados: \n', df.describe())

    df = df[['idade', 'data', 'estado', 'salario', 'nivel_educacao', 'numero_filhos', 'estado_civil', 'area_atuacao']]

    df.to_csv('clientes_v3.csv')

    print(df.head(30))

def aula2():

    df = pd.read_csv('clientes_v3.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #Using only the columns we'll need
    df = df[['idade', 'salario']]

    #A normalização é utilizada para trazer dados diferentes para a mesma faixa, para que
    #eles possam ser comparados. Por exemplo, idade e salário.

    #Normalização - MinMaxScaler
    #Aqui a normalização é feita com um padrão entre 0 e 1.
    scaler = MinMaxScaler()
    df['idadeScaler'] = scaler.fit_transform(df[['idade']])
    df['salarioScaler'] = scaler.fit_transform(df[['salario']])

    #Podemos mudar o padrão utilizado.
    #É utilizado quando queremos tratar valores negativos.
    min_max = MinMaxScaler(feature_range=(-1, 1))
    df['idadeScaler2'] = scaler.fit_transform(df[['idade']])
    df['salarioScaler2'] = scaler.fit_transform(df[['salario']])

    #Padronização - StandardScaler
    sscaler = StandardScaler()
    df['idadeSScaler'] = sscaler.fit_transform(df[['idade']])
    df['salarioSScaler'] = sscaler.fit_transform(df[['salario']])

    #Padronização - Robust Scaler
    rscaler = RobustScaler()
    df['idadeRScaler'] = rscaler.fit_transform(df[['idade']])
    df['salarioRScaler'] = rscaler.fit_transform(df[['salario']])

    print('Normalização -- MinMaxScaler (0 a 1):')
    print('Idade - Min: {:.4f} Max: {:.4f} Mean: {:.4f} STD: {:.4f}'.format(df['idadeScaler'].min(),
                                                                            df['idadeScaler'].max(),
                                                                            df['idadeScaler'].mean(),
                                                                            df['idadeScaler'].std()))
    print('Salário - Min: {:.4f} Max: {:.4f} Mean: {:.4f} STD: {:.4f}'.format(df['salarioScaler'].min(),
                                                                              df['salarioScaler'].max(),
                                                                              df['salarioScaler'].mean(),
                                                                              df['salarioScaler'].std()))

    print('\nNormalização -- MinMaxScaler (-1 a 1):')
    print('Idade - Min: {:.4f} Max: {:.4f} Mean: {:.4f} STD: {:.4f}'.format(df['idadeScaler2'].min(),
                                                                            df['idadeScaler2'].max(),
                                                                            df['idadeScaler2'].mean(),
                                                                            df['idadeScaler2'].std()))
    print('Salário - Min: {:.4f} Max: {:.4f} Mean: {:.4f} STD: {:.4f}'.format(df['salarioScaler2'].min(),
                                                                              df['salarioScaler2'].max(),
                                                                              df['salarioScaler2'].mean(),
                                                                              df['salarioScaler2'].std()))

    print('\nPadronização -- Standard Scaler (Ajuste a média 0 e o padrão 1):')
    print('Idade - Min: {:.4f} Max: {:.4f} Mean: {:.18f} STD: {:.4f}'.format(df['idadeSScaler'].min(),
                                                                            df['idadeSScaler'].max(),
                                                                            df['idadeSScaler'].mean(),
                                                                            df['idadeSScaler'].std()))
    print('Salário - Min: {:.4f} Max: {:.4f} Mean: {:.18f} STD: {:.4f}'.format(df['salarioSScaler'].min(),
                                                                              df['salarioSScaler'].max(),
                                                                              df['salarioSScaler'].mean(),
                                                                              df['salarioSScaler'].std()))

    print('\nPadronização -- Robust Scaler (Ajuste a mediana e IQR):')
    print('Idade - Min: {:.4f} Max: {:.4f} Mean: {:.4f} STD: {:.4f}'.format(df['idadeRScaler'].min(),
                                                                            df['idadeRScaler'].max(),
                                                                            df['idadeRScaler'].mean(),
                                                                            df['idadeRScaler'].std()))
    print('Salário - Min: {:.4f} Max: {:.4f} Mean: {:.4f} STD: {:.4f}'.format(df['salarioRScaler'].min(),
                                                                              df['salarioRScaler'].max(),
                                                                              df['salarioRScaler'].mean(),
                                                                              df['salarioRScaler'].std()))
    # print(df.head(30))

def aula3():
    df = pd.read_csv('clientes_v3.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #Codificação one-hot para 'estado_civil' -- Transformação do dado para True(1) e False(0)
    df = pd.concat([df, pd.get_dummies(df['estado_civil'], prefix='ES')], axis=1)

    #Codificação ordinal para 'nivel_educacao' -- Transformação da informação string para
    #números representantes criados em um dicionário.
    educacao_ordem = {
        'Ensino Fundamental': 1, 'Ensino Médio': 2, 'Ensino Superior': 3, 'Pós-graduação': 4
    }
    df['nivel_educacao_orig'] = df['nivel_educacao'].map(educacao_ordem)

    #CAT.CODES -- Transformação da string em números, porém não temos mapeados quais números
    #representam quais strings.
    df['area_atuacao_orig'] = df['area_atuacao'].astype('category').cat.codes

    #Label Encoder
    #Converte cada valor ÚNICO em número de 0 a n_classes-1
    label_encoder = LabelEncoder()
    df['estado_cod'] = label_encoder.fit_transform(df['estado'])

    print(df.head(30))

def aula4():
    df = pd.read_csv('clientes_v3.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #Transformação logarítmica -- Quando temos muita diferença entre os valores
    df['salario_log'] = np.log1p(df['salario']) #LOG1P é usado para evitar problemas com valores 0

    #Transformação Box-Cox - Função estatística
    df['salario_boxcox'], _ = stats.boxcox(df['salario'] + 1) #O +1 é utilizado para evitar valores negativos

    #Codificação de Frequência para 'estado' -- Utilizado quando você quer misturar campos
    estado_freq = df['estado'].value_counts() / len(df) #Cálculo: quantas vezes o estado aparece, dividido pelo número de registros.
    df['estado_freq'] = df['estado'].map(estado_freq)

    #Interações -- relação entre campos
    df['interacao_idade_filhos'] = df['idade'] * df['numero_filhos']

    print(df.head(30))

#aula1
#aula2()
#aula3()
#aula4()