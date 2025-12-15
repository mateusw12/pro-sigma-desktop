Vamos fazer a ferramenta para analise de custo de garantia.
teremso que selecionar a coluna de quantidade de produto, periodo de tempo, quantidade de falha, datas de falha e agrupamento
Unica seleção terá varias colunas é o datas de falha

meu backend reaproveitavel

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.common.common import remove_punctuation
from utils.warranty_costs.calculate import (
    add_predicted_reliability,
    create_data_frames_list,
    create_linearization_chart,
    create_mountain_chart,
    create_parameters,
    create_probability_chart,
    create_probability_data_frame,
    create_reliability_chart,
    split_dataframe_by_group,
)

warranty_costs_router = APIRouter()


class WarrantyCosts(BaseModel):
    inputData: object
    failureCount: list[str]
    groupingColumn: str
    productionCountColumn: str
    timeStampColumn: str


@warranty_costs_router.post(
    "/calculate",
    tags=["Custo de Garantia"],
    description=""" Calcula Custo da garantia
            rounds: numero de rodadas;
            factors: objeto das caracteristicas
            """,
    response_model=object,
)
def calculate_warranty_costs(body: WarrantyCosts):
    try:
        input_data = body.inputData
        failure_count = body.failureCount
        grouping_column = body.groupingColumn
        production_count_column = body.productionCountColumn
        time_stamp_column = body.timeStampColumn

        original_df = pd.DataFrame(input_data)
        original_df = remove_punctuation(original_df)

        if grouping_column == "":
            original_df["Grouping"] = 1
            grouping_column = "Grouping"

        list_dfs = split_dataframe_by_group(original_df, grouping_column)

        warranty: dict[str, object] = {}

        for _, df in enumerate(list_dfs):

            key = str(df[grouping_column].iloc[0])

            dfs = create_data_frames_list(
                failure_count, production_count_column, time_stamp_column, df
            )

            # soma falhas
            sums_failure_columns = {col: df[col].sum() for col in failure_count}

            prob_df = create_probability_data_frame(
                production_count_column, df, sums_failure_columns
            )

            # Remover valores None ou NaN para ajuste
            beta, alpha, mttf, ci_min, ci_max = create_parameters(prob_df)

            # Cria coluna de f(T) predito e confiabilidade
            prob_df = add_predicted_reliability(prob_df, alpha, beta, ci_min, ci_max)

            # Cria gráfico de probabilidade
            mountain_chart = create_mountain_chart(dfs)

            # Cria gráfico de linearização
            linearization_chart = create_linearization_chart(prob_df)

            # Cria gráfico de probabilidade
            probability_chart, probability_chart_min, probability_chart_max = (
                create_probability_chart(prob_df)
            )

            # Cria gráfcio de confiabilidade
            reliability_chart, reliability_chart_min, reliability_chart_max = (
                create_reliability_chart(prob_df)
            )

            warranty[key] = {
                "alpha": alpha,
                "beta": beta,
                "mttf": mttf,
                "betaLower": ci_min,
                "betaUpper": ci_max,
                "totalSales": float(df[production_count_column].sum()),
                "mountain": mountain_chart,
                "linearization": linearization_chart,
                "probability": probability_chart,
                "reliability": reliability_chart,
                "probabilityLower": probability_chart_min,
                "probabilityUpper": probability_chart_max,
                "reliabilityLower": reliability_chart_min,
                "reliabilityUpper": reliability_chart_max,
            }

        return {"result": warranty}

    except Exception as e:
        print("e", e)
        raise HTTPException(status_code=500, detail="warrantyError")


import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def create_parameters(df: pd.DataFrame):
    """
    Realiza uma regressão linear para estimar os parâmetros beta e alpha 
    de um modelo de confiabilidade, além de calcular o tempo médio até a falha (MTTF)
    e o intervalo de confiança de beta.
    
    df : pd.DataFrame
        DataFrame contendo os dados com as colunas "ln(f(t))" e "ln(1-f(t))",
        que representam os valores transformados das taxas de falha.
        
    - A regressão é feita utilizando `ln(f(t))` como variável independente (X) 
      e `ln(1-f(t))` como variável dependente (Y).
    - O erro padrão de beta é obtido a partir da matriz de variância-covariância 
      do modelo de regressão usando `statsmodels`.
    - O intervalo de confiança de 95% para beta é calculado como:
      `beta ± (erro padrão * 1.96)`, assumindo uma distribuição normal dos coeficientes.
    """
    prob_df_clean = df.dropna()

    # Definir X e Y para a regressão
    X = np.array(prob_df_clean["ln(f(t))"]).reshape(-1, 1)
    Y = np.array(prob_df_clean["ln(1-f(t))"])

    model = LinearRegression()
    model.fit(X, Y)

    beta = model.coef_[0]  
    alpha = math.exp(abs(model.intercept_) / beta)
    mttf = alpha * math.gamma(1 + 1 / beta)

    # Cálculo do erro padrão de beta
    X_with_const = sm.add_constant(X)  
    ols_model = sm.OLS(Y, X_with_const).fit()
    beta_std_error = ols_model.bse[1]  

    ci_min = beta - beta_std_error * 1.96
    ci_max = beta + beta_std_error * 1.96

    return beta, alpha, mttf, ci_min, ci_max

def create_probability_data_frame(production_count_column: str, df: pd.DataFrame, sums_failure_columns: dict):
    """
    A função calcula a fração acumulada de falhas ao longo do tempo (`f(t)`) e transforma esses valores 
    utilizando logaritmos naturais para análise de confiabilidade. O cálculo de `f(t)` é realizado 
    dividindo o número acumulado de falhas pelo total de produção acumulado. Caso `f(t)` esteja dentro do 
    intervalo (0,1), o valor `ln(1 - f(t))` também é calculado. Os valores `ln(f(t))` e `ln(t)` são 
    utilizados para modelagem estatística de confiabilidade.

    O DataFrame resultante contém as seguintes colunas:
    - `"f(t)"`: Fração acumulada de falhas ao longo do tempo.
    - `"mes"`: Período (exemplo: meses, ciclos, etc.).
    - `"ln(1-f(t))"`: Transformação logarítmica da fração de falhas para análise estatística.
    - `"ln(f(t))"`: Logaritmo natural do tempo para modelagem.
    Parâmetros:
    -----------
    production_count_column : str
        Nome da coluna no DataFrame original que contém a contagem de produção em cada período.

    df : pd.DataFrame
        DataFrame contendo os dados de produção e falhas.

    sums_failure_columns : dict
        Dicionário onde as chaves representam os períodos (por exemplo, meses) e os valores correspondem 
        ao número de falhas observadas em cada período.
    """
    
    ft_values = []
    months = []
    log_values = []
    log_t = []

    total_failure_values = list(sums_failure_columns.values())

    cumulative_failures = 0 
    total_production = 0  

    for i, total_failures in enumerate(total_failure_values, start=1):
        cumulative_failures += total_failures 
            
        total_production = df[production_count_column].iloc[:i].sum()

        ft = cumulative_failures / total_production if total_production > 0 else 0

        if 0 < ft < 1:
            log_value = math.log(-1 * math.log(1 - ft))
        else:
            log_value = None           
                
        log_i = math.log(i)

        ft_values.append(ft)
        months.append(i)
        log_values.append(log_value)
        log_t.append(log_i)

        # Criar DataFrame final
    prob_df = pd.DataFrame({
            "f(t)": ft_values,
            "mes": months,
            "ln(1-f(t))": log_values,
            "ln(f(t))": log_t
        })
    
    return prob_df

# Cria listas de data frames com base na coluna time stamp
def create_data_frames_list(failure_count: list[str], production_count_column: str, time_stamp_column: str, df: pd.DataFrame):
    """
    A função percorre cada linha do DataFrame de entrada e calcula os seguintes valores para cada período:
    - `"Em risco"`: Quantidade de produtos restantes após a ocorrência de falhas acumuladas ao longo do tempo.
    - `"Falhas"`: Contagem de falhas observadas em cada período.
    - `"Falhas Acumuladas"`: Soma cumulativa das falhas ao longo do tempo.
    - `"Taxas de falha"`: Razão entre falhas acumuladas e produtos restantes no início de cada período.
    - `"Tempo Meses"`: Índice do tempo representando cada período.
    - `"Mes"`: Período correspondente no DataFrame original.
    
    Parâmetros:
    -----------
    failure_count : list[str]
        Lista de colunas do DataFrame original que representam a contagem de falhas em diferentes períodos.

    production_count_column : str
        Nome da coluna que contém a quantidade total de produtos produzidos ou em operação.

    time_stamp_column : str
        Nome da coluna que indica o período de tempo (exemplo: meses, ciclos, anos).

    df : pd.DataFrame
        DataFrame contendo os dados de produção e falhas.

    """
    
    dfs = []

    for _, row in df.iterrows():
        qtd_produtos = row[production_count_column] 
        falhas = row[failure_count].values 

        em_risco = [qtd_produtos - falhas[:i].sum() for i in range(len(falhas) + 1)]

        falhas_acumuladas = falhas.cumsum()

        taxas_falha = falhas_acumuladas / em_risco[1:] 

        df_temp = pd.DataFrame({
                "Em risco": em_risco[:-1],  
                "Falhas": falhas.tolist(),  
                "Falhas Acumuladas": falhas_acumuladas.tolist(),  
                "Taxas de falha": taxas_falha.tolist(),  
                "Tempo Meses": list(range(len(falhas))),
                "Mes": row[time_stamp_column]
            })

        dfs.append(df_temp)

    return dfs

# Gráfico de probabilidade
def create_mountain_chart(data_frames: list[pd.DataFrame]):
    """
    Para cada DataFrame na lista, a função extrai os valores de `"Tempo Meses"` e `"Taxas de falha"`, 
    armazenando-os em um dicionário onde a chave é o período (`"Mes"`) e o valor é uma lista de coordenadas `{x, y}`.
    
    Parâmetros:
    -----------
    data_frames : list[pd.DataFrame]
        Lista de DataFrames contendo os dados processados de falhas e tempo.
    """
    
    rates: dict[str, list] = {}

    for df in data_frames:
        if df.empty:
            continue 

        timestamp_key = str(df["Mes"].iloc[0])
        rate_values = [{"x": float(x), "y": float(y)} for x, y in zip(df["Tempo Meses"], df["Taxas de falha"])]
        rates[timestamp_key] = rate_values

    return rates

# Gráfuci de linearização
def create_linearization_chart(df: pd.DataFrame):
    """
    A função cria uma lista de pontos `{x, y}` para o gráfico, onde:
    - `x` representa os valores de `"ln(f(t))"`.
    - `y` representa os valores de `"ln(1-f(t))"`.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as colunas `"ln(f(t))"` e `"ln(1-f(t))"`.
    """
    linearization_chart = [{"x": float(x), "y": float(y)} for x, y in zip(df["ln(f(t))"], df["ln(1-f(t))"])]
    return linearization_chart

# Gráfico de probabilidade
def create_probability_chart(df: pd.DataFrame):
    """
    Gera os dados para um gráfico de probabilidade, incluindo intervalos de confiança.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as colunas `"mes"`, `"f(t) predi"`, `"f(t) predi lower"` e `"f(t) predi upper"`.
    """
    probability_chart = [{"x": float(x), "y": float(y)} for x, y in zip(df["mes"], df["f(t) predi"])]
    probability_chart_min = [{"x": float(x), "y": float(y)} for x, y in zip(df["mes"], df["f(t) predi lower"])]
    probability_chart_max = [{"x": float(x), "y": float(y)} for x, y in zip(df["mes"], df["f(t) predi upper"])]
    return probability_chart, probability_chart_min, probability_chart_max

# Gráfico de confiabilidade
def create_reliability_chart(df: pd.DataFrame):
    """
    Gera os dados para um gráfico de confiabilidade, incluindo intervalos de confiança.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as colunas `"mes"`, `"r(t)"`, `"r(t) lower"` e `"r(t) upper"`.
    """
    reliability_chart = [{"x": float(x), "y": float(y)} for x, y in zip(df["mes"], df["r(t)"])]
    reliability_chart_min = [{"x": float(x), "y": float(y)} for x, y in zip(df["mes"], df["r(t) lower"])]
    reliability_chart_max = [{"x": float(x), "y": float(y)} for x, y in zip(df["mes"], df["r(t) upper"])]
    return reliability_chart, reliability_chart_min, reliability_chart_max

# Cria a coluna f(t) predito e reliability
def add_predicted_reliability(df: pd.DataFrame, alpha: float, beta: float, ci_min: float, ci_max: float):
    """
    Adiciona colunas ao DataFrame para representar a função de falha predita (`f(t) predi`) 
    e a confiabilidade (`r(t)`), incluindo intervalos de confiança.
    
    A função calcula as seguintes colunas para cada período de tempo (`"mes"`):
    - `"f(t) predi"`: Estimativa da função de falha.
    - `"f(t) predi lower"` e `"f(t) predi upper"`: Intervalo de confiança para a função de falha.
    - `"r(t)"`: Confiabilidade estimada (1 - f(t)).
    - `"r(t) lower"` e `"r(t) upper"`: Intervalo de confiança para a confiabilidade.
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo a coluna `"mes"`, que representa o tempo.

    alpha : float
        Parâmetro `alpha` do modelo Weibull.

    beta : float
        Parâmetro `beta` do modelo Weibull.

    ci_min : float
        Limite inferior do intervalo de confiança para `beta`.

    ci_max : float
        Limite superior do intervalo de confiança para `beta`.
    """
    
    df["f(t) predi"] = 1 - np.exp(-1 * (df["mes"] / alpha) ** beta)
    df["f(t) predi lower"] = 1 - np.exp(-1 * (df["mes"] / alpha) ** ci_min)
    df["f(t) predi upper"] = 1 - np.exp(-1 * (df["mes"] / alpha) ** ci_max)
    df["r(t)"] = 1 - df["f(t) predi"]
    df["r(t) lower"] = 1 - df["f(t) predi lower"]
    df["r(t) upper"] = 1 - df["f(t) predi upper"]
    return df

def split_dataframe_by_group(df: pd.DataFrame, grouping_column: str):
    """
    Divide o DataFrame em uma lista de DataFrames, onde cada um contém apenas um dos valores únicos da coluna `grouping_column`.

    Parâmetros:
    -----------
    df: DataFrame original.
    grouping_column: Nome da coluna usada para separar os DataFrames.
    """
    data_frames = [df[df[grouping_column] == value].copy() for value in df[grouping_column].unique()]
    return data_frames


meu front end

import React, { useEffect, useState } from "react";
import { useAuth } from "hooks/useAuth";
import { useTranslation } from "next-i18next";
import ContentHeader from "shared/contentHeader";
import { Spin } from "shared/spin"
import { useRouter } from "next/router";
import { getItem } from "utils/database";
import { DataToCompile } from "components/insertData/inteface";
import { Col, Row, message } from "antd";
import axios from "axios";
import ResponseContentHeader from "shared/responseContentHeader";
import { createSummaryTable } from "./table";
import Table from "shared/table";
import CostProbability from "./costProbability";
import { WarrantyCostsExportData, WarrantyCostsResult } from "./interface";
import Simulator from "./simulator";
import {
  HighChartCustomSeries,
  HighChartTemplate,
} from "shared/widget/chartHub/interface";
import dynamic from "next/dynamic";

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const fetcher = axios.create({
  baseURL: "/api",
});

export const WarrantyCosts: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });

  const { user } = useAuth();
  const router = useRouter();

  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [groupingData, setGroupingData] = useState<Record<string, any>>({});

  const [timeStampColumnName, setTimeStampColumnName] = useState("");
  const [contentVisibility, setContentVisibility] = useState<boolean[]>(
    new Array(Object.keys(groupingData).length).fill(false)
  );

  const [chartData, setChartData] = useState<{
    [key: string]: HighChartTemplate;
  }>({});

  useEffect(() => {
    const getData = async () => {
      const parsedUrlQuery = router?.query;
      if (Object.keys(parsedUrlQuery).length > 0) {
        const tool = parsedUrlQuery.tool as string;
        const uid = parsedUrlQuery.uid as string;

        const item = (await getItem(tool, uid)) as WarrantyCostsExportData;

        if (item) {
          const variables = Object.keys(item.dataToExport[0]);

          const dataToSend: DataToCompile = { obj: {}, itens: variables };

          variables.map((variable) => {
            item.dataToExport.map((item: any) => {
              if (dataToSend.obj[`${variable}`] === undefined) {
                dataToSend.obj[`${variable}`] = [];
              }
              dataToSend.obj[`${variable}`].push(item[variable]);
            });
          });

          setTimeStampColumnName(item.timeStampColumn);

          try {
            const warrantyCostsCalculate = {
              inputData: dataToSend.obj,
              failureCount: item.failureCount,
              groupingColumn: item.groupingColumn ? item.groupingColumn : "",
              productionCountColumn: item.productionCountColumn,
              timeStampColumn: item.timeStampColumn,
            };

            const { data } = await fetcher.post<WarrantyCostsResult>(
              "reliability/warrantyCosts/calculate",
              warrantyCostsCalculate
            );

            setLoadingPage(false);

            const warranty = data.result;
            const warrantyKeys = Object.keys(warranty);

            const warrantyData: Record<string, any> = {};

            for (const key of warrantyKeys) {
              const item = warranty[key];

              warrantyData[key] = {};

              const summaryTranslate: Record<string, string> = {
                alpha: commonT("warrantyCosts.alpha"),
                beta: commonT("warrantyCosts.beta"),
                ciLower: commonT("warrantyCosts.ciLower"),
                ciUpper: commonT("warrantyCosts.ciUpper"),
                mttf: commonT("warrantyCosts.mttf"),
              };

              warrantyData[key]["summary"] = createSummaryTable(
                item.alpha,
                item.beta,
                item.mttf,
                item.betaLower,
                item.betaUpper,
                summaryTranslate
              );

              setLoadingSummary(false);

              warrantyData[key]["alpha"] = item.alpha;
              warrantyData[key]["beta"] = item.beta;
              warrantyData[key]["totalSales"] = item.totalSales;
              warrantyData[key]["mountain"] = item.mountain;
              warrantyData[key]["linearization"] = item.linearization;
              warrantyData[key]["probability"] = item.probability;
              warrantyData[key]["reliability"] = item.reliability;
              warrantyData[key]["probabilityLower"] = item.probabilityLower;
              warrantyData[key]["probabilityUpper"] = item.probabilityUpper;
              warrantyData[key]["reliabilityLower"] = item.reliabilityLower;
              warrantyData[key]["reliabilityUpper"] = item.reliabilityUpper;
            }

            setGroupingData(warrantyData);
          } catch (error) {
            console.error(error);
            message.error({
              content: commonT("error.general.unexpectedMsg"),
            });
          }
        }
      }
    };

    getData().catch(console.error);
  }, [router.isReady, router.query]);

  useEffect(() => {
    const buildChart = () => {
      const updatedChartData: Record<string, any> = {};

      if (Object.keys(groupingData).length > 0) {
        for (const key of Object.keys(groupingData)) {

          const datasetsMountain = createMountainDatasets(
            groupingData[key].mountain,
            true
          );

          const chartsForVariable: HighChartTemplate = buildHighChart(
            key,
            datasetsMountain,
            groupingData[key].linearization,
            groupingData[key].probability,
            groupingData[key].probabilityUpper,
            groupingData[key].probabilityLower,
            groupingData[key].reliability,
            groupingData[key].reliabilityUpper,
            groupingData[key].reliabilityLower
          );

          updatedChartData[key] = chartsForVariable;
        }

        setChartData(updatedChartData);
      }
    };

    buildChart();
  }, [groupingData]);

  function buildHighChart(
    key: string,
    datasetsMountain: any[],
    linearization: any[],
    probability: any[],
    probabilityUpper: any[],
    probabilityLower: any[],
    reliability: any[],
    reliabilityUpper: any[],
    reliabilityLower: any[]
  ) {
    const chartsForVariable: HighChartTemplate = {};

    chartsForVariable["mountain"] = {
      seriesData: datasetsMountain,
      options: {
        title: commonT("widget.chartType.mountain"),
        xAxisTitle: timeStampColumnName,
        yAxisTitle: commonT("widget.chartType.rate"),
        titleFontSize: 24,
        chartName: key,
        legendEnabled: true,
        useVerticalLimits: false,
        showLine: true,
      },
      type: "scatter",
      displayName: commonT("widget.chartType.mountain"),
    };

    chartsForVariable["linearization"] = {
      seriesData: [
        {
          data: linearization,
          name: commonT("warrantyCosts.data"),
          type: "scatter",
          showLine: true,
          tension: 0.3,
        },
      ],
      options: {
        title: commonT("widget.chartType.linearizationPlot"),
        xAxisTitle: "Ln ( f (t))",
        yAxisTitle: "Ln (-1* (1 - f (t)))",
        titleFontSize: 24,
        chartName: key,
        legendEnabled: true,
        useVerticalLimits: false,
        showLine: true,
      },
      type: "scatter",
      displayName: commonT("widget.chartType.linearizationPlot"),
    };

    chartsForVariable["probability"] = {
      seriesData: [
        {
          data: probability,
          name: commonT("warrantyCosts.data"),
          type: "scatter",
          showLine: true,
        },
        {
          data: probabilityUpper,
          name: commonT("warrantyCosts.dataCiMax"),
          showLine: true,
          type: "scatter",
          color: "silver",
        },
        {
          data: probabilityLower,
          name: commonT("warrantyCosts.dataCiMin"),
          type: "scatter",
          showLine: true,
          color: "silver",
        },
      ],
      options: {
        title: commonT("widget.chartType.probPlot"),
        xAxisTitle: timeStampColumnName,
        yAxisTitle: "f (t)",
        titleFontSize: 24,
        chartName: key,
        legendEnabled: true,
        useVerticalLimits: false,
        showLine: true,
      },
      type: "scatter",
      displayName: commonT("widget.chartType.probPlot"),
    };

    chartsForVariable["reliability"] = {
      seriesData: [
        {
          data: reliability,
          name: commonT("warrantyCosts.data"),
          type: "scatter",
          showLine: true,
        },
        {
          data: reliabilityUpper,
          name: commonT("warrantyCosts.dataCiMax"),
          showLine: true,
          type: "scatter",
          color: "silver",
        },
        {
          data: reliabilityLower,
          name: commonT("warrantyCosts.dataCiMin"),
          type: "scatter",
          showLine: true,
          color: "silver",
        },
      ],
      options: {
        title: commonT("widget.chartType.reliabilityPlot"),
        xAxisTitle: timeStampColumnName,
        yAxisTitle: "R(t)",
        titleFontSize: 24,
        chartName: key,
        legendEnabled: true,
        useVerticalLimits: false,
        showLine: true,
      },
      type: "scatter",
      displayName: commonT("widget.chartType.reliabilityPlot"),
    };

    return chartsForVariable;
  }

  const createMountainDatasets = (
    mountains: Record<string, any>,
    showLine: boolean
  ) => {
    const datasets: HighChartCustomSeries[] = [];
    for (const key of Object.keys(mountains)) {
      const item = mountains[key];
      datasets.push({
        name: key,
        data: item,
        showLine: showLine,
        tension: 0.3,
        type: "scatter",
        lineWidth: 1,
      });
    }

    return datasets;
  };

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        await fetcher.post("access_counter", {
          user: user?.username,
          operation: "warrantyCosts",
        });
      }
    };
    countAccess().catch(console.error);
  }, [user]);

  const onToggleContent = (index: number) => {
    const updatedVisibility = [...contentVisibility];
    updatedVisibility[index] = !updatedVisibility[index];
    setContentVisibility(updatedVisibility);
  };
  return (
    <>
      <ContentHeader
        title={commonT("warrantyCosts.title")}
        tool={"warrantyCosts"}
      />
      {loadingPage ? (
        <Spin />
      ) : (
        <>
          <div id="content">
            {groupingData &&
              Object.keys(groupingData).map((key, index) => (
                <div key={index} style={{ marginBottom: 20 }}>
                  <ResponseContentHeader
                    title={key}
                    index={index}
                    onClick={() => onToggleContent(index)}
                  />
                  <Row
                    style={{ marginBottom: 20 }}
                    hidden={contentVisibility[index]}
                  >
                    <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={11}>
                      <ChartHub
                        key={key}
                        chartConfigurations={chartData[key]}
                        tool="warrantyCosts"
                        showLimits
                      />
                    </Col>

                    <Col
                      xs={24}
                      sm={24}
                      md={24}
                      lg={24}
                      xl={24}
                      xxl={{ offset: 1, span: 12 }}
                    >
                      <Table
                        key={index}
                        dataSource={groupingData[key].summary.dataSource}
                        columns={groupingData[key].summary.columns}
                        loading={loadingSummary}
                        title={commonT("warrantyCosts.summaryTable")}
                      />
                    </Col>
                  </Row>

                  <Row hidden={contentVisibility[index]}>
                    <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={11}>
                      <CostProbability
                        key={index}
                        alpha={groupingData[key].alpha}
                        beta={groupingData[key].beta}
                        totalSales={groupingData[key].totalSales}
                      />
                    </Col>
                  </Row>
                </div>
              ))}

            <Row>
              <Col span={24}>
                <Simulator data={groupingData} />
              </Col>
            </Row>
          </div>
        </>
      )}
    </>
  );
};

import { ColumnsType } from "antd/es/table";
import { formatNumber } from "utils/formatting";

export function createSummaryTable(
  alpha: number,
  beta: number,
  mttf: number,
  betaLower: number,
  betaUpper: number,
  translate: Record<string, string>
) {
  const columns: ColumnsType<any> = [
    {
      title: translate.alpha,
      dataIndex: "alpha",
      key: "alpha",
    },
    {
      title: translate.beta,
      dataIndex: "beta",
      key: "beta",
    },
    {
      title: translate.mttf,
      dataIndex: "mttf",
      key: "mttf",
    },
    {
      title: translate.ciLower,
      dataIndex: "ciLower",
      key: "ciLower",
    },
    {
      title: translate.ciUpper,
      dataIndex: "ciUpper",
      key: "ciUpper",
    },
  ];

  const dataSource = [
    {
      key: "1",
      alpha: formatNumber(alpha, 4),
      beta: formatNumber(beta, 4),
      mttf: formatNumber(mttf, 4),
      ciLower: formatNumber(betaLower, 4),
      ciUpper: formatNumber(betaUpper, 4),
    },
  ];
  return { dataSource, columns };
}

import React, { useEffect, useState } from "react";
import * as Styled from "./styled";
import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Title,
  Tooltip as TooltipJS,
  Filler,
  ScatterController,
} from "chart.js";
import annotationPlugin from "chartjs-plugin-annotation";
import ChartDataLabels from "chartjs-plugin-datalabels";
import { useTranslation } from "next-i18next";
import { formatNumber } from "utils/formatting";
import { InputNumber } from "antd";
import { Spin } from "shared/spin"
import { Scatter } from "react-chartjs-2";
import { exp } from "mathjs";

ChartJS.register(
  CategoryScale,
  Filler,
  LinearScale,
  PointElement,
  Title,
  TooltipJS,
  Legend,
  LineElement,
  annotationPlugin,
  ChartDataLabels,
  ScatterController
);

const Simulator = (props: { data: Record<string, any> }) => {
  const { t: commonT } = useTranslation("common");

  const { data } = props;

  const [chartDataProbability, setChartDataProbability] = useState<
    Record<string, any>
  >({});
  const [chartOptionsProbability, setChartOptionsProbability] = useState<
    Record<string, any>
  >({});

  const [probabilityFailureValue, setProbabilityFailureValue] = useState<
    Record<string, number>
  >({});

  const [probabilityFailureAlpha, setProbabilityFailureAlpha] = useState<
    Record<string, number>
  >({});

  const [probabilityFailureBeta, setProbabilityFailureBeta] = useState<
    Record<string, number>
  >({});

  const [probabilityFailureValueMin, setProbabilityFailureValueMin] = useState<
    Record<string, number>
  >({});

  const [probabilityFailureValueMax, setProbabilityFailureValueMax] = useState<
    Record<string, number>
  >({});

  const [reliabilityFailureValue, setReliabilityFailureValue] = useState<
    Record<string, number>
  >({});

  const [reliabilityFailureValueMin, setReliabilityFailureValueMin] = useState<
    Record<string, number>
  >({});

  const [reliabilityFailureValueMax, setReliabilityFailureValueMax] = useState<
    Record<string, number>
  >({});

  const [chartDataReliability, setChartDataReliability] = useState<
    Record<string, any>
  >({});

  const [chartOptionsReliability, setChartOptionsReliability] = useState<
    Record<string, any>
  >({});

  const [probabilityFailureResult, setProbabilityFailureResult] = useState<
    Record<string, number>
  >({});

  const [reliabilityFailureResult, setReliabilityFailureResult] = useState<
    Record<string, number>
  >({});

  useEffect(() => {
    const setData = () => {
      for (const key of Object.keys(data)) {
        const item = data[key];

        setProbabilityFailureAlpha((prevState) => ({
          ...prevState,
          [key]: item.alpha,
        }));

        setProbabilityFailureBeta((prevState) => ({
          ...prevState,
          [key]: item.beta,
        }));

        setProbabilityFailureValue((prevState) => ({
          ...prevState,
          [key]: Math.min(...item.probability.map((el) => el.x)),
        }));

        setProbabilityFailureValueMin((prevState) => ({
          ...prevState,
          [key]: Math.min(...item.probability.map((el) => el.x)),
        }));

        setProbabilityFailureValueMax((prevState) => ({
          ...prevState,
          [key]: Math.max(...item.probability.map((el) => el.x)),
        }));

        setReliabilityFailureValue((prevState) => ({
          ...prevState,
          [key]: Math.min(...item.reliability.map((el) => el.x)),
        }));

        setReliabilityFailureValueMin((prevState) => ({
          ...prevState,
          [key]: Math.min(...item.reliability.map((el) => el.x)),
        }));

        setReliabilityFailureValueMax((prevState) => ({
          ...prevState,
          [key]: Math.max(...item.reliability.map((el) => el.x)),
        }));
      }
    };

    setData();
  }, [data]);

  useEffect(() => {
    const buildProbabilityChart = () => {
      for (const key of Object.keys(data)) {
        const item = data[key];
        const result = item.probability;
        const buildChartData: any = {
          datasets: [
            {
              data: result,
              label: commonT("warrantyCosts.simulator.data"),
              showLine: true,
              tension: 0.3,
              borderColor: "black",
              backgroundColor: "black",
            },
          ],
        };

        const buildChartOptions = {
          responsive: true,
          plugins: {
            datalabels: {
              display: false,
            },
            legend: {
              display: false,
            },
            annotation: {
              annotations: [
                {
                  type: "line",
                  scaleID: "x",
                  value: probabilityFailureValue[key]
                    ? probabilityFailureValue[key]
                    : 0,
                  borderColor: "red",
                  borderWidth: 2,
                },
              ],
            },
          },
        };

        setChartDataProbability((prevState) => ({
          ...prevState,
          [key]: buildChartData,
        }));

        setChartOptionsProbability((prevState) => ({
          ...prevState,
          [key]: buildChartOptions,
        }));
      }
    };
    buildProbabilityChart();
  }, [data, probabilityFailureValue]);

  useEffect(() => {
    const buildChartReliability = () => {
      for (const key of Object.keys(data)) {
        const item = data[key];
        const result = item.reliability;

        const buildChartData: any = {
          datasets: [
            {
              data: result,
              label: commonT("warrantyCosts.simulator.data"),
              showLine: true,
              tension: 0.3,
              borderColor: "black",
              backgroundColor: "black",
            },
          ],
        };

        const buildChartOptions = {
          responsive: true,
          plugins: {
            datalabels: {
              display: false,
            },
            legend: {
              display: false,
            },
            annotation: {
              annotations: [
                {
                  type: "line",
                  scaleID: "x",
                  value: reliabilityFailureValue[key]
                    ? reliabilityFailureValue[key]
                    : 0,
                  borderColor: "red",
                  borderWidth: 2,
                },
              ],
            },
          },
        };

        setChartDataReliability((prevState) => ({
          ...prevState,
          [key]: buildChartData,
        }));

        setChartOptionsReliability((prevState) => ({
          ...prevState,
          [key]: buildChartOptions,
        }));
      }
    };

    buildChartReliability();
  }, [data, reliabilityFailureValue]);

  useEffect(() => {
    calculateProbabilityEquation();
    calculateReliabilityEquation();
  }, [probabilityFailureValue, reliabilityFailureValue]);

  const calculateReliabilityEquation = () => {
    for (const key of Object.keys(probabilityFailureValue)) {
      const value = reliabilityFailureValue[key];

      const failure =
        1 -
        exp(
          -1 *
            (value / probabilityFailureAlpha[key]) **
              probabilityFailureBeta[key]
        );

      setReliabilityFailureResult((prevState) => ({
        ...prevState,
        [key]: 1 - failure,
      }));
    }
  };

  const calculateProbabilityEquation = () => {
    for (const key of Object.keys(probabilityFailureValue)) {
      const value = probabilityFailureValue[key];

      const result =
        1 -
        exp(
          -1 *
            (value / probabilityFailureAlpha[key]) **
              probabilityFailureBeta[key]
        );

      setProbabilityFailureResult((prevState) => ({
        ...prevState,
        [key]: result,
      }));
    }
  };

  const getChartDataProbability = (key: string) => {
    return chartDataProbability ? chartDataProbability[key] : {};
  };

  const getChartDataReliability = (key: string) => {
    return chartDataReliability ? chartDataReliability[key] : {};
  };

  const handleProbabilityFailureValue = (value: number, key: string) => {
    setProbabilityFailureValue((prevState) => ({
      ...prevState,
      [key]: value,
    }));
  };

  const handleReliabilityFailureValue = (value: number, key: string) => {
    setReliabilityFailureValue((prevState) => ({
      ...prevState,
      [key]: value,
    }));
  };

  return (
    <>
      <Styled.ContainerFather
        title={commonT("lifeDistribution.simulator.title")}
      >
        <Styled.LayoutContainer>
          <Styled.LayoutContainerEquation>
            <Styled.EquationContainer>
              <h2>
                {commonT("lifeDistribution.simulator.probabilityFailure")}:
              </h2>
              {Object.keys(data).map((key) => (
                <div key={`equation-${key}`}>
                  <h3>
                    Y {key}=
                    <span style={{ color: "red" }}>
                      {formatNumber(probabilityFailureResult[key])}
                    </span>
                  </h3>
                </div>
              ))}
            </Styled.EquationContainer>

            <Styled.EquationContainer>
              <h2>{commonT("lifeDistribution.simulator.reliability")}:</h2>
              {Object.keys(data).map((key) => (
                <div key={`equation-${key}`}>
                  <h3>
                    Y {key}=
                    <span style={{ color: "red" }}>
                      {formatNumber(reliabilityFailureResult[key])}
                    </span>
                  </h3>
                </div>
              ))}
            </Styled.EquationContainer>
          </Styled.LayoutContainerEquation>
          <Styled.LayoutContainerCharts>
            {Object.keys(data).map((key, index) => (
              <React.Fragment key={`charts-${index}`}>
                <Styled.Container key={`${index}-fail`}>
                  <Styled.Header>
                    <h3>
                      {`${commonT(
                        "lifeDistribution.simulator.probabilityFailure"
                      )} ${key}`}
                    </h3>
                  </Styled.Header>
                  <Styled.Card>
                    {chartDataProbability &&
                    Object.keys(chartDataProbability).length > 0 ? (
                      <Scatter
                        options={chartOptionsProbability[key]}
                        data={getChartDataProbability(key)}
                      />
                    ) : (
                      <Spin />
                    )}

                    <Styled.InputContainer>
                      {commonT("warrantyCosts.time")}
                      <InputNumber
                        key={index}
                        value={probabilityFailureValue[key]}
                        onChange={(value) =>
                          handleProbabilityFailureValue(value, key)
                        }
                        max={probabilityFailureValueMax[key] ?? 0}
                        min={probabilityFailureValueMin[key] ?? 0}
                      />
                    </Styled.InputContainer>
                  </Styled.Card>
                </Styled.Container>

                <Styled.Container key={`${index}-fail`}>
                  <Styled.Header>
                    <h3>
                      {`${commonT(
                        "lifeDistribution.simulator.reliability"
                      )} ${key}`}
                    </h3>
                  </Styled.Header>
                  <Styled.Card key={index}>
                    {chartDataReliability &&
                    Object.keys(chartDataReliability).length > 0 ? (
                      <Scatter
                        options={chartOptionsReliability[key]}
                        data={getChartDataReliability(key)}
                      />
                    ) : (
                      <Spin />
                    )}

                    <Styled.InputContainer key={index}>
                      {commonT("warrantyCosts.time")}
                      <InputNumber
                        key={index}
                        value={reliabilityFailureValue[key]}
                        onChange={(value) =>
                          handleReliabilityFailureValue(value, key)
                        }
                        max={reliabilityFailureValueMax[key] ?? 0}
                        min={reliabilityFailureValueMin[key] ?? 0}
                      />
                    </Styled.InputContainer>
                  </Styled.Card>
                </Styled.Container>
              </React.Fragment>
            ))}
          </Styled.LayoutContainerCharts>
        </Styled.LayoutContainer>
      </Styled.ContainerFather>
    </>
  );
};

export default Simulator;

import styled from "@emotion/styled";
import { Card as AntdCard } from "antd";

export const Container = styled.div`
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  border-radius: 10px;
  text-align: center;
  padding: 0px 10px 10px 10px;
  width: 450px;
  margin-right: 30px;
  max-height: fit-content;
`;

export const ContainerFather = styled(AntdCard)`
  background-color: white;
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);

  .ant-card-head {
    background: #00579d;
  }

  .ant-card-head-title {
    color: white;
  }

  .ant-tooltip {
    color: white;
  }
`;

export const Header = styled.div`
  background: #00579d;
  color: white;
  font-size: 12px;
  border-radius: 6px;
`;

export const EquationContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
  background-color: #ffffff;
  border: 1px solid #ddd;
  width: calc(50vw - 700px);
`;

export const LayoutContainer = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 20px;
`;

export const InputContainer = styled.div`
  display: flex;
  gap: 10px;
  margin-top: 10px;
  justify-content: flex-start;
  align-items: center;
`;

export const Card = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export const LayoutContainerCharts = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
`;

export const LayoutContainerEquation = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

import React, { useState } from "react";
import { useTranslation } from "next-i18next";
import * as Styled from "./../styled";
import { Button, Form, InputNumber } from "antd";
import { AiOutlineDelete } from "react-icons/ai";
import { MdAdd } from "react-icons/md";
import { BsCalculator } from "react-icons/bs";

const CostProbability = (props: {
  alpha: number;
  beta: number;
  totalSales: number;
}) => {
  const { alpha, beta, totalSales } = props;
  const { t: commonT } = useTranslation("common");

  const [rows, setRows] = useState([
    { id: 1, cost: 0, time: 0, fT: 0, totalCost: 0 },
  ]);

  const handleAddRow = () => {
    setRows([
      ...rows,
      { id: rows.length + 1, cost: 0, time: 0, fT: 0, totalCost: 0 },
    ]);
  };

  const handleRemoveRow = (id: number) => {
    setRows(rows.filter((row) => row.id !== id));
  };

  const handleInputChange = (
    rowId: number,
    field: "cost" | "time",
    value: number
  ) => {
    setRows((prevRows) =>
      prevRows.map((row) =>
        row.id === rowId ? { ...row, [field]: value || 0 } : row
      )
    );
  };

  const handleCalculateTotal = (rowId: number) => {
    setRows((prevRows) =>
      prevRows.map((row) => {
        if (row.id === rowId) {
          const fT = 1 - Math.exp(-1 * Math.pow(row.time / alpha, beta));
          const totalCost = row.cost * fT * totalSales;
          return { ...row, fT, totalCost };
        }
        return row;
      })
    );
  };

  return (
    <Styled.Card title={commonT("warrantyCosts.costProbaility.title")}>
      <Form name="warrantyCostsForm">
        {rows.map((row) => (
          <div
            key={row.id}
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 10,
              marginBottom: 10,
            }}
          >
            <Form.Item label={commonT("warrantyCosts.costProbaility.cost")}>
              <InputNumber
                value={row.cost}
                onChange={(value) =>
                  handleInputChange(row.id, "cost", value || 0)
                }
              />
            </Form.Item>

            <Form.Item label={commonT("warrantyCosts.costProbaility.time")}>
              <InputNumber
                value={row.time}
                onChange={(value) =>
                  handleInputChange(row.id, "time", value || 0)
                }
              />
            </Form.Item>

            <Form.Item label={"f(t)"}>
              <InputNumber readOnly value={row.fT} precision={5} />
            </Form.Item>

            <Form.Item
              label={commonT("warrantyCosts.costProbaility.totalCost")}
            >
              <InputNumber readOnly value={row.totalCost} precision={2} />
            </Form.Item>

            <Button
              type="primary"
              icon={<BsCalculator />}
              onClick={() => handleCalculateTotal(row.id)}
            >
              {commonT("warrantyCosts.costProbaility.calculate")}
            </Button>

            <Button
              danger
              icon={<AiOutlineDelete />}
              onClick={() => handleRemoveRow(row.id)}
            />
          </div>
        ))}

        <Button icon={<MdAdd />} type="primary" onClick={handleAddRow}>
          {commonT("warrantyCosts.costProbaility.add")}
        </Button>
      </Form>
    </Styled.Card>
  );
};

export default CostProbability;

