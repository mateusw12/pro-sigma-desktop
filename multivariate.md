Vamos fazer a analise de multivariedade, o usuário pode colocar varias colunas X.

meu back-end que da para reaproveitar

from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.common.common import remove_punctuation
from utils.multiple_regression.multiple_regression_calculate import (
    calculate_mean_column_values,
)
from utils.multivariate_calculate.multivariate_calculate import (
    calculate_correlation_matrix,
    calculate_squared_diff,
    calculate_x_normalized_all_columns,
    calculate_x_nortalized_transpose,
)

multivariate_router = APIRouter()


class Item(BaseModel):
    obj: dict[str, List[float]]


@multivariate_router.post(
    "/calculate",
    tags=["Multivariate"],
    description="""Calcula Analise de Correlação,
                obj: dataframe;
              """,
    response_model=object,
)
def calculate_multivariate(body: Item):

    df = pd.DataFrame(data=body.obj)
    df = remove_punctuation(df)

    try:
        # Calcula médias das colunas de valor
        mean_columns = calculate_mean_column_values(df)

        # Calcula Media **2
        square = calculate_squared_diff(df, mean_columns)

        # Calcula X Normaliazda
        x_normalized = calculate_x_normalized_all_columns(df, mean_columns, square)

        # Calcula X normalizada transposta
        x_nortalized_transpose = calculate_x_nortalized_transpose(x_normalized)

        # Calcula matriz de correlação
        correlation_matrix = calculate_correlation_matrix(
            x_nortalized_transpose, x_normalized
        )

        return {"correlationMatrix": correlation_matrix.tolist()}

    except HTTPException as http_error:
        raise http_error
    except Exception as error:
        print("erro", error)
        raise HTTPException(status_code=500, detail="correlationMatrixError")


def calculate_mean_column_values(df: pd.DataFrame):
    """
    Calcula a média de todas as colunas numéricas do DataFrame e retorna um dicionário
    com o nome da coluna como chave e a média da coluna como valor.

    Parâmetros:
    - df (pd.DataFrame): O DataFrame contendo os dados para os quais as médias serão calculadas.
    """
    result: dict[str, float] = df.mean().to_dict()
    return result

import numpy as np
import pandas as pd

# Calcula X Normaliazda
def calculate_x_normalized_all_columns(df: pd.DataFrame, mean_columns: dict[str, float], square: dict[str, float]):
    """
    Normaliza todas as colunas de um DataFrame utilizando a fórmula de normalização (X - média) / desvio padrão.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas que serão normalizadas.
        mean_columns (dict[str, float]): Dicionário contendo a média de cada coluna. A chave é o nome da coluna e o valor é a média.
        square (dict[str, float]): Dicionário contendo a soma dos quadrados de cada coluna. A chave é o nome da coluna e o valor é a soma dos quadrados.
    """
    normalized_df = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        normalized_values = (df[column] - mean_columns[column]) / (square[column] ** 0.5)
        normalized_df[column] = normalized_values
    return normalized_df

# Calcula X Normaliazda
def calculate_x_normalized(df: pd.DataFrame, mean_columns, square):
    """
    Normaliza as colunas de um DataFrame (exceto a primeira coluna) utilizando a fórmula de normalização (X - média) / desvio padrão.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas a serem normalizadas.
        mean_columns (dict[str, float]): Dicionário com as médias das colunas do DataFrame.
        square (dict[str, float]): Dicionário com as somas dos quadrados das colunas do DataFrame.
    """
    df = df.iloc[:, 1:]
    normalized_df = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        normalized_values = (df[column] - mean_columns[column]) / (square[column] ** 0.5)
        normalized_df[column] = normalized_values
    return normalized_df

# Calcula X normalizada transposta 
def calculate_x_nortalized_transpose(df: pd.DataFrame):
    """
    Retorna a transposta de um DataFrame de dados normalizados.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas e as linhas a serem transpostas.
    """
    return df.transpose()

# Calcula matriz de correlação
def calculate_correlation_matrix(transposed_matrix, df: pd.DataFrame):
    """
    Calcula a matriz de correlação entre as colunas de um DataFrame usando a multiplicação de matrizes.

    Parâmetros:
    - transposed_matrix (np.ndarray): Matriz transposta que será multiplicada pela matriz original do DataFrame.
    - df (pd.DataFrame): DataFrame original contendo as colunas cujas correlações serão calculadas.
    """
    np.set_printoptions(suppress=True)
    transposed_array = np.array(transposed_matrix)
    df_array = df.to_numpy()
    result_matrix = np.matmul(transposed_array, df_array)
    return np.round(result_matrix, 3)

# Calcula Media **2
def calculate_squared_diff(df: pd.DataFrame, mean_dict: dict[str, float]):
    """
    Calcula a soma das diferenças ao quadrado entre os valores das colunas do DataFrame e suas respectivas médias.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados das variáveis.
        mean_dict (dict[str, float]): Dicionário contendo as médias das colunas. A chave é o nome da coluna e o valor é a média.
    """
    result_dict: dict[str, float] = {}
    for column in df.columns:
        squared_diff_sum = np.sum((df[column] - mean_dict[column]) ** 2)
        result_dict[column] = squared_diff_sum
    return result_dict

meu front-end

import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { useTranslation } from "next-i18next";
import { useAuth } from "hooks/useAuth";
import ContentHeader from "shared/contentHeader";
import { Spin } from "shared/spin"
import { Col, Row, message } from "antd";
import { DataToCompile, MultivariateData } from "./interface";
import Table from "shared/table";
import { createCorrelationMatrixTable } from "./tables";
import * as Styled from "./styled";
import { HeatMapGrid } from "react-grid-heatmap";
import dynamic from "next/dynamic";
import { useRouter } from "next/router";
import { MultipleRegressionExportData } from "components/multipleRegression/interface";
import { getItem } from "utils/database";

const ChartBuilder = dynamic(() => import("./chartBuilder"), { ssr: false });

const fetcher = axios.create({
  baseURL: "/api",
});

export const Multivariate: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });
  const { user } = useAuth();
  const router = useRouter();

  const [loadingPage, setLoadingPage] = useState(true);
  const [datasets, setDatasets] = useState<number[][] | null>(null);
  const [dataTable, setDataTable] = useState<any>({});
  const [uniqueCorrelationKeys, setUniqueCorrelationKeys] = useState<string[]>(
    []
  );
  const [loadingCorrelationMatrix, setLoadingCorrelationMatrix] =
    useState(true);
  const correlationMatrix = useRef<any>({});

  useEffect(() => {
    const getData = async () => {
      if (router.isReady) {
        const parsedUrlQuery = router.query;
        if (Object.keys(parsedUrlQuery).length > 0) {
          const tool = parsedUrlQuery.tool as string;
          const uid = parsedUrlQuery.uid as string;

          const item = (await getItem(
            tool,
            uid
          )) as MultipleRegressionExportData;

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

          try {
            const multivariateCalculate = { obj: dataToSend.obj };
            const { data } = await fetcher.post<MultivariateData>(
              "multivariate/calculate",
              multivariateCalculate
            );

            setLoadingPage(false);

            const correlationMatrixVariables = variables.filter(
              (el) => !el.includes("Intercept")
            );
            const correlationMatrixTranslate = {
              term: commonT("multipleRegression.term"),
            };

            correlationMatrix.current = createCorrelationMatrixTable(
              data.correlationMatrix,
              correlationMatrixVariables,
              correlationMatrixTranslate
            );

            setLoadingCorrelationMatrix(false);
            setDataTable(dataToSend.obj);

            setDatasets(data.correlationMatrix);
            setUniqueCorrelationKeys(correlationMatrixVariables);
          } catch (error: any) {
            console.error(error);
            const errorMessage =
              error.response?.data === "correlationMatrixError"
                ? commonT("error.general.proccesData")
                : commonT("error.general.unexpectedMsg");

            message.error({ content: errorMessage });
          }
        }
      }
    };

    getData().catch(console.error);
  }, [router.isReady, router.query]);

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        await fetcher.post("access_counter", {
          user: user?.username,
          operation: "multivariate",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  return (
    <>
      <ContentHeader
        title={commonT("multivariate.title")}
        tool={"multivariate"}
      />
      {loadingPage ? (
        <Spin />
      ) : (
        <div id="content">
          <Row>
            <Col xs={24} sm={24} md={24} lg={24} xl={13} xxl={13}>
              {uniqueCorrelationKeys.length > 0 ? (
                <ChartBuilder
                  columns={uniqueCorrelationKeys}
                  dataTable={dataTable}
                />
              ) : (
                <Spin />
              )}
            </Col>
            <Col
              xs={{ span: 24 }}
              sm={{ span: 24 }}
              md={{ span: 24 }}
              lg={{ span: 24 }}
              xl={{ span: 10, offset: 1 }}
              xxl={{ span: 10 }}
            >
              <div style={{ marginBottom: 20 }}>
                <Table
                  loading={loadingCorrelationMatrix}
                  dataSource={
                    correlationMatrix.current.correlataionMatrixDataSource
                  }
                  columns={correlationMatrix.current.correlationMatrixColumns}
                  title={commonT("multipleRegression.correlationMatrix")}
                />
              </div>
              <Styled.Container>
                <Styled.Title>
                  <b>{commonT("multivariate.correlationChartTitle")}</b>
                </Styled.Title>
                {datasets && uniqueCorrelationKeys.length > 0 ? (
                  <div
                    style={{
                      overflow: "auto",
                      maxHeight: "400px",
                      maxWidth: "100%",
                    }}
                  >
                    <HeatMapGrid
                      data={datasets}
                      xLabels={uniqueCorrelationKeys}
                      yLabels={uniqueCorrelationKeys}
                      cellHeight="50px"
                      xLabelsPos="bottom"
                      xLabelsStyle={() => ({
                        textAlign: "left",
                        whiteSpace: "nowrap",
                        paddingBottom: "60px",
                        marginRight: "30px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      })}
                      cellStyle={(_x: number, _y: number, ratio: number) => ({
                        background: `rgba(0, 87, 154, ${Math.abs(
                          ratio / 0.7
                        )})`,
                      })}
                    />
                  </div>
                ) : (
                  <Spin />
                )}
              </Styled.Container>
            </Col>
          </Row>
        </div>
      )}
    </>
  );
};

import styled from "@emotion/styled";

export const Container = styled.div`
  box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.4);
  margin-bottom: 20px;
  min-height: 10vw;
  overflwo: auto;
  padding: 10px;
`;

export const Title = styled.div`
  display: flex;
  justify-content: center;
  padding-bottom: 10px;
  color: ${({ theme }) => theme.color.primary.bg};
`;

export const CorrelationMatrixPlot = styled.div`
  box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.4);
  border: 2px solid #ccc;
  padding: 5px;
  position: relative;
  border-radius: 12px;
  overflow: auto;
  overflow-y: auto;
  max-height: 900px;
  min-width: 950px;
  height: 750px;
  margin-bottom: 20px;

  @media (max-width: 1768px) {
    min-width: 500px;
  }

  @media (max-width: 1268px) {
    min-width: 300px;
  }

`;

export const Toolbar = styled.div`
  display: flex;
  padding-top: 10px;
  padding-right: 10px;
  gap: 10px;
  justify-content: flex-end;
`;

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { useTranslation } from "next-i18next";
import { Button, Tooltip } from "antd";
import { AiOutlineEye, AiOutlineLineChart } from "react-icons/ai";
import * as Styled from "./styled";
import {
  ScatterMatricesLayout,
  ScatterMatricesPlotConfig,
  ScatterMatricesTrendLine,
  SplomPlotData,
  SplomPlotDimensions,
} from "./interface";
import { Spin } from "shared/spin"

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <Spin />,
}) as any;

const ChartBuilder = (props: {
  columns: string[];
  dataTable: Record<string, number[]>;
}) => {
  const { columns, dataTable } = props;

  const { t: commonT } = useTranslation("common");

  const [visible, setVisible] = useState(false);
  const [visibleTrendLine, setVisibleTrendLine] = useState(false);
  const [data, setData] = useState<
    (ScatterMatricesTrendLine | SplomPlotData)[]
  >([]);

  const dimensions: SplomPlotDimensions[] = columns.map((column) => ({
    label: column,
    values: dataTable[column],
  }));

  const splomData: SplomPlotData[] = [
    {
      type: "splom",
      dimensions: dimensions,
      marker: {
        color: "black",
      },
      diagonal: { visible: visible },
      showlegend: false,
    },
  ];

  const calculateTrendline = (xData: number[], yData: number[]) => {
    const n = xData.length;
    const sumX = xData.reduce((sum, val) => sum + val, 0);
    const sumY = yData.reduce((sum, val) => sum + val, 0);
    const sumXY = xData.reduce(
      (sum, val, index) => sum + val * yData[index],
      0
    );
    const sumXX = xData.reduce((sum, val) => sum + val * val, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return xData.map((x) => slope * x + intercept);
  };

  useEffect(() => {
    let plotlyData = [];

    const trendlines: ScatterMatricesTrendLine[] = [];
    columns.forEach((xCol, i) => {
      columns.forEach((yCol, j) => {
        if (i !== j) {
          const xData = dataTable[xCol];
          const yData = dataTable[yCol];
          const trendline = calculateTrendline(xData, yData);
          trendlines.push({
            x: xData,
            y: trendline,
            xaxis: `x${i + 1}`,
            yaxis: `y${j + 1}`,
            mode: "lines",
            type: "scatter",
            line: {
              color: visibleTrendLine ? "red" : "transparent",
              width: 1,
            },
            showlegend: false,
            hoverinfo: "none",
          });
        }
      });
    });

    plotlyData = [...splomData, ...trendlines];

    setData(plotlyData);
  }, [visible, visibleTrendLine, dataTable]);

  const layout: ScatterMatricesLayout = {
    title: commonT("multivariate.scatterPlotMatrix.title"),
    autosize: true,
    hovermode: "closest",
    dragmode: false,
    plot_bgcolor: "#fff",
    xaxis: {
      zeroline: false,
    },
    yaxis: {
      zeroline: false,
    },
    grid: {
      rows: columns.length,
      columns: columns.length,
      pattern: "independent",
    },
  };

  const plotConfig: ScatterMatricesPlotConfig = {
    displayModeBar: false,
    responsive: true,
  };

  return (
    <>
      {typeof window !== "undefined" && dimensions && columns && (
        <Styled.CorrelationMatrixPlot>
          <Styled.Toolbar>
            <Tooltip title={commonT("widget.visibleTrendLine")} placement="top">
              <Button
                onClick={() => setVisibleTrendLine(!visibleTrendLine)}
                size="small"
              >
                <AiOutlineLineChart style={{ transform: "scale(1.3)" }} />
              </Button>
            </Tooltip>
            <Tooltip
              title={commonT("widget.visibleCorrelationTerm")}
              placement="top"
            >
              <Button onClick={() => setVisible(!visible)} size="small">
                <AiOutlineEye style={{ transform: "scale(1.3)" }} />
              </Button>
            </Tooltip>
          </Styled.Toolbar>
          <Plot
            key={`plot-${visibleTrendLine}`}
            data={data}
            layout={layout}
            config={plotConfig}
            style={{
              minWidth: "calc(100vw-100px)",
              minHeight: 700,
            }}
          />
        </Styled.CorrelationMatrixPlot>
      )}
    </>
  );
};

export default ChartBuilder;

import React from "react";
import { formatNumber } from "utils/formatting";

export function createCorrelationMatrixTable(
  correlationMatrix: number[][],
  correlationMatrixVariables: string[],
  translate: any
) {
  const correlationMatrixColumns = [
    {
      title: translate.term,
      dataIndex: "term",
      key: "term",
    },
    ...correlationMatrixVariables.map((variable, index) => ({
      title: variable,
      dataIndex: `column${index}`,
      key: `column${index}`,
      render: (value: number) => {
        const cellStyle: React.CSSProperties = {};

        if (value <= -0.75) {
          cellStyle.color = "red";
        } else if (value > 0.75) {
          cellStyle.color = "#096dd9";
        }

        return <span style={cellStyle}>{formatNumber(value)}</span>;
      },
    })),
  ];

  const correlataionMatrixDataSource: any[] = [];

  correlationMatrixVariables.forEach((variable, rowIndex) => {
    const row: any = {
      key: `row${rowIndex}`,
      term: variable,
    };

    const rowValues = correlationMatrix[rowIndex];
    if (Array.isArray(rowValues)) {
      rowValues.forEach((value, colIndex) => {
        row[`column${colIndex}`] = formatNumber(value);
      });
    }
    correlataionMatrixDataSource.push(row);
  });

  return { correlataionMatrixDataSource, correlationMatrixColumns };
}
