Vamos fazer a analise do COV EMS agora.

A mesma ideia do capability posso selecionar varios X e varios Y, para fazer a analise, e tenho a oção de fazer analise cruzada ou hierarquica

meu antigo python, que pode ser reaproveitado

NESTED

from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.common.common import (
    new_split_dataframes_by_response_column_add_response_doe,
    remove_punctuation,
)
from utils.cov.cov_calculate import (
    calculate_mean_and_amplitude,
    calculate_variation_table,
)

nested_router = APIRouter()


class Item(BaseModel):
    itens: List[str]
    obj: dict[str, List[str | float]]
    responseColumn: List[str]


@nested_router.post(
    "/ems/nested",
    tags=["COV"],
    description="""Calcula COV EMS Nested, parametros: itens: Nome das colunas; obj: DataFrame dos dados """,
    response_model=object,
)
def calculate_ems_nested(body: Item):
    old_df = pd.DataFrame(data=body.obj)
    columns = body.itens
    response_columns = body.responseColumn

    old_df = remove_punctuation(old_df)

    covs = {}
    dataFrames = new_split_dataframes_by_response_column_add_response_doe(
        old_df, response_columns
    )

    for _i, df in enumerate(dataFrames):
        df = pd.DataFrame(df)
        columns = df.columns.tolist()
        response_column = df.columns[-1]
        columnsX = columns.copy()
        del columnsX[-1]
        try:

            lineQuantity = list(np.arange(0, len(df[columnsX[-1]])))

            df.insert(len(columns), "line", lineQuantity)

            balancedDataVerifier = True
            for col in columnsX:
                repeatedItens = df.pivot_table(index=[col], aggfunc="size")
                iTester = -1
                for repeatedIten in repeatedItens:
                    if iTester == -1:
                        iTester = repeatedIten
                    else:
                        if iTester != repeatedIten:
                            balancedDataVerifier = False

            if balancedDataVerifier:
                rBar = calculate_mean_and_amplitude(df, columns, columnsX, col)
                variation_table = calculate_variation_table(rBar, columnsX)
                covs[response_column] = variation_table

        except Exception as error:
            raise HTTPException(status_code=500, detail="covError")

    return covs


CROSSED

from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.common.common import (
    new_split_dataframes_by_response_column_add_response_doe,
    remove_punctuation,
)
from utils.cov.cov_calculate import (
    calculate_anova_table,
    calculate_mean_square,
    calculate_percent_total,
    combine_strings,
    construct_interaction_effects_formula,
    construct_main_effects_formula,
    fit_linear_regression,
    replace_data_frame,
)

crossed_router = APIRouter()


class Item(BaseModel):
    itens: List[str]
    obj: dict[str, List[str | float]]
    responseColumn: List[str]


@crossed_router.post(
    "/ems/crossed",
    tags=["COV"],
    description="""Calcula COV EMS Crossed, parametros: itens: Nome das colunas; obj: DataFrame dos dados """,
    response_model=object,
)
def calculate_ems_crossed(body: Item):
    old_df = pd.DataFrame(data=body.obj)
    old_df = replace_data_frame(body.itens, old_df)
    response_columns = body.responseColumn

    old_df = remove_punctuation(old_df)

    covs = {}
    dataFrames = new_split_dataframes_by_response_column_add_response_doe(
        old_df, response_columns
    )

    for _, df in enumerate(dataFrames):
        df = pd.DataFrame(df)
        response_column = df.columns[-1]

        df_without_response = df.drop(columns=[response_column])

        are_rows_unique = not df_without_response.duplicated().any()
        if are_rows_unique:
            last_column_name = df_without_response.columns[-1]
            df = df.drop(columns=[last_column_name])

        try:
            # faz o replace da coluna de resposta para sempre ser Y
            df.columns = [*df.columns[:-1], "Y"]
            columns = df.columns.to_list()
            body.itens = columns

            mainEffects = construct_main_effects_formula(body.itens)
            interactionEffects = construct_interaction_effects_formula(body.itens)

            sStringSplited = interactionEffects.split("+")
            combinedString = combine_strings(sStringSplited)

            totalString = body.itens[-1] + " ~ " + mainEffects
            if combinedString != "":
                totalString += " + " + combinedString

        except Exception as error:
            raise HTTPException(status_code=500, detail="covFormulaError")

        try:
            moore_lm = fit_linear_regression(df, totalString)

            try:
                tableToRead = calculate_anova_table(moore_lm)
                sum_sq = tableToRead["sum_sq"]
                df_values = tableToRead["df"]

                sum_mq = calculate_mean_square(sum_sq, df_values)
                tableToRead.insert(2, "sum_mq", sum_mq)
                total_sum_sq = sum(sum_sq)
                total_df = sum(df_values)

                totalPercentual = calculate_percent_total(sum_sq, total_sum_sq)
                totalPercentual.append("")
                tableToRead.insert(5, "totalPercentual", totalPercentual)
                tableToRead.loc["Total"] = [
                    total_sum_sq,
                    total_df,
                    (total_sum_sq / total_df),
                    "",
                    "",
                    "",
                ]
                covs[response_column] = {"table": tableToRead.to_dict()}

            except Exception as e:
                print("error", e)
                raise HTTPException(status_code=500, detail="covAnovaError")

        except HTTPException as http_error:
            print("http_error", http_error)
            raise http_error
        except:
            raise HTTPException(status_code=500, detail="covError")

    return covs

CHECK BALANCED

from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.common.common import remove_punctuation
from utils.cov.cov_calculate import check_balanced

check_balanced_router = APIRouter()


class CheckBalanced(BaseModel):
    itens: List[str]
    obj: dict[str, List[str | float]]
    type: str


@check_balanced_router.post(
    "/check_balanced",
    tags=["COV"],
    description="""Valida se os dados estão balanceados, parametros: itens: Nome das colunas; obj: DataFrame dos dados """,
    response_model=object,
)
def check_balanced_data(body: CheckBalanced):

    try:
        df = pd.DataFrame(data=body.obj)
        df = remove_punctuation(df)

        check = check_balanced(df, body.type)
        return {
            "check": check,
        }

    except Exception as error:
        print(error)
        raise HTTPException(status_code=500, detail="covBalanceError")

UTILS

from typing import List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Remove colunas do data frame
def replace_data_frame(itens: List[str], df: pd.DataFrame):
    """
    Remove colunas de um DataFrame onde todos os valores são idênticos, 
    atualizando a lista de itens correspondente.

    Parâmetros:
    ----------
    itens : List[str]
        Uma lista de nomes de colunas presentes no DataFrame. Representa as colunas
        que serão verificadas e possivelmente removidas se todos os seus valores
        forem idênticos.

    df : pd.DataFrame
        O DataFrame do pandas a ser processado. Contém as colunas que serão
        verificadas para valores constantes.
    """
    for col in itens:
        if(np.all(df[col] == df[col][0])):
            itens.remove(col)
            df.drop(col, inplace=True, axis=1)
    return df        

# Ajusta o modelo de regressão linear
def fit_linear_regression(df: pd.DataFrame, formula: str):
    """
    Ajusta um modelo de regressão linear utilizando a fórmula especificada.

    Parâmetros:
    ----------
    df : pd.DataFrame
        O DataFrame do pandas contendo os dados para o modelo.
    formula : str
        Uma string representando a fórmula no formato `patsy`, usada para
        especificar as variáveis dependentes e independentes no modelo.
        Exemplo: 'y ~ x1 + x2'.
    """
    try:
        model = smf.ols(formula, data=df).fit()
        return model
    except:
        return None

# Calcula a tabela ANOVA
def calculate_anova_table(model):
    """
    Calcula a tabela ANOVA com base em um modelo ajustado.

    Parâmetros:
    ----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Um objeto de modelo ajustado retornado por `statsmodels.ols().fit()`.
        Representa o modelo de regressão linear usado para calcular a tabela ANOVA.
    """
    tableToRead = sm.stats.anova_lm(model, typ=2)
    tableToRead = tableToRead.fillna('')
    return tableToRead

# Calcula a fórmula de efeitos principais
def construct_main_effects_formula(items: List[str]):
    """
    Constrói uma fórmula para análise de efeitos principais, incluindo variáveis categóricas.

    Parâmetros:
    ----------
    items : List[str]
        Uma lista de nomes de variáveis (strings) que devem ser incluídas na fórmula.
        As variáveis serão tratadas como categóricas na construção da fórmula.
    """
    formula = ''
    for index, item in enumerate(items):
        if index < len(items) - 2:
            formula += 'C(' + item + ') + '
        elif index == len(items) - 2:
            formula += 'C(' + item + ')'
    return formula

# Calcula a fórmula de efeitos de interação
def construct_interaction_effects_formula(items: List[str]):
    """
    Constrói uma fórmula para análise de efeitos de interação entre variáveis categóricas.

    Parâmetros:
    ----------
    items : List[str]
        Uma lista de nomes de variáveis (strings) que devem ser incluídas na fórmula.
        Cada variável será tratada como categórica, e pares únicos de interações
        entre variáveis serão construídos e adicionados à fórmula.
    """
    formula = ''
    for x in range(len(items) - 1):
        for y in range(len(items) - 1):
            if items[x] != items[y]:
                partial_string_r = 'C(' + items[x] + ')*C(' + items[y] + ')'
                partial_string_l = 'C(' + items[y] + ')*C(' + items[x] + ')'
                if partial_string_r not in formula and partial_string_l not in formula:
                    if y < len(items) - 2:
                        formula += partial_string_r + '+'
    return formula

# Combina strings separadas por '+'
def combine_strings(string_splited: str, separator=' + '):
    """
    Combina uma sequência de strings separadas por um delimitador.

    Parâmetros:
    ----------
    sStringSplited : str
        Uma string contendo partes separadas a serem combinadas.
    separator : str, opcional
        O separador usado para combinar as strings. O padrão é `' + '`.
    """
    combined_string = ""
    for x in range(len(string_splited) - 1):
        if x < len(string_splited) - 2:
            combined_string += string_splited[x] + separator
        elif x == len(string_splited) - 2:
            combined_string += string_splited[x]
    return combined_string

# Função para calcular a média e amplitude
def calculate_mean_and_amplitude(df: pd.DataFrame, columns: List[str], columnsX: List[str], x):
    """
    Calcula a média e a amplitude de grupos de dados em um DataFrame, organizados por agrupamentos sucessivos.

    Parâmetros:
    ----------
    df : pd.DataFrame
        O DataFrame contendo os dados a serem processados. É necessário que a coluna 'line' esteja presente para indexação.
    columns : List[str]
        Uma lista de nomes de colunas, onde a última coluna será utilizada para calcular a média e a amplitude.
    columnsX : List[str]
        Uma lista de nomes de colunas para agrupamento. A cada iteração, uma coluna é removida para criar agrupamentos sucessivos.
    x : qualquer
        Um parâmetro auxiliar cujo papel não está claramente definido nesta função.
    """
    groupedItemsFirst = {}
    groupedItemsLast = {}
    rBar = {}
    xArrColRevToCut = columnsX.copy()
    del xArrColRevToCut[-1]

    for v in range(len(xArrColRevToCut)):
        firstGroup = df.groupby(xArrColRevToCut).first().reset_index()
        lastGroup = df.groupby(xArrColRevToCut).last().reset_index()
        groupedItemsFirst[v] = firstGroup
        groupedItemsLast[v] = lastGroup
        amp = []
        mean = []

        for x in range(len(firstGroup)):
            lowerValue = None
            greaterValue = None
            dataToMean = []
            lenT = 0
            if v == 0:
                for y in range(firstGroup['line'][x], lastGroup['line'][x]+1):
                    dataToMean.append(df[columns[-1]][y])
                    if greaterValue is None or greaterValue < df[columns[-1]][y]:
                        greaterValue = df[columns[-1]][y]
                    if lowerValue is None or lowerValue > df[columns[-1]][y]:
                        lowerValue = df[columns[-1]][y]
                amp.append(greaterValue - lowerValue)
                mean.append(sum(dataToMean)/len(dataToMean))
                lenT = len(df) // len(amp)
            else:
                lenT = len(groupedItemsFirst[v-1]) // len(groupedItemsFirst[v])
                for z in range(x*lenT, ((x+1)*lenT)):
                    if greaterValue is None or greaterValue < groupedItemsFirst[v-1]['mean'][z]:
                        greaterValue = groupedItemsFirst[v-1]['mean'][z]
                    if lowerValue is None or lowerValue > groupedItemsFirst[v-1]['mean'][z]:
                        lowerValue = groupedItemsFirst[v-1]['mean'][z]
                for y in range(firstGroup['line'][x], lastGroup['line'][x]+1):
                    dataToMean.append(df[columns[-1]][y])
                amp.append(greaterValue - lowerValue)
                mean.append(sum(dataToMean)/len(dataToMean))

        groupedItemsFirst[v].insert(len(columnsX)+2, 'mean', mean)
        groupedItemsFirst[v].insert(len(columnsX)+3, 'amp', amp)
        rBar[v] = {'rBar': sum(amp)/len(amp), 'size': lenT}
        del xArrColRevToCut[-1]

    lowerValue = None
    greaterValue = None
    for x in groupedItemsFirst[len(columnsX)-2]['mean']:
        if greaterValue is None or greaterValue < x:
            greaterValue = x
        if lowerValue is None or lowerValue > x:
            lowerValue = x
    rBar[len(columnsX)-1] = {'rBar': greaterValue - lowerValue, 'size': 2}

    return rBar

tabelaSubgroup =  {
        2: {'A2': 1.880, 'd2': 1.128, 'D3': 0, 'D4': 3.267},
        3: {'A2': 1.023, 'd2': 1.693, 'D3': 0, 'D4': 2.574},
        4: {'A2': 0.729, 'd2': 2.059, 'D3': 0, 'D4': 2.282},
        5: {'A2': 0.577, 'd2': 2.326, 'D3': 0, 'D4': 2.114},
        6: {'A2': 0.483, 'd2': 2.534, 'D3': 0, 'D4': 2.004},
        7: {'A2': 0.419, 'd2': 2.704, 'D3': 0.076, 'D4': 1.924},
        8: {'A2': 0.373, 'd2': 2.847, 'D3': 0.136, 'D4': 1.864},
        9: {'A2': 0.337, 'd2': 2.970, 'D3': 0.184, 'D4': 1.816},
        10: {'A2': 0.308, 'd2': 3.078, 'D3': 0.223, 'D4': 1.777},
        11: {'A2': 0.285, 'd2': 3.173, 'D3': 0.256, 'D4': 1.744},
        12: {'A2': 0.266, 'd2': 3.258, 'D3': 0.283, 'D4': 1.717},
        13: {'A2': 0.249, 'd2': 3.336, 'D3': 0.307, 'D4': 1.693},
        14: {'A2': 0.235, 'd2': 3.407, 'D3': 0.328, 'D4': 1.672},
        15: {'A2': 0.223, 'd2': 3.472, 'D3': 0.347, 'D4': 1.653},
        16: {'A2': 0.212, 'd2': 3.532, 'D3': 0.363, 'D4': 1.637},
        17: {'A2': 0.203, 'd2': 3.588, 'D3': 0.378, 'D4': 1.622},
        18: {'A2': 0.194, 'd2': 3.640, 'D3': 0.391, 'D4': 1.608},
        19: {'A2': 0.187, 'd2': 3.689, 'D3': 0.403, 'D4': 1.597},
        20: {'A2': 0.180, 'd2': 3.735, 'D3': 0.415, 'D4': 1.585},
        21: {'A2': 0.173, 'd2': 3.778, 'D3': 0.425, 'D4': 1.575},
        22: {'A2': 0.167, 'd2': 3.819, 'D3': 0.434, 'D4': 1.566},
        23: {'A2': 0.162, 'd2': 3.858, 'D3': 0.443, 'D4': 1.557},
        24: {'A2': 0.157, 'd2': 3.895, 'D3': 0.451, 'D4': 1.548},
        25: {'A2': 0.153, 'd2': 3.931, 'D3': 0.459, 'D4': 1.541}
    }

# Função para calcular a tabela de variação
def calculate_variation_table(rBar: dict, xArrCol: List[str]):
    """
    Calcula a tabela de variação a partir de informações de amplitude e fatores de correção.

    Parâmetros:
    ----------
    rBar : dict
        Um dicionário contendo informações de amplitude média ('rBar') e tamanho do grupo ('size') 
        para diferentes níveis de agrupamento. As chaves representam os níveis.
    xArrCol : List[str]
        Uma lista de nomes de colunas (strings) que representam os níveis de agrupamento em ordem.
    """
    result = {}
    totalVariance = 0

    for x in rBar.keys():
        if x == 0:
            rBarSize = rBar[x]['size']
            if rBarSize < 2:
                rBarSize = 2
            d2 = tabelaSubgroup[rBarSize]['d2']
            rBarC = rBar[x]['rBar']
            variance = (rBarC / d2)**2
            desvpad = variance ** 2
            result[x] = {'variance': variance, 'desvpad': desvpad, 'percentage': 0.0}
        else:
            varianceBefore = 0.0
            sizeC = 1
            for y in range(x, 0, -1):
                sizeC = sizeC * rBar[y-1]['size']
                varianceBefore += result[y-1]['variance']/sizeC
            d2 = tabelaSubgroup[rBar[x]['size']]['d2']
            rBarC = rBar[x]['rBar']
            variance = (rBarC / d2)**2 - varianceBefore
            desvpad = variance ** 2
            result[x] = {'variance': variance, 'desvpad': desvpad, 'percentage': 0.0}

        if variance > 0:
            totalVariance += variance

    xArrColRev = xArrCol.copy()
    xArrColRev.reverse()

    for x in range(len(xArrColRev)):
        if result[x]['variance'] > 0:
            result[x]['percentage'] = (result[x]['variance']/totalVariance)*100
        result[xArrColRev[x]] = result.pop(x)

    result['total'] = totalVariance
    return result

# valida se os dados estão balanceados
def check_balanced(df: pd.DataFrame, type: str):
    """
    Verifica se os dados em um DataFrame estão balanceados, com base no número de valores únicos por coluna.

    Parâmetros:
    ----------
    df : pd.DataFrame
        O DataFrame contendo os dados a serem verificados. Pode incluir colunas adicionais que serão excluídas
        dependendo do tipo especificado.
    type : str
        O tipo de estrutura dos dados:
        - "crossed": remove a última coluna antes de verificar os valores únicos.
        - "nested": remove as duas últimas colunas antes de verificar os valores únicos.
    """
    if type == "crossed":
        df = df.iloc[:, :-1]
    elif type == "nested":
        df = df.iloc[:, :-2]
    
    unique_value_counts = df.apply(lambda col: col.nunique())

    if unique_value_counts.nunique() == 1:
        return True  # Se todas as colunas tiverem o mesmo número de valores únicos
    else:
        return False  # Se o número de valores únicos em qualquer coluna for diferente

# Calcula quadrado médio
def calculate_mean_square(sum_sq, df_values: pd.Series):
    """
    Calcula o quadrado médio (mean square) para cada elemento baseado na soma dos quadrados e nos valores de graus de liberdade.

    Parâmetros:
    ----------
    sum_sq : list ou array-like
        Uma lista contendo os valores da soma dos quadrados (sum of squares) para diferentes fontes de variação.
    df_values : pd.Series
        Uma série contendo os graus de liberdade (df) correspondentes a cada soma dos quadrados.
    """
    return [sum_sq[x]/df_values[x] for x in range(len(sum_sq))]

# Calcula total de percentual
def calculate_percent_total(sum_sq, total_sum_sq: int):
    """
    Calcula a porcentagem de cada soma dos quadrados em relação à soma total dos quadrados.

    Parâmetros:
    ----------
    sum_sq : list ou array-like
        Uma lista contendo os valores da soma dos quadrados (sum of squares) para diferentes fontes de variação.
    total_sum_sq : int
        O valor total da soma dos quadrados (total sum of squares), que será utilizado para calcular a porcentagem.
    """
    values = [round(sum_sq[x]*100/total_sum_sq, 3) for x in range(len(sum_sq)-1)]
    return values



meu antigo front end

import React, { useEffect, useState } from "react";
import { useTranslation } from "next-i18next";
import { Row, Col, message, notification } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useAuth } from "hooks/useAuth";
import axios from "axios";
import { Spin } from "shared/spin"
import { CalculateMessageData } from "shared/calculateMessage/interface";
import ContentHeader from "shared/contentHeader";
import Table from "shared/table";
import {
  CovExportData,
  DataToCompile,
  SourcesVarianceComponents,
  VarianceComponents,
} from "../interface";
import { P_VALUE_LIMIT } from "utils/constant";
import ResponseContentHeader from "shared/responseContentHeader";
import { WarningOutlined } from "@ant-design/icons";
import dynamic from "next/dynamic";
import { formatNumber } from "utils/formatting";
import {
  toDicitionary,
  dataSortOrder,
  sortDictionaryFiltered,
} from "utils/core";
import { useRouter } from "next/router";
import { getItem } from "utils/database";
import { HighChartTemplate } from "shared/widget/chartHub/interface";
import { COLORS } from "utils/color";

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
});

const CalculateMessage = dynamic(() => import("shared/calculateMessage"), {
  ssr: false,
});

const fetcher = axios.create({
  baseURL: "/api",
});

export const COV: React.FC = () => {
  const { t: commonT } = useTranslation("common");

  const { user } = useAuth();
  const router = useRouter();

  const [sources, setSources] = useState<
    Record<string, SourcesVarianceComponents[]>
  >({});
  const [variances, setVariances] = useState<
    Record<string, VarianceComponents[]>
  >({});

  const [dataToShow, setDataToShow] = useState<HighChartTemplate>({});
  const [covOption, setCovOption] = useState("");
  const [messageApi, contextHolder] = message.useMessage();
  const [isLoaded, setIsLoaded] = useState(false);

  const [loadingPage, setLoadingPage] = useState(true);
  const [isBalanced, setIsBalanced] = useState(false);
  const [isNegativeVariance, setIsNegativeVariance] = useState(false);

  const [loadingAnova, setLoadingAnova] = useState(true);
  const [loadingVarianceComponents, setLoadingVarianceComponents] =
    useState(true);

  const [responseColumns, setResponseColumns] = useState<string[]>([]);
  const [contentVisibility, setContentVisibility] = useState<boolean[]>(
    new Array(responseColumns.length).fill(false)
  );
  const LIMIT_TO_SHOW = 0.0001;

  const columnsToSources: ColumnsType<any> = [
    {
      title: commonT("cov.sources"),
      dataIndex: "key",
      key: "key",
      align: "center",
      sorter: (a, b) => a.key.localeCompare(b.key),
      onFilter: (value, record) => record.key.startsWith(value),
      filterSearch: true,
    },
    {
      title: commonT("cov.degreesOfFreedom"),
      dataIndex: "df",
      key: "df",
      align: "center",
      sorter: (a, b) => a.df - b.df,
    },
    {
      title: commonT("cov.sSquare"),
      dataIndex: "sSquare",
      key: "sSquare",
      align: "center",
      render: (item) => (
        <div>{typeof item === "number" ? formatNumber(item) : item}</div>
      ),
      sorter: (a, b) => a.sSquare - b.sSquare,
    },
    {
      title: commonT("cov.mSquare"),
      dataIndex: "mSquare",
      key: "mSquare",
      align: "center",
      render: (item) => (
        <div>{typeof item === "number" ? formatNumber(item) : item}</div>
      ),
      sorter: (a, b) => a.mSquare - b.mSquare,
    },
    {
      title: commonT("cov.fRatio"),
      dataIndex: "fRatio",
      key: "fRatio",
      align: "center",
      render: (item) => (
        <div>{typeof item === "number" ? formatNumber(item) : item}</div>
      ),
      sorter: (a, b) => a.fRatio - b.fRatio,
    },
    {
      title: commonT("cov.probF"),
      dataIndex: "probF",
      key: "probF",
      align: "center",
      render: (item) => (
        <div>
          {typeof item !== "number"
            ? item
            : item >= LIMIT_TO_SHOW
            ? formatNumber(item)
            : "< 0.0001*"}
        </div>
      ),
      onCell: (item: any) => {
        return {
          ["style"]: {
            color:
              typeof item.probF === "number" && item.probF < P_VALUE_LIMIT
                ? "red"
                : "",
          },
        };
      },
      sorter: (a, b) => a.probF - b.probF,
    },
  ];

  const columnsToVariances: ColumnsType<any> = [
    {
      title: commonT("cov.components"),
      dataIndex: "key",
      key: "key",
      align: "center",
      sorter: (a, b) => a.key.localeCompare(b.key),
      onFilter: (value, record) => record.key.startsWith(value),
      filterSearch: true,
      render: (item) => (
        <b>
          {!commonT("cov." + item).includes("cov")
            ? commonT("cov." + item)
            : item}
        </b>
      ),
    },
    {
      title: commonT("cov.total%"),
      dataIndex: "total",
      key: "total",
      align: "center",
      render: (item) => (
        <div>{typeof item === "number" ? item.toFixed(4) : item}</div>
      ),
      sorter: (a, b) => a.total - b.total,
    },
  ];

  // const terms = variances.map((el) => el.key);
  const columnsToVariancesNested: ColumnsType<any> = [
    {
      title: commonT("cov.components"),
      dataIndex: "key",
      key: "key",
      align: "center",
      sorter: (a, b) => a.key.localeCompare(b.key),
      onFilter: (value, record) => record.key.startsWith(value),
      filterSearch: true,
      render: (item) => (
        <b>{item === "measurements" ? commonT("cov." + item) : item}</b>
      ),
    },
    {
      title: commonT("cov.variance"),
      dataIndex: "variance",
      key: "variance",
      align: "center",
      render: (item) => (
        <div>{item >= 0 ? item.toExponential(4) : (0.0).toFixed(4)}</div>
      ),
      sorter: (a, b) => a.variance - b.variance,
    },
    {
      title: commonT("cov.standardDeviation"),
      dataIndex: "desvpad",
      key: "desvpad",
      align: "center",
      render: (item) => <div>{item ? item.toExponential(4) : ""}</div>,
      sorter: (a, b) => a.desvpad - b.desvpad,
    },
    {
      title: commonT("cov.total%"),
      dataIndex: "total",
      key: "total",
      align: "center",
      render: (item) => <div>{item ? item.toFixed(2) : ""}</div>,
      sorter: (a, b) => a.total - b.total,
    },
  ];

  const getReplaceLabelCrossed = (label: string) => {
    return label
      .replace("C(", "")
      .replaceAll(")", "")
      .replaceAll(") |", "*")
      .replaceAll(" | C(", "*");
  };

  useEffect(() => {
    const getData = async () => {
      const parsedUrlQuery = router?.query;

      if (Object.keys(parsedUrlQuery).length > 0) {
        const tool = parsedUrlQuery.tool as string;
        const uid = parsedUrlQuery.uid as string;

        const item = (await getItem(tool, uid)) as CovExportData;

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

          setCovOption(item.covOption);

          setLoadingPage(false);
          const responseColumn = item.responseColumn;

          let isBalanced = false;

          // filtra colunas de resposta para nao ser removidas caso tenha somente 1 subgrupo
          const keys = Object.keys(dataToSend.obj).filter((el) =>
            responseColumn.find((responseKey) => responseKey !== el)
          );

          // remove fontes que só tem um subgrupo
          keys.forEach((key) => {
            const values = dataToSend.obj[key];
            const uniqueValues = new Set(values).size === 1;
            if (uniqueValues) {
              delete dataToSend.obj[key];
              const index = dataToSend.itens.indexOf(key);
              if (index !== -1) {
                dataToSend.itens.splice(index, 1);
              }
            }
          });

          if (keys.length <= 2) {
            notification.open({
              message: commonT("cov.factorWithOneLevelTitle"),
              description: commonT("cov.factorWithOneLevelMessage"),
              duration: 15,
              icon: <WarningOutlined style={{ color: "#ffcc00" }} />,
            });
          }

          try {
            const balancedData = {
              obj: dataToSend.obj,
              itens: dataToSend.itens,
              type: item.covOption,
            };

            const { data } = await fetcher.post<CalculateMessageData>(
              "cov/validateBalanced",
              balancedData
            );
            isBalanced = data.check;
          } catch (error: any) {
            console.error(error);
            if (error.response.data === "covBalanceError") {
              message.error({
                content: commonT("error.cov.balanceErrorMsg"),
              });
            } else {
              message.error({
                content: commonT("error.general.proccesData"),
              });
            }
          }

          const crossedData = {
            obj: dataToSend.obj,
            responseColumn: responseColumn,
            itens: dataToSend.itens,
          };

          setIsBalanced(isBalanced);
          if (covOption === "crossed") {
            try {
              const { data: covCrossedData } = await fetcher.post(
                "cov/emsCrossed",
                crossedData
              );

              for (const key of Object.keys(covCrossedData)) {
                const data = covCrossedData[key].table;

                if (data !== "notCrossed") {
                  const keys = Object.keys(data?.sum_sq);
                  const sourcesArray: SourcesVarianceComponents[] = [];
                  const varianceArray: VarianceComponents[] = [];

                  let within = 0;
                  keys.map((key) => {
                    sourcesArray.push({
                      key:
                        key === "Residual" ? "Erro" : key.replace(":", " | "),
                      sSquare: data?.sum_sq[key],
                      df: data?.df[key],
                      mSquare: data?.sum_mq[key],
                      probF: data["PR(>F)"]?.[key],
                      fRatio: data?.F[key],
                    });
                    varianceArray.push({
                      key:
                        key === "Residual" ? "Within" : key.replace(":", " | "),
                      total: key === "Total" ? 100 : data?.totalPercentual[key],
                    });
                    within += data?.totalPercentual[key];
                  });
                  const withinIndex = varianceArray.findIndex(
                    (e) => e.key === "Within"
                  );
                  varianceArray[withinIndex].total = 100 - within;

                  const newSources = [];
                  const newVariances = [];

                  for (const iterator of sourcesArray) {
                    const data = {
                      key: getReplaceLabelCrossed(String(iterator.key)),
                      sSquare: iterator.sSquare,
                      df: iterator.df,
                      fRatio: iterator.fRatio,
                      mSquare: iterator.mSquare,
                      probF: iterator.probF,
                    };
                    newSources.push(data);
                  }

                  setSources((prevState) => ({
                    ...prevState,
                    [key]: newSources,
                  }));

                  setLoadingAnova(false);

                  for (const iterator of varianceArray) {
                    const data = {
                      key: getReplaceLabelCrossed(String(iterator.key)),
                      total: iterator.total,
                    };
                    newVariances.push(data);
                  }

                  setVariances((prevState) => ({
                    ...prevState,
                    [key]: newVariances,
                  }));

                  setLoadingVarianceComponents(false);

                  const dataHistogram: number[] = [];
                  const labelsHistogram: string[] = [];

                  varianceArray.map((data) => {
                    labelsHistogram.push(getReplaceLabelCrossed(data.key));
                    dataHistogram.push(data.total);
                  });

                  for (const item of newVariances) {
                    if (item.total <= 0) {
                      setIsNegativeVariance(true);
                    }
                  }

                  const histogramDictionary = toDicitionary(
                    labelsHistogram,
                    dataHistogram
                  );

                  const sortedDictionary: { [key: string]: number } =
                    dataSortOrder(histogramDictionary, "asc", false);

                  const sortedDictionaryFiltered: { [key: string]: number } =
                    sortDictionaryFiltered(sortedDictionary, "ems");

                  const sortedDataHistogram: number[] = Object.entries(
                    sortedDictionaryFiltered
                  ).map(([, value]) => value);

                  const sortedLabelsHistogram = Object.entries(
                    sortedDictionaryFiltered
                  ).map(([key]) => key);

                  const chartVariable = buildChart(
                    sortedDataHistogram,
                    sortedLabelsHistogram,
                    key
                  );

                  setDataToShow(chartVariable);
                } else {
                  messageApi.open({
                    type: "error",
                    content: commonT("cov.notCrossed") + ".",
                  });
                }
              }

              setTimeout(() => {
                setResponseColumns(Object.keys(covCrossedData));
              }, 300);
            } catch (error: any) {
              if (error.response.data === "covError") {
                message.error({
                  content: commonT("error.general.proccesData"),
                });
              } else if (error.response.data === "covAnovaError") {
                message.error({
                  content: commonT("error.cov.covFormulaErrorMsg"),
                });
              } else {
                message.error({
                  content: commonT("error.general.unexpectedMsg"),
                });
              }
            }
          }

          if (covOption === "nested") {
            try {
              const emsCalculate = {
                obj: dataToSend.obj,
                itens: dataToSend.itens,
                responseColumn: responseColumn,
              };

              const { data } = await fetcher.post(
                "cov/emsNested",
                emsCalculate
              );

              for (const key of Object.keys(data)) {
                const emsData = data[key];

                if (emsData !== "notBalanced") {
                  const varianceArray = [];
                  const variablesReverse = [
                    ...dataToSend.itens.filter(
                      (el) => !responseColumn.includes(el)
                    ),
                  ];

                  variablesReverse.reverse();
                  variablesReverse.map((el) => {
                    varianceArray.push({
                      key: el,
                      variance: emsData[el].variance,
                      desvpad: emsData[el].desvpad,
                      total: emsData[el].percentage,
                    });
                  });
                  varianceArray.push({
                    key: "Total",
                    variance: emsData["total"],
                    total: 100,
                  });

                  const dataHistogram: number[] = [];
                  const labelsHistogram: string[] = [];
                  varianceArray.map((data) => {
                    labelsHistogram.push(
                      data.key === "measurements"
                        ? commonT("cov." + data.key)
                        : data.key
                    );
                    dataHistogram.push(data.total);
                  });

                  const histogramDictionary = toDicitionary(
                    labelsHistogram,
                    dataHistogram
                  );

                  const sortedDictionary: { [key: string]: number } =
                    dataSortOrder(histogramDictionary, "asc", false);

                  const sortedDictionaryFiltered: { [key: string]: number } =
                    sortDictionaryFiltered(sortedDictionary, "ems");

                  const sortedDataHistogram: number[] = Object.entries(
                    sortedDictionaryFiltered
                  ).map(([, value]) => value);

                  const sortedLabelsHistogram = Object.entries(
                    sortedDictionaryFiltered
                  ).map(([key]) => key);

                  const chartVariable = buildChart(
                    sortedDataHistogram,
                    sortedLabelsHistogram,
                    key
                  );

                  setDataToShow(chartVariable);

                  for (const item of varianceArray) {
                    if (item.variance <= 0) {
                      setIsNegativeVariance(true);
                    }
                  }

                  setVariances((prevState) => ({
                    ...prevState,
                    [key]: varianceArray,
                  }));

                  setLoadingVarianceComponents(false);
                } else {
                  messageApi.open({
                    type: "error",
                    content: commonT("cov.notBalanced") + ".",
                  });
                }
              }

              setTimeout(() => {
                setResponseColumns(Object.keys(data));
              }, 300);
            } catch (error: any) {
              if (error.response.data === "covError") {
                message.error({
                  content: commonT("error.general.proccesData"),
                });
              } else if (error.response.data === "covAnovaError") {
                message.error({
                  content: commonT("error.cov.covFormulaErrorMsg"),
                });
              } else {
                message.error({
                  content: commonT("error.general.unexpectedMsg"),
                });
              }
            }
          }

          setIsLoaded(true);
        }
      }
    };
    getData().catch(console.error);
  }, [covOption, router.query]);

  function buildChart(values: number[], labels: string[], key: string) {
    const chartsForVariable: HighChartTemplate = {};

    chartsForVariable["bar"] = {
      seriesData: [
        {
          data: values,
          name: commonT("cov.dataLegend"),
          type: "bar",
          showLegend: true,
        },
      ],
      options: {
        title: commonT("cov.chartTitle"),
        xAxisTitle: "",
        yAxisTitle: commonT("varianceCompare.variance") + " (%)",
        colors: COLORS,
        titleFontSize: 24,
        chartName: key,
        legendEnabled: true,
        categories: labels,
        barIsVertical: false,
        dataLabelsEnabled: true,
      },
      type: "bar",
      displayName: commonT("cov.chartTitle"),
    };

    return chartsForVariable;
  }

  useEffect(() => {
    const countAccess = async () => {
      const isStorageRecalculate =
        localStorage.getItem("redo") && localStorage.getItem("redo") === "true"
          ? true
          : false;
      if (isStorageRecalculate) return;
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "cov",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  useEffect(() => {
    // empty use Effect
  }, [responseColumns, sources, variances]);

  useEffect(() => {
    // empty use effect
  }, [isNegativeVariance, isBalanced]);

  const covTitle = commonT("cov.title") + " - " + commonT("cov." + covOption);

  const buttonProps = {
    name: commonT("redo"),
  };

  const onToggleContent = (index: number) => {
    const updatedVisibility = [...contentVisibility];
    updatedVisibility[index] = !updatedVisibility[index];
    setContentVisibility(updatedVisibility);
  };

  return (
    <>
      {contextHolder}
      <ContentHeader
        title={covTitle}
        tool={"cov"}
        enableRecalculate={true}
        recalculateButtonProps={buttonProps}
      />

      {loadingPage ? (
        <>
          <Spin />
        </>
      ) : (
        <div id="content">
          {responseColumns.length > 0 ? (
            <>
              {responseColumns.map((key, index) => (
                <>
                  <ResponseContentHeader
                    title={key}
                    index={index}
                    onClick={() => onToggleContent(index)}
                  />
                  {covOption === "crossed" ? (
                    <>
                      {sources[key].length > 0 ? (
                        <>
                          <h3 style={{ marginBottom: "20px" }}>
                            {commonT("cov.covCrossedMessage")}
                          </h3>
                          <Row
                            style={{ marginBottom: "30px" }}
                            hidden={contentVisibility[index]}
                          >
                            <Col
                              xs={24}
                              sm={24}
                              md={24}
                              lg={24}
                              xl={24}
                              xxl={13}
                            >
                              <div style={{ marginBottom: "20px" }}>
                                <Table
                                  loading={loadingAnova}
                                  dataSource={sources[key]}
                                  columns={columnsToSources}
                                  title={commonT("cov.analysisOfVariance")}
                                />
                              </div>
                            </Col>
                            <Col
                              xs={24}
                              sm={24}
                              md={24}
                              lg={24}
                              xl={24}
                              xxl={{ offset: 1, span: 10 }}
                            >
                              <Table
                                loading={loadingVarianceComponents}
                                dataSource={variances[key]}
                                columns={columnsToVariances}
                                title={commonT("cov.varianceComponentsTitle")}
                              />
                            </Col>
                          </Row>
                          <Row
                            style={{ marginBottom: 20 }}
                            hidden={contentVisibility[index]}
                          >
                            <Col
                              xs={24}
                              sm={24}
                              md={24}
                              lg={24}
                              xl={24}
                              xxl={13}
                            >
                              {Object.keys(dataToShow).length > 0 && (
                                <>
                                  <ChartHub
                                    chartConfigurations={dataToShow}
                                    tool={"cov"}
                                  />
                                </>
                              )}
                            </Col>
                            <Col offset={1} span={10}>
                              {!isNegativeVariance && isBalanced ? (
                                <></>
                              ) : (
                                <>
                                  <CalculateMessage
                                    isBalanced={isBalanced}
                                    covType={"ems"}
                                    isNegativeVariance={isNegativeVariance}
                                  />
                                </>
                              )}
                            </Col>
                          </Row>
                        </>
                      ) : (
                        <>{commonT("cov.notCrossed")}</>
                      )}
                    </>
                  ) : (
                    <>
                      <h3 style={{ marginBottom: "20px" }}>
                        {commonT("cov.covNestedMessage")}
                      </h3>
                      {variances[key].length > 0 ? (
                        <>
                          <Row
                            style={{ marginBottom: "30px" }}
                            hidden={contentVisibility[index]}
                          >
                            <Col
                              xs={24}
                              sm={24}
                              md={24}
                              lg={24}
                              xl={24}
                              xxl={11}
                            >
                              <Table
                                loading={false}
                                dataSource={variances[key]}
                                columns={columnsToVariancesNested}
                                title={commonT("cov.varianceComponentsTitle")}
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
                              {isLoaded ? (
                                <>
                                  {Object.keys(dataToShow).length > 0 && (
                                    <>
                                      <ChartHub
                                        chartConfigurations={dataToShow}
                                        tool={"cov"}
                                      />
                                    </>
                                  )}
                                </>
                              ) : (
                                <>
                                  <Spin />
                                </>
                              )}
                            </Col>
                          </Row>
                          <Row hidden={contentVisibility[index]}>
                            <Col
                              xs={24}
                              sm={24}
                              md={24}
                              lg={24}
                              xl={24}
                              xxl={11}
                            >
                              {!isNegativeVariance && !isBalanced ? (
                                <></>
                              ) : (
                                <div style={{ marginBottom: 20 }}>
                                  <CalculateMessage
                                    isBalanced={isBalanced}
                                    covType={"ems"}
                                    isNegativeVariance={isNegativeVariance}
                                  />
                                </div>
                              )}
                            </Col>
                          </Row>
                        </>
                      ) : (
                        <>
                          {commonT("cov.notBalanced")}
                          <>
                            <Spin />
                          </>
                        </>
                      )}
                    </>
                  )}
                </>
              ))}
            </>
          ) : (
            <></>
          )}
        </div>
      )}
    </>
  );
};
