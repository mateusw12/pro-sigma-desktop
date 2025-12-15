# Vamos fazer a analise de central composite design com interações que o usuário pode escolher como opcional (sempre interações quadraticas), varios X e vários Y, gere gráficos de superficie também. Vai ter a parte de gerar o experimento também igual no space filling

meu back end antigo em python 

from typing import List
import numpy as np
import pandas as pd
from scipy.stats import f

# Calcula erro puro
def calculate_pure_error(df: pd.DataFrame):
    """
    Calcula os erros quadráticos puros (squared pure errors) para as combinações duplicadas de valores
    das linhas do DataFrame, ignorando a última coluna.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados. A última coluna é considerada a variável dependente.
    """
    #Removera primeira coluna e colunas com hífen no nome
    df.drop(columns=['Intercept'], inplace=True)

    # Percorre as linhas do DataFrame, ignorando a última coluna
    duplicates = {}
    for index, row in df.iloc[:, :-1].iterrows():
        concatenated_values = '-'.join([str(value) for value in row])
        duplicates.setdefault(concatenated_values, []).append(index)

    duplicates = {key: value for key, value in duplicates.items() if len(value) >= 2}

    # Percorre as chaves do dicionário de duplicatas
    sq_pure_error: List[float] = []
    for duplicate_indexes in duplicates.values():
        last_column_values = df.loc[duplicate_indexes, df.columns[-1]].tolist()
        average = np.mean(last_column_values)
        squared_diff_sum = sum((value - average) ** 2 for value in last_column_values)
        sq_pure_error.append(squared_diff_sum)

    return sq_pure_error

# Calcula erro puro
def calculate_degreess_freedom_pure_error(df: pd.DataFrame):
    """
    Calcula os graus de liberdade associados ao erro puro para as combinações duplicadas de valores
    das linhas do DataFrame, ignorando a última coluna.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados. A última coluna é considerada a variável dependente.
    """
    # Percorre as linhas do DataFrame, ignorando a última coluna
    duplicates = {}

    for index, row in df.iloc[:, :-1].iterrows():
        concatenated_values = '-'.join([str(value) for value in row])
        duplicates.setdefault(concatenated_values, []).append(index)

    duplicates = {key: value for key, value in duplicates.items() if len(value) >= 2}
    keys = 0

    for _, value in duplicates.items():
        keys = len(value)

    return keys -1 if keys > 2 else keys

# Calcula lack os fit
def calculate_lack_of_fit(degress_freedom_pure_error: float, degress_freedom_error: int, sq_erro_total: float, pure_error_total: float):
    """
    Calcula o Lack of Fit e suas estatísticas associadas.

    Parâmetros:
        degress_freedom_pure_error (float): Graus de liberdade do erro puro.
        degress_freedom_error (int): Graus de liberdade total do erro.
        sq_erro_total (float): Soma total de quadrados do erro.
        pure_error_total (float): Soma de quadrados do erro puro.
    """

    s_square_lack_of_fit = sq_erro_total - pure_error_total
    degress_freedom_lack_of_fit = degress_freedom_error - degress_freedom_pure_error

    degress_freedom_total = degress_freedom_lack_of_fit + degress_freedom_pure_error

    if degress_freedom_error > 0 and degress_freedom_lack_of_fit > 0:
        m_square_lack_of_fit = s_square_lack_of_fit / degress_freedom_lack_of_fit if degress_freedom_error != 0 else 0
    else:
        m_square_lack_of_fit = 0

    if degress_freedom_pure_error > 0:
        m_square_pure_error = pure_error_total / degress_freedom_pure_error if degress_freedom_pure_error != 0 else 0
    else:
        m_square_pure_error = 0 
    
    if m_square_pure_error > 0:
        f_ratio = m_square_lack_of_fit / m_square_pure_error if m_square_pure_error != 0 else 0
    else:
        f_ratio = 0

    if(degress_freedom_pure_error != 0):
        
        if f_ratio > 0:
            prob_f: float = round(f.sf(f_ratio, degress_freedom_error, degress_freedom_pure_error), 6)
        else: 
            prob_f = 0

        return {
        "grausLiberdade": {
            "lackOfFit": degress_freedom_lack_of_fit,
            "erroPuro": degress_freedom_pure_error,
            "total": degress_freedom_total
        },
        "sQuadrados":{
            "lackOfFit": s_square_lack_of_fit,
            "erroPuro": pure_error_total,
            "total": sq_erro_total
        },
        "mQuadrados":{
            "lackOfFit": m_square_lack_of_fit,
            "erroPuro": m_square_pure_error
        },
        "fRatio": f_ratio,
        "probF": prob_f
    }
    else:
        return {
        "grausLiberdade": {
            "lackOfFit": 0,
            "erroPuro": 0,
            "total": 0
        },
        "sQuadrados":{
            "lackOfFit": 0,
            "erroPuro": 0,
            "total": 0
        },
        "mQuadrados":{
            "lackOfFit": 0,
            "erroPuro": 0
        },
        "fRatio": 0,
        "probF": 0
        }

def create_interaction_columns(df: pd.DataFrame, columns_used: List[str]):
    """
    Cria colunas de interação no DataFrame com base em colunas especificadas.

    Parâmetros:
        df (pd.DataFrame): DataFrame original.
    """
    df = df.copy()
    
    for col in columns_used:
        if "_interaction_" in col:
            # Obter as colunas envolvidas na interação
            interaction_cols = col.replace("_interaction_", "-").split("-")
            
            # Garantir que as colunas existam no DataFrame
            if all(c in df.columns for c in interaction_cols):
                # Criar a nova coluna com o nome modificado
                new_col_name = col.replace("_interaction_", "-")
                
                # Calcular o produto das colunas
                df[new_col_name] = np.prod(df[interaction_cols], axis=1)
    
    return df

def create_quadratic_column(df: pd.DataFrame, columns_used: List[str]):
    """
    Cria colunas quadráticas no DataFrame com base em colunas especificadas.

    Parâmetros:
        df (pd.DataFrame): DataFrame original.
        columns_used (List[str]): Lista de nomes de colunas quadráticas a ser criada.
    """
    df = df.copy()
    
    for col in columns_used:
        if "_squared" in col:
            # Obter o nome da coluna original (antes de "_squared")
            base_col = col.replace("_squared", "")
            
            # Verificar se a coluna base existe no DataFrame
            if base_col in df.columns:
                # Criar o novo nome da coluna no formato "A-A"
                new_col_name = f"{base_col}-{base_col}"
                
                # Calcular o quadrado da coluna e adicioná-la ao DataFrame
                df[new_col_name] = df[base_col] ** 2
    
    return df


tenho meu back end em R, se for possivel altere para python

library(rsm)
#' Gera um design de experimentos composto central (CCD) ou Box-Behnken (BBD) com base nos parâmetros fornecidos.
#' Permite adicionar colunas de respostas com valores simulados ou padrões.
#'
#' @param basis Número de fatores no experimento. Deve ser um inteiro positivo.
#' @param centerPoint Número de pontos centrais no design. Deve ser um inteiro positivo.
#' @param type Tipo de design a ser gerado. Pode ser:
#'   - \code{"orthogonal"} ou \code{"rotatable"} para CCD padrão, indicando o tipo de alpha.
#'   - \code{"bbd"} para o design Box-Behnken.
#' @param columnResponseNumber Número de colunas de resposta a serem adicionadas ao design.
#' @param responseValues Vetor de valores a serem usados nas colunas de resposta. Se vazio, valores padrão serão adicionados.
#'
#' @return Um \code{data.frame} contendo o design gerado com as colunas de fatores e as colunas de resposta.
#' 
#' @details 
#' - Para designs CCD, o parâmetro \code{type} define o valor de alpha:
#'   - \code{"orthogonal"} para designs ortogonais.
#'   - \code{"rotatable"} para designs rotacionáveis.
#' - Para designs Box-Behnken, \code{type} deve ser \code{"bbd"}.
#' - Se \code{responseValues} for fornecido, os valores são atribuídos aleatoriamente às colunas de resposta.
#' - Se \code{responseValues} estiver vazio, um valor padrão de 99999 será atribuído.
generateCCD <- function(basis, centerPoint, type, columnResponseNumber, responseValues) {
  
  if(type == "bbd") {
    ccd_design <- bbd(k = basis, n0 = centerPoint, randomize = TRUE) # box-behnken
  } else {
    ccd_design <- ccd(basis = basis, n0 = centerPoint, alpha = type, randomize = TRUE) # ccd-padrão
  }
  
  df <- as.data.frame(ccd_design)

  df <- df[, grepl("x", names(df))]

  if (length(responseValues) > 0) {
    for (i in seq_len(columnResponseNumber)) {
      col_name <- paste0("Y", i)
      df[, col_name] <- sample(c(responseValues[1:min(length(responseValues), nrow(df))]))
    }
    
    df[is.na(df)] <- 0
    
  } else {
    for (i in seq_len(columnResponseNumber)) {
      col_name <- paste0("Y", i)
      df[, col_name] <- 99999
    }
  }
  
  result = df
  
  return(result) 
}

#' Calcula modelo Central Composite Design (CCD)
#'
#' Esta função cria um modelo usando Central Composite Design (CCD) com base em um determinado conjunto de dados. 
#' Ele gera termos quadráticos e de interação, ajusta um modelo linear, realiza ANOVA e retorna parâmetros do modelo, estatísticas resumidas e valores previstos.
#'
#' @param df Um data frame contendo os dados.
#' @param columnsUsed Um vetor de nomes de colunas usado para recalcular interação ou termos quadráticos. Ony usado se `recalculate` para TRUE.
#' @param recalculate Um booleano que indica se os termos quadráticos e de interação devem ser recalculados com base em `columnsUsed`. O padrão é FALSO.
#' @param responseColumn O nome da coluna de resposta (Y) no conjunto de dados.
calculateCcd <- function(df, columnsUsed, recalculate, responseColumn) {
  num_columns <- ncol(df)
  colnames(df)[num_columns] <- "Y"

  df_copy <- df
  
  colunas_sem_Y <- names(df)[!grepl("Y", names(df))]
  response_column <- names(df)[ncol(df)]

  if(recalculate == FALSE){
    
    for (coluna_sem_Y in colunas_sem_Y) {
      nome_coluna_quadratica <- paste0(coluna_sem_Y, "_squared")
      df[[nome_coluna_quadratica]] <- df[[coluna_sem_Y]] * df[[coluna_sem_Y]]
    }

    colunas_interacao <- combn(colunas_sem_Y, 2, FUN = function(x) paste(x, collapse = "_interaction_"), simplify = TRUE)
    
    for (coluna_interacao in colunas_interacao) {
      nome_coluna_interacao <- paste0(coluna_interacao)
      df[[nome_coluna_interacao]] <- df[[strsplit(coluna_interacao, "_interaction_")[[1]][1]]] * df[[strsplit(coluna_interacao, "_interaction_")[[1]][2]]]
    }
    
    # remova o termo Y da formula
    colunas_modelo <- setdiff(names(df), response_column)
    
    formula <- as.formula(paste(response_column, "~", paste0(colunas_modelo, collapse = " + ")))
    formula <- update(formula, . ~ . - response_column)
  
    model <- lm(formula, data = df)

    summaryModel = summary(model)
    
    parameterEstimates <- as.data.frame(summary(model)$coef)
    
    anova_result <- anova(model)
    anovaTable <- as.data.frame(anova_result[-nrow(anova_result), ])

    anova_summary <- data.frame(
      df = c(sum(anova_result[-nrow(anova_result), "Df"]),  
             anova_result[nrow(anova_result), "Df"],       
             sum(anova_result[-nrow(anova_result), "Df"], anova_result[nrow(anova_result), "Df"])), 
      
      sm = c(sum(anova_result[-nrow(anova_result), "Sum Sq"]),
             anova_result[nrow(anova_result), "Sum Sq"],      
             sum(anova_result[-nrow(anova_result), "Sum Sq"], anova_result[nrow(anova_result), "Sum Sq"])), 
      
      
      ms = c(sum(anova_result[-nrow(anova_result), "Sum Sq"]) / sum(anova_result[-nrow(anova_result), "Df"]), 
             anova_result[nrow(anova_result), "Mean Sq"],      
             sum(anova_result[-nrow(anova_result), "Mean Sq"], anova_result[nrow(anova_result), "Mean Sq"])), 
      
      
      fValue = c(sum(anova_result[-nrow(anova_result), "F value"]), 
                 anova_result[nrow(anova_result), "F value"],     
                 sum(anova_result[-nrow(anova_result), "F value"], anova_result[nrow(anova_result), "F value"])), 
      prob = c(sum(anova_result[-nrow(anova_result), "Pr(>F)"]),  
               anova_result[nrow(anova_result), "Pr(>F)"],       
               sum(anova_result[-nrow(anova_result), "Pr(>F)"], anova_result[nrow(anova_result), "Pr(>F)"])) 
    )
    
    row.names(anova_summary) <- c("Modelo", "Erro", "Total")

    names(parameterEstimates) <- gsub(" ", "_", names(parameterEstimates))
    
    names(parameterEstimates)[names(parameterEstimates) == "Pr(>|t|)"] <- "prob"
    names(parameterEstimates)[names(parameterEstimates) == "Std._Error"] <- "stdError"
    names(parameterEstimates)[names(parameterEstimates) == "Estimater"] <- "estimate"
    
  } else {
    
    colunas_interacao <- combn(colunas_sem_Y, 2, FUN = function(x) paste(x, collapse = "_interaction_"), simplify = TRUE)
  
    for (coluna in columnsUsed) {
      if (grepl("/", coluna)) {
        nome_coluna_quadratica <- sub("^.*/", "", coluna)
        colunas <- unlist(strsplit(coluna, "/"))
        nome_coluna_quadratica <- paste0(nome_coluna_quadratica, "_squared")
        
        df[[nome_coluna_quadratica]] <- df[[colunas[1]]] * df[[colunas[2]]]
      }
      
      if (grepl("-", coluna)) {
        nome_coluna_interacao <- gsub("-", "_interaction_", coluna)
        colunas <- unlist(strsplit(coluna, "-")) 
        df[[nome_coluna_interacao]] <- df[[colunas[1]]] * df[[colunas[2]]]
     }
      
    }
  
    colunas_modelo <- setdiff(names(df), response_column)
    
    formula <- as.formula(paste(response_column, "~", paste0(colunas_modelo, collapse = " + ")))
    formula <- update(formula, . ~ . - response_column)
    
    model <- lm(formula, data = df)
    
    summaryModel = summary(model)
    
    parameterEstimates <- as.data.frame(summary(model)$coef)
    
    anova_result <- anova(model)
    anovaTable <- as.data.frame(anova_result[-nrow(anova_result), ])
    
    anova_summary <- data.frame(
      df = c(sum(anova_result[-nrow(anova_result), "Df"]),  
             anova_result[nrow(anova_result), "Df"],       
             sum(anova_result[-nrow(anova_result), "Df"], anova_result[nrow(anova_result), "Df"])), 
      
      sm = c(sum(anova_result[-nrow(anova_result), "Sum Sq"]),
             anova_result[nrow(anova_result), "Sum Sq"],      
             sum(anova_result[-nrow(anova_result), "Sum Sq"], anova_result[nrow(anova_result), "Sum Sq"])), 
      
      
      ms = c(sum(anova_result[-nrow(anova_result), "Sum Sq"]) / sum(anova_result[-nrow(anova_result), "Df"]), 
             anova_result[nrow(anova_result), "Mean Sq"],      
             sum(anova_result[-nrow(anova_result), "Mean Sq"], anova_result[nrow(anova_result), "Mean Sq"])), 
      
      
      fValue = c(sum(anova_result[-nrow(anova_result), "F value"]), 
                 anova_result[nrow(anova_result), "F value"],     
                 sum(anova_result[-nrow(anova_result), "F value"], anova_result[nrow(anova_result), "F value"])), 
      prob = c(sum(anova_result[-nrow(anova_result), "Pr(>F)"]),  
               anova_result[nrow(anova_result), "Pr(>F)"],       
               sum(anova_result[-nrow(anova_result), "Pr(>F)"], anova_result[nrow(anova_result), "Pr(>F)"])) 
    )
    
    row.names(anova_summary) <- c("Modelo", "Erro", "Total")
    
    names(parameterEstimates) <- gsub(" ", "_", names(parameterEstimates))
    
    names(parameterEstimates)[names(parameterEstimates) == "Pr(>|t|)"] <- "prob"
    names(parameterEstimates)[names(parameterEstimates) == "Std._Error"] <- "stdError"
    names(parameterEstimates)[names(parameterEstimates) == "Estimater"] <- "estimate"
    
  }
  
  r2 <- summaryModel$r.squared
  
  # Calcula R² Ajustado
  num_obs <- nrow(df)
  num_predictors <- length(coef(model)) - 1
  r2_adjusted <- 1 - ((1 - r2) * (num_obs - 1) / (num_obs - num_predictors - 1))
  
  # Calcula RMSE
  rmse <- sqrt(anova_result[nrow(anova_result), "Mean Sq"])
  
  # Calcula média do Y
  mean_y <- mean(df$Y)
  
  # Número de observações
  num_observations <- nrow(df)
  
  summary_Of_fit <- data.frame(
    df = c(r2, r2_adjusted, rmse, mean_y, num_observations)
  )

  row.names(summary_Of_fit) <- c("r2", "r2adjust", "rmse", "mean", "observation")  

  # Gerar a equação do modelo
  equation <- paste0("Y = ", round(parameterEstimates[1, "Estimate"], 4))  # Intercepto
  for (i in 2:nrow(parameterEstimates)) {
    term <- paste0(" + ", round(parameterEstimates[i, "Estimate"], 4), "*", rownames(parameterEstimates)[i])
    equation <- paste0(equation, term)
  }
  
  # Substituir "_interaction_" por " * "
  equation <- gsub("_interaction_", " * ", equation)
  
  # Substituir "x1_squared" por "x1 * x1"
  equation <- gsub("_squared", "", equation)
  equation <- gsub("([a-zA-Z0-9]+)$", "\\1 * \\1", equation)

  df_copy$`Y Predicted` <- predict(model)

  # Ordena a tabela pelo Y Predito
  df_copy <- df_copy[order(df_copy$`Y`), ]
  
  # Extrai os valores do Y e do Y Predito para listas de numeros
  y_values <- as.numeric(df_copy$Y)
  y_predicted_values <- as.numeric(df_copy$`Y Predicted`)
  
  results <- list(parameterEstimates = parameterEstimates, anovaTable = anova_summary, summaryOfFit = summary_Of_fit, y=y_values, yPredicted=y_predicted_values)
  return(results)
}

#' Divide um DataFrame em vários DataFrames baseados em colunas de resposta
#'
#' Esta função divide um DataFrame em uma lista de DataFrames, onde cada DataFrame
#' contém todas as colunas, exceto outras colunas de resposta, além da coluna de
#' resposta específica para aquele DataFrame.
#'
#' @param df DataFrame de entrada que será dividido.
#' @param response_columns Vetor de strings com os nomes das colunas de resposta.
#'
#' @return Uma lista onde cada elemento é um DataFrame, indexado pelo nome da
#'         coluna de resposta correspondente.
split_dataframes_by_response_column <- function(df, response_columns) {
  dfs <- list()

  for (response_col in response_columns) {
    selected_cols <- setdiff(names(df), response_columns) 
    selected_cols <- c(selected_cols, response_col) 
    new_df <- df[selected_cols]
    dfs[[response_col]] <- new_df
  }
  
  return(dfs)
}


e meu antigo front end

import React, { Suspense, useEffect, useState } from "react";
import { useAuth } from "hooks/useAuth";
import axios from "axios";
import { useTranslation } from "next-i18next";
import { Col, Row, message } from "antd";
import ContentHeader from "shared/contentHeader";
import {
  CcdData,
  CcdExportData,
  CcdLackOfFitSummaryData,
  DataToCompile,
} from "./interface";
import {
  createAnovaTable,
  createLackOfFitTable,
  createParameterEstimateTable,
  createSummaryOfFitTable,
} from "./table";
import { Spin } from "shared/spin"
import ResponseContentHeader from "shared/responseContentHeader";
import Table from "shared/table";
import Equation from "shared/equation";
import dynamic from "next/dynamic";
import { formatNumber } from "utils/formatting";
import {
  createDataFrames,
  createProfilerDataObj,
  toDicitionary,
  dataSortOrder,
} from "utils/core";
import { useRouter } from "next/router";
import { getItem } from "utils/database";
import { HighChartTemplate } from "shared/widget/chartHub/interface";

const ChartBuilder = dynamic(() => import("./chartBuilder"), { ssr: false });

const GenerateChart = dynamic(() => import("./generateChartModal"), {
  ssr: false,
});

const Profiler = dynamic(() => import("shared/profiler"), {
  ssr: false,
  loading: () => <Spin />,
});

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const fetcher = axios.create({
  baseURL: "/api",
});
export const CentralCompositeDesign: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });

  const { user } = useAuth();
  const router = useRouter();

  const [responseColumnData, setResponseColumnData] = useState<any>({});
  const [responseColumnKey, setResponseColumnKey] = useState<string[]>([]);
  const [contentVisibility, setContentVisibility] = useState<boolean[]>(
    new Array(responseColumnKey.length).fill(false)
  );

  const [loadingAnalisysOfVariance, setLoadingAnalisysOfVariance] =
    useState(true);
  const [loadingParamterEstimate, setLoadingParamterEstimate] = useState(true);
  const [loadingLackOfFit, setLoadingLackOfFit] = useState(true);
  const [loadingSummaryOfFit, setLoadingSummaryOfFit] = useState(true);

  const [loadingPage, setLoadingPage] = useState(true);
  const [recalculate, setIsRecalculate] = useState(false);
  const [colNumber, setColNumber] = useState(3);

  const [chartData, setChartData] = useState<{
    [key: string]: HighChartTemplate;
  }>({});
  const [chartDataBuilder, setChartDataBuilder] = useState<any>({});
  const [profilerData, setProfilerData] = useState<any>({});

  const [isOpenGenerateChart, setIsOpenGenerateChart] = useState(false);
  const [constructorChart, setConstructorChart] = useState<any>({});
  const [isChartBuild, setIsChartBuild] = useState(false);
  const [generateChartList, setGenerateChartList] = useState<string[]>([]);

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "ccd",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  useEffect(() => {
    const getData = async () => {
      setIsChartBuild(false);

      const parsedUrlQuery = router?.query;

      if (Object.keys(parsedUrlQuery).length > 0) {
        const tool = parsedUrlQuery.tool as string;
        const uid = parsedUrlQuery.uid as string;

        const item = (await getItem(tool, uid)) as CcdExportData;
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

          setLoadingPage(false);

          try {
            const columns = dataToSend.itens;

            for (const iterator of item.iterationColumns) {
              columns.push(iterator);
            }

            const ccdCalculate = {
              columnsUsed: columns,
              recalculate: item.recalculate,
              obj: dataToSend.obj,
              responseColumn: item.responseColumns,
            };

            setIsRecalculate(item.recalculate);

            setColNumber(
              Object.keys(dataToSend.obj).length - item.responseColumns.length
            );

            const allColumns = Object.keys(dataToSend.obj);

            const { data } = await fetcher.post<CcdData>(
              "centralCompositeDesign/calculate",
              ccdCalculate
            );

            const ccd = data.result;
            const responseColumnsData: Record<string, any> = {};
            const profilerObj: Record<string, any> = {};

            if (Object.keys(ccd).length <= 0) {
              return;
            }

            for (const key of Object.keys(ccd)) {
              profilerObj[key] = [];

              const errorModel = ccd[key].anovaTable.find(
                (el) => el._row === "Erro"
              );

              const columnsUsed = Object.values(ccd[key].parameterEstimates)
                .filter((el) => !el._row.includes("Intercept"))
                .map((el) => el._row);

              const lackOfFitCalculate = {
                obj: dataToSend.obj,
                degreesFreedomError: errorModel?.df,
                sqErrorTotal: errorModel?.sm,
                responseColumn: item.responseColumns,
                columnsUsed: columnsUsed,
                tool: "ccd",
              };

              const anovaTranslate = {
                term: commonT("ccd.term"),
                degreesOfFreedom: commonT("ccd.degreesOfFreedom"),
                sSquare: commonT("ccd.sSquare"),
                mSquare: commonT("ccd.mSquare"),
                fRatio: commonT("ccd.fRatio"),
                probF: commonT("ccd.probF"),
                model: commonT("ccd.model"),
                error: commonT("ccd.error"),
                total: commonT("ccd.total"),
              };

              responseColumnsData[key] = createAnovaTable(
                ccd[key].anovaTable,
                anovaTranslate
              );
              setLoadingAnalisysOfVariance(false);

              const parameterEstimateTranslate = {
                term: commonT("ccd.term"),
                estimates: commonT("ccd.estimate"),
                pValue: commonT("ccd.pValue"),
                stdError: commonT("ccd.stdError"),
                tRatio: commonT("ccd.tRatio"),
              };

              responseColumnsData[key]["parameterEstimate"] =
                createParameterEstimateTable(
                  ccd[key].parameterEstimates,
                  parameterEstimateTranslate
                );

              setLoadingParamterEstimate(false);

              const lackOfFitTranslate = {
                term: commonT("ccd.term"),
                degreesOfFreedom: commonT("ccd.degreesOfFreedom"),
                sSquare: commonT("ccd.sSquare"),
                mSquare: commonT("ccd.mSquare"),
                fRatio: commonT("ccd.fRatio"),
                probF: commonT("ccd.probF"),
                lackOfFit: commonT("ccd.lackOfFit"),
                pureError: commonT("ccd.pureError"),
                total: commonT("ccd.total"),
              };

              setLoadingLackOfFit(false);

              const summaryOfFitTranslate = {
                term: commonT("ccd.term"),
                rSquare: commonT("ccd.rSquare"),
                rSquareAdjust: commonT("ccd.rSquareAdjust"),
                rmse: commonT("ccd.rmse"),
                mean: commonT("ccd.mean"),
                observations: commonT("ccd.observations"),
                value: commonT("ccd.value"),
              };

              responseColumnsData[key]["summaryOfFit"] =
                createSummaryOfFitTable(
                  ccd[key].summaryOfFit,
                  summaryOfFitTranslate
                );

              const rmse = ccd[key].summaryOfFit.find(
                (el) => el._row === "rmse"
              );
              profilerObj[key]["rmse"] = Math.sqrt(rmse?.df ?? 0);

              setLoadingSummaryOfFit(false);

              const dataHistogram: number[] = [];
              const labelsHistogram: string[] = [];

              ccd[key].parameterEstimates.forEach((el) => {
                if (el._row !== "(Intercept)") {
                  dataHistogram.push(Math.abs(el.Estimate));
                  labelsHistogram.push(
                    el._row
                      .replaceAll("_interaction_", "-")
                      .replaceAll(
                        "_squared",
                        `*${el._row.replace("_squared", "")}`
                      )
                  );
                }
              });

              responseColumnsData[key]["sortedEstimates"] = dataHistogram;
              responseColumnsData[key]["columnList"] = labelsHistogram;
              responseColumnsData[key]["y"] = ccd[key].y;
              responseColumnsData[key]["yPredicted"] = ccd[key].yPredicted;

              if (item.recalculate) {
                const { data: lackOfFit } =
                  await fetcher.post<CcdLackOfFitSummaryData>(
                    "centralCompositeDesign/lackOfFitCalculate",
                    lackOfFitCalculate
                  );

                responseColumnsData[key]["lackOfFit"] = createLackOfFitTable(
                  lackOfFit.ccds[key].lackOfFit,
                  lackOfFitTranslate
                );
              }

              const correlationMatrixEquationVariables: string[] = [];

              for (const item of ccd[key].parameterEstimates.map(
                (el) => el._row
              )) {
                correlationMatrixEquationVariables.push(
                  item.replace("(Intercept)", "Intercept")
                );
              }

              const betaMatrix: number[] = ccd[key].parameterEstimates.map(
                (el) => el.Estimate
              );

              // Cria a equação
              let equationCalc = "Y = ";
              let equationProfiler = "Y = ";
              let equationCalcOrigin = "Y = ";

              for (let i = 0; i < betaMatrix.length; i++) {
                const variable = correlationMatrixEquationVariables[i];
                const beta = parseFloat(formatNumber(Number(betaMatrix[i])));

                const variableReplace = variable
                  .replaceAll("_interaction_", " * ")
                  .replaceAll(
                    "_squared",
                    ` * ${variable.replace("_squared", "")} `
                  );

                if (!variable) continue;

                if (variable === "Intercept") {
                  equationCalc += beta;
                  equationCalcOrigin += beta;
                  equationProfiler += beta;
                }

                if (variable !== "Intercept") {
                  const interation = variable.split(" * ");
                  if (interation.length <= 1) {
                    equationCalcOrigin +=
                      beta < 0
                        ? ` - ${Math.abs(beta)} * (${variable})`
                        : ` + ${beta} * ${variable}`;

                    equationProfiler +=
                      beta < 0
                        ? ` - ${Math.abs(beta)} * ${variable}`
                        : ` + ${beta} * ${variable}`;
                  }

                  equationCalc +=
                    beta < 0
                      ? ` + (- ${Math.abs(beta)}) * ${variableReplace}`
                      : ` + ${beta} * ${variableReplace}`;
                }
              }

              responseColumnsData[key]["equation"] = equationCalc;
              responseColumnsData[key]["equationOrigin"] = equationCalcOrigin;

              profilerObj[key]["equation"] = equationCalc;
              profilerObj[key]["equationOrigin"] = equationProfiler;

              // Cria toolip da equação
              // Verifica se há uma coluna "intercept" no correlationMatrixEquationVariables

              let equationTooltipCalc = "Y = ";

              for (let i = 0; i < betaMatrix.length; i++) {
                const variable = correlationMatrixEquationVariables[i];
                const beta = parseFloat(formatNumber(Number(betaMatrix[i])));

                const variableReplace = variable
                  .replace("_interaction_", "-")
                  .replace("_squared", `*${variable.replace("_squared", "")}`);

                if (variable === "Intercept") {
                  equationTooltipCalc += "Intercept";
                }

                if (variable !== "Intercept") {
                  const betaNumber = i + 1;
                  equationTooltipCalc +=
                    beta < 0
                      ? ` - ${variableReplace.replace(
                          "-",
                          "*"
                        )} * Beta ${betaNumber}`
                      : ` + ${variableReplace.replace(
                          "-",
                          "*"
                        )} * Beta ${betaNumber}`;
                }
              }

              responseColumnsData[key]["equationTooltip"] = equationTooltipCalc;

              profilerObj[key]["responseTitle"] = key;

              setGenerateChartList(variables);

              profilerObj[key]["data"] = createDataFrames(
                allColumns,
                dataToSend.obj,
                item.responseColumns
              );

              const profilerDataObj = profilerObj[key]["data"];

              const profilerDataObjKeys = Object.keys(profilerDataObj).filter(
                (el) => el !== key
              );

              const createProfilerData = createProfilerDataObj(
                profilerDataObjKeys,
                profilerDataObj,
                profilerObj,
                key
              );
              delete profilerObj[key]["data"];
              profilerObj[key]["data"] = createProfilerData;
            }

            setResponseColumnData(responseColumnsData);
            setResponseColumnKey(Object.keys(responseColumnsData));
            setChartDataBuilder(dataToSend.obj);
            setProfilerData(profilerObj);
          } catch (error: any) {
            if (error.response.data === "ccdLackOfFitError") {
              message.error({
                content: commonT("error.general.lackOfFitErrorMsg"),
              });
            } else {
              message.error({
                content: commonT("error.general.proccesData"),
              });
            }
          }
        }
      }
    };
    getData().catch(console.error);
  }, [router.isReady, router.query]);

  useEffect(() => {
    const updatedChartData: Record<string, any> = {};

    if (responseColumnData && Object.keys(responseColumnData).length > 0) {
      responseColumnKey.map((key) => {
        if (
          !responseColumnData[key].parameterEstimate
            .columnsToParameterEstimates ||
          responseColumnData[key].parameterEstimate.columnsToParameterEstimates
            .length <= 0
        )
          return;

        const dictionary = toDicitionary(
          responseColumnData[key].columnList,
          responseColumnData[key].sortedEstimates
        );

        const sortedList: { [key: string]: number } = dataSortOrder(
          dictionary,
          "asc",
          false
        );

        const sortedEstimates: number[] = Object.entries(sortedList).map(
          ([, value]) => value
        );

        const sortedEstimatesLabels: string[] = Object.entries(sortedList).map(
          ([key]) => key
        );

        const chartsForVariable: HighChartTemplate = buildChart(
          key,
          sortedEstimates,
          sortedEstimatesLabels,
          responseColumnData[key].y,
          responseColumnData[key].yPredicted
        );

        updatedChartData[key] = chartsForVariable;
      });

      setChartData(updatedChartData);
    }
  }, [loadingPage, responseColumnData, recalculate]);

  function buildChart(
    key: string,
    estimates: number[],
    estimateLabels: string[],
    yValues: any[],
    yPredictedValues: any[]
  ) {
    const chartsForVariable: HighChartTemplate = {};

    chartsForVariable["overlay"] = {
      seriesData: [
        {
          data: yValues,
          name: commonT("widget.chartType.y"),
          type: "line",
        },
        {
          data: yPredictedValues,
          name: commonT("widget.chartType.yPredicted"),
          type: "line",
        },
      ],
      options: {
        title: commonT("widget.chartType.overlay"),
        xAxisTitle: commonT("widget.chartType.rowNumber"),
        yAxisTitle: commonT("widget.chartType.y"),
        titleFontSize: 24,
        chartName: key,
        legendEnabled: true,
        useVerticalLimits: false,
      },
      type: "line",
      displayName: commonT("widget.chartType.overlay"),
    };

    chartsForVariable["estimate"] = {
      seriesData: [
        {
          data: estimates,
          name: commonT("multipleRegression.chart.dataLegend"),
          type: "bar",
          showLegend: true,
        },
      ],
      options: {
        title: commonT("multipleRegression.chart.titleChart"),
        xAxisTitle: "",
        yAxisTitle: commonT("multipleRegression.estimate"),
        titleFontSize: 24,
        chartName: key,
        legendEnabled: true,
        categories: estimateLabels,
        useVerticalLimits: false,
        barIsVertical: false,
      },
      type: "bar",
      displayName: commonT("multipleRegression.chart.titleChart"),
    };

    return chartsForVariable;
  }

  useEffect(() => {
    // empty use effect
  }, [chartData]);

  const onToggleContent = (index: number) => {
    const updatedVisibility = [...contentVisibility];
    updatedVisibility[index] = !updatedVisibility[index];
    setContentVisibility(updatedVisibility);
  };

  const handleGenerateChart = (
    xVariables: string | string[],
    yVariables: string,
    chartType: string
  ) => {
    setConstructorChart({
      xVariables: xVariables,
      yVariables: yVariables,
      chartType: chartType,
    });

    setIsOpenGenerateChart(false);
    setIsChartBuild(true);
  };

  return (
    <>
      <ContentHeader
        title={commonT("ccd.title")}
        tool={"ccd"}
        enableRecalculate={true}
        enableGenerateChart={true}
        generateChart={() => setIsOpenGenerateChart(true)}
      />

      {loadingPage ? (
        <Spin />
      ) : (
        <div id="content">
          {responseColumnKey.map((key, index) => (
            <>
              <ResponseContentHeader
                title={key}
                index={index}
                onClick={() => onToggleContent(index)}
              />
              <Row hidden={contentVisibility[index]}>
                <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={10}>
                  <div
                    style={{
                      marginBottom: "20px",
                      minHeight: "50vh",
                    }}
                  >
                    <ChartHub
                      key={key}
                      chartConfigurations={chartData[key]}
                      tool="ccd"
                      showLimits
                    />
                  </div>
                  <Equation
                    loading={false}
                    tooltipTitle={responseColumnData[key].equationTooltip}
                    equation={responseColumnData[key].equation}
                    width={undefined}
                  />
                </Col>
                <Col
                  xs={24}
                  sm={24}
                  md={24}
                  lg={24}
                  xl={24}
                  xxl={{ offset: 1, span: 13 }}
                >
                  <div style={{ marginBottom: "20px" }}>
                    <Table
                      loading={loadingAnalisysOfVariance}
                      dataSource={
                        responseColumnData[key].analysisOfVarianceDataSource
                      }
                      columns={
                        responseColumnData[key].columnsToAnalysisOfVariance
                      }
                      title={commonT("ccd.analysisOfVariance")}
                    />
                  </div>
                  <div style={{ marginBottom: "20px" }}>
                    <Table
                      loading={loadingParamterEstimate}
                      dataSource={
                        responseColumnData[key].parameterEstimate
                          .parameterEstimatesDataSource
                      }
                      columns={
                        responseColumnData[key].parameterEstimate
                          .columnsToParameterEstimates
                      }
                      title={commonT("ccd.parameterEstimates")}
                    />
                  </div>

                  <div style={{ marginBottom: "20px" }}>
                    <Table
                      loading={loadingSummaryOfFit}
                      dataSource={
                        responseColumnData[key].summaryOfFit
                          ?.summaryOfFitDataSource
                      }
                      columns={
                        responseColumnData[key].summaryOfFit
                          ?.columnsToSummaryOfFit
                      }
                      title={commonT("ccd.summaryOfFit")}
                    />
                  </div>

                  {recalculate && (
                    <>
                      <div style={{ marginBottom: "20px" }}>
                        <Table
                          loading={loadingLackOfFit}
                          dataSource={
                            responseColumnData[key].lackOfFit
                              ?.lackOfFitDataSource
                          }
                          columns={
                            responseColumnData[key].lackOfFit
                              ?.columnsToLackOfFit
                          }
                          title={commonT("ccd.lackOfFit")}
                        />
                      </div>
                    </>
                  )}
                </Col>
              </Row>
            </>
          ))}

          {recalculate && (
            <>
              <Suspense fallback={<Spin />}>
                <Profiler
                  defaultColNumber={colNumber}
                  optimizationData={profilerData}
                  profilerData={profilerData}
                  loading={false}
                  type={"ccd"}
                />
              </Suspense>
            </>
          )}

          {isChartBuild ? (
            <>
              <ResponseContentHeader
                title={commonT("ccd.chartBuilder.chartTitle")}
                index={10}
                onClick={() => onToggleContent(10)}
              />
              <Row hidden={contentVisibility[10]}>
                <Col span={12}>
                  <ChartBuilder
                    chartType={constructorChart.chartType as string}
                    xVariable1={constructorChart.xVariables}
                    yVariable={constructorChart.yVariables}
                    chartData={chartDataBuilder}
                  />
                </Col>
              </Row>
            </>
          ) : (
            <></>
          )}
        </div>
      )}

      <GenerateChart
        showOpenModal={isOpenGenerateChart}
        onModalClose={() => setIsOpenGenerateChart(false)}
        variables={generateChartList}
        onSaveClick={handleGenerateChart}
      />
    </>
  );
};

import React, { useEffect, useState } from "react";
import { Chart as ChartJS } from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";
import annotationPlugin from "chartjs-plugin-annotation";
import * as Styled from "./styled";
import { useTranslation } from "next-i18next";
import {
  WidgetGeneralDisabled,
  WidgetColorDisable,
  WidgetLabelsDisabled,
} from "shared/widget/widget/interface";
import { Button, Dropdown, MenuProps, Tooltip } from "antd";
import { EditOutlined } from "@ant-design/icons";
import dynamic from "next/dynamic";
import { AiOutlineSetting } from "react-icons/ai";
import { ChartData2D, ChartType3d } from "./interface";
import { BiScatterChart } from "react-icons/bi";
import { GiMeshBall, GiMeshNetwork } from "react-icons/gi";
import {
  blackBodyGradient,
  blueredGradient,
  earthGradient,
  greysGradient,
  jetGradient,
  rdbuGradient,
  portlandGradient,
  ylOrRdGradient,
} from "utils/color";
import { Spin } from "shared/spin"

const Widget = dynamic(() => import("shared/widget/widget"), {
  ssr: false,
  loading: () => <Spin />,
});

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <Spin />,
}) as any;

ChartJS.register(annotationPlugin, ChartDataLabels);

const ChartBuilder = (props: {
  chartType: string;
  xVariable1: string | string[];
  yVariable: string;
  chartData: any;
}) => {
  const { chartType, xVariable1, yVariable, chartData } = props;

  const { t: commonT } = useTranslation("common");

  const [isHovered, setIsHovered] = useState(false);
  const [backgroundColorChart3D, setBackgroundColorChart3D] =
    useState("Portland");

  const [chartType3d, setChartType3d] = useState<ChartType3d>({
    mode: "markers",
    type: "mesh3d",
  });

  const widgetLabelsDisabled: WidgetLabelsDisabled = {
    disabledAxisX: true,
    disabledAxisY: true,
    disabledEnableAxisX: true,
    disabledEnableAxisY: true,
    enableDataLabel: true,
    disabledTitle: true,
    disabledEnableTitle: true,
    disabledDataLabel: true,
  };

  const widgetGeneralDisabled: WidgetGeneralDisabled = {
    disabledMaxCharacterLimitLegend: true,
    disabledAxisBeginAtZero: true,
    disabledYMaxScale: true,
    disabledYMinScale: true,
    disabledPointerStyleSize: true,
    disabledXMaxScale: true,
    disabledXMinScale: true,
    disabledOrderColumn: true,
  };

  const widgetColorDisabled: WidgetColorDisable = {
    subGroupMeansColor: true,
    groupMeansColor: true,
    pseLineColor: true,
    stdErrorLineColor: true,
    modelLineColor: true,
    lieColor: true,
    lseColor: true,
  };

  const generateDataToChart3D = (variables?: string) => {
    const dataToChart: any[] = [];
    if (!variables) return dataToChart;

    const xValues = chartData[variables];
    if (!xValues) return [];
    for (let index = 0; index < xValues.length; index++) {
      dataToChart.push(xValues[index]);
    }

    return dataToChart;
  };

  const dataToChart3D = [
    {
      mode: chartType3d.mode,
      type: chartType3d.type,
      marker: chartType3d.type
        ? {
            color: generateDataToChart3D(yVariable),
            size: 12,
            symbol: "circle",
            line: {
              color: generateDataToChart3D(yVariable),
              width: 1,
            },
            opacity: 0.8,
          }
        : null,
      z: generateDataToChart3D(yVariable),
      x: generateDataToChart3D(xVariable1[0]),
      y: generateDataToChart3D(xVariable1[1]),
      intensity: generateDataToChart3D(yVariable),
      colorscale: backgroundColorChart3D,
    },
  ];

  const configChart3D = {
    displayModeBar: false,
  };

  const layoutChart3D = {
    title: {
      text: `Surface Plot (${xVariable1[0]} - ${xVariable1[1]})`,
      font: {
        family: "Arial, sans-serif",
        size: 18,
      },
      y: 0.95,
    },

    autosize: true,
    width: 650,
    height: 600,
    margin: {
      l: 50,
      r: 0,
      b: 10,
      t: 40,
      pad: 10,
    },
    scene: {
      xaxis: {
        title: xVariable1[0],
      },
      yaxis: {
        title: xVariable1[1],
      },
      zaxis: {
        title: yVariable,
      },
    },
  };

  const generateDataChart2D = () => {
    return {
      data: {
        axisColor: "#00579D",
        borderColor: "#00579D",
        label: commonT("spaceFilling.chartBuilder.legend"),
        data: generateDataToChart2D(xVariable1 as string, yVariable),
        pointBackgroundColor: "black",
        pointBorderColor: "black",
        pointRadius: 3,
        showLine: false,
      },
    };
  };

  const generateDataToChart2D = (xVariable?: string, x2Variable?: string) => {
    const dataToChart: ChartData2D[] = [];
    if (!xVariable || !x2Variable) return dataToChart;

    const xValues = chartData[xVariable];
    const x2Values = chartData[x2Variable];

    for (let index = 0; index < xValues.length; index++) {
      const x = xValues[index];
      const y = x2Values[index];
      dataToChart.push({
        x: x,
        y: y,
      });
    }

    return dataToChart;
  };

  useEffect(() => {
    // empty use effect
  }, [chartType, xVariable1, yVariable, chartData, backgroundColorChart3D]);

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  const chartTypesItems: MenuProps["items"] = [
    {
      key: "mesh3d",
      label: (
        <Tooltip title="Surface" placement="right">
          <>
            <GiMeshBall style={{ transform: "scale(1.3)", paddingRight: 10 }} />
            Surface
          </>
        </Tooltip>
      ),
    },
    {
      key: "scatter3d",
      label: (
        <>
          <Tooltip
            title={commonT("spaceFilling.chartBuilder.bubble")}
            placement="right"
          >
            <>
              <BiScatterChart
                style={{ transform: "scale(1.3)", paddingRight: 10 }}
              />
              {commonT("spaceFilling.chartBuilder.bubble")}
            </>
          </Tooltip>
        </>
      ),
    },
    {
      key: "line",
      label: (
        <>
          <Tooltip
            title={commonT("spaceFilling.chartBuilder.bubbleLine")}
            placement="right"
          >
            <>
              <GiMeshNetwork
                style={{ transform: "scale(1.3)", paddingRight: 10 }}
              />
              {commonT("spaceFilling.chartBuilder.bubbleLine")}
            </>
          </Tooltip>
        </>
      ),
    },
  ];

  const itemsStyles: MenuProps["items"] = [
    {
      key: "Blackbody",
      label: <div style={blackBodyGradient} />,
    },
    {
      key: "Bluered",
      label: <div style={blueredGradient} />,
    },
    {
      key: "Earth",
      label: <div style={earthGradient} />,
    },
    {
      key: "Greys",
      label: <div style={greysGradient} />,
    },
    {
      key: "Jet",
      label: <div style={jetGradient} />,
    },
    {
      key: "RdBu",
      label: <div style={rdbuGradient} />,
    },
    {
      key: "Portland",
      label: <div style={portlandGradient} />,
    },
    {
      key: "YlOrRd",
      label: <div style={ylOrRdGradient} />,
    },
  ];

  const handleChartTypes: MenuProps["onClick"] = ({ key }) => {
    switch (key) {
      case "scatter3d":
        setChartType3d({
          type: "scatter3d",
          mode: "markers",
        });
        break;
      case "mesh3d":
        setChartType3d({
          type: "mesh3d",
          mode: "markers",
        });
        break;
      case "line":
        setChartType3d({
          type: "scatter3d",
          mode: "lines+markers",
        });
        break;
    }
  };

  const handleStyle: MenuProps["onClick"] = ({ key }) => {
    switch (key) {
      case "Blackbody":
        setBackgroundColorChart3D("Blackbody");
        break;
      case "Bluered":
        setBackgroundColorChart3D("Bluered");
        break;
      case "Earth":
        setBackgroundColorChart3D("Earth");
        break;
      case "Greys":
        setBackgroundColorChart3D("Greys");
        break;
      case "Jet":
        setBackgroundColorChart3D("Jet");
        break;
      case "RdBu":
        setBackgroundColorChart3D("RdBu");
        break;
      case "Portland":
        setBackgroundColorChart3D("Portland");
        break;
      case "YlOrRd":
        setBackgroundColorChart3D("YlOrRd");
        break;
      default:
        setBackgroundColorChart3D("Bluered");
        break;
    }
  };

  return (
    <>
      <Styled.AlignChart>
        {chartType === "2d" ? (
          <>
            <Widget
              chartType={"scatter"}
              chartData={generateDataChart2D()}
              chartName={"space-filling-chart"}
              height={"540px"}
              showSettings={false}
              currentLse={0}
              currentLie={0}
              widgetColorDisable={widgetColorDisabled}
              widgetGeneralDisabled={widgetGeneralDisabled}
              widgetLabelDisabled={widgetLabelsDisabled}
              lie={undefined}
              lse={undefined}
              axisNames={[
                "Space Filling Chart",
                xVariable1 as string,
                yVariable,
              ]}
              tool={"spaceFilling"}
              disabledChartType={true}
            />
          </>
        ) : (
          <>
            {typeof window !== "undefined" && xVariable1 && yVariable && (
              <div
                style={{
                  border: "2px solid #ccc",
                  boxShadow: "0px 0px 8px rgba(0, 0, 0, 0.4)",
                  padding: "5px",
                  position: "relative",
                  borderRadius: "12px",
                  overflow: "auto",
                  overflowY: "hidden",
                  width: "750px",
                  height: "650px",
                }}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
              >
                {isHovered && (
                  <div
                    style={{
                      position: "absolute",
                      top: "10px",
                      right: "10px",
                      display: "flex",
                      justifyContent: "flex-end",
                      opacity: isHovered ? 1 : 0,
                      transition: "opacity 10s ease",
                    }}
                  >
                    <div style={{ marginRight: "5px" }}>
                      <Tooltip
                        title={commonT("widget.stylization.stylization")}
                        placement="top"
                      >
                        <Dropdown
                          menu={{ items: itemsStyles, onClick: handleStyle }}
                          disabled={chartType3d.type !== "mesh3d"}
                        >
                          <Button style={{ width: "100%" }} size="small">
                            <EditOutlined style={{ transform: "scale(1.3)" }} />
                          </Button>
                        </Dropdown>
                      </Tooltip>
                    </div>
                    <div>
                      <Tooltip
                        title={commonT("widget.chart.type")}
                        placement="top"
                      >
                        <Dropdown
                          menu={{
                            items: chartTypesItems,
                            onClick: handleChartTypes,
                          }}
                        >
                          <Button style={{ width: "100%" }} size="small">
                            <AiOutlineSetting
                              style={{ transform: "scale(1.3)" }}
                            />
                          </Button>
                        </Dropdown>
                      </Tooltip>
                    </div>
                  </div>
                )}

                <Plot
                  data={dataToChart3D}
                  layout={layoutChart3D}
                  config={configChart3D}
                />
              </div>
            )}
          </>
        )}
      </Styled.AlignChart>
    </>
  );
};

export default ChartBuilder;

a geração do ccd no front 

import React, { useEffect, useState } from "react";
import { useTranslation } from "next-i18next";
import axios from "axios";
import {
  Button,
  Checkbox,
  Col,
  Form,
  InputNumber,
  Row,
  Select,
  message,
} from "antd";
import { CheckboxChangeEvent } from "antd/es/checkbox";
import { saveAsExcel } from "utils/export";
import Modal from "shared/modal";
import { generateRandomNumbers } from "utils/core";
import { DataToCompile } from "components/insertData/inteface";

interface GenerateCcdForm {
  type: string;
  factor: number;
  centerPoint: number;
  responseColumnNumber: number;
}

const fetcher = axios.create({
  baseURL: "/api",
});

const GenerateCcdModal = (props: {
  showOpenModal: boolean;
  onModalClose: () => void;
}) => {
  const { showOpenModal, onModalClose } = props;

  const { t: commonT } = useTranslation("common");
  const { t: layoutT } = useTranslation("layout");

  const [isModalOpen, setIsModalOpen] = useState(showOpenModal);
  const [loading, setLoading] = useState<boolean | undefined>(undefined);
  const [isRandomColumnResponse, setIsRandomColumnResponse] = useState(false);
  const [initialValue, setInitialValue] = useState<number | undefined>(
    undefined
  );
  const [finalValue, setFinalValue] = useState<number | undefined>(undefined);
  const [intervalValue, setIntervalValue] = useState<number | undefined>(
    undefined
  );
  const [ccdType, setCcdType] = useState("bbd");
  const [centerPoints, setCenterPoints] = useState<number | undefined>(
    undefined
  );

  const [factors, setFactors] = useState<number | undefined>(undefined);

  const [form] = Form.useForm<GenerateCcdForm>();

  const ccdTypes = [
    {
      label: commonT("ccd.boxBehnken"),
      value: "bbd",
    },
    {
      label: commonT("ccd.orthogonal"),
      value: "orthogonal",
    },
    {
      label: commonT("ccd.rotatable"),
      value: "rotatable",
    },
    {
      label: commonT("ccd.spherical"),
      value: "spherical",
    },
    {
      label: commonT("ccd.faces"),
      value: "faces",
    },
  ];

  const handleCancel = () => {
    setIsModalOpen(false);
    onModalClose();
  };

  useEffect(() => {
    setIsModalOpen(showOpenModal);
    form.resetFields();
    form.setFieldValue("type", ccdType);
  }, [showOpenModal]);

  const onFinish = async (formValue: GenerateCcdForm) => {
    setLoading(true);
    const responseValues = handleResponseValues();

    const generateCcd = {
      basis: formValue.factor,
      centerPoint: formValue.centerPoint,
      type: ccdType,
      responseValues: responseValues,
      columnResponseNumber: formValue.responseColumnNumber,
    };

    try {
      const { data: ccdTable } = await fetcher.post(
        "centralCompositeDesign/generateCcd",
        generateCcd
      );

      const data = ccdTable.result;
      const variables = Object.keys(data[0]);

      const dataToSend: DataToCompile = { obj: {}, itens: variables };

      variables.map((variable) => {
        data.map((item: { [x: string]: any }) => {
          if (dataToSend.obj[`${variable}`] === undefined) {
            dataToSend.obj[`${variable}`] = [];
          }
          const is999 = item[variable] === 99999;
          item[variable] = is999 ? "" : parseFloat(item[variable]);
          dataToSend.obj[`${variable}`].push(item[variable]);
        });
      });

      setLoading(false);
      saveAsExcel(dataToSend.obj, "central-composite-design");
    } catch (error) {
      message.error({
        content: commonT("errorLoadingData", { error }),
      });
    }
    onModalClose();
    setLoading(false);
  };

  const handleIsRandomColumnResponse = (e: CheckboxChangeEvent) => {
    setIsRandomColumnResponse(e.target.checked);
  };

  const handleResponseValues = () => {
    const interval = (intervalValue ?? 0) <= 0 ? 1 : intervalValue;
    if (!finalValue || !interval || !initialValue) return [];

    const roundsLimit = 1000;
    let randomNumbers: number[] = [];

    if (isRandomColumnResponse) {
      randomNumbers = generateRandomNumbers(
        initialValue,
        finalValue,
        interval,
        roundsLimit
      );
    }

    return randomNumbers;
  };

  function createCenterPointsBbd(factorNumber: number) {
    if (factorNumber <= 4) {
      setCenterPoints(3);
      form.setFieldValue("centerPoint", 3);
    } else if (factorNumber <= 7) {
      setCenterPoints(6);
      form.setFieldValue("centerPoint", 6);
    }
  }

  function createCenterPointsSpherical(factorNumber: number) {
    if (factorNumber === 2) {
      setCenterPoints(5);
      form.setFieldValue("centerPoint", 5);
    } else if (factorNumber === 3) {
      setCenterPoints(6);
      form.setFieldValue("centerPoint", 6);
    } else if (factorNumber === 4) {
      setCenterPoints(7);
      form.setFieldValue("centerPoint", 7);
    } else if (factorNumber === 5) {
      setCenterPoints(6);
      form.setFieldValue("centerPoint", 6);
    } else if (factorNumber <= 7) {
      setCenterPoints(14);
      form.setFieldValue("centerPoint", 6);
    } else if (factorNumber === 8) {
      setCenterPoints(13);
      form.setFieldValue("centerPoint", 13);
    }
  }

  useEffect(() => {
    const selectedCenterPoints = () => {
      switch (ccdType) {
        case "bbd":
          createCenterPointsBbd(factors);
          break;

        case "orthogonal":
        case "faces":
        case "rotatable":
          setCenterPoints(2);
          form.setFieldValue("centerPoint", 2);
          break;

        case "spherical":
          createCenterPointsSpherical(factors);
          break;

        default:
          break;
      }
    };
    selectedCenterPoints();
  }, [ccdType, factors]);

  return (
    <>
      <Modal
        title={commonT("ccd.generateTable")}
        open={isModalOpen}
        onCancel={handleCancel}
        width={"750px"}
        footer={[
          <Button key="cancel" onClick={handleCancel}>
            {layoutT("buttons.cancel")}
          </Button>,
          <Button
            htmlType="submit"
            type="primary"
            key="generate"
            loading={loading as boolean}
            form="form"
          >
            {layoutT("buttons.generate")}
          </Button>,
        ]}
      >
        <Form name="form" form={form} onFinish={onFinish}>
          <Row style={{ marginTop: 20 }}>
            <Col span={12}>
              <Form.Item label={commonT("ccd.type")} name="type">
                <Select
                  style={{ width: "100%" }}
                  onChange={setCcdType}
                  options={ccdTypes}
                  defaultValue={"bbd"}
                  value={ccdType}
                />
              </Form.Item>
            </Col>
            <Col offset={1} span={11}>
              <Form.Item
                label={commonT("ccd.factors")}
                name="factor"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber
                  onChange={(value) => setFactors(value)}
                  style={{ width: "100%" }}
                  min={ccdType === "bbd" ? 3 : 2}
                  max={ccdType === "bbd" ? 7 : 8}
                />
              </Form.Item>
            </Col>
          </Row>
          <Row>
            <Col span={12}>
              <Form.Item
                label={commonT("ccd.centerPoints")}
                name="centerPoint"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber
                  value={centerPoints}
                  onChange={(value) => setCenterPoints(value)}
                  style={{ width: "100%" }}
                  min={2}
                />
              </Form.Item>
            </Col>
            <Col offset={1} span={11}>
              <Form.Item
                label={commonT("ccd.responseColumnNumber")}
                name="responseColumnNumber"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber style={{ width: "100%" }} min={1} />
              </Form.Item>
            </Col>
          </Row>
          <Row>
            <Col span={12}>
              <Form.Item
                name="useValuesColumnY"
                valuePropName="checked"
                label={commonT("ccd.useValuesColumnY")}
              >
                <Checkbox onChange={handleIsRandomColumnResponse} />
              </Form.Item>
            </Col>
          </Row>
          {isRandomColumnResponse ? (
            <>
              <Row>
                <Col span={6}>
                  <Form.Item
                    name="minValue"
                    label={commonT("ccd.minValue")}
                    rules={[{ required: isRandomColumnResponse, message: "" }]}
                  >
                    <InputNumber
                      style={{ width: "100%" }}
                      min={0.01}
                      defaultValue={
                        isRandomColumnResponse ? initialValue : undefined
                      }
                      onChange={(value) => setInitialValue(Number(value))}
                    />
                  </Form.Item>
                </Col>
                <Col offset={3} span={6}>
                  <Form.Item
                    name="maxValue"
                    label={commonT("ccd.maxValue")}
                    rules={[{ required: isRandomColumnResponse, message: "" }]}
                  >
                    <InputNumber
                      defaultValue={
                        isRandomColumnResponse ? finalValue : undefined
                      }
                      onChange={(value) => setFinalValue(Number(value))}
                      style={{ width: "100%" }}
                      min={initialValue ?? 0}
                    />
                  </Form.Item>
                </Col>
                <Col offset={2} span={7}>
                  <Form.Item
                    name="intervalValue"
                    label={commonT("ccd.intervalValue")}
                    rules={[{ required: isRandomColumnResponse, message: "" }]}
                  >
                    <InputNumber
                      defaultValue={
                        isRandomColumnResponse ? intervalValue : undefined
                      }
                      style={{ width: "100%" }}
                      onChange={(value) => setIntervalValue(Number(value))}
                      min={1}
                    />
                  </Form.Item>
                </Col>
              </Row>
            </>
          ) : (
            <></>
          )}
        </Form>
      </Modal>
    </>
  );
};

export default GenerateCcdModal;
