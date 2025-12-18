Vamos fazer a ferramenta no plano pro, mixture design, posso colocar varios X e vários Y, cada coluna Y é uma analise diferente, coloque interação um termo * outro

tenho um codigo em R, consegue converter para python

library(mixexp)
library(lhs)
library(dplyr)
library(AlgDesign)

#' Geração de Design de Mistura
#'
#' A função `generateMixtureDesign` cria um design experimental para mistura, onde os fatores podem ser distribuídos de maneira otimizada ou preenchendo o espaço de forma aleatória.
#' 
#' @param isReplicated Lógico (TRUE ou FALSE). Indica se o design deve ser replicado (ou seja, os experimentos serão repetidos um número específico de vezes).
#' @param repeatNumber Número de repetições do design (somente se `isReplicated` for TRUE).
#' @param level Número de níveis do design (geralmente o número de pontos experimentais).
#' @param type Tipo de design a ser gerado, podendo ser:
#'   - "spaceFilling": Design preenchendo o espaço experimental de maneira aleatória e uniforme.
#'   - "optimal": Design ótimo baseado em critérios de Fedorov (geração ótima de pontos experimentais).
#' @param factorNumber Número de fatores no experimento (quantidade de variáveis independentes).
#' @param responseColumnNumber Número de colunas de resposta no experimento.
#' @param isResponseColumnValue Lógico (TRUE ou FALSE). Indica se as colunas de resposta devem ser preenchidas com valores específicos ou permanecer em branco.
#' @param responseValues Vetor contendo os valores possíveis das colunas de resposta.
#' @param listData Data frame contendo informações sobre os fatores, incluindo os níveis mínimo e máximo de cada fator.
generateMixtureDesign <- function(isReplicated, repeatNumber, level, type, factorNumber, responseColumnNumber, isResponseColumnValue, responseValues, listData) {
  min_level_value_rand <- runif(1, min = 0.001, max = 0.1)
  
  # Valida se existe valores padrões
  if (nrow(listData) == 0) {
    listData <- data.frame(factor = character(), minLevelValue = numeric(), maxLevelValue = numeric())
    for (i in 1:factorNumber) {
      new_factor <- paste0("x", i)
      listData <- rbind(listData, data.frame(factor = new_factor, minLevelValue = min_level_value_rand, maxLevelValue = 1))
    }
  }else {
    # Se listData não estiver vazio, verifique quais fatores estão faltando e adicione-os
    existing_factors <- unique(listData$factor)
    for (i in 0:factorNumber) {
      new_factor <- paste0("x", i)
      if (!new_factor %in% existing_factors) {
        
        listData <- rbind(listData, data.frame(factor = new_factor, minLevelValue = min_level_value_rand, maxLevelValue = 1))
      }
    }
  }

  if (type == "spaceFilling") {
    design <- lhs::randomLHS(n = level, k = factorNumber)
    design <- normalize(design)
    design <- apply_constraints(design, listData)
    colnames(design) <- paste0("X", 1:factorNumber)
    df <- as.data.frame(design)
  }
  
  if (type == "optimal") {
    constraints <- listData
    constraint_matrix <- as.matrix(constraints[, c("minLevelValue", "maxLevelValue")])
    
    # Definir as restrições para a soma dos fatores
    sum_constraint <- rep(1, factorNumber)
    
    # Gerar o design
    design <- optFederov(
      ~ ., 
      data = expand.grid(replicate(factorNumber, seq(0, 1, length.out = 10), simplify = FALSE)), 
      nTrials = level
    )$design
    
    design <- apply_constraints(design, constraints)
    colnames(design) <- paste0("X", 1:factorNumber)
    df <- as.data.frame(design)
  }
  
  for (i in 1:responseColumnNumber) {
    colName <- paste0("Y", i)
    df[[colName]] <- ""
  }
  
  totalRows <- nrow(df)
  
  if(isResponseColumnValue) {
    for (i in 1:responseColumnNumber) {
      colName <- paste0("Y", i)
      startIdx <- (i - 1) * totalRows + 1
      endIdx <- i * totalRows
      df[[colName]] <- sample(responseValues[startIdx:endIdx], totalRows)
    }
  } else {
    for (i in 1:responseColumnNumber) {
      colName <- paste0("Y", i)
      df[[colName]] <- ""
    }
  }
  
  if (isReplicated) {
    df <- df[rep(seq_len(totalRows), each = repeatNumber + 1), ]
  }

  result = df
  return(result) 
}

#' Normaliza os Dados para o Método de Space Filling
#'
#' A função `normalize` é utilizada para normalizar uma matriz de dados, garantindo que a soma de cada linha seja igual a 1. Este processo é frequentemente usado no método de "space filling" para garantir que os pontos experimentais estejam distribuídos de forma adequada.
#'
#' @param matrix Uma matriz de dados numéricos onde cada linha representa um ponto experimental e cada coluna representa uma variável (fator).
normalize <- function(matrix) {
  matrix / rowSums(matrix)
}

#' Aplica Restrições aos Dados Gerados
#'
#' A função `apply_constraints` ajusta os valores de um design experimental gerado (por exemplo, um design de mistura) para que eles atendam a restrições de nível mínimo e máximo definidas para cada fator. Além disso, normaliza os dados após a aplicação das restrições, garantindo que a soma de cada linha seja igual a 1.
#'
#' @param design Uma matriz de dados gerados (por exemplo, pontos experimentais), onde cada coluna representa um fator e cada linha representa uma combinação experimental.
#' @param constraints Um data frame contendo as restrições para cada fator, com as colunas `minLevelValue` e `maxLevelValue` que especificam os limites inferior e superior para cada fator.
#' 
#' @details A função aplica as restrições de nível mínimo e máximo a cada coluna do design gerado, multiplicando os valores pela faixa definida para cada fator (diferença entre `maxLevelValue` e `minLevelValue`). Após a aplicação das restrições, os dados são normalizados para garantir que a soma de cada linha seja igual a 1.
apply_constraints <- function(design, constraints) {
  for (i in seq_len(ncol(design))) {
    min_val <- constraints$minLevelValue[i]
    max_val <- constraints$maxLevelValue[i]
    range_val <- max_val - min_val
    design[, i] <- min_val + design[, i] * range_val
  }
  normalize(design)
}

#' Calcula o Design de Mistura e Ajuste de Modelo
#'
#' A função `calculateMixtureDesign` calcula o design de mistura para um conjunto de dados, ajusta um modelo de mistura usando a função `MixModel` e realiza uma análise de variância (ANOVA) para avaliar o desempenho do modelo. A função também gera previsões para os valores de resposta com base no modelo ajustado e organiza os resultados para análise posterior.
#'
#' @param df Um data frame contendo os dados experimentais, onde as colunas representam fatores e a última coluna é a variável resposta (Y).
#' @param recalculate Um valor lógico (TRUE ou FALSE) que determina se o modelo deve ser recalculado. Se TRUE, interações entre colunas especificadas serão criadas.
#' @param columnsUsed Um vetor de nomes de colunas que devem ser consideradas para a criação de interações, caso `recalculate` seja TRUE.
#' 
#' @details A função realiza os seguintes passos:
#' 1. **Criação de interações**: Se `recalculate` for TRUE, a função cria colunas de interações entre fatores que possuem um nome com o caractere "-". A interação é criada multiplicando os valores das duas colunas envolvidas.
#' 2. **Ajuste do modelo**: Dependendo do valor de `recalculate`, a função ajusta um modelo de mistura utilizando a função `MixModel`. O modelo é ajustado com base nas colunas especificadas para as composições de mistura.
#' 3. **Análise de Variância (ANOVA)**: A função realiza uma ANOVA para avaliar o desempenho do modelo ajustado, fornecendo informações sobre a significância dos fatores e interações.
#' 4. **Cálculo de métricas**: São calculados R², R² ajustado, RMSE (Erro Quadrático Médio da Raiz), a média dos valores de Y e o número de observações.
#' 5. **Predições**: A função gera previsões dos valores de Y usando o modelo ajustado e adiciona essas previsões ao data frame original.
#' 6. **Dados agregados de previsões**: A função agrega os dados preditos para cada fator, calculando a média das previsões para cada valor único do fator.
calculateMixtureDesign <- function(df, recalculate, columnsUsed) {
  original_df <- df
  
  response_col <- names(df)[ncol(df)]

  if(recalculate == TRUE){
    for (coluna in columnsUsed) {
      if (grepl("-", coluna)) {
        nome_coluna_interacao <- gsub("-", "_interaction_", coluna)
        colunas <- unlist(strsplit(coluna, "-")) 
        df[[nome_coluna_interacao]] <- df[[colunas[1]]] * df[[colunas[2]]]
      }
    }

    response_col_data <- df[[response_col]]
    
    # Remove a coluna de resposta do data frame temporariamente
    df[[response_col]] <- NULL
    
    # Adiciona a coluna de resposta como a última coluna
    df[[response_col]] <- response_col_data
    
  }

  mix_cols <- names(df)[-ncol(df)]

  if (recalculate == TRUE) {
    model <- MixModel(df, response_col, mixcomps = mix_cols, model = 1)
  } else {
    model <- MixModel(df, response_col, mixcomps = mix_cols, model = 2)
  }
  
  summaryModel <- summary(model)

  parameterEstimates <- as.data.frame(summary(model)$coef)
  
  # Crie a fórmula para o modelo com todas as interações
  predictors <- paste(names(df)[-ncol(df)], collapse = " + ")
  

  if (recalculate == TRUE) {
    formula <- as.formula(paste(response_col, "~ (", predictors, ")"))
  }else {
    formula <- as.formula(paste(response_col, "~ (", predictors, ")^2"))
  }

  # Ajusta o modelo usando lm()

  model_lm <- lm(formula, data = df)
  
  anova_result <- anova(model_lm)

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
  
  r2 <- summaryModel$r.squared
  
  # Calcula R² Ajustado
  num_obs <- nrow(df)
  summaryModel_lm <- summary(model_lm)
  r2 <- summaryModel_lm$r.squared
  r2_adjusted <- summaryModel_lm$adj.r.squared
  
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
  
  # Previsão dos valores Y usando o modelo MixModel
  predicted_values_MixModel <- predict(model, newdata = df)
  
  # Adiciona a coluna de predições à cópia do DataFrame original
  original_df$predicted <- predicted_values_MixModel

  original_df <- original_df[order(original_df[[response_col]]), ]
  
  # Extrai os valores da coluna de resposta e da coluna 'predicted'
  response_values_sorted <- as.list(original_df[[response_col]])
  predicted_values_sorted <- as.list(original_df$predicted)

  prediction_data <- list()

  for (col in names(original_df)) {
    if (col != response_col && col != "predicted") {
      prediction_data[[col]] <- list()
      for (i in 1:nrow(original_df)) {
        prediction_data[[col]][[i]] <- list(x = original_df[[col]][i], y = original_df$predicted[i])
      }
    }
  }

  aggregated_prediction_data <- list()
  for (col in names(prediction_data)) {
    temp_data <- prediction_data[[col]]
    temp_df <- do.call(rbind, lapply(temp_data, as.data.frame))
    aggregated_df <- aggregate(temp_df$y, by = list(temp_df$x), FUN = mean) 
    names(aggregated_df) <- c("x", "y")
    aggregated_prediction_data[[col]] <- aggregated_df 
  }
  
  results <- list(parameterEstimates = parameterEstimates, anovaTable = anova_summary, summaryOfFit = summary_Of_fit, y = response_values_sorted, predictValues = predicted_values_sorted, predictionData = aggregated_prediction_data)
  return(results)
}


meu front-end


import React, { useEffect, useState } from "react";
import { InboxOutlined } from "@ant-design/icons";
import {
  Button,
  Checkbox,
  Col,
  Form,
  Input,
  InputNumber,
  List,
  Row,
  Select,
  message,
} from "antd";
import { CheckboxChangeEvent } from "antd/es/checkbox";
import axios from "axios";
import { ListData } from "components/spaceFilling/interface";
import { useTranslation } from "next-i18next";
import { AiOutlineDelete } from "react-icons/ai";
import { MdAdd } from "react-icons/md";
import { saveAsExcel } from "utils/export";
import * as Styled from "./styled";
import { ListLocale } from "antd/es/list";
import Modal from "shared/modal";
import { generateRandomNumbers } from "utils/core";
import { DataToCompile } from "components/insertData/inteface";

interface GenerateMixtureDesignForm {
  isReplicated: boolean;
  repeatNumber?: number;
  level?: number;
  type: string;
  factorNumber: number;
  responseColumnNumber: number;
  useValuesColumnY: boolean;
}

const fetcher = axios.create({
  baseURL: "/api",
});

const GenerateMixtureDesign = (props: {
  showOpenModal: boolean;
  onModalClose: () => void;
}) => {
  const { showOpenModal, onModalClose } = props;

  const { t: commonT } = useTranslation("common");
  const { t: layoutT } = useTranslation("layout");

  const [isModalOpen, setIsModalOpen] = useState(showOpenModal);
  const [loading, setLoading] = useState<boolean | null>(false);
  const [isReplicated, setIsReplicated] = useState(false);
  const [isConstraint, setIsConstraint] = useState(false);

  const [mixtureDesignType, setMixtureDesignType] = useState("optimal");
  const [isRandomColumnResponse, setIsRandomColumnResponse] = useState(false);
  const [initialValue, setInitialValue] = useState<number | undefined>(
    undefined
  );
  const [finalValue, setFinalValue] = useState<number | undefined>(undefined);
  const [intervalValue, setIntervalValue] = useState<number | undefined>(
    undefined
  );

  const [minLevelValue, setMinLevelValue] = useState<number | undefined>(0);
  const [maxLevelValue, setMaxLevelValue] = useState<number | undefined>(1);
  const [listData, setListData] = useState<ListData[]>([]);
  const [selectFactor, setSelectFactor] = useState<any>(null);
  const [factorOptions, setFactorOptions] = useState<any[]>([]);
  const [factorNumber, setFactorNumber] = useState(0);
  const [rounds, setRounds] = useState(0);

  const [form] = Form.useForm<GenerateMixtureDesignForm>();

  const listLocale: ListLocale = {
    emptyText: (
      <>
        <div>
          <InboxOutlined style={{ transform: "scale(3)" }} />
          <h3>{commonT("transfer.notFound")}</h3>
        </div>
      </>
    ),
  };

  const mixtureDesignTypes = [
    {
      label: commonT("mixtureDesign.optimal"),
      value: "optimal",
    },
    {
      label: commonT("mixtureDesign.spaceFilling"),
      value: "spaceFilling",
    },
  ];

  const handleCancel = () => {
    setIsModalOpen(false);
    form.resetFields();
    onModalClose();
  };

  useEffect(() => {
    form.resetFields();
    clearFields();
    setIsModalOpen(showOpenModal);
  }, [showOpenModal]);

  const clearFields = () => {
    setListData([]);
    setIntervalValue(undefined);
    setMinLevelValue(undefined);
    setMaxLevelValue(undefined);
    setFinalValue(undefined);
    setInitialValue(undefined);
    setIsRandomColumnResponse(false)
  };

  const handleIsRandomColumnResponse = (e: CheckboxChangeEvent) => {
    setIsRandomColumnResponse(e.target.checked);
  };

  const handleResponseValues = () => {
    let randomNumbers: number[] = [];
    if (!initialValue || !intervalValue || !finalValue) return randomNumbers;
    const interval = intervalValue <= 0 ? 1 : intervalValue;

    const roundsLimit = 200;
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

  const onFinish = async (formValue: GenerateMixtureDesignForm) => {
    setLoading(true);
    const responseValues = handleResponseValues();

    const mixtureDesignGenerate = {
      isReplicated: formValue.isReplicated ? true : false,
      repeatNumber: formValue.repeatNumber ? formValue.repeatNumber : 0,
      level: formValue.level ? formValue.level : 0,
      type: formValue.type as string,
      factorNumber: formValue.factorNumber,
      responseColumnNumber: formValue.responseColumnNumber
        ? formValue.responseColumnNumber
        : 1,
      isResponseColumnValue: formValue.useValuesColumnY ? true : false,
      responseValues: responseValues,
      listData: listData,
    };

    try {
      const { data: mixtureDesignData } = await fetcher.post(
        "mixtureDesign/generateDataTable",
        mixtureDesignGenerate
      );
      const dataTile = handleDoeFileName(mixtureDesignGenerate.type);

      const data = mixtureDesignData.result;
      const variables = Object.keys(data[0]);

      const dataToSend: DataToCompile = { obj: {}, itens: variables };

      variables.map((variable) => {
        data.map((item: { [x: string]: any }) => {
          if (dataToSend.obj[`${variable}`] === undefined) {
            dataToSend.obj[`${variable}`] = [];
          }
          dataToSend.obj[`${variable}`].push(item[variable]);
        });
      });

      setLoading(false);
      saveAsExcel(dataToSend.obj, dataTile);
    } catch (error) {
      console.error(error);
      message.error({
        content: commonT("error.doe.generateDoeError"),
      });
    } finally {
      setLoading(false);
    }

    onModalClose();
  };

  useEffect(() => {
    const letterArray = [];
    for (let i = 1; i <= factorNumber; i++) {
      const item = `x${i}`;
      letterArray.push({
        value: item,
        label: item,
      });
    }
    setFactorOptions(letterArray);
  }, [factorNumber]);

  const handleDoeFileName = (type: string) => {
    const baseKey = "mixtureDesign.export";
    return commonT(`${baseKey}.${type}`);
  };

  const handleIsReplicated = (e: CheckboxChangeEvent) => {
    setIsReplicated(e.target.checked);
  };

  const handleIsConstraint = (e: CheckboxChangeEvent) => {
    setIsConstraint(e.target.checked);
  };

  const onRemoveList = (item: ListData) => {
    const newListData = listData.filter((el) => el !== item);
    setListData(newListData);

    setFactorOptions((prevOptions) => [
      ...prevOptions,
      { label: item.factor, value: item.factor },
    ]);
  };

  const handleSelectFactor = (value: any) => {
    setSelectFactor(value);
  };

  const onAddClick = () => {
    if (minLevelValue && maxLevelValue && minLevelValue > maxLevelValue) {
      message.error({
        content: commonT("spaceFilling.minMaxErrorMsg"),
      });
      return;
    }

    const selectedFactor = factorOptions.find(
      (el) => el.value === selectFactor
    );

    if (!selectedFactor) {
      message.error({
        content: commonT("spaceFilling.notFactorErrorMsg"),
      });
      return;
    }

    const newItem: ListData = {
      factor: selectedFactor.value,
      minLevelValue: minLevelValue as number,
      maxLevelValue: maxLevelValue as number,
    };

    setFactorOptions((prevOptions) =>
      prevOptions.filter((el) => el.value !== selectedFactor.value)
    );

    setListData([...listData, newItem]);
    setSelectFactor({ value: null, label: null });
    form.resetFields(["selectFactor"]);
  };

  useEffect(() => {
    const calculateRounds = () => {
      if (mixtureDesignType === "spaceFilling") {
        const newRounds = factorNumber * 10;
        setRounds(newRounds);
        form.setFieldValue("level", newRounds);
      }
    };
    calculateRounds();
  }, [mixtureDesignType, factorNumber]);

  return (
    <>
      <Modal
        title={commonT("mixtureDesign.generateTable")}
        open={isModalOpen}
        onCancel={handleCancel}
        width={"850px"}
        footer={[
          <Button key="cancel" onClick={handleCancel}>
            {layoutT("buttons.cancel")}
          </Button>,
          <Button
            type="primary"
            htmlType="submit"
            form="form"
            key="generate"
            loading={loading as boolean}
          >
            {layoutT("buttons.generate")}
          </Button>,
        ]}
      >
        <Form name="form" form={form} onFinish={onFinish}>
          <Row>
            <Col span={10}>
              <Form.Item
                label={commonT("mixtureDesign.type")}
                rules={[{ required: true, message: "" }]}
                name="type"
              >
                <Select
                  style={{ width: "100%" }}
                  options={mixtureDesignTypes}
                  onChange={setMixtureDesignType}
                  value={mixtureDesignType}
                />
              </Form.Item>
            </Col>
            <Col
              offset={1}
              span={mixtureDesignType === "spaceFilling" ? 6 : 13}
            >
              <Form.Item
                label={commonT("mixtureDesign.factorNumber")}
                name="factorNumber"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber
                  style={{ width: "100%" }}
                  min={3}
                  max={15}
                  onChange={(value) => setFactorNumber(Number(value))}
                />
              </Form.Item>
            </Col>

            {mixtureDesignType === "spaceFilling" && (
              <>
                <Col offset={1} span={6}>
                  <Form.Item
                    label={commonT("mixtureDesign.rounds")}
                    name="level"
                    rules={[{ required: true, message: "" }]}
                  >
                    <InputNumber
                      style={{ width: "100%" }}
                      min={10}
                      value={rounds}
                      onChange={(value) => setRounds(Number(value))}
                    />
                  </Form.Item>
                </Col>
              </>
            )}
          </Row>
          <Row>
            <Col span={12}>
              <Form.Item
                label={commonT("mixtureDesign.isReplicated")}
                name="isReplicated"
                valuePropName="checked"
              >
                <Checkbox onChange={handleIsReplicated} />
              </Form.Item>
            </Col>
            <Col offset={1} span={11}>
              {isReplicated ? (
                <Form.Item
                  label={commonT("mixtureDesign.repeatNumber")}
                  name="repeatNumber"
                  rules={[{ required: isReplicated, message: "" }]}
                >
                  <InputNumber style={{ width: "100%" }} min={1} max={5} />
                </Form.Item>
              ) : (
                <></>
              )}
            </Col>
          </Row>
          <Row>
            <Col span={12}>
              <Form.Item
                name="useValuesColumnY"
                valuePropName="checked"
                label={commonT("mixtureDesign.useValuesColumnY")}
              >
                <Checkbox onChange={handleIsRandomColumnResponse} />
              </Form.Item>
            </Col>
            <Col offset={1} span={11}>
              <Form.Item
                label={commonT("mixtureDesign.responseColumnNumber")}
                name="responseColumnNumber"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber style={{ width: "100%" }} min={1} max={12} />
              </Form.Item>
            </Col>
          </Row>
          {isRandomColumnResponse ? (
            <>
              <Row>
                <Col span={6}>
                  <Form.Item
                    name="minValue"
                    label={commonT("mixtureDesign.minValue")}
                    rules={[{ required: isRandomColumnResponse, message: "" }]}
                  >
                    <InputNumber
                      min={0.01}
                      style={{ width: "100%" }}
                      defaultValue={
                        isRandomColumnResponse ? initialValue : undefined
                      }
                      onChange={(value) => setInitialValue(Number(value))}
                    />
                  </Form.Item>
                </Col>
                <Col offset={1} span={8}>
                  <Form.Item
                    name="maxValue"
                    label={commonT("mixtureDesign.maxValue")}
                    rules={[{ required: isRandomColumnResponse, message: "" }]}
                  >
                    <InputNumber
                      style={{ width: "100%" }}
                      defaultValue={
                        isRandomColumnResponse ? finalValue : undefined
                      }
                      onChange={(value) => setFinalValue(Number(value))}
                      min={initialValue}
                    />
                  </Form.Item>
                </Col>
                <Col offset={1} span={8}>
                  <Form.Item
                    name="intervalValue"
                    label={commonT("mixtureDesign.intervalValue")}
                    rules={[{ required: isRandomColumnResponse, message: "" }]}
                  >
                    <InputNumber
                      style={{ width: "100%" }}
                      defaultValue={
                        isRandomColumnResponse ? intervalValue : undefined
                      }
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
          {mixtureDesignType === "optimal" ||
          mixtureDesignType === "spaceFilling" ? (
            <>
              <Row>
                <Col span={12}>
                  <Form.Item
                    label={commonT("mixtureDesign.constraint")}
                    name="isConstraint"
                    valuePropName="checked"
                  >
                    <Checkbox onChange={handleIsConstraint} />
                  </Form.Item>
                </Col>
              </Row>
            </>
          ) : (
            <></>
          )}
          {isConstraint ? (
            <>
              <Row>
                <Col span={6}>
                  <Form.Item
                    label={commonT("designSpaceFillingTable.minLevel")}
                    name="minLevel"
                  >
                    <Input
                      value={minLevelValue}
                      onChange={(e) =>
                        setMinLevelValue(
                          parseFloat(e.target.value.replace(",", "."))
                        )
                      }
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col offset={1} span={6}>
                  <Form.Item
                    label={commonT("designSpaceFillingTable.maxLevel")}
                    name="maxLevel"
                  >
                    <Input
                      value={maxLevelValue}
                      onChange={(e) =>
                        setMaxLevelValue(
                          parseFloat(e.target.value.replace(",", "."))
                        )
                      }
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col offset={1} span={6}>
                  <Form.Item
                    label={commonT("designSpaceFillingTable.selectFactor")}
                    name="selectFactor"
                  >
                    <Select
                      value={selectFactor}
                      onChange={handleSelectFactor}
                      options={factorOptions}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col offset={1} span={3}>
                  <Button
                    type="primary"
                    icon={
                      <MdAdd
                        style={{ color: "white", transform: "scale(1.2)" }}
                      />
                    }
                    onClick={onAddClick}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                    }}
                  >
                    {commonT("spaceFilling.add")}
                  </Button>
                </Col>
              </Row>
              <Row>
                <Col span={24}>
                  <Styled.ListContainer>
                    <List
                      itemLayout="horizontal"
                      dataSource={listData}
                      bordered
                      locale={listLocale}
                      header={
                        <>
                          <b>
                            <p>{commonT("spaceFilling.factorLevelValues")}</p>
                          </b>
                        </>
                      }
                      renderItem={(item) => (
                        <List.Item>
                          <List.Item.Meta
                            avatar={
                              <>
                                <div
                                  style={{
                                    position: "relative",
                                    bottom: 12,
                                  }}
                                >
                                  <Styled.AddButton
                                    shape="circle"
                                    danger
                                    type="primary"
                                    onClick={() => onRemoveList(item)}
                                    icon={<AiOutlineDelete />}
                                  />
                                </div>
                              </>
                            }
                            title={`${item.factor} (${item.minLevelValue} - ${item.maxLevelValue}) `}
                          />
                        </List.Item>
                      )}
                    />
                  </Styled.ListContainer>
                </Col>
              </Row>
            </>
          ) : (
            <></>
          )}
          <></>
        </Form>
      </Modal>
    </>
  );
};

export default GenerateMixtureDesign;


import React, { Suspense, useEffect, useState } from "react";
import { useAuth } from "hooks/useAuth";
import { useTranslation } from "next-i18next";
import axios from "axios";
import ContentHeader from "shared/contentHeader";
import { Col, Row, message } from "antd";
import ResponseContentHeader from "shared/responseContentHeader";
import { Spin } from "shared/spin"
import {
  DataToCompile,
  MixtureDesignExportData,
  MixtureDesignResponse,
  MixtureGenerateChart,
} from "./interface";
import {
  createAnovaTable,
  createParameterEstimateTable,
  createSummaryOfFitTable,
} from "./table";
import Table from "shared/table";
import Equation from "shared/equation";
import dynamic from "next/dynamic";
import {
  createDataFrames,
  createProfilerDataObj,
  toDicitionary,
  dataSortOrder,
} from "utils/core";
import { useRouter } from "next/router";
import { getItem } from "utils/database";
import { HighChartTemplate } from "shared/widget/chartHub/interface";

const GenerateChart = dynamic(() => import("./generateChartModal"), {
  ssr: false,
});

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const ChartBuilder = dynamic(() => import("./chartBuilder"), { ssr: false });

const Profiler = dynamic(() => import("shared/profiler"), { ssr: false });

const fetcher = axios.create({
  baseURL: "/api",
});

export const MixtureDesign: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });
  const { user } = useAuth();
  const router = useRouter();

  const [responseColumnData, setResponseColumnData] = useState<
    Record<string, any>
  >({});
  const [responseColumnKey, setResponseColumnKey] = useState<string[]>([]);
  const [contentVisibility, setContentVisibility] = useState<boolean[]>(
    new Array(responseColumnKey.length).fill(false)
  );

  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingParamterEstimate, setLoadingParamterEstimate] = useState(true);
  const [loadingAnalisysOfVariance, setLoadingAnalisysOfVariance] =
    useState(true);
  const [loadingSummaryOfFit, setLoadingSummaryOfFit] = useState(true);

  const [recalculate, setIsRecalculate] = useState(false);
  const [loadingChart, setLoadingChart] = useState(true);
  const [colNumber, setColNumber] = useState(3);

  const [chartData, setChartData] = useState<{
    [key: string]: HighChartTemplate;
  }>({});

  const [loadingProfiler, setLoadingProfiler] = useState(true);
  const [profilerData, setProfilerData] = useState<any>({});

  const [isOpenGenerateChart, setIsOpenGenerateChart] = useState(false);
  const [generateChartList, setGenerateChartList] = useState<string[]>([]);
  const [constructorChart, setConstructorChart] =
    useState<MixtureGenerateChart>({ xVariables: [], yVariable: "" });
  const [isChartBuild, setIsChartBuild] = useState(false);
  const [chartBuilderData, setChartBuilderData] = useState<
    Record<string, number[]>
  >({});

  useEffect(() => {
    const getData = async () => {
      const parsedUrlQuery = router?.query;
      if (Object.keys(parsedUrlQuery).length > 0) {
        const tool = parsedUrlQuery.tool as string;
        const uid = parsedUrlQuery.uid as string;

        const item = (await getItem(tool, uid)) as MixtureDesignExportData;

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

          const columns = dataToSend.itens;

          for (const iterator of item.iterationColumns) {
            columns.push(iterator);
          }

          try {
            const mixtureDesignCalculate = {
              obj: dataToSend.obj,
              responseColumn: item.responseColumns,
              recalculate: item.recalculate,
              columnsUsed: columns,
            };

            const allColumns = Object.keys(dataToSend.obj);

            const { data } = await fetcher.post<MixtureDesignResponse>(
              "mixtureDesign/calculate",
              mixtureDesignCalculate
            );

            const mixtureDesign = data.result;
            const responseColumnsData: Record<string, any> = {};
            const profilerObj: Record<string, any> = {};

            if (Object.keys(mixtureDesign).length <= 0) {
              return;
            }
            setIsRecalculate(item.recalculate);

            setGenerateChartList(variables);
            setChartBuilderData(dataToSend.obj);

            Object.keys(mixtureDesign).forEach(async (key) => {
              profilerObj[key] = [];

              const anovaTranslate = {
                term: commonT("mixtureDesign.term"),
                degreesOfFreedom: commonT("mixtureDesign.degreesOfFreedom"),
                sSquare: commonT("mixtureDesign.sSquare"),
                mSquare: commonT("mixtureDesign.mSquare"),
                fRatio: commonT("mixtureDesign.fRatio"),
                probF: commonT("mixtureDesign.probF"),
                model: commonT("mixtureDesign.model"),
                error: commonT("mixtureDesign.error"),
                total: commonT("mixtureDesign.total"),
              };

              responseColumnsData[key] = createAnovaTable(
                mixtureDesign[key].anovaTable,
                anovaTranslate
              );
              setLoadingAnalisysOfVariance(false);

              const parameterEstimateTranslate = {
                term: commonT("mixtureDesign.term"),
                estimates: commonT("mixtureDesign.estimate"),
                pValue: commonT("mixtureDesign.pValue"),
                stdError: commonT("mixtureDesign.stdError"),
                tRatio: commonT("mixtureDesign.tRatio"),
              };

              responseColumnsData[key]["parameterEstimate"] =
                createParameterEstimateTable(
                  mixtureDesign[key].parameterEstimates,
                  parameterEstimateTranslate
                );

              setLoadingParamterEstimate(false);

              const summaryOfFitTranslate = {
                term: commonT("mixtureDesign.term"),
                rSquare: commonT("mixtureDesign.rSquare"),
                rSquareAdjust: commonT("mixtureDesign.rSquareAdjust"),
                rmse: commonT("mixtureDesign.rmse"),
                mean: commonT("mixtureDesign.mean"),
                observations: commonT("mixtureDesign.observations"),
                value: commonT("mixtureDesign.value"),
              };

              responseColumnsData[key]["summaryOfFit"] =
                createSummaryOfFitTable(
                  mixtureDesign[key].summaryOfFit,
                  summaryOfFitTranslate
                );

              setLoadingSummaryOfFit(false);

              const rmse = mixtureDesign[key].summaryOfFit.find(
                (el) => el._row === "rmse"
              );
              profilerObj[key]["rmse"] = Math.sqrt(rmse?.df ?? 0);

              const dataHistogram: number[] = [];
              const labelsHistogram: string[] = [];
              mixtureDesign[key].parameterEstimates.forEach((el) => {
                dataHistogram.push(Math.abs(el.Estimate));
                labelsHistogram.push(
                  el._row.replaceAll(":", "*").replace("_interaction_", "*")
                );
              });

              responseColumnsData[key]["sortedEstimates"] = dataHistogram;
              responseColumnsData[key]["columnList"] = labelsHistogram;

              responseColumnsData[key]["yPredito"] = Object.values(
                mixtureDesign[key].predictValues
              );
              responseColumnsData[key]["y"] = mixtureDesign[key].y;

              setColNumber(
                Object.keys(dataToSend.obj).length - item.responseColumns.length
              );

              // Cria a equação
              let equationCalc = "Y = ";
              let equationCalcOrigin = "Y = ";

              const estimatesParameters = mixtureDesign[key].parameterEstimates;
              for (const item of estimatesParameters) {
                item._row = item._row
                  .replace("_interaction_", "*")
                  .replace("-", "*")
                  .replace(":", "*");

                equationCalc += `+ ${item._row} * (${item.Estimate})`;
                equationCalcOrigin += `+ ${item._row} * (${item.Estimate})`;
              }

              equationCalc = equationCalc.replaceAll("Y = + ", "Y = ");
              equationCalc = equationCalc.replaceAll("Y = - ", "Y = -");

              equationCalcOrigin = equationCalcOrigin.replaceAll(
                "Y = + ",
                "Y = "
              );
              equationCalcOrigin = equationCalcOrigin.replaceAll(
                "Y = - ",
                "Y = -"
              );

              responseColumnsData[key]["equationTooltip"] = equationCalcOrigin;
              responseColumnsData[key]["equation"] = equationCalc;

              profilerObj[key]["equation"] = equationCalc;
              profilerObj[key]["equationOrigin"] = equationCalcOrigin;

              profilerObj[key]["responseTitle"] = key;
              profilerObj[key]["predictionData"] =
                mixtureDesign[key].predictionData;

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

              profilerObj[key]["yPredicted"] = Object.values(
                mixtureDesign[key].predictValues
              );

              delete profilerObj[key]["data"];
              profilerObj[key]["data"] = createProfilerData;
              setLoadingProfiler(false);
            });

            setResponseColumnData(responseColumnsData);
            setResponseColumnKey(Object.keys(responseColumnsData));
            setProfilerData(profilerObj);
          } catch (error) {
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
    const countAccess = async () => {
      if (user) {
        await fetcher.post("access_counter", {
          user: user?.username,
          operation: "mixtureDesign",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  useEffect(() => {
    const updatedChartData: Record<string, any> = {};
    if (!responseColumnKey || responseColumnKey.length <= 0) return;
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
        "desc",
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
        recalculate ? responseColumnData[key].yPredito : []
      );

      updatedChartData[key] = chartsForVariable;
    });

    setChartData(updatedChartData);
    // setTimeout(() => {
    setLoadingChart(false);
    // }, 500);
  }, [responseColumnKey, recalculate]);

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

  const handleGenerateChart = (xVariables: string[], yVariables: string) => {
    setConstructorChart({
      xVariables: xVariables,
      yVariable: yVariables,
    });

    setIsOpenGenerateChart(false);
    setIsChartBuild(true);
  };

  return (
    <>
      <ContentHeader
        title={commonT("mixtureDesign.title")}
        tool={"mixtureDesign"}
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
                    {!loadingChart && (
                      <>
                        <ChartHub
                          key={key}
                          chartConfigurations={chartData[key]}
                          tool="mixtureDesign"
                          showLimits
                        />
                      </>
                    )}
                  </div>

                  <div style={{ marginBottom: "20px" }}>
                    <Equation
                      loading={false}
                      tooltipTitle={responseColumnData[key].equationTooltip}
                      equation={responseColumnData[key].equation}
                      width={undefined}
                    />
                  </div>

                  <div style={{ marginBottom: "20px" }}>
                    {isChartBuild ? (
                      <ChartBuilder
                        xVariable1={constructorChart.xVariables}
                        yVariable={constructorChart.yVariable}
                        chartData={chartBuilderData}
                      />
                    ) : (
                      <></>
                    )}
                  </div>
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
                      title={commonT("mixtureDesign.analysisOfVariance")}
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
                      title={commonT("mixtureDesign.parameterEstimates")}
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
                      title={commonT("mixtureDesign.summaryOfFit")}
                    />
                  </div>
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
                  loading={loadingProfiler}
                  type={"mixtureDesign"}
                />
              </Suspense>
            </>
          )}
        </div>
      )}

      <GenerateChart
        showOpenModal={isOpenGenerateChart}
        onModalClose={() => setIsOpenGenerateChart(false)}
        onSaveClick={handleGenerateChart}
        variables={generateChartList}
      />
    </>
  );
};

import { ColumnsType } from "antd/es/table";
import {
  MixtureDesignAnovaTable,
  MixtureDesignParamterEstimate,
  MixtureDesignSummaryOfFit,
  ParameterEstimatesRow,
} from "../interface";
import { formatNumber } from "utils/formatting";
import { P_VALUE_LIMIT, P_VALUE_NOT_REPLICATED_LIMIT } from "utils/constant";

export function createAnovaTable(
  analysisOfVariance: MixtureDesignAnovaTable[],
  translate: any
) {
  const columnsToAnalysisOfVariance: ColumnsType<any> = [
    {
      title: translate.term,
      dataIndex: "source",
      key: "source",
    },
    {
      title: translate.degreesOfFreedom,
      dataIndex: "degreesOfFreedom",
      key: "degreesOfFreedom",
    },
    {
      title: translate.sSquare,
      dataIndex: "sSquare",
      key: "sSquare",
    },
    {
      title: translate.mSquare,
      dataIndex: "mSquare",
      key: "mSquare",
    },
    {
      title: translate.fRatio,
      dataIndex: "fRatio",
      key: "fRatio",
    },
    {
      title: translate.probF,
      dataIndex: "probF",
      key: "probF",
      render: (item) => (
        <div>
          {typeof item !== "number"
            ? item
            : item > P_VALUE_LIMIT
            ? formatNumber(item)
            : "< 0.0001*"}
        </div>
      ),
      onCell: (item: any) => {
        return {
          ["style"]: { color: item.probF > P_VALUE_LIMIT ? "" : "red" },
        };
      },
    },
  ];

  const model = analysisOfVariance.find((el) => el._row === "Modelo");
  const error = analysisOfVariance.find((el) => el._row === "Erro");
  const total = analysisOfVariance.find((el) => el._row === "Total");

  const analysisOfVarianceDataSource = [
    {
      key: "model",
      source: translate.model,
      degreesOfFreedom: formatNumber(model?.df ?? 0),
      sSquare: (model?.sm ?? 0).toFixed(4),
      mSquare: (model?.ms ?? 0).toFixed(4),
      fRatio: formatNumber((model?.fValue ?? 0) / (model ? model.df : 0)),
      probF: (model?.prob ?? 0) / (model?.df ?? 0),
    },
    {
      key: "error",
      source: translate.error,
      degreesOfFreedom: formatNumber(error?.df ?? 0),
      sSquare: formatNumber(error?.sm ?? 0),
      mSquare: formatNumber(error?.ms ?? 0),
      fRatio: "",
      probF: "",
    },
    {
      key: "total",
      source: translate.total,
      degreesOfFreedom: formatNumber(total?.df ?? 0),
      sSquare: formatNumber(total?.sm ?? 0),
      mSquare: "",
      fRatio: "",
      probF: "",
    },
  ];
  return { analysisOfVarianceDataSource, columnsToAnalysisOfVariance };
}

export function createParameterEstimateTable(
  data: MixtureDesignParamterEstimate[],
  translate: any
) {
  const rows: any[] = [];

  for (const iterator of data) {
    const row = {
      estimate: iterator.Estimate,
      _row: iterator._row
        .replace("(Intercept)", "Intercept")
        .replace("_interaction_", "-")
        .replace("_squared", `*${iterator._row.replace("_squared", "")}`),
      stdError: iterator.stdError,
      t_value: iterator.t_value,
      prob: iterator.prob,
    };
    rows.push(row);
  }

  const terms = rows.map((el) => el._row.replace(":", "*"));

  const columnsToParameterEstimates: ColumnsType<any> = [
    {
      title: translate.term,
      dataIndex: "source",
      key: "source",
      filters: terms.map((term) => ({ text: term, value: term })),
    },
    {
      title: translate.estimates,
      dataIndex: "estimates",
      key: "estimates",
    },
    {
      title: translate.stdError,
      dataIndex: "stdError",
      key: "stdError",
    },
    {
      title: translate.tRatio,
      dataIndex: "tRatio",
      key: "tRatio",
    },
    {
      title: translate.pValue,
      dataIndex: "pValue",
      key: "pValue",
      onCell: (item: any) => {
        return {
          ["style"]: {
            color: item.pValue > P_VALUE_NOT_REPLICATED_LIMIT ? "" : "red",
          },
        };
      },
    },
  ];

  const parameterEstimatesDataSource: ParameterEstimatesRow[] = [];

  for (const item of rows) {
    const row: ParameterEstimatesRow = {
      key: item._row.replace(":", "*").replace("_interaction_", "*"),
      source: item._row.replace(":", "*").replace("_interaction_", "*"),
      estimates: formatNumber(item.estimate),
      stdError: formatNumber(item.stdError),
      tRatio: formatNumber(item.t_value),
      pValue: formatNumber(item.prob),
    };
    parameterEstimatesDataSource.push(row);
  }

  return { parameterEstimatesDataSource, columnsToParameterEstimates };
}

export function createSummaryOfFitTable(
  summaryOfFit: MixtureDesignSummaryOfFit[],
  translate: any
) {
  const columnsToSummaryOfFit: ColumnsType = [
    {
      title: translate.term,
      dataIndex: "source",
      key: "source",
    },
    {
      title: translate.rSquare,
      dataIndex: "rSquare",
      key: "rSquare",
    },
    {
      title: translate.rSquareAdjust,
      dataIndex: "rSquareAdjust",
      key: "rSquareAdjust",
    },
    {
      title: translate.rmse,
      dataIndex: "rmse",
      key: "rmse",
    },
    {
      title: translate.mean,
      dataIndex: "mean",
      key: "mean",
    },
    {
      title: translate.observations,
      dataIndex: "observations",
      key: "observations",
    },
  ];

  const r2 = summaryOfFit.find((el) => el._row === "r2");
  const r2adjust = summaryOfFit.find((el) => el._row === "r2adjust");
  const rmse = summaryOfFit.find((el) => el._row === "rmse");
  const mean = summaryOfFit.find((el) => el._row === "mean");
  const observation = summaryOfFit.find((el) => el._row === "observation");

  const summaryOfFitDataSource = [
    {
      key: "1",
      source: translate.value,
      rSquare: formatNumber(r2?.df ?? 0),
      rSquareAdjust: formatNumber(r2adjust?.df ?? 0),
      rmse: formatNumber(rmse?.df ?? 0),
      mean: formatNumber(mean?.df ?? 0),
      observations: observation?.df ?? 0,
    },
  ];

  return { summaryOfFitDataSource, columnsToSummaryOfFit };
}

