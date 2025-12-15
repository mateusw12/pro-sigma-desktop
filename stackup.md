Vamos fazer a ferramenta de stackup 2D.

Meu antigo front end

import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import ContentHeader from "shared/contentHeader";
import { useAuth } from "hooks/useAuth";
import { useTranslation } from "next-i18next";
import {
  Col,
  Row,
  Form,
  Select,
  InputNumber,
  Button,
  Upload,
  message,
  notification,
} from "antd";
import * as Styled from "./styled";
import { BsCalculatorFill } from "react-icons/bs";
import { UploadOutlined } from "@ant-design/icons";
import Input from "antd/es/input/Input";
import { CharacteristicData, StackUpCalculate, StackUpData } from "./interface";
import {
  AiOutlineDownload,
  AiOutlineRedo,
  AiOutlineSend,
} from "react-icons/ai";
import Table from "shared/table";
import { CreateDistributionFittingTable } from "./table";
import { Spin } from "shared/spin"
import Equation from "shared/equation";
import { ExcelProps, saveAsExcel } from "utils/export";
import ExcelJS from "exceljs";
import {
  convertWorksheetToCSV,
  parseCSVToDictionary,
} from "components/insertData/shared/upload/uploaderUtils";

const fetcher = axios.create({
  baseURL: "/api",
});

export const StackUp: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });
  const { user } = useAuth();

  const fittingDistribution = useRef<any>({});

  const [factors, setFactors] = useState<number>(0);
  const [rounds, setRounds] = useState<number>(5000);
  const [equation, setEquation] = useState<string>("");
  const [dataFrame, setDataFrame] = useState<string>("");
  const [file, setFile] = useState<any>(null);

  const [factorData, setFactorData] = useState<{
    [key: string]: CharacteristicData;
  }>({});

  const [visibleFittingDistribution, setVisibleFittingDistribution] =
    useState(false);

  const [loadingFittingDistribution, setLoadingFittingDistribution] =
    useState(true);

  const [loadingCalculateContent, setLoadingCalculateContent] = useState(false);
  const [loadingFile, setLoadingFile] = useState(false);

  const [form] = Form.useForm();

  const quotaOptions = [
    {
      label: commonT("stackUp.standard"),
      value: "1",
    },
    {
      label: commonT("stackUp.ctq"),
      value: "1.33",
    },
    {
      label: commonT("stackUp.cts"),
      value: "2",
    },
  ];

  const quotaMapping: { [key: string]: string } = {
    Standard: "1",
    CTS: "1.33",
    CTQ: "2",
  };

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "stackUp",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  const handleGenerateFactors = () => {
    const newFactorData: { [key: string]: CharacteristicData } = {};
    for (let i = 0; i < factors; i++) {
      const label = String.fromCharCode(65 + i);
      newFactorData[label] = {
        min: 0,
        max: 0,
        sensitivity: 0,
        quota: "1",
        name: "",
      };
    }

    setFactorData(newFactorData);
  };

  const handleCalculate = async () => {
    if (Object.keys(factorData).length <= 0) {
      message.error({
        content: commonT("stackUp.notValuesErrorMsg"),
      });
      return;
    }

    for (const key in factorData) {
      const { min, max } = factorData[key];

      if (min > max) {
        message.error({
          content: commonT("stackUp.minMaxValidationMsg"),
        });

        return;
      }
    }

    const stackUpCalculate: StackUpCalculate = {
      factors: factorData,
      rounds: rounds,
    };

    setLoadingFittingDistribution(true);
    setLoadingCalculateContent(true);
    try {
      const { data } = await fetcher.post<StackUpData>(
        "stackUp/calculate",
        stackUpCalculate
      );
      setLoadingCalculateContent(false);

      const fittingTranslate = {
        term: commonT("stackUp.term"),
        mean: commonT("stackUp.mean"),
        std: commonT("stackUp.std"),
      };

      fittingDistribution.current = CreateDistributionFittingTable(
        data.means,
        data.stds,
        fittingTranslate
      );

      setLoadingFittingDistribution(false);
      setVisibleFittingDistribution(true);

      const newEquation = `Y = ${data.equation}`;
      setEquation(newEquation);
      setDataFrame(data.dataFrame);
    } catch (error: any) {
      console.error(error);
      message.error({
        content: commonT("error.general.proccesData"),
      });
    }
  };

  const handleChange = (
    label: string,
    field: keyof CharacteristicData,
    value: any
  ) => {
    setFactorData((prevData) => ({
      ...prevData,
      [label]: {
        ...prevData[label],
        [field]: value,
      },
    }));
  };

  const handleExportDataTable = async () => {
    await saveAsExcel(JSON.parse(dataFrame), commonT("stackUp.title"));
  };

  const handleGenerateStandardFile = async () => {
    const key1 = commonT("stackUp.standardFile.characteristic");
    const key2 = commonT("stackUp.standardFile.minValue");
    const key3 = commonT("stackUp.standardFile.maxValue");
    const key4 = commonT("stackUp.standardFile.sensitivity");
    const key5 = commonT("stackUp.standardFile.quota");
    const standardFile = {
      [key1]: {
        "0": "",
      },
      [key2]: {
        "0": "",
      },
      [key3]: {
        "0": "",
      },
      [key4]: {
        "0": "",
      },
      [key5]: {
        "0": "",
      },
    };

    const excelProps: ExcelProps = {
      formulaValues: {
        columnsIndex: [Object.keys(standardFile).length],
        errorTitle: commonT("stackUp.standardFile.errorTitle"),
        errorMsg: commonT("stackUp.standardFile.errorMsg"),
        showErrorMessage: true,
        values: ['"Standard,CTS,CTQ"'],
      },
    };

    await saveAsExcel(standardFile, commonT("stackUp.title"), excelProps);
  };

  useEffect(() => {
    if (file) {
      const rows = 5000;

      const characteristicsIndex = 0;
      const minValueIndex = 1;
      const maxValueIndex = 2;
      const sensitivityIndex = 3;
      const quotaIndex = 4;

      const variables = Object.keys(file);

      const newFactorData: { [key: string]: CharacteristicData } =
        populateFactorData(
          file,
          variables,
          minValueIndex,
          maxValueIndex,
          sensitivityIndex,
          quotaMapping,
          quotaIndex,
          characteristicsIndex
        );

      setFactorData(newFactorData);
      setRounds(rows);
      setFactors(Object.keys(newFactorData).length);
      populateFormValues(newFactorData, rows);

      setLoadingFile(false);
    }
  }, [file]);

  const handleClearValues = () => {
    setFactorData({});
    setRounds(0);
    setFactors(0);
    setVisibleFittingDistribution(false);
    form.resetFields();
  };

  function populateFormValues(
    newFactorData: { [key: string]: CharacteristicData },
    rows: number
  ) {
    const formValues = Object.keys(newFactorData).reduce((acc, label) => {
      acc["characteristicQuantity"] = Object.keys(newFactorData).length;
      acc["roundsNumber"] = rows;
      acc[`characteristicName-${label}`] = label;
      acc[`minValue-${label}`] = newFactorData[label].min;
      acc[`maxValue-${label}`] = newFactorData[label].max;
      acc[`sensitivity-${label}`] = newFactorData[label].sensitivity;
      acc[`quota-${label}`] = newFactorData[label].quota;
      acc[`quota-${label}`] = newFactorData[label].quota;
      return acc;
    }, {});

    form.setFieldsValue(formValues);
  }

  function populateFactorData(
    dataSource: Record<string, any[]>,
    variables: string[],
    minValueIndex: number,
    maxValueIndex: number,
    sensitivityIndex: number,
    quotaMapping: { [key: string]: string },
    quotaIndex: number,
    characteristicsIndex: number
  ): { [key: string]: CharacteristicData } {
    const newFactorData: { [key: string]: CharacteristicData } = {};

    const rowCount = dataSource[variables[0]].length;

    for (let i = 0; i < rowCount; i++) {
      const name = dataSource[variables[characteristicsIndex]][i];
      const min = dataSource[variables[minValueIndex]][i];
      const max = dataSource[variables[maxValueIndex]][i];
      const sensitivity = dataSource[variables[sensitivityIndex]][i];
      const quotaKey = dataSource[variables[quotaIndex]][i];
      const quota = quotaMapping[quotaKey] || "1";

      newFactorData[name] = {
        min,
        max,
        sensitivity,
        quota,
        name,
      };
    }

    return newFactorData;
  }

  return (
    <>
      <ContentHeader title={commonT("stackUp.title")} tool={"stackUp"} />
      <div id="content">
        <Row>
          <Col xs={24} sm={24} md={24} xl={24} xxl={13}>
            <Styled.Container>
              <Form name="form" form={form} layout="inline">
                <Styled.ToolBar>
                  <Form.Item
                    name="characteristicQuantity"
                    label={commonT("stackUp.characteristicQuantity")}
                    rules={[{ required: true, message: "" }]}
                  >
                    <InputNumber
                      min={1}
                      max={50}
                      value={factors}
                      onChange={(value) => setFactors(value || 0)}
                    />
                  </Form.Item>
                  <Form.Item
                    name="roundsNumber"
                    rules={[{ required: true, message: "" }]}
                    label={commonT("stackUp.roundsNumber")}
                  >
                    <InputNumber
                      min={1}
                      max={250000}
                      value={rounds}
                      onChange={(value) => setRounds(value || 0)}
                    />
                  </Form.Item>
                  <Button
                    type="primary"
                    style={{ marginLeft: "8px" }}
                    icon={<AiOutlineSend style={{ transform: "scale(1.3)" }} />}
                    onClick={() => {
                      setFactorData({});
                      handleGenerateFactors();
                    }}
                  >
                    {commonT("stackUp.generateCharacteristic")}
                  </Button>
                </Styled.ToolBar>

                <Styled.TollBarFile>
                  <Button
                    type="primary"
                    style={{ marginLeft: "8px" }}
                    icon={
                      <AiOutlineDownload style={{ transform: "scale(1.3)" }} />
                    }
                    onClick={handleGenerateStandardFile}
                  >
                    {commonT("stackUp.downloadStandardFile")}
                  </Button>
                  <Upload
                    maxCount={1}
                    name="file"
                    showUploadList={false}
                    accept=".xlsx, .csv"
                    beforeUpload={async (file) => {
                      const maxSize = 50 * 1024 * 1024; // 50MB
                      const size5Mb = 5 * 1024; // 5MB

                      if (file.size > maxSize) {
                        message.error({
                          content: commonT("fileSizeErrorMsg"),
                        });
                        setLoadingFile(false);
                        return false;
                      }

                      if (file.size >= size5Mb) {
                        setLoadingFile(false);
                        notification.info({
                          message: commonT("startLoadFile"),
                        });
                      }

                      const isCSV = file.name.endsWith(".csv");
                      const isExcel = file.name.endsWith(".xlsx");

                      const reader = new FileReader();

                      reader.onload = async (e: ProgressEvent<FileReader>) => {
                        const data = e.target?.result;
                        if (!data) {
                          setLoadingFile(false);
                          return;
                        }

                        try {
                          if (isCSV) {
                            const csvText = data as string;
                            const dictionary = parseCSVToDictionary(csvText);
                            setFile(dictionary);
                            notification.success({
                              message: commonT("sheetSuccess"),
                            });
                          } else if (isExcel) {
                            const workbook = new ExcelJS.Workbook();
                            const arrayBuffer = data as ArrayBuffer;

                            await workbook.xlsx.load(arrayBuffer);
                            const worksheet = workbook.worksheets[0];

                            if (!worksheet) {
                              notification.error({
                                message: commonT("sheetNotFound"),
                              });
                              return;
                            }
                            const csvData = await convertWorksheetToCSV(
                              worksheet
                            );
                            const dictionary = parseCSVToDictionary(csvData);
                            setFile(dictionary);
                            notification.success({
                              message: commonT("sheetSuccess"),
                            });
                          }
                        } catch (error) {
                          notification.error({
                            message: commonT("fileProcessError"),
                          });
                          console.error("Erro ao processar o arquivo:", error);
                        }
                      };

                      if (isCSV) {
                        reader.readAsText(file);
                      } else {
                        reader.readAsArrayBuffer(file);
                      }

                      return false;
                    }}
                  >
                    <Button
                      type="primary"
                      style={{ marginLeft: "8px" }}
                      icon={
                        <UploadOutlined
                          name="file"
                          style={{ transform: "scale(1.3)" }}
                        />
                      }
                      onClick={() => {
                        setFactorData({});
                        handleGenerateFactors();
                      }}
                    >
                      {commonT("stackUp.attachFile")}
                    </Button>
                  </Upload>
                </Styled.TollBarFile>

                <Styled.BodyContainer>
                  {loadingFile ? (
                    <>
                      <Spin />
                    </>
                  ) : (
                    <>
                      {Object.keys(factorData).map((label, index) => (
                        <Styled.CharacteristicContainer key={label}>
                          <h3>
                            {commonT("stackUp.characteristic")} {index + 1}
                          </h3>

                          <div style={{ paddingBottom: 20 }}>
                            <Form.Item
                              name={`characteristicName-${label}`}
                              rules={[{ required: true, message: "" }]}
                              label={commonT("stackUp.characteristicName")}
                            >
                              <Input
                                style={{ width: "100%" }}
                                value={factorData[label].name}
                                onChange={(e) =>
                                  handleChange(label, "name", e.target.value)
                                }
                              />
                            </Form.Item>
                          </div>

                          <Styled.CharacteristicContainerBody>
                            <Form.Item
                              name={`minValue-${label}`}
                              rules={[{ required: true, message: "" }]}
                              label={commonT("stackUp.minValue")}
                            >
                              <InputNumber
                                value={factorData[label].min}
                                onChange={(value) =>
                                  handleChange(label, "min", value)
                                }
                              />
                            </Form.Item>
                            <Form.Item
                              name={`maxValue-${label}`}
                              rules={[{ required: true, message: "" }]}
                              label={commonT("stackUp.maxValue")}
                            >
                              <InputNumber
                                value={factorData[label].max}
                                onChange={(value) =>
                                  handleChange(label, "max", value)
                                }
                              />
                            </Form.Item>
                            <Form.Item
                              name={`sensitivity-${label}`}
                              rules={[{ required: true, message: "" }]}
                              label={commonT("stackUp.sensitivity")}
                            >
                              <InputNumber
                                value={factorData[label].sensitivity}
                                onChange={(value) =>
                                  handleChange(label, "sensitivity", value)
                                }
                              />
                            </Form.Item>
                            <Form.Item
                              name={`quota-${label}`}
                              rules={[{ required: true, message: "" }]}
                              label={commonT("stackUp.quota")}
                            >
                              <Select
                                options={quotaOptions}
                                defaultValue="1"
                                value={factorData[label].quota}
                                onChange={(value) =>
                                  handleChange(label, "quota", value)
                                }
                              />
                            </Form.Item>
                          </Styled.CharacteristicContainerBody>
                        </Styled.CharacteristicContainer>
                      ))}
                    </>
                  )}
                </Styled.BodyContainer>
              </Form>

              <Styled.FooterLine />
              <Styled.FooterContainer>
                <Button
                  icon={
                    <BsCalculatorFill style={{ transform: "scale(1.3)" }} />
                  }
                  type="primary"
                  onClick={handleCalculate}
                >
                  {commonT("stackUp.calculate")}
                </Button>
                <Button
                  type="primary"
                  onClick={handleExportDataTable}
                  disabled={dataFrame.length <= 0}
                  icon={
                    <AiOutlineDownload style={{ transform: "scale(1.3)" }} />
                  }
                >
                  {commonT("stackUp.downloadDataTable")}
                </Button>

                <Button
                  type="primary"
                  onClick={handleClearValues}
                  icon={<AiOutlineRedo style={{ transform: "scale(1.3)" }} />}
                >
                  {commonT("stackUp.clear")}
                </Button>
              </Styled.FooterContainer>
            </Styled.Container>
          </Col>

          <Col xs={24} sm={24} md={24} xl={24} xxl={{ offset: 1, span: 10 }}>
            {loadingCalculateContent ? (
              <>
                <Spin />
              </>
            ) : (
              <>
                {visibleFittingDistribution && (
                  <>
                    <div style={{ marginBottom: 20 }}>
                      <Table
                        dataSource={fittingDistribution.current.datasource}
                        columns={fittingDistribution.current.columns}
                        loading={loadingFittingDistribution}
                        title={commonT("stackUp.distributionSummary")}
                      />
                    </div>

                    <div style={{ marginBottom: 20 }}>
                      <Equation
                        equation={equation}
                        loading={false}
                        tooltipTitle={equation}
                        width={undefined}
                      />
                    </div>
                  </>
                )}
              </>
            )}
          </Col>
        </Row>
      </div>
    </>
  );
};

import { ColumnsType } from "antd/es/table";
import { formatNumber } from "utils/formatting";

export const CreateDistributionFittingTable = (
  means: Record<string, number>,
  stds: Record<string, number>,
  translate: any
) => {
  const columns: ColumnsType<any> = [
    {
      title: translate.term,
      dataIndex: "term",
      key: "term",
      align: "center",
    },
    {
      title: translate.mean,
      dataIndex: "mean",
      key: "mean",
      align: "center",
    },
    {
      title: translate.std,
      dataIndex: "std",
      key: "std",
      align: "center",
    },
  ];

  const datasource = [];

  for (const key of Object.keys(means)) {
    datasource.push({
      term: key,
      mean: formatNumber(means[key]),
      std: formatNumber(stds[key], 4),
    });
  }

  return {
    datasource,
    columns,
  };
};


meu back-end que da pra reaproveitar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.stack_up.stack_up_calculate import (
    calculate_equation,
    calculate_equation_not_normalized,
    calculate_means_std,
    generate_data_frame,
    generate_distributions,
    normalize_column_name,
    trim_named,
)

stack_up_router = APIRouter()


class Factors(BaseModel):
    name: str
    min: float
    max: float
    sensitivity: float
    quota: str


class Item(BaseModel):
    rounds: int
    factors: dict[str, Factors]


@stack_up_router.post(
    "/calculate",
    tags=["Empilhamento de Tolerância"],
    description=""" Calcula Empilhamento de tolerância
            rounds: numero de rodadas;
            factors: objeto das caracteristicas
            """,
    response_model=object,
)
def calculate_stack_up(body: Item):
    try:
        rounds = body.rounds
        factors = body.factors

        # Remove espaços em brancos
        new_factors = trim_named(factors)

        # Calcula média e desvio padrão
        means, stds = calculate_means_std(new_factors)

        # Gera equação
        equation = calculate_equation(new_factors)

        # Calcula as distribuições
        distributions = generate_distributions(rounds, means, stds)

        # Gera data frame
        df = generate_data_frame(distributions)

        df.columns = [normalize_column_name(col) for col in df.columns]

        # Calcula coluna resposta
        df["Y"] = df.eval(equation)

        new_equation = calculate_equation_not_normalized(factors)

        return {
            "means": means,
            "stds": stds,
            "equation": new_equation,
            "dataFrame": df.to_json(),
        }

    except Exception as e:
        print("e", e)
        raise HTTPException(status_code=500, detail="stackUpError")

import re
import numpy as np
import pandas as pd
from typing import Any, List

# Gera data frame
def generate_data_frame(distributions: dict[str, List[float]]):
    """
    Gera um DataFrame a partir de um dicionário contendo distribuições de valores.

    Parâmetros:
    ----------
    distributions : dict[str, List[float]]
        Dicionário onde as chaves representam os nomes das colunas e os valores são listas de números.

    Retorno:
    -------
    pd.DataFrame
        DataFrame contendo as distribuições fornecidas, com as chaves como nomes das colunas.
    """
    column_names = {key: key for key, _ in distributions.items()}
    data = {column_names[key]: values for key, values in distributions.items()}
    df = pd.DataFrame(data)
    return df

# Gera as distribuições normais
def generate_distributions(rounds: int, means: dict[str, float], stds: dict[str, float]):
    """
    Gera distribuições normais baseadas em médias e desvios padrão.

    Parâmetros:
    ----------
    rounds : int
        Número de amostras a serem geradas para cada distribuição.
    means : dict[str, float]
        Dicionário contendo as médias de cada fator.
    stds : dict[str, float]
        Dicionário contendo os desvios padrão de cada fator.
    """
    distributions: dict[str, List[float]] = {}
    
    for key, (mean, std) in zip(means.keys(), zip(means.values(), stds.values())):
        distributions[key] = np.random.normal(loc=mean, scale=std, size=rounds).tolist()
    return distributions

# Calcula equação
def normalize_column_name(column_name: str):
    """
    Normaliza o nome de uma coluna, substituindo caracteres especiais e espaços por underscores.

    Parâmetros:
    ----------
    column_name : str
        Nome original da coluna.
    """
    return re.sub(r'\W|^(?=\d)', '_', column_name)

# Função para calcular a equação
def calculate_equation(factors: dict[str, Any]):
    """
    Calcula a equação com base nos fatores fornecidos, usando nomes normalizados.

    Parâmetros:
    ----------
    factors : dict[str, Any]
        Dicionário onde cada valor contém:
        - `name` (str): Nome do fator.
        - `sensitivity` (float): Sensibilidade do fator.
    """
    equation_parts = []
    for _, value in factors.items():
        column_name = normalize_column_name(value.name)
        equation_parts.append(f"{column_name} * ({value.sensitivity})")
    
    return " + ".join(equation_parts)

def calculate_equation_not_normalized(factors: dict[str, Any]):
    """
    Calcula a equação com base nos fatores fornecidos, sem normalizar os nomes das colunas.

    Parâmetros:
    ----------
    factors : dict[str, Any]
        Dicionário onde cada valor contém:
        - `name` (str): Nome do fator.
        - `sensitivity` (float): Sensibilidade do fator.
    """
    equation_parts = []
    for _, value in factors.items():
        equation_parts.append(f"{value.name} * ({value.sensitivity})")
    
    return " + ".join(equation_parts)

# Calcula média e desvio padrão das caracteristicas
def calculate_means_std(factors: dict[str, Any]):
    """
    Calcula as médias e os desvios padrão para os fatores fornecidos.

    Parâmetros:
    ----------
    factors : dict[str, Any]
        Dicionário onde cada valor contém:
        - `name` (str): Nome do fator.
        - `min` (float): Valor mínimo do fator.
        - `max` (float): Valor máximo do fator.
        - `quota` (float): Quota para cálculo do desvio padrão.
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for _, value in factors.items():
        means[value.name] = (value.max + value.min) / 2
        stds[value.name] = (value.max - value.min) / (6 * float(value.quota))

    return means, stds

def trim_named(factors: dict[str, Any]):
    """
    Remove espaços dos nomes dos fatores.

    Parâmetros:
    ----------
    factors : dict[str, Any]
        Dicionário onde cada valor contém:
        - `name` (str): Nome do fator.
    """
    for _, factor in factors.items():
        factor.name = factor.name.replace(" ", "")
    return factors