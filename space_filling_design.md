Vamos fazer a analise de Space Filling Design

o usuário pode colcoar varias colunas X e interações e varios Y. Se tiver mais de um 1 Y é feito uma analise para cada Y.
O usuário também pode ter a opção de gerar um experimento

meu front end

import React, { useEffect, useState, Suspense } from "react";
import ContentHeader from "shared/contentHeader";
import { Spin } from "shared/spin";
import { useTranslation } from "next-i18next";
import { useAuth } from "hooks/useAuth";
import axios from "axios";
import { Col, Row, message } from "antd";
import Table from "shared/table";
import {
  DataToCompile,
  SpaceFillingExportData,
  SpaceFillingParameterEstimateTranslate,
  SpaceFillingResponseData,
} from "./interface";
import {
  createAnovaTable,
  createParameterEstimateTable,
  createSummaryOfFitTable,
} from "./table";
import ResponseContentHeader from "shared/responseContentHeader";
import Equation from "shared/equation";
import { formatNumber } from "utils/formatting";
import dynamic from "next/dynamic";
import {
  dataSortOrder,
  createDataFrames,
  createProfilerDataObj,
  getStaticProfilerDataConfig,
} from "utils/core";
import { useRouter } from "next/router";
import { getItem } from "utils/database";
import { HighChartTemplate } from "shared/widget/chartHub/interface";

const GenerateChart = dynamic(() => import("./generateChartModal"), {
  ssr: false,
});

const ChartBuilder = dynamic(() => import("./chartBuilder"), {
  ssr: false,
});

const Profiler = dynamic(() => import("shared/profiler"), {
  ssr: false,
});

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const fetcher = axios.create({
  baseURL: "/api",
});

export const SpaceFilling: React.FC = () => {
  const { t: commonT } = useTranslation("common");
  const { user } = useAuth();
  const router = useRouter();

  const [responseColumnData, setResponseColumnData] = useState<
    Record<string, any>
  >({});
  const [responseColumnKey, setResponseColumnKey] = useState<string[]>([]);
  const [contentVisibility, setContentVisibility] = useState<boolean[]>(
    new Array(responseColumnKey.length).fill(false)
  );

  const [generateChartList, setGenerateChartList] = useState<string[]>([]);

  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingSummaryOfFit, setLoadingSummaryOfFit] = useState(true);
  const [loadingParamterEstimate, setLoadingParamterEstimate] = useState(true);
  const [loadingAnalisysOfVariance, setLoadingAnalisysOfVariance] =
    useState(true);
  const [loadingProfiler, setLoadingProfiler] = useState(true);

  const [isOpenGenerateChart, setIsOpenGenerateChart] = useState(false);
  const [constructorChart, setConstructorChart] = useState<any>({});
  const [isChartBuild, setIsChartBuild] = useState(false);

  const [chartBuilderData, setChartBuilderData] = useState<
    Record<string, number[]>
  >({});
  const [profilerData, setProfilerData] = useState<Record<string, any>>({});
  const [isRecalculate, setIsRecalculate] = useState(false);
  const [colNumber, setColNumber] = useState(3);
  const [rows, setRows] = useState(0);

  const [chartData, setChartData] = useState<{
    [key: string]: HighChartTemplate;
  }>({});

  useEffect(() => {
    const getData = async () => {
      setIsChartBuild(false);

      const parsedUrlQuery = router?.query;
      if (Object.keys(parsedUrlQuery).length > 0) {
        const tool = parsedUrlQuery.tool as string;
        const uid = parsedUrlQuery.uid as string;

        const item = (await getItem(tool, uid)) as SpaceFillingExportData;

        if (item) {
          const variables = Object.keys(item.dataToExport[0]);

          const dataToSend: DataToCompile = { obj: {}, itens: variables };

          variables.map((variable, index) => {
            item.dataToExport.map((item: any) => {
              if (dataToSend.obj[`${variable}`] === undefined) {
                dataToSend.obj[`${variable}`] = [];
              }
              if (
                index === variables.length - 1 &&
                typeof item[variable] !== "number"
              ) {
                item[variable] = parseFloat(item[variable]);
              }
              dataToSend.obj[`${variable}`].push(item[variable]);
            });
          });

          const allColumns = Object.keys(dataToSend.obj);

          setColNumber(
            Object.keys(dataToSend.obj).length - item.responseColumns.length
          );

          const firstColumnRow =
            dataToSend.obj[Object.keys(dataToSend.obj)[0]].length;
          setRows(firstColumnRow);

          try {
            const spaceFillingCalculate = {
              data: dataToSend.obj,
              responseColumn: item.responseColumns,
              recalculate: item.recalculate,
              interactionColumn: item.iterationColumns
                ? item.iterationColumns
                : [],
            };

            const { data } = await fetcher.post<SpaceFillingResponseData>(
              "spaceFilling/calculate",
              spaceFillingCalculate
            );

            const spaceFilling = data.spaceFilling;

            const responseColumnsData: Record<string, any> = {};
            const profilerObj: Record<string, any> = {};

            if (Object.keys(spaceFilling).length <= 0) {
              return;
            }

            setLoadingPage(false);

            for (const key of Object.keys(spaceFilling)) {
              setIsRecalculate(spaceFilling[key].isRecalculate);

              const createAnovaSend = JSON.parse(
                JSON.stringify(spaceFilling[key].anovaTable)
              );

              createAnovaSend["translate"] = {
                term: commonT("spaceFilling.term"),
                degreesOfFreedom: commonT("spaceFilling.degreesOfFreedom"),
                sSquare: commonT("spaceFilling.sSquare"),
                mSquare: commonT("spaceFilling.mSquare"),
                fRatio: commonT("spaceFilling.fRatio"),
                probF: commonT("spaceFilling.probF"),
                model: commonT("spaceFilling.model"),
                error: commonT("spaceFilling.error"),
                total: commonT("spaceFilling.total"),
              };

              responseColumnsData[key] = createAnovaTable(createAnovaSend);
              setLoadingAnalisysOfVariance(false);

              const createLackOfFitSend = JSON.parse(
                JSON.stringify(spaceFilling[key].lackOfFit)
              );

              createLackOfFitSend["translate"] = {
                term: commonT("spaceFilling.term"),
                degreesOfFreedom: commonT("spaceFilling.degreesOfFreedom"),
                sSquare: commonT("spaceFilling.sSquare"),
                mSquare: commonT("spaceFilling.mSquare"),
                fRatio: commonT("spaceFilling.fRatio"),
                probF: commonT("spaceFilling.probF"),
                lackOfFit: commonT("spaceFilling.lackOfFit"),
                pureError: commonT("spaceFilling.pureError"),
                total: commonT("spaceFilling.total"),
              };

              const createSummaryOfFitSend = JSON.parse(
                JSON.stringify(spaceFilling[key].summarOfFit)
              );

              createSummaryOfFitSend["translate"] = {
                term: commonT("spaceFilling.term"),
                rSquare: commonT("spaceFilling.rSquare"),
                rSquareAdjust: commonT("spaceFilling.rSquareAdjust"),
                rmse: commonT("spaceFilling.rmse"),
                mean: commonT("spaceFilling.mean"),
                observations: commonT("spaceFilling.observations"),
                value: commonT("spaceFilling.value"),
              };

              responseColumnsData[key]["summaryOfFit"] =
                createSummaryOfFitTable(createSummaryOfFitSend);
              setLoadingSummaryOfFit(false);

              profilerObj[key] = [];

              profilerObj[key]["rmse"] = Math.sqrt(
                spaceFilling[key].summarOfFit.rmse
              );

              profilerObj[key]["minValue"] = Math.min(
                ...spaceFilling[key].yPredicteds
              );
              profilerObj[key]["maxValue"] = Math.max(
                ...spaceFilling[key].yPredicteds
              );

              variables.unshift("Intercept");

              const parameterEstimates = spaceFilling[key].parameterEstimates;

              const estimatesList = Object.entries(
                spaceFilling[key].parameterEstimates
              ).reduce((acc: Record<string, number>, [key, value]) => {
                acc[key] = Math.abs(value.estimates);
                return acc;
              }, {});

              const sortedList: { [key: string]: number } = dataSortOrder(
                estimatesList,
                "desc",
                true
              );

              const correlationMatrixEquationVariables: string[] =
                Object.keys(parameterEstimates);
              const columnList = Object.entries(sortedList).map(([key]) =>
                key.replaceAll("/", "*")
              );

              responseColumnsData[key]["estimatesList"] = estimatesList;
              responseColumnsData[key]["columnList"] = columnList;

              responseColumnsData[key]["y"] = spaceFilling[key].y;
              responseColumnsData[key]["yPredito"] =
                spaceFilling[key].yPredictedsOdered;

              setGenerateChartList(variables);

              const createParameterEstimateSend = JSON.parse(
                JSON.stringify(spaceFilling[key].parameterEstimates)
              );

              const parameterEstimateTranslate: SpaceFillingParameterEstimateTranslate =
                {
                  term: commonT("spaceFilling.term"),
                  estimates: commonT("spaceFilling.estimate"),
                  pValue: commonT("spaceFilling.pValue"),
                  stdError: commonT("spaceFilling.stdError"),
                  tRatio: commonT("spaceFilling.tRatio"),
                };

              responseColumnsData[key]["parameterEstimate"] =
                createParameterEstimateTable(
                  createParameterEstimateSend,
                  parameterEstimateTranslate
                );

              setLoadingParamterEstimate(false);

              const betaMatrix: number[][] = Object.values(
                spaceFilling[key].betas
              );

              // Cria a equação
              const means = spaceFilling[key].mean;

              // Cria a equação
              let equationCalcOrigin = "Y = ";

              for (let i = 0; i < betaMatrix.length; i++) {
                const variable = correlationMatrixEquationVariables[i];
                const beta = parseFloat(formatNumber(Number(betaMatrix[i])));

                if (!variable) continue;

                if (variable === "Intercept") {
                  equationCalcOrigin += beta;
                }

                if (variable !== "Intercept" && !variable.includes("/")) {
                  const interation = variable.split("-");
                  if (interation.length <= 1) {
                    equationCalcOrigin +=
                      beta < 0
                        ? ` - ${formatNumber(Math.abs(beta))} * ${variable}`
                        : ` + ${formatNumber(beta)} * ${variable}`;
                  }
                }

                if (variable.includes("/")) {
                  const variableSplited = variable.split("/")[0];
                  const parameterEstimateQuadratic = (
                    spaceFilling[key].parameterEstimates as any
                  )[variable];

                  if (parameterEstimateQuadratic) {
                    equationCalcOrigin += ` + (${variableSplited} - (${formatNumber(
                      means[variableSplited]
                    )})) * ((${variableSplited} - (${formatNumber(
                      means[variableSplited]
                    )})) * (${formatNumber(
                      parameterEstimateQuadratic.estimates
                    )}))`;
                  }
                }
              }

              responseColumnsData[key]["equation"] = equationCalcOrigin;
              responseColumnsData[key]["equationOrigin"] = equationCalcOrigin;

              profilerObj[key]["equation"] = equationCalcOrigin;
              profilerObj[key]["equationOrigin"] = equationCalcOrigin;
              profilerObj[key]["optimizationEquation"] = equationCalcOrigin;

              profilerObj[key]["responseTitle"] = key;

              // Cria toolip da equação
              // Verifica se há uma coluna "intercept" no correlationMatrixEquationVariables

              let equationTooltipCalc = "Y = ";

              for (let i = 0; i < betaMatrix.length; i++) {
                const variable = correlationMatrixEquationVariables[i];
                const beta = parseFloat(formatNumber(Number(betaMatrix[i])));

                if (variable === "Intercept") {
                  equationTooltipCalc += "Intercept";
                }

                if (variable !== "Intercept") {
                  const betaNumber = i + 1;
                  equationTooltipCalc +=
                    beta < 0
                      ? ` - ${variable} * ${commonT(
                          "spaceFilling.beta"
                        )} ${betaNumber}`
                      : ` + ${variable} * ${commonT(
                          "spaceFilling.beta"
                        )} ${betaNumber}`;
                }
              }

              responseColumnsData[key]["equationTooltip"] = equationTooltipCalc;

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

            setProfilerData(profilerObj);
            setChartBuilderData(dataToSend.obj);
            setResponseColumnData(responseColumnsData);
            setResponseColumnKey(Object.keys(responseColumnsData));
            setLoadingProfiler(false);
          } catch (error: any) {
            if (error.response.data === "spaceFillingSplitDfError") {
              message.error({
                content: commonT("error.general.proccesData"),
              });
            } else if (error.response.data === "spaceFillingAnovaError") {
              message.error({
                content: commonT("error.general.anovaErrorMsg"),
              });
            } else if (error.response.data === "spaceFillingSummaryError") {
              message.error({
                content: commonT("error.general.summaryOfFitErrorMsg"),
              });
            } else if (error.response.data === "spaceFillingLackOfFitError") {
              message.error({
                content: commonT("error.general.lackOfFitErrorMsg"),
              });
            } else if (error.response.data === "spaceFillingMeanSquareError") {
              message.error({
                content: commonT("error.general.meanSquareError"),
              });
            } else {
              message.error({
                content: commonT("error.general.unexpectedMsg"),
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
    if (Object.keys(responseColumnData).length > 0) {
      responseColumnKey.map((key) => {
        const columnList = responseColumnData[key].columnList;
        const estimatesList: { [key: string]: number } =
          responseColumnData[key].estimatesList;

        delete estimatesList["Intercept"];

        const estimates = Object.entries(estimatesList)
          .map(([, value]) => value)
          .sort((a, b) => a - b);

        const chartsForVariable: HighChartTemplate = buildChart(
          key,
          columnList.filter((el) => el !== "Intercept"),
          estimates,
          responseColumnData[key].y,
          responseColumnData[key].yPredito
        );

        updatedChartData[key] = chartsForVariable;
      });
      setChartData(updatedChartData);
    }
  }, [loadingPage]);

  function buildChart(
    key: string,
    estimateLabels: string[],
    estimates: number[],
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
    const countAccess = async () => {
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "spaceFilling",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

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
        title={commonT("spaceFilling.title")}
        tool={"spaceFilling"}
        enableRecalculate={true}
        enableGenerateChart={true}
        generateChart={() => setIsOpenGenerateChart(true)}
      />
      {loadingPage ? (
        <Spin />
      ) : (
        <>
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
                    <div style={{ marginBottom: 20 }}>
                      <ChartHub
                        key={key}
                        chartConfigurations={chartData[key]}
                        tool="spaceFilling"
                        showLimits
                      />
                    </div>

                    <div style={{ marginBottom: 20 }}>
                      <Equation
                        loading={false}
                        tooltipTitle={responseColumnData[key].equationTooltip}
                        equation={responseColumnData[key].equation}
                        width={undefined}
                      />
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
                        loading={loadingParamterEstimate}
                        dataSource={
                          responseColumnData[key].parameterEstimate
                            .parameterEstimatesDataSource
                        }
                        columns={
                          responseColumnData[key].parameterEstimate
                            .columnsToParameterEstimates
                        }
                        title={commonT("spaceFilling.parameterEstimates")}
                      />
                    </div>

                    <div style={{ marginBottom: "20px" }}>
                      <Table
                        loading={loadingAnalisysOfVariance}
                        dataSource={
                          responseColumnData[key].analysisOfVarianceDataSource
                        }
                        columns={
                          responseColumnData[key].columnsToAnalysisOfVariance
                        }
                        title={commonT("spaceFilling.analysisOfVariance")}
                      />
                    </div>

                    <div style={{ marginBottom: 20 }}>
                      <Table
                        loading={loadingSummaryOfFit}
                        dataSource={
                          responseColumnData[key].summaryOfFit
                            .summaryOfFitDataSource
                        }
                        columns={
                          responseColumnData[key].summaryOfFit
                            .columnsToSummaryOfFit
                        }
                        title={commonT("spaceFilling.summaryOfFit")}
                      />
                    </div>
                  </Col>
                </Row>
              </>
            ))}

            {isRecalculate ? (
              <Row>
                <Col span={24}>
                  {!loadingProfiler && (
                    <Suspense fallback={<Spin />}>
                      <Profiler
                        optimizationData={profilerData}
                        profilerData={profilerData}
                        loading={false}
                        type={"spaceFilling"}
                        defaultColNumber={colNumber}
                        isStatic={getStaticProfilerDataConfig(colNumber, rows)}
                      />
                    </Suspense>
                  )}
                </Col>
              </Row>
            ) : (
              <></>
            )}

            {isChartBuild ? (
              <>
                <ResponseContentHeader
                  title={commonT("spaceFilling.chartBuilder.chartTitle")}
                  index={10}
                  onClick={() => onToggleContent(10)}
                />
                <Row hidden={contentVisibility[10]}>
                  <Col span={12}>
                    <ChartBuilder
                      chartType={constructorChart.chartType as string}
                      xVariable1={constructorChart.xVariables}
                      yVariable={constructorChart.yVariables}
                      chartData={chartBuilderData}
                    />
                  </Col>
                </Row>
              </>
            ) : (
              <></>
            )}
          </div>
        </>
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
import { ChartType3d, DataToChart2D } from "./interface";
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
  chartData: Record<string, number[]>;
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
    const dataToChart: number[] = [];
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
    const dataToChart: DataToChart2D[] = [];
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

import { ColumnsType } from "antd/es/table";
import { formatNumber } from "utils/formatting";
import {
  AnovaGridRow,
  ParameterEstimateGridRow,
  SpaceFillingAnovaTable,
  SpaceFillingParameterEstimateTranslate,
  SpaceFillingParameterEstimatesObject,
  SpaceFillingSummaryOfFit,
  SummaryOfFitGridRow,
} from "../interface";
import { P_VALUE_LIMIT, P_VALUE_NOT_REPLICATED_LIMIT } from "utils/constant";

export function createAnovaTable(analysisOfVariance: SpaceFillingAnovaTable) {
  const columnsToAnalysisOfVariance: ColumnsType<any> = [
    {
      title: analysisOfVariance.translate.term,
      dataIndex: "source",
      key: "source",
    },
    {
      title: analysisOfVariance.translate.degreesOfFreedom,
      dataIndex: "degreesOfFreedom",
      key: "degreesOfFreedom",
    },
    {
      title: analysisOfVariance.translate.sSquare,
      dataIndex: "sSquare",
      key: "sSquare",
    },
    {
      title: analysisOfVariance.translate.mSquare,
      dataIndex: "mSquare",
      key: "mSquare",
    },
    {
      title: analysisOfVariance.translate.fRatio,
      dataIndex: "fRatio",
      key: "fRatio",
    },
    {
      title: analysisOfVariance.translate.probF,
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

  const analysisOfVarianceDataSource: AnovaGridRow[] = [
    {
      key: "model",
      source: analysisOfVariance.translate.model,
      degreesOfFreedom: formatNumber(
        analysisOfVariance?.grausLiberdade?.modelo ?? 0
      ),
      sSquare: formatNumber(analysisOfVariance?.sQuadrados?.modelo ?? 0),
      mSquare: formatNumber(analysisOfVariance?.mQuadrados?.modelo ?? 0),
      fRatio: formatNumber(analysisOfVariance?.fRatio ?? 0),
      probF: analysisOfVariance?.probF ?? 0,
    },
    {
      key: "errror",
      source: analysisOfVariance.translate.error,
      degreesOfFreedom: formatNumber(
        analysisOfVariance?.grausLiberdade?.erro ?? 0
      ),
      sSquare: formatNumber(analysisOfVariance?.sQuadrados?.erro ?? 0),
      mSquare: formatNumber(analysisOfVariance?.mQuadrados?.erro ?? 0),
      fRatio: "",
      probF: "",
    },
    {
      key: "total",
      source: analysisOfVariance.translate.total,
      degreesOfFreedom: formatNumber(
        analysisOfVariance?.grausLiberdade?.total ?? 0
      ),
      sSquare: formatNumber(analysisOfVariance?.sQuadrados?.total ?? 0),
      mSquare: "",
      fRatio: "",
      probF: "",
    },
  ];
  return { analysisOfVarianceDataSource, columnsToAnalysisOfVariance };
}

export function createSummaryOfFitTable(
  summaryOfFit: SpaceFillingSummaryOfFit
) {
  const columnsToSummaryOfFit: ColumnsType = [
    {
      title: summaryOfFit.translate.term,
      dataIndex: "source",
      key: "source",
    },
    {
      title: summaryOfFit.translate.rSquare,
      dataIndex: "rSquare",
      key: "rSquare",
    },
    {
      title: summaryOfFit.translate.rSquareAdjust,
      dataIndex: "rSquareAdjust",
      key: "rSquareAdjust",
    },
    {
      title: summaryOfFit.translate.rmse,
      dataIndex: "rmse",
      key: "rmse",
    },
    {
      title: summaryOfFit.translate.mean,
      dataIndex: "mean",
      key: "mean",
    },
    {
      title: summaryOfFit.translate.observations,
      dataIndex: "observations",
      key: "observations",
    },
  ];

  const summaryOfFitDataSource: SummaryOfFitGridRow[] = [
    {
      key: "1",
      source: summaryOfFit.translate.value,
      rSquare: formatNumber(summaryOfFit?.rQuadrado ?? 0),
      rSquareAdjust: formatNumber(summaryOfFit?.rQuadradoAjustado ?? 0),
      rmse: formatNumber(summaryOfFit?.rmse ?? 0),
      mean: formatNumber(summaryOfFit?.media ?? 0),
      observations: summaryOfFit?.observacoes ?? 0,
    },
  ];
  return { summaryOfFitDataSource, columnsToSummaryOfFit };
}

export function createParameterEstimateTable(
  data: { [key: string]: SpaceFillingParameterEstimatesObject },
  translate: SpaceFillingParameterEstimateTranslate
) {
  const terms = Object.keys(data).map((term) => term.replace(/\//g, "*"));

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
      render: (item) => (
        <div>
          {typeof item !== "number"
            ? item
            : item > P_VALUE_NOT_REPLICATED_LIMIT
            ? formatNumber(item)
            : 0.0001}
        </div>
      ),
      onCell: (item: any) => {
        return {
          ["style"]: {
            color: item.pValue > P_VALUE_NOT_REPLICATED_LIMIT ? "" : "red",
          },
        };
      },
    },
  ];

  const parameterEstimatesDataSource: ParameterEstimateGridRow[] = [];

  Object.keys(data).forEach((key, index) => {
    const row: ParameterEstimateGridRow = {
      key: `row${index + 1}`,
      source: key.replace("/", "*"),
      estimates: formatNumber(data[key].estimates),
      stdError: formatNumber(data[key].stdError),
      tRatio: formatNumber(data[key].tRatio),
      pValue:
        data[key].pValue > P_VALUE_NOT_REPLICATED_LIMIT
          ? formatNumber(data[key].pValue)
          : P_VALUE_NOT_REPLICATED_LIMIT,
    };

    parameterEstimatesDataSource.push(row);
  });

  return { parameterEstimatesDataSource, columnsToParameterEstimates };
}

import { Button, Col, Modal, Row, Select } from "antd";
import Transfer from "shared/transfer";
import { useTranslation } from "next-i18next";
import React, { useState } from "react";

const GenerateChart = (props: {
  showOpenModal: boolean;
  onModalClose: () => void;
  onSaveClick: (
    xVariables: string[] | string,
    yVariables: string,
    chartTypeValue: string
  ) => void;
  variables: string[];
}) => {
  const { showOpenModal, onModalClose, variables, onSaveClick } = props;

  const { t: commonT } = useTranslation("common");
  const { t: layoutT } = useTranslation("layout");

  const newVariables = [];
  for (const iterator of variables) {
    if (!iterator.includes("-") && iterator !== "Intercept") {
      newVariables.push({
        key: iterator,
        title: iterator,
      });
    }
  }

  const chartTypes = [
    {
      label: "2D",
      value: "2d",
    },
    {
      label: "3D",
      value: "3d",
    },
  ];

  const [chartType, setChartType] = useState("2d");

  const [geerateChartTargetKeys, setGenerateChartTargetKeys] = useState<
    string[]
  >([]);
  const [generateChartSelectedKeys, setGenerateChartSelectedKeys] = useState<
    string[]
  >([]);
  const [generateChartTargetKeysResponse, setGenerateChartTargetKeysResponse] =
    useState<string[]>([]);
  const [
    generateChartSelectedKeysResponse,
    setGenerateChartSelectedKeysResponse,
  ] = useState<string[]>([]);

  const generateChartOnChange = (nextTargetKeys: string[]) => {
    setGenerateChartTargetKeys(nextTargetKeys);
  };

  const generateChartOnSelectChange = (
    sourceSelectedKeys: string[],
    targetSelectedKeys: string[]
  ) => {
    setGenerateChartSelectedKeys([
      ...sourceSelectedKeys,
      ...targetSelectedKeys,
    ]);
  };

  const generateChartOnChangeResponse = (nextTargetKeys: string[]) => {
    setGenerateChartTargetKeysResponse(nextTargetKeys);
  };

  const generateChartOnSelectChangeResponse = (
    sourceSelectedKeys: string[],
    targetSelectedKeys: string[]
  ) => {
    setGenerateChartSelectedKeysResponse([
      ...sourceSelectedKeys,
      ...targetSelectedKeys,
    ]);
  };

  const handleCancel = () => {
    onModalClose();
  };

  const handleGenerateClick = () => {
    if (chartType === "2d") {
      onSaveClick(
        geerateChartTargetKeys[0],
        generateChartTargetKeysResponse[0],
        chartType
      );
    } else {
      onSaveClick(
        geerateChartTargetKeys,
        generateChartTargetKeysResponse[0],
        chartType
      );
    }
  };

  return (
    <>
      <Modal
        title={commonT("spaceFilling.generateChart.title")}
        open={showOpenModal}
        onCancel={handleCancel}
        width={700}
        footer={[
          <Button key="cancel" onClick={handleCancel}>
            {layoutT("buttons.cancel")}
          </Button>,
          <Button
            type="primary"
            key="generate"
            onClick={handleGenerateClick}
            disabled={
              chartType === "2d"
                ? geerateChartTargetKeys.length !== 1 ||
                  generateChartTargetKeysResponse.length !== 1
                : chartType === "3d"
                ? geerateChartTargetKeys.length < 2 ||
                  generateChartTargetKeysResponse.length !== 1
                : false
            }
          >
            {layoutT("buttons.generate")}
          </Button>,
        ]}
      >
        <Row style={{ fontWeight: "bolder" }}>
          <Col span={12}>{commonT("spaceFilling.generateChart.type")}</Col>
        </Row>
        <Row>
          <Col span={12}>
            <Select
              value={chartType}
              onChange={setChartType}
              options={chartTypes}
              style={{ width: 305 }}
            />
          </Col>
        </Row>

        <>
          <Transfer
            dataSource={newVariables}
            titles={[
              commonT("spaceFilling.xSelect"),
              commonT("spaceFilling.xSelected"),
            ]}
            targetKeys={geerateChartTargetKeys}
            selectedKeys={generateChartSelectedKeys}
            onChange={generateChartOnChange}
            onSelectChange={generateChartOnSelectChange}
          />
          <Transfer
            dataSource={newVariables}
            titles={
              chartType === "3d"
                ? [
                    commonT("spaceFilling.variableResponse"),
                    commonT("spaceFilling.variableSelectedResponse"),
                  ]
                : [
                    commonT("spaceFilling.xSelect"),
                    commonT("spaceFilling.xSelected"),
                  ]
            }
            targetKeys={generateChartTargetKeysResponse}
            selectedKeys={generateChartSelectedKeysResponse}
            onChange={generateChartOnChangeResponse}
            onSelectChange={generateChartOnSelectChangeResponse}
          />
        </>
      </Modal>
    </>
  );
};

export default GenerateChart;

a geração do space filling
import React, { useState, useEffect } from "react";
import axios from "axios";
import { useTranslation } from "next-i18next";
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
import { AiOutlineDelete } from "react-icons/ai";
import { MdAdd } from "react-icons/md";
import { InboxOutlined } from "@ant-design/icons";
import { ListLocale } from "antd/es/list";
import * as Styled from "./styled";
import { saveAsExcel } from "utils/export";
import Modal from "shared/modal";
import { generateRandomNumbers } from "utils/core";

interface generateSpaceFillingForm {
  type: string;
  factors: number;
  rounds: number;
  responseColumnNumber: number;
}

const fetcher = axios.create({
  baseURL: "/api",
});
const GenerateSpaceFillingModal = (props: {
  showOpenModal: boolean;
  onModalClose: () => void;
}) => {
  const { showOpenModal, onModalClose } = props;

  const { t: commonT } = useTranslation("common");
  const { t: layoutT } = useTranslation("layout");
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  const [isModalOpen, setIsModalOpen] = useState(showOpenModal);
  const [loading, setLoading] = useState<boolean | null>(false);

  const [isRandomColumnResponse, setIsRandomColumnResponse] = useState(false);
  const [initialValue, setInitialValue] = useState<number | undefined>(
    undefined
  );
  const [finalValue, setFinalValue] = useState<number | undefined>(undefined);
  const [intervalValue, setIntervalValue] = useState<number | undefined>(
    undefined
  );
  const [minLevelValue, setMinLevelValue] = useState<number | undefined>(-1);
  const [maxLevelValue, setMaxLevelValue] = useState<number | undefined>(1);
  const [columnName, setColumnName] = useState<string>("");
  const [roundsValue, setRoundsValue] = useState<number | undefined>(undefined);

  const [listData, setListData] = useState<ListDataSP[]>([]);
  const [selectFactor, setSelectFactor] = useState<any>(null);
  const [factorOptions, setFactorOptions] = useState<any[]>([]);

  const [factorNumber, setFactorNumber] = useState(0);

  const [form] = Form.useForm<generateSpaceFillingForm>();

  interface ListDataSP {
    factor: string;
    maxLevelValue: number;
    minLevelValue: number;
    columnName: string;
  }

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

  const spaceFillingTypes = [
    {
      label: "Latin Hypercube",
      value: "lhs",
    },
    {
      label: commonT("designSpaceFillingTable.lhsMin"),
      value: "lhsMin",
    },
    {
      label: commonT("designSpaceFillingTable.lhsMax"),
      value: "lhsMax",
    },
    {
      label: commonT("designSpaceFillingTable.sp"),
      value: "sp",
    },
  ];

  useEffect(() => {
    setIsModalOpen(showOpenModal);
    setFactorOptions([]);
    setMaxLevelValue(1);
    setMinLevelValue(-1);
  }, [showOpenModal]);

  const handleCancel = () => {
    setIsModalOpen(false);
    setListData([]);
    onModalClose();
  };

  useEffect(() => {
    const calculateRounds = () => {
      const roundsCalculate = factorNumber * 10;
      setRoundsValue(roundsCalculate);
      form.setFieldValue("rounds", roundsCalculate);
    };

    calculateRounds();
  }, [factorNumber]);

  const onFinish = async (formValue: generateSpaceFillingForm) => {
    setLoading(true);
    const responseValues = handleResponseValues();

    const generateSpaceFilling = {
      responseValues: responseValues,
      factors: formValue.factors,
      rounds: formValue.rounds,
      responseColumnNumber: formValue.responseColumnNumber,
      listData: listData,
      type: formValue.type,
    };

    try {
      const { data } = await fetcher.post(
        "spaceFilling/generateDataTable",
        generateSpaceFilling
      );

      setLoading(false);
      saveAsExcel(data, "space-filling");
    } catch (error: any) {
      if (error.response.data === "spaceFillingGenerateTableError") {
        message.error({
          content: commonT("error.spaceFilling.generateTableError"),
        });
      } else {
        message.error({
          content: commonT("error.general.unexpectedMsg"),
        });
      }
    }
    setLoading(false);
    onModalClose();
  };

  const handleIsRandomColumnResponse = (e: CheckboxChangeEvent) => {
    setIsRandomColumnResponse(e.target.checked);
  };

  const handleResponseValues = () => {
    if (!intervalValue || !finalValue || !initialValue) return [];
    const interval = intervalValue <= 0 ? 1 : intervalValue;

    let randomNumbers: number[] = [];
    if (!roundsValue) return randomNumbers;

    if (isRandomColumnResponse) {
      randomNumbers = generateRandomNumbers(
        initialValue,
        finalValue,
        interval,
        roundsValue
      );
    }

    return randomNumbers;
  };

  useEffect(() => {
    const letterArray = [];
    for (const item of Array.from(alphabet).slice(0, factorNumber)) {
      letterArray.push({
        value: item,
        label: item,
      });
    }
    setFactorOptions(letterArray);
  }, [factorNumber]);

  const onRemoveList = (item: ListDataSP) => {
    const newListData = listData.filter((el) => el !== item);
    setListData(newListData);

    setFactorOptions((prevOptions) => [
      ...prevOptions,
      { label: item.factor, value: item.factor },
    ]);
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

    const newItem: ListDataSP = {
      factor: selectedFactor.value,
      minLevelValue: minLevelValue as number,
      maxLevelValue: maxLevelValue as number,
      columnName: columnName ? columnName : selectedFactor.value,
    };

    setFactorOptions((prevOptions) =>
      prevOptions.filter((el) => el.value !== selectedFactor.value)
    );

    setListData([...listData, newItem]);
    setSelectFactor({ value: null, label: null });
    form.resetFields(["selectFactor"]);
    form.resetFields(["columnName"]);
    setColumnName("");
  };

  useEffect(() => {
    // empty use Effect
  }, [factorOptions, listData]);

  const handleSelectFactor = (value: any) => {
    setSelectFactor(value);
  };

  useEffect(() => {
    // empty useEffect
  }, [minLevelValue, maxLevelValue, columnName]);

  return (
    <>
      <Modal
        title={commonT("designSpaceFillingTable.title")}
        open={isModalOpen}
        onCancel={handleCancel}
        width={"650px"}
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
            <Col span={14}>
              <Form.Item
                label={commonT("designSpaceFillingTable.type")}
                name="type"
                rules={[{ required: true, message: "" }]}
              >
                <Select options={spaceFillingTypes} />
              </Form.Item>
            </Col>
            <Col offset={1} span={9}>
              <Form.Item
                label={commonT("designSpaceFillingTable.factors")}
                name="factors"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber
                  onChange={(value) => setFactorNumber(Number(value))}
                  style={{ width: "100%" }}
                  min={2}
                  max={25}
                />
              </Form.Item>
            </Col>
          </Row>
          <Row>
            <Col span={8}>
              <Form.Item
                label={commonT("designSpaceFillingTable.rounds")}
                name="rounds"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber
                  onChange={(value) => setRoundsValue(Number(value))}
                  style={{ width: "100%" }}
                  min={6}
                />
              </Form.Item>
            </Col>
            <Col offset={5} span={11}>
              <Form.Item
                label={commonT("designSpaceFillingTable.responseColumnNumber")}
                name="responseColumnNumber"
                rules={[{ required: true, message: "" }]}
              >
                <InputNumber style={{ width: "100%" }} min={1} max={10} />
              </Form.Item>
            </Col>
          </Row>

          <Row>
            <Col span={12}>
              <Form.Item
                name="useValuesColumnY"
                valuePropName="checked"
                label={commonT("designSpaceFillingTable.useValuesColumnY")}
              >
                <Checkbox onChange={handleIsRandomColumnResponse} />
              </Form.Item>
            </Col>
          </Row>

          {isRandomColumnResponse ? (
            <>
              <Row>
                <Col span={7}>
                  <Form.Item
                    name="minValue"
                    label={commonT("designSpaceFillingTable.minValue")}
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
                <Col offset={1} span={7}>
                  <Form.Item
                    name="maxValue"
                    label={commonT("designSpaceFillingTable.maxValue")}
                    rules={[{ required: isRandomColumnResponse, message: "" }]}
                  >
                    <InputNumber
                      defaultValue={
                        isRandomColumnResponse ? finalValue : undefined
                      }
                      onChange={(value) => setFinalValue(Number(value))}
                      style={{ width: "100%" }}
                      min={initialValue}
                    />
                  </Form.Item>
                </Col>
                <Col offset={1} span={8}>
                  <Form.Item
                    name="intervalValue"
                    label={commonT("designSpaceFillingTable.intervalValue")}
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

          <Row>
            <Col span={7}>
              <Form.Item
                label={commonT("designSpaceFillingTable.minLevel")}
                name="minLevel"
              >
                <InputNumber
                  min={0.01}
                  value={minLevelValue}
                  onChange={(value) => setMinLevelValue(Number(value))}
                  style={{ width: "100%" }}
                />
              </Form.Item>
            </Col>
            <Col offset={1} span={7}>
              <Form.Item
                label={commonT("designSpaceFillingTable.maxLevel")}
                name="maxLevel"
              >
                <InputNumber
                  value={maxLevelValue}
                  onChange={(value) => setMaxLevelValue(Number(value))}
                  style={{ width: "100%" }}
                />
              </Form.Item>
            </Col>
            <Col offset={1} span={8}>
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
          </Row>
          <Row>
            <Col span={15}>
              <Form.Item
                label={commonT("designSpaceFillingTable.columnName")}
                name="columnName"
              >
                <Input
                  value={columnName}
                  onChange={(e) => setColumnName(e.target.value)}
                  style={{ width: "100%" }}
                />
              </Form.Item>
            </Col>
            <Col offset={1} span={8}>
              <Button
                type="primary"
                icon={
                  <MdAdd style={{ color: "white", transform: "scale(1.2)" }} />
                }
                onClick={onAddClick}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  width: "100%",
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
                            <Styled.AddButton
                              shape="circle"
                              danger
                              type="primary"
                              onClick={() => onRemoveList(item)}
                              icon={<AiOutlineDelete />}
                            />
                          </>
                        }
                        style={{ display: "flex", alignItems: "center" }}
                        title={`${item.factor} (${item.minLevelValue} - ${item.maxLevelValue}) - ${item.columnName} `}
                      />
                    </List.Item>
                  )}
                />
              </Styled.ListContainer>
            </Col>
          </Row>
        </Form>
      </Modal>
    </>
  );
};

export default GenerateSpaceFillingModal;

meu backend reaproveitavel

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.doe_full.doe_generate_dataframe import rename_column_name
from utils.space_filling.generate_dataframe_space_filling import (
    generate_lhs,
    generate_lhs_max,
    generate_lhs_min,
    generate_sphere_packing,
)
from utils.space_filling.space_filling_calculate import calculate_natural_dataFrame


class ListData(BaseModel):
    factor: str
    maxLevelValue: float
    minLevelValue: float
    columnName: str


class GenerateDataFrame(BaseModel):
    rounds: int
    factors: int
    type: str
    responseColumnNumber: int
    responseValues: List[float]
    listData: List[ListData]


generate_space_filling_router = APIRouter()


@generate_space_filling_router.post(
    "/generate_dataframe",
    tags=["Space Filling"],
    description="""Calculo Space Filling,
                rounds: numero de rodadas;
                type: lhs, lhsMin, lhsMax;
                responseColumnNumber: numero de colunas de respostas;
                responseValues: valores das colunas de respostas, se nao houver envie [];
                listData: lista de valores para cada subgrupo;
              """,
    response_model=object,
)
def generate_dataFrame(body: GenerateDataFrame):
    rounds = body.rounds
    factors = body.factors
    type = body.type
    responseValues = body.responseValues
    responseColumnNumber = body.responseColumnNumber
    listData = body.listData

    try:
        if type == "lhs":
            df = generate_lhs(rounds, factors, responseValues, responseColumnNumber)
        elif type == "lhsMin":
            df = generate_lhs_min(
                rounds, factors, 5, responseValues, responseColumnNumber
            )
        elif type == "lhsMax":
            df = generate_lhs_max(
                rounds, factors, 5, responseValues, responseColumnNumber
            )
        elif type == "sp":
            df = generate_sphere_packing(
                factors, rounds, responseColumnNumber, responseValues
            )
        elif type == "fastFilling":
            df = {}

        # Calcula numeros naturais
        if len(listData) > 0:
            df = calculate_natural_dataFrame(df, listData)

        # Renomeia colunas
        df = rename_column_name(df, listData)

        return df.to_json()

    except Exception as error:
        print(error)
        raise HTTPException(status_code=500, detail=f"spaceFillingGenerateTableError")


from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.common.common import (
    calculate_sq_error_total,
    calculate_sq_model_total,
    calculate_sq_models,
    new_split_dataframes_by_response_column_add_response_doe,
    remove_punctuation,
)
from utils.multiple_regression.multiple_regression_calculate import (
    add_intercept_column,
    calculate_anova_table,
    calculate_degress_freedom,
    calculate_mean_response_column,
    calculate_pure_error_total,
    calculate_response_columns_values,
    calculate_rounds,
    calculate_sq_error,
    calculate_summary_of_fit,
    calculate_y_predicteds,
    concat_data_frame,
    separate_column,
)
from utils.space_filling.space_filling_calculate import (
    calculate_betas,
    calculate_matrix_inverse,
    calculate_means,
    calculate_multiply_matrices_column_value,
    calculate_multiply_matrices_response_column_value,
    calculate_parameter_estimates,
    calculate_pure_error_list,
    calculate_space_filling_lack_of_fit,
    calculate_space_filling_pure_error,
    calculate_square_column,
    calculate_square_recalculate_column,
    calculate_standard_error_matrix,
    calculate_transpose_x,
)

space_filling_router = APIRouter()


class SpaceFillingCalculate(BaseModel):
    data: dict[str, List[float]]
    responseColumn: List[str]
    recalculate: bool
    interactionColumn: List[str]


@space_filling_router.post(
    "/calculate",
    tags=["Space Filling"],
    description="""Calculo Space Filling,
                responseColumn: nome das colunas de respostas;
                data: dataframe;
                recalculate: é modelo reduzido;
                interactionColumn: nome das colunas de interação, se nao houver envie [];
              """,
    response_model=object,
)
def calculate(body: SpaceFillingCalculate):

    response_columns = body.responseColumn
    recalculate = body.recalculate
    interactionColumn = body.interactionColumn

    df = pd.DataFrame(data=body.data)
    df = remove_punctuation(df)

    try:
        try:
            # Separa dataFrame em varios conforme o numero de resposta
            dataFrames = new_split_dataframes_by_response_column_add_response_doe(
                df, response_columns
            )
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"spaceFillingSplitDfError")

        space_filling_response = {}

        for i, df in enumerate(dataFrames):
            response_column = df.columns[-1]
            # calcula médias
            means = calculate_means(df, response_column)

            # Cria data frame da coluna de resposta
            data_frames = separate_column(df, response_column)

            # Atualiza data frames com os valores
            response_data_frame = data_frames[1]
            df = data_frames[0]

            # Calcula médias da colunas de resposta
            respose_mean_columns = calculate_mean_response_column(response_data_frame)

            # Retorna os valores da coluna de resposta
            response_columns_values = calculate_response_columns_values(
                response_data_frame
            )

            if recalculate:
                if len(interactionColumn) > 0:
                    df = calculate_square_recalculate_column(df, interactionColumn)
                else:
                    df = calculate_square_recalculate_column(df, [])
            else:
                df = calculate_square_column(df, response_column)

            # Cria coluna intercept
            df = add_intercept_column(df)

            # Calcula transposta de X
            transpose_df_columns_value = calculate_transpose_x(df)

            # Calcula matriz multiplicadora
            multiply_matrices_column_value = calculate_multiply_matrices_column_value(
                transpose_df_columns_value, df
            )

            # Calcula inversa da transposta de coluna de valor * matriz de valor
            inverse_column_value = calculate_matrix_inverse(
                multiply_matrices_column_value
            )

            # Calcula matrix resultante de X transposta * Y
            multiply_matrices_response_column_value = (
                calculate_multiply_matrices_response_column_value(
                    transpose_df_columns_value, response_data_frame
                )
            )

            # Calcula betas
            betas = calculate_betas(
                inverse_column_value, multiply_matrices_response_column_value
            )

            # Calcula Y preditos
            y_predicteds = calculate_y_predicteds(betas, df)

            # Gera Y Predito e aplica ordenção
            new_df = response_data_frame
            new_df[f"{response_column} Predicted"] = y_predicteds
            new_df = new_df.sort_values(by=f"{response_column}")

            y_predicteds_ordered = new_df[f"{response_column} Predicted"].to_list()
            y = new_df[response_column].to_list()

            # Calcula Sq erro
            sq_error = calculate_sq_error(y_predicteds, response_data_frame)

            # Calcula Sq erro total
            sq_erro_total = calculate_sq_error_total(sq_error)

            # Calcula Sq Modelo
            sq_models = calculate_sq_models(y_predicteds, respose_mean_columns)

            # Calcula Sq Modelo total
            sq_model_total = calculate_sq_model_total(sq_models)

            # Concatena dataFrame
            df_concat = concat_data_frame(df, response_data_frame)

            # Calcula erro puro
            pure_error_regression = calculate_pure_error_list(df_concat)

            # Calcula graus de liberdade do erro puro
            degress_freedom_pure_error = calculate_space_filling_pure_error(df_concat)

            # Calcula erro puro total
            pure_error_total = calculate_pure_error_total(pure_error_regression)

            # Calcula graus de liberdade
            degress_freedom = calculate_degress_freedom(df)

            # Calcula numero de rodadas
            rounds = calculate_rounds(df)

            try:
                # Calcula anova table
                anova_table = calculate_anova_table(
                    degress_freedom, rounds - 1, sq_model_total, sq_erro_total
                )
            except Exception as anovaError:
                raise HTTPException(status_code=500, detail=f"spaceFillingAnovaError")

            try:
                # Calcula summary of fit
                summary_of_fit = calculate_summary_of_fit(
                    sq_model_total,
                    sq_erro_total,
                    degress_freedom,
                    anova_table["mQuadrados"]["erro"],
                    response_columns_values,
                    respose_mean_columns,
                )
            except Exception as summaryError:
                raise HTTPException(status_code=500, detail=f"spaceFillingSummaryError")

            try:
                # Calcula lack of fit
                lack_of_fit = calculate_space_filling_lack_of_fit(
                    degress_freedom_pure_error,
                    anova_table["grausLiberdade"]["erro"],
                    sq_erro_total,
                    pure_error_total,
                )
            except Exception as lackOfFitError:
                raise HTTPException(
                    status_code=500, detail=f"spaceFillingLackOfFitError"
                )

            # Calcula erro padrão
            standard_error_matrix = calculate_standard_error_matrix(
                inverse_column_value, anova_table["mQuadrados"]["erro"]
            )

            # calcula parameter estimates
            parameterEstimates = calculate_parameter_estimates(
                df,
                betas,
                standard_error_matrix.tolist(),
                anova_table["grausLiberdade"]["total"],
            )

            space_filling_response[response_column] = {
                "betas": betas.tolist(),
                "anovaTable": anova_table,
                "summarOfFit": summary_of_fit,
                "lackOfFit": lack_of_fit,
                "parameterEstimates": parameterEstimates,
                "isRecalculate": recalculate,
                "yPredicteds": y_predicteds,
                "mean": means,
                "yPredictedsOdered": y_predicteds_ordered,
                "y": y,
            }

        return {"spaceFilling": space_filling_response}

    except HTTPException as http_error:
        raise http_error
    except Exception as error:
        print("error", error)
        raise HTTPException(status_code=500, detail=f"spaceFillingError")
