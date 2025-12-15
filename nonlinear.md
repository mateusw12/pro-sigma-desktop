Vamos fazer a analise de regressão não linear, onde posso colocar 1 coluna X e 1 coluna Y
Posso fazer diversas analises com diversas curvas.

meu front end antigo

import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "next-i18next";
import { useAuth } from "hooks/useAuth";
import { useRouter } from "next/router";
import axios from "axios";
import ContentHeader from "shared/contentHeader";
import { Spin } from "shared/spin"
import {
  NonlinearRegressionCoeff,
  NonlinearRegressionExportData,
  NonlinearRegressionResult,
} from "./interface";
import { getItem } from "utils/database";
import { DataToCompile } from "components/insertData/inteface";
import { Col, Row, message } from "antd";
import { createParamsTable, createSummaryTable } from "./table";
import Table from "shared/table";
import { HighChartTemplate } from "shared/widget/chartHub/interface";
import dynamic from "next/dynamic";
import Equation from "shared/equation";
import { getCurveName } from "./utils";

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const fetcher = axios.create({
  baseURL: "/api",
});

export const NonlinearRegression: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });

  const { user } = useAuth();
  const router = useRouter();

  const summaryTable = useRef<any>({});
  const paramsTable = useRef<any>({});

  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingParams, setLoadingParams] = useState(true);

  const [chartData, setChartData] = useState<HighChartTemplate>({});

  const [curve, setCurve] = useState<string>("");
  const [equation, setEquation] = useState<string>("");
  const [responseColumn, setResponseColumn] = useState<string>("");

  const [metrics, setMetrics] = useState<
    Record<string, NonlinearRegressionCoeff>
  >({});

  const [predictions, setPredictions] =
    useState<Record<string, { x: number; y: number }[]>>();

  const summaryTranslate = {
    curve: commonT("nonlinear.curve"),
    value: commonT("nonlinear.value"),
  };

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
          )) as NonlinearRegressionExportData;

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

            const nonlinearCalculate = {
              inputData: dataToSend.obj,
              responseColumn: item.responseColumn,
            };

            try {
              const { data } = await fetcher.post<NonlinearRegressionResult>(
                "nonlinear/calculate",
                nonlinearCalculate
              );
              setLoadingPage(false);

              const result = data.result;

              setResponseColumn(item.responseColumn);

              summaryTable.current = createSummaryTable(
                result.metrics,
                summaryTranslate,
                commonT
              );

              setLoadingSummary(false);

              const defaultCurve = Object.keys(result.predictions)[0];

              setCurve(defaultCurve);

              paramsTable.current = createParamsTable(
                result.metrics[defaultCurve].coef,
                summaryTranslate,
                commonT
              );

              setLoadingParams(false);

              setPredictions(result.predictions);
              setMetrics(result.metrics);

              setEquation(result.metrics[defaultCurve].equation);

              const chartsForVariable = buildChartBase(result.original);
              setChartData(chartsForVariable);
            } catch (error) {
              console.error(error);
              message.error({
                content: commonT("error.general.unexpectedMsg"),
              });
            }
          }
        }
      }
    };

    getData().catch(console.error);
  }, [router.query, router.isReady]);

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "nonlinear",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  useEffect(() => {
    if (curve && predictions && chartData["predicted"]) {
      setChartData((prev) => {
        const updatedChart = { ...prev };
        updatedChart["predicted"] = {
          ...updatedChart["predicted"],
          seriesData: [
            {
              ...updatedChart["predicted"].seriesData[0],
              name: responseColumn,
            },
            {
              ...updatedChart["predicted"].seriesData[1],
              data: predictions[curve] || [],
              name: getCurveName(curve, commonT),
            },
          ],
        };
        return updatedChart;
      });

      paramsTable.current = createParamsTable(
        metrics[curve].coef,
        summaryTranslate,
        commonT
      );

      setEquation(metrics[curve].equation);
    }
  }, [curve, predictions]);

  function buildChartBase(originalDataSeries: any[]) {
    const chartsForVariable: HighChartTemplate = {};

    chartsForVariable["predicted"] = {
      seriesData: [
        {
          name: commonT("nonlinear.data"),
          type: "scatter",
          showLegend: true,
          data: originalDataSeries ? originalDataSeries : [],
        },
        {
          name: getCurveName(curve, commonT),
          type: "scatter",
          showLegend: true,
          data: [],
        },
      ],
      options: {
        title: commonT("nonlinear.chart.title"),
        xAxisTitle: commonT("nonlinear.chart.xAxis"),
        yAxisTitle: responseColumn,
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
        useVerticalLimits: false,
      },
      type: "scatter",
      displayName: commonT("descriptiveStatistics.histogramTitle"),
    };

    return chartsForVariable;
  }

  return (
    <>
      <ContentHeader title={commonT("nonlinear.title")} tool={"nonlinear"} />
      {loadingPage ? (
        <Spin />
      ) : (
        <div id="content">
          <Row>
            <Col span={12}>
              {chartData && Object.keys(chartData).length > 0 && curve ? (
                <>
                  <ChartHub
                    chartConfigurations={chartData}
                    tool={"doe"}
                    showLimits
                  />
                </>
              ) : (
                <>
                  <Spin />
                </>
              )}
            </Col>
            <Col offset={1} span={11}>
              <div style={{ marginBottom: "20px" }}>
                <Table
                  dataSource={summaryTable.current.dataSource}
                  columns={summaryTable.current.columns}
                  loading={loadingSummary}
                  title={commonT("nonlinear.summaryTable")}
                  type={"radio"}
                  rowSelection={{
                    type: "radio",
                    onChange: (
                      _selectedRowKeys: React.Key[],
                      selectedRows: any[]
                    ) => {
                      setCurve(selectedRows[0].key);
                    },
                    getCheckboxProps: (record: any) => ({
                      name: record.distribution,
                    }),
                  }}
                />
              </div>
            </Col>
          </Row>
          <Row>
            <Col span={12}>
              <Equation
                tooltipTitle={equation}
                equation={equation}
                width={0}
                loading={false}
              />
            </Col>
            <Col offset={1} span={11}>
              <Table
                dataSource={paramsTable.current.dataSource}
                columns={paramsTable.current.columns}
                loading={loadingParams}
                title={commonT("nonlinear.paramsTable")}
              />
            </Col>
          </Row>
        </div>
      )}
    </>
  );
};

export const getCurveName = (type: string, commonT) => {
  switch (type) {
    case "polynomial_2":
      return commonT("nonlinear.models.polynomial_2");
    case "polynomial_3":
      return commonT("nonlinear.models.polynomial_3");
    case "gamma":
      return commonT("nonlinear.models.gamma");
    case "exponential_2p":
      return commonT("nonlinear.models.exponential_2p");
    case "exponential_3p":
      return commonT("nonlinear.models.exponential_3p");
    case "biexponential":
      return commonT("nonlinear.models.biexponential");
    case "mechanistic_growth":
      return commonT("nonlinear.models.mechanistic_growth");
    case "cell_growth":
      return commonT("nonlinear.models.cell_growth");
    case "weibull_growth":
      return commonT("nonlinear.models.weibull_growth");
    case "gompertz":
      return commonT("nonlinear.models.gompertz");
    case "logistic_2p":
      return commonT("nonlinear.models.logistic_2p");
    case "logistic_4p":
      return commonT("nonlinear.models.logistic_4p");
    case "probit_2p":
      return commonT("nonlinear.models.probit_2p");
    case "probit_4p":
      return commonT("nonlinear.models.probit_4p");
    case "gaussian_peak":
      return commonT("nonlinear.models.gaussian_peak");
    case "asymmetric_gaussian_peak":
      return commonT("nonlinear.models.asymmetric_gaussian_peak");
    case "lorentzian_peak":
      return commonT("nonlinear.models.lorentzian_peak");
    case "logarithmic":
      return commonT("nonlinear.models.logarithmic");
    case "inverse":
      return commonT("nonlinear.models.inverse");
    case "sqrt_model":
      return commonT("nonlinear.models.sqrt_model");
    case "power_intercept":
      return commonT("nonlinear.models.power_intercept");
    case "michaelis_menten":
      return commonT("nonlinear.models.michaelis_menten");
    case "tanh_model":
      return commonT("nonlinear.models.tanh_model");
    case "weibull":
      return commonT("nonlinear.models.weibull");
    default:
      return "Y"
  }
};


import { formatNumber } from "utils/formatting";
import { NonlinearRegressionCoeff } from "../interface";
import { ColumnsType } from "antd/lib/table";

export function createSummaryTable(
  metrics: Record<string, NonlinearRegressionCoeff>,
  translate: any,
  commonT: any
): { columns: ColumnsType<any>; dataSource: any[] } {
  const dataSource: any[] = Object.entries(metrics).map(([curve, coeff]) => ({
    key: curve,
    curve: commonT(`nonlinear.models.${curve}`),
    rSquared: formatNumber(coeff.rSquared),
    aic: formatNumber(coeff.aic),
    bic: formatNumber(coeff.bic),
  }));

  const columns: ColumnsType<any> = [
    {
      title: translate.curve,
      dataIndex: "curve",
      key: "curve",
    },
    {
      title: "R²",
      dataIndex: "rSquared",
      key: "rSquared",
    },
    {
      title: "AIC",
      dataIndex: "aic",
      key: "aic",
    },
    {
      title: "BIC",
      dataIndex: "bic",
      key: "bic",
    },
  ];

  return { columns, dataSource };
}

export function createParamsTable(
  coeff: Record<string, number>,
  translate: any,
  commonT
) {
  const dataSource: any[] = [];

  const columns: ColumnsType<any> = [
    {
      title: translate.curve,
      dataIndex: "curve",
      key: "curve",
    },
    {
      title: translate.value,
      dataIndex: "value",
      key: "value",
    },
  ];

  for (const key of Object.keys(coeff)) {
    dataSource.push({
      curve: getTranslatePolynomial(key.replaceAll("(Intercept)", "Intercept"), commonT),
      value: coeff[key] ?? 0,
    });
  }

  return { columns, dataSource };
}

const getTranslatePolynomial = (key: string, commonT) => {
  switch (key) {
    case "linear":
      return commonT("nonlinear.linear");
    case "quadratic":
      return commonT("nonlinear.quadratic");
    case "cube":
      return commonT("nonlinear.cube");
    default:
      return key;
  }
};


eu tenho o back end em R, se fosse possivel fazer em python


#' Calcula regressão não linear
#' @serializer unboxedJSON list(na = NULL)
#' @tag Regressao_Nao_Linear
#' @param inputData:object
#' @param responseColumn:string
#' @post /nonlinear/calculate
function(inputData, responseColumn) {
  df <- as.data.frame(inputData)

  non_linear_result <- calculate_nonlinear_regression(df, responseColumn)
  
  list(result = non_linear_result)
}


# Função para ajustar múltiplos modelos, extrair métricas e gerar dados de predição para cada curva.
calculate_nonlinear_regression <- function(df, response_column, n_points = 200) {
  # Identifica colunas X e Y
  x_col <- setdiff(names(df), response_column)
  x_name <- x_col[1]
  y_name <- response_column
  
  x <- df[[x_name]]
  y <- df[[y_name]]
  
  # Safe fit: captura erros e suprime warnings
  safe_fit <- function(expr) {
    suppressWarnings(
      tryCatch(expr, error = function(e) NULL)
    )
  }
  
  equations <- list(
    polynomial_2 = "Y = β₀ + β₁·x + β₂·x²",
    polynomial_3 = "Y = β₀ + β₁·x + β₂·x² + β₃·x³",
    gamma = "Y = exp(β₀ + β₁·x)",
    exponential_2p = "Y = a·exp(b·x)",
    exponential_3p = "Y = a·(1 - exp(-b·x))",
    biexponential = "Y = a·exp(b·x) + c·exp(d·x)",
    mechanistic_growth = "Y = a·(1 - exp(-b·x))",
    cell_growth = "Y = a·exp(b·x) / (1 + exp(c·x))",
    weibull_growth = "Y = a·(1 - exp(-b·xᶜ))",
    gompertz = "Y = a·exp(-b·exp(-c·x))",
    weibull = "Y = a·xᵇ",
    logistic_2p = "Y = a / (1 + exp(-b·(x - c)))",
    logistic_4p = "Y = d + (a - d) / (1 + exp(-b·(x - c)))",
    probit_2p = "Y = Φ(a + b·x)",
    probit_4p = "Y = d + (a - d)·Φ(b·(x - c))",
    gaussian_peak = "Y = a·exp(-((x - b)²) / (2·c²))",
    asymmetric_gaussian_peak = "Y = a·exp(-((x - b)²) / (2·c²)) + d·(x - b)",
    skew_normal_peak = "Y = a·exp(-((x - b)²) / (2·c²)) + d·(x - b)³",
    lorentzian_peak = "Y = a / (1 + ((x - b)/c)²)",
    logarithmic = "Y = a + b·log(x)",
    inverse = "Y = a + b/x",
    sqrt_model = "Y = a + b·√x",
    power_intercept = "Y = a + b·xᶜ",
    michaelis_menten = "Y = Vmax·x / (Km + x)",
    tanh_model = "Y = A + B·tanh(C·(x - D))"
  )
  
  # Extrai coef, R², AIC, BIC
  extract_metrics <- function(model, model_type = "lm", model_name) {
    if (is.null(model)) return(NULL)
    coef_vals <- tryCatch(coef(model), error = function(e) NULL)
    coef_list <- as.list(coef_vals)
    names(coef_list) <- names(coef_vals)
    aic_val <- tryCatch(AIC(model), error = function(e) NA)
    bic_val <- tryCatch(BIC(model), error = function(e) NA)
    r2_val <- NA
    if (model_type == "lm") {
      r2_val <- summary(model)$r.squared
    } else if (model_type == "nls") {
      yhat <- tryCatch(predict(model), error = function(e) rep(NA, length(y)))
      ss_res <- sum((y - yhat)^2)
      ss_tot <- sum((y - mean(y))^2)
      r2_val <- 1 - ss_res/ss_tot
    } else if (model_type == "glm") {
      ll0  <- logLik(update(model, . ~ 1))
      llm  <- logLik(model)
      r2_val <- 1 - as.numeric(llm/ll0)
    }
    
    if (model_name == "polynomial_2") {
      names(coef_list) <- c("Intercept", "linear", "quadratic")
    } else if (model_name == "polynomial_3") {
      names(coef_list) <- c("Intercept", "linear", "quadratic", "cubic")
    } else {
      names(coef_list) <- names(coef_vals)
    }
    
    equation <- equations[[model_name]]
    
    list(coef = coef_list, rSquared = r2_val, aic = aic_val, bic = bic_val, equation = equation)
  }
  
  # Especificações de modelos
  specs <- list(
    # Linear e Polinomial
    polynomial_2             = list(expr=quote(lm(y~poly(x,2,raw=TRUE))), type="lm"),
    polynomial_3             = list(expr=quote(lm(y~poly(x,3,raw=TRUE))), type="lm"),
    
    # GLM
    gamma                    = list(expr=quote(glm(y~x, family=Gamma(link="identity"))), type="glm"),
    
    # Exponenciais
    exponential_2p           = list(expr=quote(nls(y~a*exp(b*x), start=list(a=y[1],b=0.1))), type="nls"),
    exponential_3p           = list(expr=quote(nls(y~a*(1-exp(-b*x)), start=list(a=max(y),b=0.1))), type="nls"),
    biexponential            = list(expr=quote(nls(y~a*exp(b*x)+c*exp(d*x), start=list(a=y[1],b=0.1,c=y[1]/2,d=-0.1))), type="nls"),
    mechanistic_growth       = list(expr=quote(nls(y~a*(1-exp(-b*x)), start=list(a=max(y),b=0.1))), type="nls"),
    cell_growth              = list(expr=quote(nls(y~a*exp(b*x)/(1+exp(c*x)), start=list(a=max(y),b=0.1,c=0.1))), type="nls"),
    
    # Weibull e Gompertz
    weibull_growth           = list(expr=quote(nls(y~a*(1-exp(-b*x^c)), start=list(a=max(y),b=1,c=1))), type="nls"),
    gompertz                 = list(expr=quote(nls(y~a*exp(-b*exp(-c*x)), start=list(a=max(y),b=1,c=0.1))), type="nls"),
    weibull                  = list(expr = quote(nls(y ~ a * x^b, start = list(a = 1, b = 1))), type = "nls"),
    
    # Sigmoides
    logistic_2p              = list(expr=quote(nls(y~a/(1+exp(-b*(x-c))), start=list(a=max(y),b=1,c=mean(x)))), type="nls"),
    logistic_4p              = list(expr=quote(nls(y~d+(a-d)/(1+exp(-b*(x-c))), start=list(a=max(y),d=min(y),b=1,c=mean(x)))), type="nls"),
    probit_2p                = list(expr=quote(nls(y~pnorm(a+b*x), start=list(a=0,b=1))), type="nls"),
    probit_4p                = list(expr=quote(nls(y~d+(a-d)*pnorm(b*(x-c)), start=list(a=max(y),d=min(y),b=1,c=mean(x)))), type="nls"),
    
    # Modelos de pico
    gaussian_peak            = list(expr=quote(nls(y~a*exp(-((x-b)^2)/(2*c^2)), start=list(a=max(y),b=mean(x),c=sd(x)))), type="nls"),
    asymmetric_gaussian_peak = list(expr=quote(nls(y~a*exp(-((x-b)^2)/(2*c^2))+d*(x-b), start=list(a=max(y),b=mean(x),c=sd(x),d=0.1))), type="nls"),
    skew_normal_peak         = list(expr=quote(nls(y~a*exp(-((x-b)^2)/(2*c^2))+d*(x-b)^3, start=list(a=max(y),b=mean(x),c=sd(x),d=0.1))), type="nls"),
    lorentzian_peak          = list(expr=quote(nls(y~a/(1+((x-b)/c)^2), start=list(a=max(y),b=mean(x),c=sd(x)))), type="nls"),
    
    # Outros não-lineares
    logarithmic              = list(expr=quote(nls(y~a+b*log(x), start=list(a=min(y),b=1))), type="nls"),
    inverse                  = list(expr=quote(nls(y~a+b/x, start=list(a=mean(y),b=1))), type="nls"),
    sqrt_model               = list(expr=quote(nls(y~a+b*sqrt(x), start=list(a=min(y),b=1))), type="nls"),
    power_intercept          = list(expr=quote(nls(y~a+b*x^c, start=list(a=min(y),b=1,c=1))), type="nls"),
    michaelis_menten         = list(expr=quote(nls(y~Vmax*x/(Km+x), start=list(Vmax=max(y),Km=mean(x)))), type="nls"),
    tanh_model               = list(expr=quote(nls(y~A+B*tanh(C*(x-D)), start=list(A=mean(y),B=(max(y)-min(y))/2,C=1,D=mean(x)))), type="nls")
  )
  
  model_objs  <- list()
  metrics     <- list()
  predictions <- list()
  x_seq <- seq(min(x, na.rm=TRUE), max(x, na.rm=TRUE), length.out = n_points)
  
  # Itera e ajusta
  for (name in names(specs)) {
    spec <- specs[[name]]
    fit  <- safe_fit(eval(spec$expr))
    model_objs[[name]]  <- fit
    metrics[[name]]     <- extract_metrics(fit, spec$type, name)
    if (!is.null(fit)) {
      newdata <- data.frame(x = x_seq)
      y_pred  <- tryCatch(predict(fit, newdata), error = function(e) rep(NA, n_points))
      predictions[[name]] <- data.frame(x = x_seq, y = y_pred)
    }
  }
  
  original <- data.frame(x = x, y = y)
  
  list(
    original    = original,
    predictions = predictions,
    metrics     = metrics
  )
}

