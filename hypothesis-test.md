vamos fazer o teste t com amostra com valor esperado dentro do teste de hipotese, ele precisa selecionar somente y, e informar a media esperada


meu python que da para reaproveitar

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats

from utils.common.common import remove_punctuation

test_t_sample_router = APIRouter()


class Item(BaseModel):
    obj: dict[str, List[float | str]]
    expectedMean: float


@test_t_sample_router.post(
    "/test_t_sample",
    tags=["Teste de Hipótese"],
    description="""Calcula Test T Comparação de Amostras, obj: dataFrame com várias colunas de Y.""",
    response_model=object,
)
def calculate_test_t_sample(body: Item):
    df = pd.DataFrame(body.obj)
    df = pd.DataFrame(remove_punctuation(df))

    expected_mean = body.expectedMean

    try:
        if len(df.columns) <= 1:
            col = df.columns[0]

            mean = df[col].mean()
            std_dev = df[col].std(ddof=1)
            n = len(df[col])

            t_calculate = (mean - expected_mean) / (std_dev / np.sqrt(n))
            p_value = 2 * (1 - stats.t.cdf(abs(t_calculate), df=n - 1))

            return {
                "result": {
                    "type": "one-sample",
                    "column": col,
                    "mean": mean,
                    "std": std_dev,
                    "tCalculate": t_calculate,
                    "pValue": p_value,
                    "yValues": {col: df[col].tolist()},
                }
            }

        else:
            col_pairs = list(combinations(df.columns, 2))
            results = {}

            for col1, col2 in col_pairs:
                diff_series = abs(df[col1] - df[col2])
                mean_diff = diff_series.mean()
                std_dev = diff_series.std(ddof=1)
                n = len(diff_series)

                if std_dev == 0:
                    t_calculate = 0.0
                    p_value = 0.0
                else:
                    t_calculate = mean_diff / (std_dev / np.sqrt(n))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_calculate), df=n - 1))

                results = {
                    "type": "paired-sample",
                    "pair": f"{col1} - {col2}",
                    "mean": mean_diff,
                    "std": std_dev,
                    "tCalculate": t_calculate,
                    "pValue": p_value,
                    "yValues": {
                        col1: df[col1].tolist(),
                        col2: df[col2].tolist(),
                    },
                }

            return {"result": results}

    except Exception as error:
        print(error)
        raise HTTPException(status_code=500, detail="testTSampleError")


meu front end antigo



import { formatNumber } from "utils/formatting";
import { TestTSampleData } from "../interface";
import { P_VALUE_LIMIT } from "utils/constant";

export const createSummaryTable = (result: TestTSampleData, translate: any) => {
  const columns = [
    {
      title: translate.mean,
      dataIndex: "mean",
      key: "mean",
    },
    {
      title: translate.std,
      dataIndex: "std",
      key: "std",
    },
    {
      title: translate.tCalculate,
      dataIndex: "tCalculate",
      key: "tCalculate",
    },
    {
      title: translate.pValue,
      dataIndex: "pValue",
      key: "pValue",
      onCell: (item: any) => {
        return {
          ["style"]: { color: item.pValue > P_VALUE_LIMIT ? "" : "red" },
        };
      },
    },
  ];

  const dataSource: any[] = [];

  dataSource.push({
    mean: formatNumber(result.mean, 5),
    std: formatNumber(result.std, 5),
    tCalculate: formatNumber(result.tCalculate, 5),
    pValue: formatNumber(result.pValue, 5),
  });

  return { columns, dataSource };
};

import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { useAuth } from "hooks/useAuth";
import { useTranslation } from "next-i18next";
import { useRouter } from "next/router";
import ContentHeader from "shared/contentHeader";
import { Spin } from "shared/spin"
import { HypothesisTestExportData } from "../interface";
import { getItem } from "utils/database";
import { DataToCompile } from "components/insertData/inteface";
import { Col, Row, message } from "antd";
import { TestTSampleResult } from "./interface";
import { createSummaryTable } from "./table";
import Table from "shared/table";
import dynamic from "next/dynamic";
import {
  HighChartCustomSeries,
  HighChartTemplate,
} from "shared/widget/chartHub/interface";
import { COLORS } from "utils/color";

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const fetcher = axios.create({
  baseURL: "/api",
});

export const TestTSample: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });

  const { user } = useAuth();
  const router = useRouter();

  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingSummaryTable, setLoadingSummaryTable] = useState(true);
  const [chartData, setChartData] = useState<HighChartTemplate>({});

  const summaryTable = useRef<any>({});

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "testTSample",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  useEffect(() => {
    const getData = async () => {
      const parsedUrlQuery = router.query;
      if (Object.keys(parsedUrlQuery).length > 0) {
        const tool = parsedUrlQuery.tool as string;
        const uid = parsedUrlQuery.uid as string;

        const item = (await getItem(tool, uid)) as HypothesisTestExportData;

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

          const testTSampleCalculate = {
            obj: dataToSend.obj,
            expectedMean: item.expectedMean,
          };

          try {
            const { data } = await fetcher.post<TestTSampleResult>(
              "hypothesisTest/testTSample/calculate",
              testTSampleCalculate
            );

            const result = data.result;

            const summaryTranslate = {
              mean: commonT("testTSample.mean"),
              std: commonT("testTSample.std"),
              pValue: commonT("testTSample.pValue"),
              tCalculate: commonT("testTSample.tCalculate"),
            };

            summaryTable.current = createSummaryTable(result, summaryTranslate);
            setLoadingSummaryTable(false);

            const chartsForVariable: HighChartTemplate = buildChart(
              result.yValues
            );

            setChartData(chartsForVariable);
          } catch (error) {
            console.error(error);
            message.error({
              content: commonT("error.general.proccesData"),
            });
          }
        }
      }
    };

    getData().catch(console.error);
  }, [router.query]);

  function buildChart(yValues: Record<string, number[]>) {
    const chartsForVariable: HighChartTemplate = {};

    const series: HighChartCustomSeries[] = [];
    let index = 0;

    for (const key of Object.keys(yValues)) {
      const baseId = `raw-data-${index}`;

      series.push({
        name: `Dados brutos ${key}`,
        type: "scatter",
        visible: false,
        showLegend: false,
        showInLegend: false,
        data: yValues[key],
        id: baseId,
      });

      series.push({
        type: "histogram",
        name: key,
        baseSeries: baseId,
        zIndex: -3,
        data: yValues[key],
        showLegend: true,
        visible: true,
      });
      index += 1;
    }

    chartsForVariable["samples"] = {
      seriesData: series,
      type: "histogram",
      displayName: commonT("descriptiveStatistics.histogramTitle"),
      options: {
        title: commonT("descriptiveStatistics.histogramTitle"),
        seriesName: commonT("descriptiveStatistics.frequency"),
        xAxisTitle: commonT("descriptiveStatistics.valueRange"),
        yAxisTitle: commonT("descriptiveStatistics.count"),
        colors: COLORS,
        titleFontSize: 24,
        chartName: commonT("descriptiveStatistics.count"),
        legendEnabled: true,
      },
    };

    return chartsForVariable;
  }

  return (
    <>
      <ContentHeader
        title={commonT("testTSample.title")}
        tool={"testTSample"}
      />

      {loadingPage ? (
        <Spin />
      ) : (
        <div id="content">
          <Row style={{ marginBottom: 20 }}>
            <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={10}>
              <ChartHub
                chartConfigurations={chartData}
                tool={"hypothesisTest"}
                showLimits={true}
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
              <div style={{ marginBottom: 20 }}>
                <Table
                  loading={loadingSummaryTable}
                  dataSource={summaryTable.current.dataSource}
                  columns={summaryTable.current.columns}
                  title={commonT("testTSample.summaryTable")}
                />
              </div>
            </Col>
          </Row>
        </div>
      )}
    </>
  );
};
