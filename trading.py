import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from pykrx import stock
import re
from datetime import datetime
from collections import Counter
from dateutil.relativedelta import relativedelta
import math
import pickle
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class Trading:

    def __init__(self):
        self.loadData()

        self.targetGroup = []

    def loadData(self):
        with open('Raw_Price/change_all_df.pickle', 'rb') as handle:
            self.change = pickle.load(handle)
        with open('Raw_Price/price_all_df.pickle', 'rb') as handle:
            self.close = pickle.load(handle)
        # with open('Raw_Price/kospi_index.pickle', 'rb') as handle:
        #     self.kospi_index = pickle.load(handle)
        # with open('Raw_Finance/finance_kospi_modi.pickle', 'rb') as handle:
        #     self.finance_kospi = pickle.load(handle)
        # with open('Raw_Finance/finance_input_quarterly_data.pickle', 'rb') as handle:  # dart 정보만을 변환한 것.
        #     self.finance = pickle.load(handle)
        # with open('Raw_Finance/finance_naver_converted_202106.pickle', 'rb') as handle:
        #     self.finance_naver = pickle.load(handle)
        # with open('Raw_Finance/finance_latest.pickle', 'rb') as handle:  # naver 정보를 합친 것. (현재 매수 종목 선정 시 사용해야함)
        #     self.finance_latest = pickle.load(handle)
        # with open('Raw_Finance/finance_quantking_201809.pickle', 'rb') as handle:
        #     self.finance_quantking = pickle.load(handle)
        # self.dfFinance = self.finance

    # def defineUniverse(self):


    # def makeResult(self):


    def searchByTarget(self, targetStrategy:str, targetGroup :list = None,
                       targetPeriodStart : datetime=None, targetPeriodEnd : datetime=None):
        self.listTradeTarget = []
        # dfClose = self.close.loc[targetPeriodStart:targetPeriodEnd,targetGroup]
        if targetGroup == None:
            targetGroup = self.targetGroup
        if targetStrategy == "ma":
            daysForMABuy = 20
            dfCloseTemp = self.close.loc[targetPeriodStart-relativedelta(days=daysForMABuy*2):targetPeriodEnd, targetGroup]
            dfClose = dfCloseTemp.loc[targetPeriodStart:targetPeriodEnd,:]
            dfMA = dfCloseTemp.rolling(daysForMABuy).mean().loc[targetPeriodStart:targetPeriodEnd,:]

            for stockCode in dfClose.columns:
                seriesClose = dfClose[stockCode]
                seriesMA = dfMA[stockCode]
                seriesSignal = [1 if close>ma else 0 for close,ma in zip(seriesClose, seriesMA)]
                for i in range(len(seriesSignal)-1):
                    if (seriesSignal[i] != seriesSignal[i+1]) & (seriesSignal[i+1] == 1):
                        self.listTradeTarget.append((stockCode,seriesClose.index[i+1]))  # 매수 신호가 발생한 일을 저장.
                # 두 값을 차례대로 비교하면서 0이었다가 1이 되는 순간 매수신호로 인식.

    #  dataframe 형태로 만든다. column에는 전체 종목 명이 있고, 거래 했을 시의 각 종목 별 일자별 수익이 전체 target period에 대해서 출력.
    def makeTargetResult(self, tradeTarget : list=None, targetPeriodStart : datetime=None, targetPeriodEnd : datetime=None):
        self.dfTradeResult = pd.DataFrame(columns = self.targetGroup,index=self.change.loc[targetPeriodStart:targetPeriodEnd,:].index)
        if tradeTarget == None:
            tradeTarget = self.listTradeTarget
        listTrade = []  # 기간 중에 거래한 종목은 다시 거래하지 않게 하기 위해
        for stockCode, dateBuy in tradeTarget:

            #  listTradeTarget에는 빠른 날짜가 먼저 입력되므로, 먼저 나온 결과를 사용하고, 뒤에 같은 종목이 다시 나오면 제외
            #  매도 기준도 나중에 적용해보자.
            if stockCode not in listTrade:
                listTrade.append(stockCode)
                earning = (self.change.loc[dateBuy+relativedelta(days=1):targetPeriodEnd,stockCode]+1)
                self.dfTradeResult[stockCode] = earning

        self.dfTradeResult.fillna(value=1,inplace=True)



            #  결과는 [(stockCode, earning)]


#
# case_Trading = Trading()
# start = datetime(2020,1,1)
# end = datetime(2020,2,1)
# case_Trading.targetGroup = ['005930','000660','000220']
# case_Trading.searchByTarget(targetStrategy='ma',targetPeriodStart=start,targetPeriodEnd=end)
# case_Trading.makeTargetResult(tradeTarget=case_Trading.listTradeTarget,targetPeriodStart=start ,targetPeriodEnd=end)
# print(case_Trading.listTradeTarget)
# print(case_Trading.dfTradeResult)