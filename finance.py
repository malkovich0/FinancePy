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

class Finance():


    def __init__(self):

        self.dictPortfolio = {} # 기간별 포트폴리오 저장
        self.stocks_last = {} # 직전 기간의 보유종목

        self.loadData()

        # TEST 용 변수들
        self.listYears = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        self.caseFactors = {'val': ['PBR', 'PER', 'PCR','PSR'], 'qual': ['GP/A','AssetGrowth'], 'profitMom': ['OPQ','OPY','NPQ','NPY'], 'mom': ['Discrete'], 'size': ['Cap'], 'vol': [],
                            'quarter': True,'partial':False,'sizeTarget':None,'rebalance':True,'fscore':True,'kospi':True}
        self.methodStockWeight = 'equal'

        # make_stock_list_by_finance 변수
        self.limitOfMarketCapitalization = 10000000000 # 100억
        self.limitOfTradingPrice = 100000000 # 1억
        self.exclusiveStock = ['중국', '금융', '지주']  # 제외항목
        self.limitOfPBR = 0.2
        self.limitOfPER = 2
        self.limitOfPCR = 1
        self.limitOfPSR = 0.1

        # cal_yield_for_periods 변수
        self.noOfPartial = 5
        self.noOfStocks = 30
        #  시점, 종목명, 종목별 성과, {시점 : {'종목명':[],'성과':[],'상폐수':}},{시점 : }
        self.logSheets = {}
        self.caseStudyResult = {}


        # applyTraidingStrategy 변수
        self.daysContinuous = None
        self.tradingLoss = 1  #거래비용 및 슬리피지로 거래 마다 발생하는 손실 [%]

        # makeTradePlanByNaverFinance 변수
        self.dateBuy = datetime.now()  # 당일
        self.periodBuy = self.define_period(self.dateBuy)
        self.totalAsset = 10000000 # 천만원
        self.sizeTarget = None  # high20, low20
        self.dictPortfolioTradePlan = {}

    def printVariables(self):
        print("caseStudy 용 변수들")
        print("listYears : ", self.listYears)
        print("caseFactors : ",self.caseFactors)
        print("methodStockWeight : ",self.methodStockWeight)
        print("limitOfMarketCapitalization : ",self.limitOfMarketCapitalization)
        print("limitOfTradingPrice : ",self.limitOfTradingPrice)
        # print("dfFinance : ",self.dfFinance)
        print("makeTradePlanByNaverFinance 용 변수들")
        print("dateBuy : ", self.dateBuy)
        print("periodBuy : ",self.periodBuy)
        print("totalAsset : ",self.totalAsset)

    # dict형태로 받아온다음에 사용할때 필요하면 dataframe형태로 수정하는 것으로 변경.
    def loadData(self):
        with open('Raw_Price/change_all_df.pickle', 'rb') as handle:
            self.change = pickle.load(handle)
            # self.dictChange = pickle.load(handle)
            # self.change = pd.DataFrame(self.dictChange)
        with open('Raw_Price/price_all_df.pickle', 'rb') as handle:
            self.close = pickle.load(handle)
        with open('Raw_Price/tradingCost_all.pickle', 'rb') as handle:
            self.tradingCost = pickle.load(handle)

        with open('Raw_Price/kospi_index.pickle', 'rb') as handle:
            self.kospi_index = pickle.load(handle)
        # with open('Raw_Finance/finance_kospi_modi.pickle', 'rb') as handle:
        #     self.finance_kospi = pickle.load(handle)
        with open('Raw_Finance/finance_kospi_modi.pickle', 'rb') as handle:
            self.finance_kospi = pickle.load(handle)
        with open('Raw_Finance/finance_input_quarterly_data.pickle', 'rb') as handle:  # dart 정보만을 변환한 것.
            self.finance = pickle.load(handle)
        with open('Raw_Finance/finance_naver_converted_202106.pickle', 'rb') as handle:
            self.finance_naver = pickle.load(handle)
        with open('Raw_Finance/finance_latest.pickle', 'rb') as handle:  # naver 정보를 합친 것. (현재 매수 종목 선정 시 사용해야함)
            self.finance_latest = pickle.load(handle)
        with open('Raw_Finance/finance_quantking_201809.pickle', 'rb') as handle:
            self.finance_quantking = pickle.load(handle)
        with open('Raw_Finance/finance_dartapi.pickle', 'rb') as handle:
            self.finance_dartapi = pickle.load(handle)

        self.dfFinance = self.finance

    # self.change에 있는 종목명과 일자를 그대로 활용. 12분 예상
    def initializeData(self):
        dictChangeRaw = self.change
        dateStart = dictChangeRaw.index[0]
        dateEnd = dictChangeRaw.index[-1]
        dictChange = {}
        dictClose = {}
        for stockCode in dictChangeRaw.columns:
            dataTemp = fdr.DataReader(stockCode,start=dateStart,end=dateEnd)
            dictChange[stockCode] = dataTemp['Change']
            dictClose[stockCode] = dataTemp['Close']
        with open('Raw_Price/change_all_dict.pickle', 'wb') as handle:
            pickle.dump(dictChange, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Raw_Price/price_all_dict.pickle', 'wb') as handle:
            pickle.dump(dictClose, handle, protocol=pickle.HIGHEST_PROTOCOL)
        dfChange = pd.DataFrame(dictChange)
        dfClose = pd.DataFrame(dictClose)
        with open('Raw_Price/change_all_df.pickle', 'wb') as handle:
            pickle.dump(dfChange, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Raw_Price/price_all_df.pickle', 'wb') as handle:
            pickle.dump(dfClose, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 기존 저장된 자료에 현재까지의 일자를 더해서 자료 추가. 10분 예상.
    def resaveData(self, daysBefore : int=0):
        with open('Raw_Price/change_all_dict.pickle', 'rb') as handle:
            dictChange = pickle.load(handle)
        with open('Raw_Price/price_all_dict.pickle', 'rb') as handle:
            dictClose = pickle.load(handle)

        from pykrx import stock
        dateNow = datetime.now()-relativedelta(days=daysBefore)
        # dateStrNow = self.date_str(dateNow)
        kospi_list = stock.get_market_ticker_list(dateNow, market='KOSPI')
        kosdaq_list = stock.get_market_ticker_list(dateNow, market='KOSDAQ')
        dateFinal = dictClose['005930'].index[-1]
        for stockCode in kospi_list+kosdaq_list:
            dataTemp = fdr.DataReader(stockCode,start=dateFinal,end=dateNow).iloc[1:]
            dictChange[stockCode] = pd.concat([dictChange[stockCode], dataTemp['Change']])
            dictClose[stockCode] = pd.concat([dictClose[stockCode], dataTemp['Close']])
        with open('Raw_Price/change_all_dict.pickle', 'wb') as handle:
            pickle.dump(dictChange, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Raw_Price/price_all_dict.pickle', 'wb') as handle:
            pickle.dump(dictClose, handle, protocol=pickle.HIGHEST_PROTOCOL)
        dfChange = pd.DataFrame(dictChange)
        dfClose = pd.DataFrame(dictClose)
        with open('Raw_Price/change_all_df.pickle', 'wb') as handle:
            pickle.dump(dfChange, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Raw_Price/price_all_df.pickle', 'wb') as handle:
            pickle.dump(dfClose, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def date_str_bar(self, date):
        date_temp = str("%s-%02d-%02d" % (date.year, date.month, date.day))
        return date_temp

    def date_str(self, date):
        date_temp = str("%s%02d%02d" % (date.year, date.month, date.day))
        return date_temp

    def date_to_date(self, date):
        date = date.split('-')
        date_new = datetime(int(date[0]), int(date[1]), int(date[2]))
        return date_new

    # 대상 종목 선정
    def cap_add_preferred(self, df_cap : pd.DataFrame):
        df_cap['시가총액수정'] = df_cap['시가총액']
        # 우선주 리스트 만들기
        p = re.compile('[0-9]*5$|[0-9]*7$|[0-9]*9$|[0-9]*K$|[0-9]*L$|[0-9]*M$|[0-9]*B$')
        stock_list_preferred = []
        for i in df_cap.index:
            if p.match(i):
                stock_list_preferred.append(i)
        # 우선주의 시가총액을 보통주에 더하기
        for stock_code in stock_list_preferred:
            stock_code_common = stock_code[0:5] + '0'
            df_cap.loc[stock_code_common, '시가총액수정'] = df_cap.loc[stock_code_common, '시가총액수정'] + df_cap.loc[
                stock_code, '시가총액']
        df_cap['시가총액'] = df_cap['시가총액수정']
        df_cap.drop(columns=['시가총액수정'], inplace=True)
        return df_cap

    def make_stock_list_by_finance(self, date_buy: datetime,
                                   fs: pd.DataFrame,  # 재무제표
                                   size_target: str = None,  # high20, low20
                                   fscore: bool = True,
                                   period: int = None,
                                   value_factors: list = ['PBR', 'PER', 'PCR', 'PSR'],  # PBR, PER, PCR, PSR
                                   quality_factors: list = ['GP/A','AssetGrowth'],  # GP/A
                                   profit_momentum_factors : list = ['OPQ', 'OPY', 'NPQ', 'NPY'],
                                   momentum_factors: list = ['Discrete'],  # 일반 momentum, discrete momentum
                                   size_factors: list = ['Cap'],  # 저가주
                                   volatility_factors: list = []):  # 변동성 factor
        # fs : financial statements

        if period is None:
            period = self.define_period(date_buy)
        # periodQuarterAgo, periodYearAgo 정의
        if (str(period)[4:] == '12') or (str(period)[4:] == '09') or (str(period)[4:] == '06'):
            periodQuarterAgo = period - 3
        elif (str(period)[4:] == '03'):
            periodQuarterAgo = int(str(int(str(period)[:4]) - 1) + '12')
        periodYearAgo = int(str(int(str(period)[:4]) - 1) + str(period)[4:])

        # 기간의 재무제표와 시가총액 가져오기
        # 시가총액
        # date = self.date_str(date_buy)
        # 별도로 저장된 dict에서 가져오기.
        df_cap_kospi = stock.get_market_cap_by_ticker(date_buy, market='KOSPI')
        df_cap_kosdaq = stock.get_market_cap_by_ticker(date_buy, market='KOSDAQ')
        df_cap = pd.concat([df_cap_kospi, df_cap_kosdaq], axis=0)
        # 보통주의 시가총액에 우선주를 더해준다.
        df_cap = self.cap_add_preferred(df_cap)
        # 시가총액과 거래대금을 기준으로 제외한다.
        cap = self.limitOfMarketCapitalization
        trade = self.limitOfTradingPrice
        if cap != None:
            df_cap = df_cap[lambda df_cap: df_cap['시가총액'] > np.int64(cap)]  # 시가총액 100억 이상
        if trade != None:
            df_cap = df_cap[lambda df_cap: df_cap['거래대금'] > np.int64(trade)]  # 거래대금 1억 이상

        if size_target == "high20":
            listCap = list(df_cap['시가총액'])
            listCap.sort()
            df_cap = df_cap[lambda df_cap:df_cap['시가총액'] > np.int64(listCap[int(len(listCap) / 5 * 4)])]
        elif size_target == "low20":
            listCap = list(df_cap['시가총액'])
            listCap.sort()
            df_cap = df_cap[lambda df_cap: df_cap['시가총액'] < np.int64(listCap[int(len(listCap) / 5)])]

        df_cap['stock_code'] = df_cap.index
        df_cap = df_cap[['stock_code', '시가총액']]

        fsThis = fs[fs['period'] == period]
        fsYearAgo = fs[fs['period'] == periodYearAgo]
        fsQuarterAgo = fs[fs['period'] == periodQuarterAgo]
        finance_cap = pd.merge(fsThis, df_cap, how='left')  # 중복된 항목명 중 왼쪽을 기준으로 정렬

        # 상장이전에도 재무제표에 값이 있어, 시가총액이 있는 기준으로 정렬
        finance_cap = finance_cap[~finance_cap['시가총액'].isnull()]

        # 제외항목이 있는 경우 제거.

        if self.exclusiveStock != None:
            with open('Raw_Finance/listExclusive.pickle', 'rb') as handle:
                dfExclusive = pickle.load(handle)
            for exclusiveCategory in self.exclusiveStock:
                dfExclusiveStocks = dfExclusive[dfExclusive['Category'] == exclusiveCategory].StockCode
                listIndex = []
                for stockCode in list(dfExclusiveStocks):
                    indexName = finance_cap[finance_cap['stock_code'] == stockCode].index
                    if len(indexName) != 0:
                        listIndex.append(indexName[0])
                finance_cap.drop(listIndex,inplace=True)

        # F-score 3 점 외에는 제거.
        if fscore:
            last_year = date_buy.year - 1
            Fscore_1_first = stock.get_market_cap_by_ticker(date=str(last_year) + '0101')
            Fscore_1_first['stock_code'] = Fscore_1_first.index
            Fscore_1_first = Fscore_1_first[['stock_code', '상장주식수']]
            Fscore_1_first.rename(columns={'상장주식수': '상장주식수_first'}, inplace=True)
            Fscore_1_last = stock.get_market_cap_by_ticker(date=str(last_year) + '1231')
            Fscore_1_last['stock_code'] = Fscore_1_last.index
            Fscore_1_last = Fscore_1_last[['stock_code', '상장주식수']]
            Fscore_1_last.rename(columns={'상장주식수': '상장주식수_last'}, inplace=True)
            finance_cap = pd.merge(finance_cap, Fscore_1_first, how='left')
            finance_cap = pd.merge(finance_cap, Fscore_1_last, how='left')

            Fscore_1 = []
            for i in range(len(finance_cap.index)):
                first = finance_cap['상장주식수_first'][i]
                last = finance_cap['상장주식수_last'][i]
                if first < last:
                    Fscore_1.append(0)
                else:
                    Fscore_1.append(1)  # 만일 무상감자와 유상증자를 해서 주식수는 줄어들었지만 유상증자의 효과가 있다면 찾기 어려움 (000040, 2020)
            finance_cap['Fscore_1'] = Fscore_1

            Fscore_2 = []
            for i in range(len(finance_cap.index)):
                if finance_cap['지배순이익'][i] > 0:
                    Fscore_2.append(1)
                elif math.isnan(finance_cap['지배순이익'][i]) & (finance_cap['당기순이익'][i] > 0):
                    Fscore_2.append(1)
                else:
                    Fscore_2.append(0)
            finance_cap['Fscore_2'] = Fscore_2

            Fscore_3 = []
            for i in range(len(finance_cap.index)):
                if finance_cap['영업활동현금흐름'][i] > 0:
                    Fscore_3.append(1)
                else:
                    Fscore_3.append(0)
            finance_cap['Fscore_3'] = Fscore_3
            finance_cap['Fscore'] = finance_cap['Fscore_1'] + finance_cap['Fscore_2'] + finance_cap['Fscore_3']
            finance_cap = finance_cap[finance_cap['Fscore'] == 3]

        # 초기화, 하나의 factor에 여러개의 score를 고려하면 0을 넣어줘야 함.
        finance_cap['PBR_Score'] = 0
        finance_cap['PER_Score'] = 0
        finance_cap['PCR_Score'] = 0
        finance_cap['PSR_Score'] = 0
        finance_cap['GP/A_Score'] = 0
        finance_cap['AssetGrowth_Score'] = 0
        finance_cap['Volatility_Score'] = 0
        finance_cap['OPQ_Score'] = 0
        finance_cap['OPY_Score'] = 0
        finance_cap['NPQ_Score'] = 0
        finance_cap['NPY_Score'] = 0
        finance_cap['Momentum_Simple_Score'] = 0
        finance_cap['Momentum_Discrete_Score'] = 0

        # value_factors
        if len(value_factors) != 0:
            for value_factor in value_factors:
                if value_factor == 'PBR':
                    if value_factor not in finance_cap.columns:
                        finance_cap['PBR'] = finance_cap['시가총액'] / finance_cap['지배자산']
                        finance_cap.loc[finance_cap[finance_cap['PBR'].isnull()].index, 'PBR'] = finance_cap['시가총액'] / finance_cap['총자산']
                    # if self.limitOfPBR != None:
                    #     finance_cap = finance_cap[finance_cap['PBR'] > self.limitOfPBR]
                    finance_cap = finance_cap.dropna(subset=['PBR'])
                    finance_cap['1/PBR'] = 1/finance_cap['PBR']
                    finance_cap = finance_cap.sort_values(by='1/PBR',ascending=False)
                    if self.limitOfPBR != None:
                        finance_cap = finance_cap[finance_cap['1/PBR'] < 1/self.limitOfPBR]
                    # -1 -0.2 0.2 1 -1 -5 5 1
                    # finance_cap = pd.concat([finance_cap[finance_cap['PBR']>0].sort_values(by='PBR',ascending=True),
                    #                          finance_cap[finance_cap['PBR']<=0].sort_values(by='PBR',ascending=False)])
                    finance_cap['PBR_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                elif value_factor == 'PER':
                    if value_factor not in finance_cap.columns:
                        finance_cap['PER'] = finance_cap['시가총액'] / finance_cap['지배순이익']
                        finance_cap.loc[finance_cap[finance_cap['PER'].isnull()].index, 'PER'] = finance_cap['시가총액'] / finance_cap['당기순이익']
                    # if self.limitOfPER != None:
                    #     finance_cap = finance_cap[finance_cap['PER'] > self.limitOfPER]
                    finance_cap = finance_cap.dropna(subset=['PER'])
                    finance_cap['1/PER'] = 1 / finance_cap['PER']
                    finance_cap = finance_cap.sort_values(by='1/PER', ascending=False)
                    if self.limitOfPER != None:
                        finance_cap = finance_cap[finance_cap['1/PER'] < 1 / self.limitOfPER]

                    # finance_cap = pd.concat([finance_cap[finance_cap['PER']>0].sort_values(by='PER',ascending=True),
                    #                          finance_cap[finance_cap['PER']<=0].sort_values(by='PER',ascending=False)])
                    finance_cap['PER_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                elif value_factor == 'PCR':
                    if value_factor not in finance_cap.columns:
                        finance_cap['PCR'] = finance_cap['시가총액'] / finance_cap['영업활동현금흐름']
                    # if self.limitOfPCR != None:
                    #     finance_cap = finance_cap[finance_cap['PCR'] > self.limitOfPCR]
                    finance_cap = finance_cap.dropna(subset=['PCR'])
                    finance_cap['1/PCR'] = 1 / finance_cap['PCR']
                    finance_cap = finance_cap.sort_values(by='1/PCR', ascending=False)
                    if self.limitOfPCR != None:
                        finance_cap = finance_cap[finance_cap['1/PCR'] < 1 / self.limitOfPCR]
                    #
                    # finance_cap = pd.concat([finance_cap[finance_cap['PCR'] > 0].sort_values(by='PCR', ascending=True),
                    #                          finance_cap[finance_cap['PCR'] <= 0].sort_values(by='PCR', ascending=False)])
                    finance_cap['PCR_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                elif value_factor == 'PSR':
                    if value_factor not in finance_cap.columns:
                        finance_cap['PSR'] = finance_cap['시가총액'] / finance_cap['매출액']
                    # if self.limitOfPSR != None:
                    #     finance_cap = finance_cap[finance_cap['PSR'] > self.limitOfPSR]
                    finance_cap = finance_cap.dropna(subset=['PSR'])
                    finance_cap['1/PSR'] = 1 / finance_cap['PSR']
                    finance_cap = finance_cap.sort_values(by='1/PSR', ascending=False)
                    if self.limitOfPSR != None:
                        finance_cap = finance_cap[finance_cap['1/PSR'] < 1 / self.limitOfPSR]

                    # finance_cap = pd.concat([finance_cap[finance_cap['PSR'] > 0].sort_values(by='PSR', ascending=True),
                    #                          finance_cap[finance_cap['PSR'] <= 0].sort_values(by='PSR', ascending=False)])
                    finance_cap['PSR_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
            finance_cap['Value_Score'] = (finance_cap['PBR_Score'] + finance_cap['PER_Score'] + finance_cap[
                'PCR_Score'] + finance_cap['PSR_Score']) / len(value_factors)
        else:
            finance_cap['Value_Score'] = 0

        # quality factors
        if len(quality_factors) != 0:
            for quality_factor in quality_factors:
                if quality_factor == 'GP/A':
                    if quality_factor not in finance_cap.columns:
                        finance_cap['GP/A'] = finance_cap['매출총이익'] / finance_cap['총자산']
                        finance_cap.loc[finance_cap[finance_cap['GP/A'].isnull()].index, 'GP/A'] = (finance_cap['매출액'] - finance_cap['매출원가']) / finance_cap['총자산']
                    finance_cap = finance_cap.dropna(subset=['GP/A']).sort_values(by='GP/A', ascending=False)
                    # finance_cap = pd.concat([finance_cap[finance_cap['GP/A'] > 0].sort_values(by='GP/A', ascending=False),
                    #                          finance_cap[finance_cap['GP/A'] <= 0].sort_values(by='GP/A', ascending=True)])
                    #
                    finance_cap['GP/A_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                elif quality_factor == "AssetGrowth":  # 자산성장률이 낮은게 좋다.
                    finance_cap.drop(columns=['AssetGrowth_Score'], inplace=True) # 계산할 경우 기존에 0 넣어준 것을 지워준다.
                    fsThisAsset = fsThis.loc[:, ['stock_code', '총자산']]
                    fsThisAsset = fsThisAsset[fsThisAsset['총자산'] > 0]  # 현재 총자산이 마이너스인 경우 제외
                    fsYearAgoAsset = fsYearAgo.loc[:, ['stock_code', '총자산']]
                    fsYearAgoAsset.rename(columns={'총자산': '총자산Y'}, inplace=True)
                    # 연간 데이터간의 비교
                    fsYearAssetComparison = pd.merge(fsThisAsset, fsYearAgoAsset, on='stock_code')
                    fsYearAssetComparison['총자산YOY'] = (fsYearAssetComparison['총자산'] - fsYearAssetComparison['총자산Y']) / fsYearAssetComparison['총자산Y']
                    fsYearAssetComparison = fsYearAssetComparison.dropna(subset=['총자산YOY']).sort_values(by='총자산YOY', ascending=True)
                    fsYearAssetComparison['AssetGrowth_Score'] = np.arange(1, len(fsYearAssetComparison.index) + 1) / len(fsYearAssetComparison.index)
                    # score입력
                    finance_cap = pd.merge(finance_cap, fsYearAssetComparison.loc[:, ['stock_code', 'AssetGrowth_Score']], on='stock_code')
                elif quality_factor == 'Volatility': # 낮은게 좋다.
                    # 지난 1개월 변동성.
                    bgn = date_buy - relativedelta(months=60)
                    end = date_buy
                    change_stocks = self.change[finance_cap.stock_code]
                    yield_stocks = change_stocks.loc[self.date_str_bar(bgn):self.date_str_bar(end), :]
                    yield_index = self.kospi_index.loc[self.date_str_bar(bgn):self.date_str_bar(end), ['Change']]
                    for stock_code in yield_stocks.columns:
                        yield_stocks[stock_code] = yield_stocks[stock_code] - yield_index['Change'] # 코스피 수익률만큼 제거.
                    finance_cap['Vol_Simple'] = yield_stocks.T.std(axis=1).values
                    finance_cap = finance_cap.dropna(subset=['Vol_Simple']).sort_values(by='Vol_Simple', ascending=True)
                    finance_cap['Volatility_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
            finance_cap['Quality_Score'] = (finance_cap['GP/A_Score'] + finance_cap['AssetGrowth_Score'] + finance_cap['Volatility_Score']) / len(quality_factors)
        else:
            finance_cap['Quality_Score'] = 0

        # profit momentum factors
        if len(profit_momentum_factors) !=0:
            for profit_momentum_factor in profit_momentum_factors:
                if profit_momentum_factor == "OPQ":  # Operating Profit Quarter
                    finance_cap.drop(columns=['OPQ_Score'], inplace=True)  # 계산할 경우 기존에 0 넣어준 것을 지워준다.
                    # 각 분기의 직전 1년 데이터
                    fsThisOP = fsThis.loc[:, ['stock_code', '영업이익']]
                    fsThisOP = fsThisOP[fsThisOP['영업이익'] > 0]  # 현재 영업이익이 마이너스인 경우 제외
                    fsQuarterAgoOP = fsQuarterAgo.loc[:, ['stock_code', '영업이익']]
                    fsQuarterAgoOP = fsQuarterAgoOP[fsQuarterAgoOP['영업이익'] > 0]  # 과거 영업이익이 마이너스인 경우 제외
                    fsQuarterAgoOP.rename(columns={'영업이익': '영업이익Q'}, inplace=True)
                    # 분기 데이터간의 비교
                    fsQuarterOPComparison = pd.merge(fsThisOP, fsQuarterAgoOP, on='stock_code')
                    fsQuarterOPComparison['영업이익QOQ'] = (fsQuarterOPComparison['영업이익'] - fsQuarterOPComparison['영업이익Q']) / fsQuarterOPComparison['영업이익']
                    fsQuarterOPComparison = fsQuarterOPComparison.dropna(subset=['영업이익QOQ']).sort_values(by='영업이익QOQ', ascending=False)
                    fsQuarterOPComparison['OPQ_Score'] = np.arange(1, len(fsQuarterOPComparison.index) + 1) / len(fsQuarterOPComparison.index)
                    # score입력
                    finance_cap = pd.merge(finance_cap, fsQuarterOPComparison.loc[:, ['stock_code', 'OPQ_Score']], on='stock_code')
                elif profit_momentum_factor == "OPY":  # Operating Profit Year
                    finance_cap.drop(columns=['OPY_Score'], inplace=True)  # 계산할 경우 기존에 0 넣어준 것을 지워준다.
                    # 각 분기의 직전 1년 데이터
                    fsThisOP = fsThis.loc[:, ['stock_code', '영업이익']]
                    fsThisOP = fsThisOP[fsThisOP['영업이익'] > 0]  # 현재 영업이익이 마이너스인 경우 제외
                    fsYearAgoOP = fsYearAgo.loc[:, ['stock_code', '영업이익']]
                    fsYearAgoOP = fsYearAgoOP[fsYearAgoOP['영업이익'] > 0]  # 과거 영업이익이 마이너스인 경우 제외
                    fsYearAgoOP.rename(columns={'영업이익': '영업이익Y'}, inplace=True)
                    # 분기 데이터간의 비교
                    fsYearOPComparison = pd.merge(fsThisOP, fsYearAgoOP, on='stock_code')
                    fsYearOPComparison['영업이익YOY'] = (fsYearOPComparison['영업이익'] - fsYearOPComparison['영업이익Y']) / fsYearOPComparison['영업이익']
                    fsYearOPComparison = fsYearOPComparison.dropna(subset=['영업이익YOY']).sort_values(by='영업이익YOY', ascending=False)
                    fsYearOPComparison['OPY_Score'] = np.arange(1, len(fsYearOPComparison.index) + 1) / len(fsYearOPComparison.index)
                    # score입력
                    finance_cap = pd.merge(finance_cap, fsYearOPComparison.loc[:, ['stock_code', 'OPY_Score']], on='stock_code')
                elif profit_momentum_factor == "NPQ":  # Operating Profit Quarter
                    finance_cap.drop(columns=['NPQ_Score'], inplace=True)  # 계산할 경우 기존에 0 넣어준 것을 지워준다.
                    # 각 분기의 직전 1년 데이터
                    fsThis['순이익'] = [b if not math.isnan(b) else a for a, b in zip(fsThis['당기순이익'], fsThis['지배순이익'])]
                    fsThisNP = fsThis.loc[:, ['stock_code', '순이익']]
                    fsThisNP = fsThisNP[fsThisNP['순이익'] > 0]  # 현재 순이익이 마이너스인 경우 제외
                    fsQuarterAgo['순이익'] = [b if not math.isnan(b) else a for a, b in zip(fsQuarterAgo['당기순이익'], fsQuarterAgo['지배순이익'])]
                    fsQuarterAgoNP = fsQuarterAgo.loc[:, ['stock_code', '순이익']]
                    fsQuarterAgoNP = fsQuarterAgoNP[fsQuarterAgoNP['순이익'] > 0]  # 과거 순이익이 마이너스인 경우 제외
                    fsQuarterAgoNP.rename(columns={'순이익': '순이익Q'}, inplace=True)
                    # 분기 데이터간의 비교
                    fsQuarterNPComparison = pd.merge(fsThisNP, fsQuarterAgoNP, on='stock_code')
                    fsQuarterNPComparison['순이익QOQ'] = (fsQuarterNPComparison['순이익'] - fsQuarterNPComparison['순이익Q']) / fsQuarterNPComparison['순이익']
                    fsQuarterNPComparison = fsQuarterNPComparison.dropna(subset=['순이익QOQ']).sort_values(by='순이익QOQ', ascending=False)
                    fsQuarterNPComparison['NPQ_Score'] = np.arange(1, len(fsQuarterNPComparison.index) + 1) / len(fsQuarterNPComparison.index)
                    # score입력
                    finance_cap = pd.merge(finance_cap, fsQuarterNPComparison.loc[:, ['stock_code', 'NPQ_Score']], on='stock_code')
                elif profit_momentum_factor == "NPY":  # Operating Profit Quarter
                    finance_cap.drop(columns=['NPY_Score'], inplace=True)  # 계산할 경우 기존에 0 넣어준 것을 지워준다.
                    # 각 분기의 직전 1년 데이터
                    fsThis['순이익'] = [b if not math.isnan(b) else a for a, b in zip(fsThis['당기순이익'], fsThis['지배순이익'])]
                    fsThisNP = fsThis.loc[:, ['stock_code', '순이익']]
                    fsThisNP = fsThisNP[fsThisNP['순이익'] > 0]  # 현재 순이익이 마이너스인 경우 제외
                    fsYearAgo['순이익'] = [b if not math.isnan(b) else a for a, b in zip(fsYearAgo['당기순이익'], fsYearAgo['지배순이익'])]
                    fsYearAgoNP = fsYearAgo.loc[:, ['stock_code', '순이익']]
                    fsYearAgoNP = fsYearAgoNP[fsYearAgoNP['순이익'] > 0]  # 과거 순이익이 마이너스인 경우 제외
                    fsYearAgoNP.rename(columns={'순이익': '순이익Y'}, inplace=True)
                    # 분기 데이터간의 비교
                    fsYearNPComparison = pd.merge(fsThisNP, fsYearAgoNP, on='stock_code')
                    fsYearNPComparison['순이익YOY'] = (fsYearNPComparison['순이익'] - fsYearNPComparison['순이익Y']) / fsYearNPComparison['순이익']
                    fsYearNPComparison = fsYearNPComparison.dropna(subset=['순이익YOY']).sort_values(by='순이익YOY', ascending=False)
                    fsYearNPComparison['NPY_Score'] = np.arange(1, len(fsYearNPComparison.index) + 1) / len(fsYearNPComparison.index)
                    # score입력
                    finance_cap = pd.merge(finance_cap, fsYearNPComparison.loc[:, ['stock_code', 'NPY_Score']], on='stock_code')

            finance_cap['Profit_Momentum_Score'] = (finance_cap['OPQ_Score'] + finance_cap['OPY_Score'] + finance_cap['NPQ_Score'] + finance_cap['NPY_Score']) / len(profit_momentum_factors)
        else:
            finance_cap['Profit_Momentum_Score'] = 0

        # momentum factors
        if len(momentum_factors) != 0:
            #         change = change_all
            for momentum_factor in momentum_factors:
                if momentum_factor == 'Simple':
                    bgn = date_buy - relativedelta(months=12)
                    end = date_buy - relativedelta(months=1)
                    change_stocks = self.change[finance_cap.stock_code]
                    yield_stocks = (change_stocks.loc[self.date_str_bar(bgn):self.date_str_bar(end), :] + 1).cumprod().tail(1).T.iloc[:, 0].values
                    finance_cap['Momentum_Simple'] = yield_stocks
                    finance_cap = finance_cap.dropna(subset=['Momentum_Simple']).sort_values(by='Momentum_Simple', ascending=False)  # 현재는 NaN을 0으로 변환한게 아니기때문에, 1년 사이에 상장한 경우 그 기간의 가격변동만을 고려.
                    finance_cap['Momentum_Simple_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                    # finance_cap['Momentum_Score'] = finance_cap['Momentum_Simple_Score']
                elif momentum_factor == "Discrete":
                    bgn = date_buy - relativedelta(months=12)
                    end = date_buy - relativedelta(months=1)
                    change_stocks = self.change[finance_cap.stock_code]
                    yield_stocks = (change_stocks.loc[self.date_str_bar(bgn):self.date_str_bar(end), :] + 1).cumprod().tail(1)
                    sign_stocks = yield_stocks.T[yield_stocks.index[0]].apply(lambda x: 1 if x > 1 else -1)
                    # %neg-%pos 를 계산하는 것은 1m-12m (?) -> 아직은 2m-12m으로 함수구성됨.
                    yield_count = self.change[yield_stocks.columns].loc[self.date_str_bar(bgn):self.date_str_bar(end), :]
                    for i in yield_count.columns:
                        yield_count[i] = yield_count[i].apply(lambda x: 1 if x > 0 else -1)  # 그 날에 값이 없는 경우도 -1을 준다.
                    yield_count = yield_count.sum(axis=0) / len(yield_count.index)
                    yield_id = pd.concat([pd.DataFrame(sign_stocks), pd.DataFrame(yield_count)], axis=1)
                    yield_id['ID'] = (yield_id[yield_id.columns[0]] * yield_id[yield_id.columns[1]] + yield_id[yield_id.columns[0]])
                    yield_id['ID'] = yield_id['ID'] * abs(yield_stocks.T - 1)[yield_stocks.index[0]]
                    finance_cap['Momentum_Discrete'] = yield_id['ID'].values
                    finance_cap = finance_cap.dropna(subset=['Momentum_Discrete']).sort_values(by='Momentum_Discrete', ascending=False)  # 현재는 NaN을 0으로 변환한게 아니기때문에, 1년 사이에 상장한 경우 그 기간의 가격변동만을 고려.
                    finance_cap['Momentum_Discrete_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                    # finance_cap['Momentum_Score'] = finance_cap['Momentum_Discrete_Score']
            finance_cap['Momentum_Score'] = (finance_cap['Momentum_Simple_Score'] + finance_cap['Momentum_Discrete_Score']) / len(momentum_factors)
        else:
            finance_cap['Momentum_Score'] = 0

        # size_factors
        if len(size_factors) != 0:
            for size_factor in size_factors:
                if size_factor == 'Cap':
                    finance_cap.sort_values(by='시가총액', ascending=True, inplace=True)
                    finance_cap['Cap_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
            finance_cap['Size_Score'] = (finance_cap['Cap_Score']) / len(size_factors)
        else:
            finance_cap['Size_Score'] = 0

        # Volatility_factos
        if len(volatility_factors) != 0:
            for vol_factor in volatility_factors:
                self.tradingCost = pd.DataFrame(self.tradingCost)
                if vol_factor == "tradingCost":
                    bgn = date_buy - relativedelta(months=6)
                    end = date_buy
                    tradingCost = self.tradingCost[finance_cap.stock_code]
                    tradingCost = tradingCost.loc[bgn:end, :]
                    finance_cap['Vol_TradingCost'] = tradingCost.T.std(axis=1).values
                    finance_cap = finance_cap.dropna(subset=['Vol_TradingCost']).sort_values(by='Vol_TradingCost', ascending=True)
                    finance_cap['Vol_TradingCost_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                    finance_cap['Vol_Score'] = finance_cap['Vol_TradingCost_Score']

                # if vol_factor == 'Simple':
                #     bgn = date_buy - relativedelta(months=6)
                #     end = date_buy
                #     change_stocks = self.change[finance_cap.stock_code]
                #     yield_stocks = change_stocks.loc[self.date_str_bar(bgn):self.date_str_bar(end), :]
                #
                #     # print('yield_stocks\n',yield_stocks)
                #     # yield_index = self.kospi_index.loc[self.date_str_bar(bgn):self.date_str_bar(end), ['Change']]
                #     # for stock_code in yield_stocks.columns:
                #     #     yield_stocks[stock_code] = yield_stocks[stock_code] - yield_index['Change']
                #     # print('yield_stocks\n', yield_stocks)
                #     finance_cap['Vol_Simple'] = yield_stocks.T.std(axis=1).values
                #     # print('finance_cap\n',finance_cap)
                #     finance_cap = finance_cap.dropna(subset=['Vol_Simple']).sort_values(by='Vol_Simple', ascending=True)
                #     finance_cap['Vol_Simple_Score'] = np.arange(1, len(finance_cap.index) + 1) / len(finance_cap.index)
                #     finance_cap['Vol_Score'] = finance_cap['Vol_Simple_Score']
        else:
            finance_cap['Vol_Score'] = 0

        finance_cap['Total_Score'] = finance_cap['Value_Score'] + finance_cap['Quality_Score'] + finance_cap['Profit_Momentum_Score'] + finance_cap['Momentum_Score'] + finance_cap['Size_Score'] + finance_cap['Vol_Score']
        finance_cap.sort_values(by='Total_Score', ascending=True, inplace=True)

        # 불필요한 column 삭제
        del_columns_list = ['총자산', '현금', '부채', '지배자산', '매출액', '매출원가', '매출총이익', '판관비',
                            '영업이익', '당기순이익', '지배순이익', '영업활동현금흐름',
                            '시가총액', 'PBR', 'PER', 'PCR', 'PSR', 'GP/A', 'Momentum_Discrete', 'Vol_Simple',
                            '상장주식수_first', '상장주식수_last', 'Fscore_1', 'Fscore_2', 'Fscore_3']
        for column in del_columns_list:
            if column in finance_cap.columns:
                finance_cap.drop(columns=[column], inplace=True)

        return finance_cap

    def define_period(self, date: datetime):
        year = date.year
        month = date.month
        if (month == 4) or (month == 5):
            set_period = str(year - 1) + str(12)
        elif (month == 1) or (month == 2) or (month == 3):
            set_period = str(year - 1) + str('09')
        elif (month == 6) or (month == 7) or (month == 8):
            set_period = str(year) + str('03')
        elif (month == 9) or (month == 10) or (month == 11):
            set_period = str(year) + str('06')
        elif (month == 12):
            set_period = str(year) + str('09')
        return int(set_period)

    # 종목별 비중 계산

    # momentum_score_active (모멘텀 스코어의 비중대로 종목 비중 결정)
    def cal_portion_by_equal(self, stocks:list, bgn:datetime):
        portions = [1 / len(stocks)] * len(stocks)

        dateTemp = str(str(bgn.year) + '%02d' % (bgn.month) + '%02d' % (bgn.day))
        stocksTemp = stocks.copy()
        portionsTemp = portions.copy()
        self.dictPortfolio[dateTemp] = {'stock': stocksTemp, 'portion': portionsTemp}
        return stocks, portions

    def cal_portion_by_stock_daily_and_momentum_score_active(self, stocks: list, bgn: datetime):
        change_stocks = self.change[stocks]
        # 평가할 수 있는 내용 추가
        momentum_score = [0] * len(stocks)
        close_stocks = self.close.loc[:self.date_str_bar(bgn), stocks]
        for i in range(12):
            price_subtraction_list = (close_stocks.iloc[-1, :] - close_stocks.iloc[-1 - 20 * (i + 1), :]).to_list()
            price_subtraction_list = [1 if i > 0 else 0 for i in price_subtraction_list]
            momentum_score = [a + b for a, b in zip(momentum_score, price_subtraction_list)]

        portions = list(np.array(momentum_score) / sum(momentum_score))
        return stocks, portions

    def cal_portion_by_momentum_score_defensive(self, stocks: list, bgn: datetime):
        # 평가할 수 있는 내용 추가
        stocks = list(stocks)
        momentum_score = [0] * len(stocks)
        close_stocks = self.close.loc[:self.date_str_bar(bgn), stocks]
        months = 12
        for i in range(months):
            price_subtraction_list = (close_stocks.iloc[-1, :] - close_stocks.iloc[-1 - 20 * (i + 1), :]).to_list()
            price_subtraction_list = [1 if i > 0 else 0 for i in price_subtraction_list]
            momentum_score = [a + b for a, b in zip(momentum_score, price_subtraction_list)]

        portions = list(np.array(momentum_score) / (months * len(stocks)))
        # cash portion 추가
        stocks.append('cash')
        portions.append(1 - sum(portions))

        dateTemp = str(str(bgn.year) + '%02d' % (bgn.month) + '%02d' % (bgn.day))
        stocksTemp = stocks.copy()
        portionsTemp = portions.copy()
        self.dictPortfolio[dateTemp] = {'stock': stocksTemp, 'portion': portionsTemp}

        return stocks, portions

    # 모멘텀이 있는 종목을 포함한 30개 종목에 대해서 mementum score 기준으로 재평가 후 매수.
    def cal_portion_by_momentum_score_defensive_and_keeping_momentum_alter(self, stocks:list, bgn:datetime, end:datetime):
        if len(self.stocks_last) == 0:
            stocks_new, portion_new = self.cal_portion_by_momentum_score_defensive(stocks=stocks, bgn=bgn, end=end)
        else:
            stocks_with_momentum = []
            for stock_code in list(self.stocks_last.keys()):
                if stock_code == 'cash':
                    continue
                close_stocks = self.close.loc[:self.date_str_bar(bgn), [stock_code]]
                last_close_price = close_stocks[stock_code][-21]
                if not math.isnan(last_close_price):  # 20일 전 가격이 없으면 제외
                    if close_stocks[stock_code][-1] > last_close_price:
                        # 최소값을 더 작게 잡아야 (왜냐면 1/len이 개별 종목이 가질 수 있는 max이기때문에)
                        if self.stocks_last[stock_code] > 1 / (len(stocks) * 2):  # 너무 적게 있으면 유지하는 것보다 리밸런싱할때 추가로 기본만큼 매수하는게 낫다.
                            stocks_with_momentum.append(stock_code)
            stocks_no_new = len(stocks) - len(stocks_with_momentum)
            for stockCode in stocks_with_momentum:
                if stockCode in stocks:
                    stocks.remove(stockCode)
            stocks_new = stocks[:stocks_no_new] + stocks_with_momentum
            stocks_new, portion_new = self.cal_portion_by_momentum_score_defensive(stocks=stocks_new, bgn=bgn)

        dateTemp = str(str(bgn.year) + '%02d' % (bgn.month) + '%02d' % (bgn.day))
        stocksTemp = stocks_new.copy()
        portionsTemp = portion_new.copy()
        self.dictPortfolio[dateTemp] = {'stock': stocksTemp, 'portion': portionsTemp}
        return stocks_new, portion_new

    # exit : 기존에 보유량 중 20거래일 전보다 가격이 높고, 매수 비중이 종목별 최대비중의 50% 이상일 때 그대로 보유. 나머지는 신규종목 재매수.
    def cal_portion_by_momentum_score_defensive_and_keeping_momentum(self, stocks: list, bgn: datetime, end: datetime):
        # stock_last와 stock_last를 비교해서 동일하면 그대로 memontum_score_defensive 실시
        if len(self.stocks_last) == 0:
            print("new")
            stocks_new, portion_new = self.cal_portion_by_momentum_score_defensive(stocks=stocks, bgn=bgn)
        else:
            print("old")
            # 지난번 매수한 종목 중 현재 수익률이 + 라면 유지.
            stocks_with_momentum = []
            for stock_code in list(self.stocks_last.keys()):
                if stock_code == 'cash':
                    continue
                close_stocks = self.close.loc[:self.date_str_bar(bgn), [stock_code]]
                last_close_price = close_stocks[stock_code][-21]
                if not math.isnan(last_close_price):  # 20일 전 가격이 없으면 제외
                    if close_stocks[stock_code][-1] > last_close_price:
                        # 최소값을 더 작게 잡아야 (왜냐면 1/len이 개별 종목이 가질 수 있는 max이기때문에)
                        if self.stocks_last[stock_code] > 1 / (len(stocks) * 2):  # 너무 적게 있으면 유지하는 것보다 리밸런싱할때 추가로 기본만큼 매수하는게 낫다.
                            stocks_with_momentum.append(stock_code)
            print(bgn, "모멘텀 종목 수 : ", len(stocks_with_momentum))
            # defensive 전략이기때문에, 각 종목의 최대비중은 결정되어 있음. 때문에 30 - stocks_with_momentum 만큼 계산하고 나머지는 현금이 적절
            stocks_no_new = len(stocks) - len(stocks_with_momentum)
            portion_new_all = stocks_no_new / len(stocks)
            for stockCode in stocks_with_momentum:
                if stockCode in stocks:
                    stocks.remove(stockCode)
            # with_momentum을 포함하여 30종목으로 유지 (new는 모멘텀스코어계산해서 나온대로 비중하면 될듯)
            stocks_new = stocks[:stocks_no_new]
            stocks_new, portion_new = self.cal_portion_by_momentum_score_defensive(stocks=stocks_new, bgn=bgn)
            portion_new = list(np.array(portion_new) * portion_new_all)
            for stockCode in stocks_with_momentum:
                stocks_new.append(stockCode)
                portion_new.append(self.stocks_last[stockCode] / sum(self.stocks_last.values()))
            # stocks_with_momentum 종목에서 매수안한 현금비중도 cash에 합치기
            portion_new.pop(stocks_new.index('cash'))
            stocks_new.remove('cash')
            stocks_new.append('cash')
            # portionCash = sum(self.stocks_last.values()) - sum(portion_new)
            portion_new.append(1 - sum(portion_new))
            # portion_new = list(np.array(portion_new) / sum(portion_new))

        dateTemp = str(str(bgn.year) + '%02d' % (bgn.month) + '%02d' % (bgn.day))
        stocksTemp = stocks_new.copy()
        portionsTemp = portion_new.copy()
        self.dictPortfolio[dateTemp] = {'stock': stocksTemp, 'portion': portionsTemp}

        return stocks_new, portion_new

    # 시장의 흐름에 맞춰서 하락추세일 경우 매수 비중 감소 or 현금화. 상승 추세일경우 active. 상승 추세에 변동성이 커질 경우 defensive.
    # 이 방식이 적절한지 평가하기 위해서는 active, defensive의 각 기간 별 수익률을 펼쳐놓고,
    # 각 기간을 정의할 수 있는 방법을 고민해야함. -> 선형회귀에서 기울기와 r2값
    def calculateRegressedReturn(self, listYield : pd.Series):
        # 1년간의 cumprod() 값을 활용하여, x축은 index, y축은 수익률
        lr = LinearRegression()
        axisX = np.arange(len(listYield))/252
        axisX = axisX.reshape(-1,1)
        axisY = np.array(listYield.cumprod())
        axisY = axisY.reshape(-1,1)
        lr.fit(axisX, axisY)
        rar = lr.coef_ # Regressed Annual Return
        # rSquare = lr.score(axisX,axisY)
        return rar[0][0]

    # def calculatePortion

    # 대상 기간 가격 정보
    def cal_yield_by_stock_daily_and_portion (self, stocks : list, portions : list, bgn : datetime, end : datetime):
        logSheetTemp = {}
        logSheetTemp['Stock'] = stocks
        logSheetTemp['Portion'] = portions

        if 'cash' in stocks:
            cashPortion = portions.pop(stocks.index('cash'))
            stocks.remove('cash')
            # change_stocks = self.change[stocks]
            change_stocks = self.change[stocks].loc[self.date_str_bar(bgn):self.date_str_bar(end),:]
            yield_stocks = (change_stocks+1).fillna(0).cumprod()
            # logSheetTemp['Yield'] = yield_stocks
            for i in range(len(yield_stocks.columns)):
                yield_stocks[yield_stocks.columns[i]] = yield_stocks[yield_stocks.columns[i]]*portions[i]
            yield_stocks['cash'] = cashPortion
        else:
            # change_stocks = self.change[stocks]
            change_stocks = self.change[stocks].loc[self.date_str_bar(bgn):self.date_str_bar(end), :]
            yield_stocks = (change_stocks+1).fillna(0).cumprod()
            # logSheetTemp['Yield'] = yield_stocks
            for i in range(len(yield_stocks.columns)):
                yield_stocks[yield_stocks.columns[i]] = yield_stocks[yield_stocks.columns[i]]*portions[i]
            cashPortion = 0

        logSheetTemp['Change'] = change_stocks
        logSheetTemp['delisting'] = sum([0 if i == 0 else 1 for i in change_stocks.isnull().sum()])
        self.logSheets[self.date_str(bgn)[:6]] = logSheetTemp.copy()

        yield_sum = yield_stocks.sum(axis=1)
        yield_list_temp = []
        yield_list_temp.append(yield_sum.values[0] * (1-self.tradingLoss*(1-cashPortion)/100))
        for i in range(len(yield_sum.values)-1):
            yield_list_temp.append(yield_sum.values[i+1] / yield_sum.values[i])
        return pd.DataFrame(data=yield_list_temp,index=yield_sum.index), yield_stocks.to_dict('records')[-1]

    def cal_yield_by_stock_rebal (self, stocks : list, bgn : datetime, duration : int, rebalance : bool = True, stock_weight : str = 'equal'): # relancing안하는 기준도 함수 추가.
        df_yield = pd.DataFrame()
        if rebalance is True:
            for month in range(duration):
                bgn_temp = bgn + relativedelta(months = month)
                end_temp = bgn + relativedelta(months = (month+1))
                if end_temp > datetime.today():
                    continue
                if stock_weight == 'equal':
                    stocks, portions = self.cal_portion_by_equal(stocks=stocks, bgn=bgn_temp)
                    df_yield_partial, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks, portions=portions, bgn=bgn_temp, end=end_temp)
                elif stock_weight == 'ma': # 미완성 사용금지
                    df_yield_partial = self.cal_yield_by_stock_daily_and_sell_by_stock_ma(stocks=stocks, bgn=bgn_temp, end=end_temp)
                elif stock_weight == "momentum_score_active":
                    stocks, portions = self.cal_portion_by_stock_daily_and_momentum_score_active(stocks=stocks, bgn=bgn_temp)
                    df_yield_partial, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks, portions=portions, bgn=bgn_temp, end=end_temp)
                elif stock_weight == "momentum_score_defensive":
                    stocks, portions = self.cal_portion_by_momentum_score_defensive(stocks=stocks,bgn=bgn_temp)
                    df_yield_partial, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks,portions=portions,bgn=bgn_temp,end=end_temp)
                elif stock_weight == "keeping_momentum":
                    stocks, portions = self.cal_portion_by_momentum_score_defensive_and_keeping_momentum(stocks=stocks,bgn=bgn_temp,end=end_temp)
                    df_yield_partial, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks,portions=portions,bgn=bgn_temp,end=end_temp)
                df_yield = pd.concat([df_yield,df_yield_partial],axis=0)
        else:
            if bgn.month == 4:
                period = 2
            elif bgn.month == 6:
                period = 3
            elif bgn.month == 9:
                period = 3
            elif bgn.month == 12:
                period = 4
            else:
                print("check bgn")
            bgn_temp = bgn
            end_temp = bgn + relativedelta(months=period)
            if stock_weight == 'equal':
                stocks, portions = self.cal_portion_by_equal(stocks=stocks, bgn=bgn_temp)
                df_yield, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks, portions=portions, bgn=bgn_temp, end=end_temp)
            elif stock_weight == 'ma':  # 미완성 사용금지
                df_yield = self.cal_yield_by_stock_daily_and_sell_by_stock_ma(stocks=stocks, bgn=bgn_temp, end=end_temp)
            elif stock_weight == "momentum_score_active":
                stocks, portions = self.cal_portion_by_stock_daily_and_momentum_score_active(stocks=stocks, bgn=bgn_temp)
                df_yield, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks, portions=portions, bgn=bgn_temp, end=end_temp)
            elif stock_weight == "momentum_score_defensive":
                stocks, portions = self.cal_portion_by_momentum_score_defensive(stocks=stocks, bgn=bgn_temp)
                df_yield, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks, portions=portions, bgn=bgn_temp, end=end_temp)
            elif stock_weight == "keeping_momentum":
                stocks, portions = self.cal_portion_by_momentum_score_defensive_and_keeping_momentum(stocks=stocks, bgn=bgn_temp, end=end_temp)
                df_yield, self.stocks_last = self.cal_yield_by_stock_daily_and_portion(stocks=stocks, portions=portions, bgn=bgn_temp, end=end_temp)
        df_yield = df_yield.groupby(df_yield.index).last()
        return df_yield

    def cal_yield_by_kospi (self, bgn : datetime, duration : int):
        end = bgn + relativedelta(months = duration)
        df_yield = fdr.DataReader('KS11',start=bgn,end=end)
        df_yield = df_yield.loc[:,['Change']]+1
        return df_yield

    def cal_change_by_kospi(self, bgn: datetime, end: datetime):
        df_change = fdr.DataReader('KS11', start=bgn, end=end)
        df_change = df_change.loc[:, ['Close']]
        return df_change

    def cal_change_by_kosdaq(self, bgn: datetime, end: datetime):
        df_change = fdr.DataReader('KQ11', start=bgn, end=end)
        df_change = df_change.loc[:, ['Close']]
        return df_change

    # 기간 정보 계산
    def cal_yield_for_periods(self,
                              years_list : list,
                              finance: pd.DataFrame,  #
                              size_target: str,  # high20, low20
                              quarter_data: bool = False,  # 분기 보고서 활용할지 or 사업보고서만 활용할지
                              comparison_with_kopsi: bool = True,  # KOSPI 가격 정보 추가 or 제외
                              partial_stocks: bool = True,  # 분할 종목 선정 or 상위 종목 선정
                              fscore: bool = False,
                              stock_weight: str = 'equal',
                              # 매수 비중 디테일 (equal : 동일비중매수, ma : 직전 60이평선대비 아래있으면 현금, momentum_score : 각 종목의 momentum_score를 계산해 비중설정. )
                              rebalance : bool = True,
                              value_factors: list = ['PBR', 'PER', 'PCR'],  # PBR, PER, PCR
                              quality_factors: list = ['GP/A','Volatility'],  # GP/A
                              profit_momentum_factors: list = ['OPQ', 'OPY', 'NPQ', 'NPY'],
                              momentum_factors: list = ['Discrete'],  # 일반 momentum, discrete momentum
                              size_factors: list = ['Cap'],  # 저가주
                              volatility_factors: list = ['Simple']):  # 변동성 factor
        df_yield = pd.DataFrame()
        years = years_list
        if quarter_data:
            months = [4, 6, 9, 12]
        else:
            months = [4]

        #  사업보고서 : +90일, 반기보고서/분기보고서 : +60일(연결, 변도 : +45일)
        #  사업보고서 : 4/1, 1분기 : 5/15, 반기 : 8/15, 3분기 : 11/15
        for year in years:
            for month in months:
                date_buy = datetime(year, month, 15)
                if quarter_data:
                    month_index = months.index(month)
                    if month_index == 3:
                        duration = 4
                    else:
                        duration = months[month_index + 1] - months[month_index]
                else:
                    duration = 12

                # print(date_buy, duration)
                df = self.make_stock_list_by_finance(date_buy=date_buy, fs=finance, fscore=fscore,
                                                     size_target = size_target,
                                                value_factors=value_factors,
                                                quality_factors=quality_factors,
                                                     profit_momentum_factors = profit_momentum_factors,
                                                momentum_factors=momentum_factors,
                                                size_factors=size_factors,
                                                volatility_factors=volatility_factors)

                df_yield_duration = pd.DataFrame()

                if partial_stocks:
                    # 분할 종목 선정
                    len_all = len(df.stock_code)
                    no_of_partial = self.noOfPartial
                    for i in range(no_of_partial):
                        stock_list = list(df.stock_code[int(len_all * i / no_of_partial):int(len_all * (i + 1) / no_of_partial)])
                        # stock_list = [x for x in stock_list if x[0] != '9']
                        df_yield_temp = self.cal_yield_by_stock_rebal(stocks=stock_list, bgn=date_buy, duration=duration, rebalance=rebalance,
                                                                 stock_weight=stock_weight)  # duration : month기준
                        df_yield_duration = pd.concat([df_yield_duration, df_yield_temp], axis=1)
                    column_name = list(np.arange(1, no_of_partial + 1))
                    column_name[0] = 'Buy'
                    column_name[-1] = 'Sell'
                    df_yield_duration.columns = column_name
                else:
                    # 상위 종목 선정
                    no_of_stocks = self.noOfStocks
                    if fscore:
                        df = df[df['Fscore'] == 3]
                    stock_list = [x for x in df.stock_code if x[0] != '9'][:no_of_stocks]
                    df_yield_temp = self.cal_yield_by_stock_rebal(stocks=stock_list, bgn=date_buy,
                                                             duration=duration, rebalance=rebalance,
                                                             stock_weight=stock_weight)  # duration : month기준
                    df_yield_duration = pd.concat([df_yield_duration, df_yield_temp], axis=1)
                    df_yield_duration.columns = ['Buy']

                if comparison_with_kopsi:
                    #                 비교를 위한 KOSPI 지수 추가
                    df_yield_kospi = self.cal_yield_by_kospi(bgn=date_buy, duration=duration)
                    df_yield_kospi.columns = ['KOSPI']
                    # df_yield_kospi.index = [self.date_str_bar(date) for date in df_yield_kospi.index]
                    df_yield_duration = pd.merge(df_yield_duration, df_yield_kospi, left_index=True, right_index=True,
                                                 how='left')

                df_yield = pd.concat([df_yield, df_yield_duration], axis=0)
        # 리밸런싱하는 시점은 이전과 이후 두번이 계산되기 때문에, df_yield의 index에서 중복으로 입력되어있다.
        # 리벌랜싱 이후만을 사용하기 위해 중복 중에 마지막 것만 사용한다.
        df_yield = df_yield.groupby(df_yield.index).last()
        return df_yield

    # 테스트트
    def caseStudy(self):
        case = self.caseFactors
        self.dfFinance
        df_yield = self.cal_yield_for_periods(years_list=self.listYears, partial_stocks=case['partial'], stock_weight=self.methodStockWeight,
                                              comparison_with_kopsi=case['kospi'],
                                              size_target=case['sizeTarget'], rebalance=case['rebalance'], fscore=case['fscore'],
                                         quarter_data=case['quarter'], value_factors=case['val'],
                                         quality_factors=case['qual'], profit_momentum_factors=case['profitMom'],
                                         momentum_factors=case['mom'], size_factors=case['size'],
                                         volatility_factors=case['vol'], finance=self.dfFinance
                                         )
        self.dfCaseStudyResult = df_yield.copy()
        self.caseStudyResult['variables'] = {'listYears': self.listYears, 'caseFactors': self.caseFactors, 'methodStockWeight': self.methodStockWeight,
                                             'noOfStocks':self.noOfStocks, 'noOfPartial':self.noOfPartial,
                                             'exclusiveStock': self.exclusiveStock, 'tradingLoss': self.tradingLoss,
                                             'limitOfMarketCapitalization': self.limitOfMarketCapitalization, 'limitOfTradingPrice': self.limitOfTradingPrice,
                                             'limitOfPBR': self.limitOfPBR, 'limitOfPER': self.limitOfPER, 'limitOfPCR': self.limitOfPCR, 'limitOfPSR': self.limitOfPSR}
        self.caseStudyResult['result'] = self.dfCaseStudyResult.copy()
        self.caseStudyResult['log'] = self.logSheets.copy()
        # return self.dictCaseStudy

    def applyTraidingStrategy(self, strategy : str, data : pd.DataFrame = pd.DataFrame(), daysForMABuy : int = 20, daysForMASell : int = 10):
        self.noOfTrade = 0
        if len(data) == 0:
            data = self.dfCaseStudyResult.copy()
        if not isinstance(data.index[0], datetime):
            data.index = [self.date_to_date(date) for date in data.index]

        if strategy == "MAKospi":
            bgn = data.index[0]
            end = data.index[-1]
            dfChangeKospi = self.cal_change_by_kospi(bgn = bgn-relativedelta(days=daysForMABuy), end=end)
            if datetime(2013,12,31) in dfChangeKospi.index:  # 특정일이 거래일이 아님에도 kospi에 들어가있음.
                dfChangeKospi.drop(datetime(2013,12,31), inplace=True)
            movingAverageKospiBuy = dfChangeKospi['Close'].rolling(daysForMABuy).mean()
            movingAverageKospiSell = dfChangeKospi['Close'].rolling(daysForMASell).mean()
            listForMASignalBuy = [0 if (aboveMA < 0) else 1 for aboveMA in (dfChangeKospi['Close'][bgn:end] - movingAverageKospiBuy[bgn:end])]
            listForMASignalSell = [0 if (aboveMA < 0) else 1 for aboveMA in (dfChangeKospi['Close'][bgn:end] - movingAverageKospiSell[bgn:end])]
            # case1
            # listForMASignal = [signalBuy*signalSell for signalBuy, signalSell in zip(listForMASignalBuy, listForMASignalSell)]
            # case2 (buyMA < 현재가 < sellMA 인 경우는 이전의 상태에 따라 분리해서 고려필요)
            listForMASignal = [listForMASignalBuy[0]*listForMASignalSell[0]]  # 계산이 복잡해지니 처음엔 곱으로 설정
            for i in range(len(listForMASignalBuy)-1):
                if (listForMASignalBuy[i+1]==1) & (listForMASignalSell[i+1]==0):  # 특별히 고려해야 하는 경우
                    if not (listForMASignalBuy[i] == listForMASignalBuy[i+1]) & (listForMASignalSell[i] == listForMASignalSell[i+1]):  # 둘 중 하나라도 이전과 다르면
                        if listForMASignalBuy[i] * listForMASignalSell[i]:
                            listForMASignal.append(0)
                        else:
                            listForMASignal.append(1)
                    else:  # 이전과 동일하다면 유지
                        listForMASignal.append(listForMASignal[i])
                else:
                    listForMASignal.append(listForMASignalBuy[i+1]*listForMASignalSell[i+1])

            listForMASignal = listForMASignal[0:1] + listForMASignal[:-1]  # 매수/매도 신호가 있다고 하더라도 종가기준이라 다음날 거래하는게 현실적이기때문에 한자리씩 뒤로 미루기.
            listForTradingLoss = [1]
            for i in range(len(listForMASignal)-1):
                if listForMASignal[i] != listForMASignal[i+1]:
                    listForTradingLoss.append(1-self.tradingLoss/100)
                else:
                    listForTradingLoss.append(1)

            for i in range(len(listForMASignal)-1):
                if listForMASignal[i] != listForMASignal[i+1]:
                    self.noOfTrade += 1
            print(f"거래횟수 : {self.noOfTrade}")

            for code in data.columns:
                listForMATrading = [change*loss if ma == 1 else loss for change, ma, loss in zip(data[code],listForMASignal,listForTradingLoss)]
                data[code+'MA'] = listForMATrading
            data['MASignal'] = listForMASignal

        elif strategy == "MAKosdaq":
            bgn = data.index[0]
            end = data.index[-1]
            dfChangeKosdaq = self.cal_change_by_kosdaq(bgn = bgn-relativedelta(days=max(daysForMABuy,daysForMASell)), end=end)
            if datetime(2013,12,31) in dfChangeKosdaq.index:  # 특정일이 거래일이 아님에도 kospi에 들어가있음.
                dfChangeKosdaq.drop(datetime(2013,12,31), inplace=True)
            if datetime(2014,9,8) in dfChangeKosdaq.index:  # 특정일이 거래일이 아님에도 kospi에 들어가있음.
                dfChangeKosdaq.drop(datetime(2014,9,8), inplace=True)
            movingAverageKosdaqBuy = dfChangeKosdaq['Close'].rolling(daysForMABuy).mean()
            movingAverageKosdaqSell = dfChangeKosdaq['Close'].rolling(daysForMASell).mean()
            listForMASignalBuy = [0 if (aboveMA < 0) else 1 for aboveMA in (dfChangeKosdaq['Close'][bgn:end] - movingAverageKosdaqBuy[bgn:end])]
            listForMASignalSell = [0 if (aboveMA < 0) else 1 for aboveMA in (dfChangeKosdaq['Close'][bgn:end] - movingAverageKosdaqSell[bgn:end])]
            # case1
            # listForMASignal = [signalBuy*signalSell for signalBuy, signalSell in zip(listForMASignalBuy, listForMASignalSell)]
            # case2 (buyMA < 현재가 < sellMA 인 경우는 이전의 상태에 따라 분리해서 고려필요)
            listForMASignal = [listForMASignalBuy[0] * listForMASignalSell[0]]  # 계산이 복잡해지니 처음엔 곱으로 설정
            for i in range(len(listForMASignalBuy) - 1):
                if (listForMASignalBuy[i + 1] == 1) & (listForMASignalSell[i + 1] == 0):  # 특별히 고려해야 하는 경우
                    if not (listForMASignalBuy[i] == listForMASignalBuy[i + 1]) & (listForMASignalSell[i] == listForMASignalSell[i + 1]):  # 둘 중 하나라도 이전과 다르면
                        if listForMASignalBuy[i] * listForMASignalSell[i]:
                            listForMASignal.append(0)
                        else:
                            listForMASignal.append(1)
                    else:  # 이전과 동일하다면 유지
                        listForMASignal.append(listForMASignal[i])
                else:
                    listForMASignal.append(listForMASignalBuy[i + 1] * listForMASignalSell[i + 1])

            listForMASignal = listForMASignal[0:1] + listForMASignal[:-1]  # 매수/매도 신호가 있다고 하더라도 종가기준이라 다음날 거래하는게 현실적이기때문에 한자리씩 뒤로 미루기.
            listForTradingLoss = [1]
            for i in range(len(listForMASignal) - 1):
                if listForMASignal[i] != listForMASignal[i + 1]:
                    listForTradingLoss.append(1 - self.tradingLoss / 100)
                else:
                    listForTradingLoss.append(1)

            for i in range(len(listForMASignal)-1):
                if listForMASignal[i] != listForMASignal[i+1]:
                    self.noOfTrade += 1
            print(f"거래횟수 : {self.noOfTrade}")

            for code in data.columns:
                listForMATrading = [change*loss if ma == 1 else loss for change, ma, loss in zip(data[code],listForMASignal,listForTradingLoss)]
                data[code+'MA'] = listForMATrading
            data['MASignal'] = listForMASignal

        elif strategy == "MAKosdaqContinuous":
            bgn = data.index[0]
            end = data.index[-1]
            dfChangeKosdaq = self.cal_change_by_kosdaq(bgn=bgn - relativedelta(days=max(daysForMABuy, daysForMASell)), end=end)
            if datetime(2013, 12, 31) in dfChangeKosdaq.index:  # 특정일이 거래일이 아님에도 kospi에 들어가있음.
                dfChangeKosdaq.drop(datetime(2013, 12, 31), inplace=True)
            if datetime(2014, 9, 8) in dfChangeKosdaq.index:  # 특정일이 거래일이 아님에도 kospi에 들어가있음.
                dfChangeKosdaq.drop(datetime(2014, 9, 8), inplace=True)
            movingAverageKosdaqBuy = dfChangeKosdaq['Close'].rolling(daysForMABuy).mean()
            movingAverageKosdaqSell = dfChangeKosdaq['Close'].rolling(daysForMASell).mean()
            listForMASignalBuy = [0 if (aboveMA < 0) else 1 for aboveMA in (dfChangeKosdaq['Close'][bgn:end] - movingAverageKosdaqBuy[bgn:end])]
            listForMASignalSell = [0 if (aboveMA < 0) else 1 for aboveMA in (dfChangeKosdaq['Close'][bgn:end] - movingAverageKosdaqSell[bgn:end])]
            # case1
            # listForMASignal = [signalBuy*signalSell for signalBuy, signalSell in zip(listForMASignalBuy, listForMASignalSell)]
            # case2 (buyMA < 현재가 < sellMA 인 경우는 이전의 상태에 따라 분리해서 고려필요)
            listForMASignal = [listForMASignalBuy[0] * listForMASignalSell[0]]  # 계산이 복잡해지니 처음엔 곱으로 설정
            for i in range(len(listForMASignalBuy) - 1):
                if (listForMASignalBuy[i + 1] == 1) & (listForMASignalSell[i + 1] == 0):  # 특별히 고려해야 하는 경우
                    if not (listForMASignalBuy[i] == listForMASignalBuy[i + 1]) & (listForMASignalSell[i] == listForMASignalSell[i + 1]):  # 둘 중 하나라도 이전과 다르면
                        if listForMASignalBuy[i] * listForMASignalSell[i]:
                            listForMASignal.append(0)
                        else:
                            listForMASignal.append(1)
                    else:  # 이전과 동일하다면 유지
                        listForMASignal.append(listForMASignal[i])
                else:
                    listForMASignal.append(listForMASignalBuy[i + 1] * listForMASignalSell[i + 1])

            listForMASignal = listForMASignal[0:1] + listForMASignal[:-1]  # 매수/매도 신호가 있다고 하더라도 종가기준이라 다음날 거래하는게 현실적이기때문에 한자리씩 뒤로 미루기.

            if self.daysContinuous == None:
                daysContinuous = int(input("다음 일자로 매도/매수가 연속될 때만 거래 실시 : "))
            else:
                daysContinuous = self.daysContinuous

            listForMASignalActual = listForMASignal[:daysContinuous]
            for i in range(len(listForMASignal) - daysContinuous):
                if listForMASignal[i + daysContinuous] == listForMASignalActual[i + daysContinuous-1]:
                    listForMASignalActual.append(listForMASignalActual[i + daysContinuous-1])
                else:
                    if all([listForMASignalActual[i] == listForMASignalActual[i+k] for k in range(daysContinuous)]):
                    # if (listForMASignal[i + 2] == listForMASignal[i + 1]) & (listForMASignal[i + 1] == listForMASignal[i]):
                        listForMASignalActual.append(listForMASignal[i + daysContinuous])
                    else:
                        listForMASignalActual.append(listForMASignalActual[i + daysContinuous-1])
            listForMASignal = listForMASignalActual.copy()
            listForTradingLoss = [1]
            for i in range(len(listForMASignal) - 1):
                if listForMASignal[i] != listForMASignal[i + 1]:
                    listForTradingLoss.append(1 - self.tradingLoss / 100)
                else:
                    listForTradingLoss.append(1)

            for i in range(len(listForMASignal) - 1):
                if listForMASignal[i] != listForMASignal[i + 1]:
                    self.noOfTrade += 1
            print(f"거래횟수 : {self.noOfTrade}")

            for code in data.columns:
                listForMATrading = [change*loss if ma == 1 else loss for change, ma, loss in zip(data[code],listForMASignal,listForTradingLoss)]
                data[code + 'MA'] = listForMATrading
            data['MASignal'] = listForMASignal

        return data

    def makeTradePlanByNaverFinance(self,daysForMABuy : int = 30, daysForMASell : int = 20):
        # self.dateBuy = datetime.now()
        print(f"매수일 기준 : {self.dateBuy}\n재무제표 기준 : {self.periodBuy}")

        listStockBuy = self.make_stock_list_by_finance(fs=self.finance_dartapi, date_buy=self.dateBuy, period=self.periodBuy,
                                                       size_target=self.sizeTarget,
                                                       value_factors=self.caseFactors['val'],
                                                       quality_factors=self.caseFactors['qual'],
                                                       profit_momentum_factors=self.caseFactors['profitMom'],
                                                       momentum_factors=self.caseFactors['mom'],
                                                       size_factors=self.caseFactors['size'],
                                                       volatility_factors=self.caseFactors['vol']).head(50)
        if 'Fscore' in listStockBuy.columns:
            listStockBuy = listStockBuy[listStockBuy['Fscore'] == 3]['stock_code']
        else:
            listStockBuy = listStockBuy['stock_code']
        listStockBuy = [x for x in listStockBuy if x[0]!='9'][:30]  # 중국기업은 기업코드의 첫자리가 9로 시작해서 제외. (재무제표가 불투명)
        if self.methodStockWeight == 'equal':
            listStockBuy, listStockPortion = self.cal_portion_by_equal(stocks=listStockBuy, bgn=self.dateBuy)
        elif self.methodStockWeight == "momentum_score_active":
            listStockBuy, listStockPortion = self.cal_portion_by_stock_daily_and_momentum_score_active(stocks=listStockBuy, bgn=self.dateBuy)
        elif self.methodStockWeight == "momentum_score_defensive":
            listStockBuy, listStockPortion = self.cal_portion_by_momentum_score_defensive(stocks=listStockBuy, bgn=self.dateBuy)
        # print(listStockBuy, listStockPortion)
        totalBuy = 0
        for i in range(len(listStockBuy)):
            stockCode = listStockBuy[i]
            if stockCode == "cash":
                continue
            price = fdr.DataReader(stockCode,start=self.date_str_bar(self.dateBuy-relativedelta(days=10)),end=self.date_str_bar(self.dateBuy)).iat[-1,3]
            assetOfEachStock = self.totalAsset * listStockPortion[i]
            quantityOfEachStock = math.ceil(assetOfEachStock/price)
            totalBuy += price * quantityOfEachStock
            self.dictPortfolioTradePlan[stockCode] = {'수량' : quantityOfEachStock}
        print("총 매수 예정 금액 : ",totalBuy)
        print(self.dictPortfolioTradePlan)
        # 코스피지수의 이평선
        bgn = self.dateBuy-relativedelta(days=max(daysForMABuy,daysForMASell)+40)
        end = self.dateBuy
        dfChangeKosdaq = self.cal_change_by_kosdaq(bgn=bgn, end=end)
        movingAverageKosdaqBuy = dfChangeKosdaq['Close'].rolling(daysForMABuy).mean()
        movingAverageKosdaqSell = dfChangeKosdaq['Close'].rolling(daysForMASell).mean()

        price = dfChangeKosdaq['Close'][-1]
        maBuy = movingAverageKosdaqBuy[-1]
        maSell = movingAverageKosdaqSell[-1]
        print(f"최근 기간의 {daysForMABuy}일 이평선 매수 신호 : {['Y' if priceDay>maDay else 'N' for priceDay, maDay in zip(dfChangeKosdaq['Close'][-15:],movingAverageKosdaqBuy[-15:])]}")
        print(f"최근 기간의 {daysForMASell}일 이평선 매도 신호 : {['Y' if priceDay > maDay else 'N' for priceDay, maDay in zip(dfChangeKosdaq['Close'][-15:], movingAverageKosdaqSell[-15:])]}")
        if price > maBuy:  # buy, sell 지표 모두 20일 기준이기때문에 maBuy 만 활용.
            print(f"지수 {daysForMABuy}일 이평선 돌파 ( 가격 : {price}, 이평선 : {maBuy}, 비율 : {(price-maBuy)/maBuy})")
        else:
            print(f"지수 이평선 미돌파 ( 가격 : {price}, 이평선 : {maBuy}, 비율 : {(price-maBuy)/maBuy})")

    def outputPortfolioTradeResult(self, dateStart: datetime, dateEnd: datetime, portfolio: dict = {}):
        import matplotlib.pyplot as plt

        if len(portfolio) == 0:
            with open(f'/Users/malko/PycharmProjects/xing/TradingPlan/trade_plan_{self.periodBuy}.pickle', 'rb') as handle:
                print(f'매수기준 재무정보 : {self.periodBuy}')
                portfolio = pickle.load(handle)
        print(f'시작일 기준 : {dateStart}\n종료일 기준 : {dateEnd}')
        dfPortfolioResult = pd.DataFrame(columns=portfolio.keys())
        totalBuy = 0
        for stock_code in portfolio.keys():
            df_stock = stock.get_market_ohlcv_by_date(ticker=stock_code, fromdate=dateStart, todate=dateEnd, adjusted=False)
            change_temp = df_stock['등락률']
            priceStart = df_stock.iat[0, 0]
            totalBuy += priceStart * portfolio[stock_code]['수량']
            dfPortfolioResult[stock_code] = priceStart * portfolio[stock_code]['수량'] * ((change_temp * 0.01 + 1).cumprod())
        seriesResult = dfPortfolioResult.sum(axis=1)
        print(f'시작가 : {totalBuy}\n종가 : {int(seriesResult[-1])} ( {"{:.2%}".format(seriesResult[-1]/totalBuy - 1) } )')

        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        ax[0].plot(seriesResult)
        ax[1].plot(seriesResult / totalBuy)
        plt.show()

    def saveTradePlan(self):
        with open(f'/Users/malko/PycharmProjects/xing/TradingPlan/trade_plan_{self.periodBuy}.pickle', 'wb') as handle:
            pickle.dump(self.dictPortfolioTradePlan, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("저장 완료")

    # 결과 분석
    def simple_analyzer(self, data : pd.DataFrame = pd.DataFrame()):
        if len(data) == 0:
            data = self.dfCaseStudyResult
        analysis_result = pd.DataFrame(columns=data.columns, index=['Total Profit', 'CAGR', 'MDD', 'Std', 'Sharpe_Ratio','승률[%]'])
        trade_year = len(data) / 252

        for col in data.columns:
            if "Signal" in str(col):
                continue
            total_profit = data[col].cumprod()[-1]
            cagr = round((total_profit ** (1 / trade_year) - 1) * 100, 2)

            arr_v = np.array(data[col].cumprod())
            peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
            peak_upper = np.argmax(arr_v[:peak_lower])
            mdd = round((arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper] * 100, 3)

            std = round(data[col].std() * (252 ** 0.5) * 100, 2)

            risk_free_rate = 0
            sharpe_ratio = (cagr - risk_free_rate) / std

            countPlus = 0
            countTrade = 0
            dateStart = data.index[0]
            while True:
                countTrade += 1
                dateEnd = dateStart + relativedelta(months=1)
                dateEnd = datetime(dateEnd.year, dateEnd.month, 1) - relativedelta(days=1)
                if data.loc[dateStart:dateEnd, col].cumprod()[-1] > 1:
                    countPlus += 1
                dateStart = dateEnd + relativedelta(days=1)
                if data.index[-1] < dateStart:
                    break

            analysis_result[col] = [total_profit, cagr, mdd, std, sharpe_ratio,countPlus/countTrade]
        return analysis_result

    def simple_analyzer_yearly(self, data : pd.DataFrame = pd.DataFrame()):
        if len(data) == 0:
            data = self.dfCaseStudyResult
        if not isinstance(data.index[0], datetime):
            data.index = [self.date_to_date(date) for date in data.index]
        analyze_result_yearly = {}
        for code in data.columns:
            if "Signal" in str(code):
                continue
            year_range = range(data.index[0].year, data.index[-1].year + 1)
            analyze_result = pd.DataFrame(columns=year_range, index=['CAGR', 'MDD', 'Std', 'Sharpe_Ratio','승률[%]'])
            for year in year_range:
                data_year = data.loc[datetime(year, 1, 1):datetime(year, 12, 31), [code]]
                analyze_result[year] = self.simple_analyzer(data_year)[code][1:]
            analyze_result_yearly[code] = analyze_result
        return analyze_result_yearly

    def simple_analyzer_monthly_return(self, data : pd.DataFrame = pd.DataFrame()):
        if len(data) == 0:
            data = self.dfCaseStudyResult
        if not isinstance(data.index[0], datetime):
            data.index = [self.date_to_date(date) for date in data.index]

        year_range = range(data.index[0].year, data.index[-1].year + 1)
        month_range = range(1, 13)
        analyze_result_monthly = {}
        for code in data.columns:
            if "Signal" in str(code):
                continue
            analyze_result = pd.DataFrame(columns=month_range, index=year_range)
            for month in month_range:
                monthly_return_list = []
                for year in year_range:
                    data_month = data.loc[datetime(year, month, 1):datetime(year, month, 1) + relativedelta(months=1) - relativedelta(days=1), code]
                    if len(data_month) == 0:
                        return_month = float('nan')
                    else:
                        return_month = data_month.cumprod()[-1]
                    monthly_return_list.append(return_month)
                analyze_result[month] = monthly_return_list
            analyze_result_monthly[code] = analyze_result
        return analyze_result_monthly

    def DrawingSimpleGraph(self, data : pd.DataFrame = pd.DataFrame()):
        import matplotlib.pyplot as plt
        import matplotlib.dates as md
        if len(data) == 0:
            data = self.dfCaseStudyResult
        if not isinstance(data.index[0], datetime):
            data.index = [self.date_to_date(date) for date in data.index]
        to_show = data.columns
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for i in range(len(to_show)):
            if 'Signal' in str(to_show[i]):
                continue
            ax.plot(data.cumprod()[to_show[i]])
        ax.xaxis.set_major_locator(md.YearLocator(1, month=1, day=1))
        ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
        plt.legend(to_show)
        plt.show()




