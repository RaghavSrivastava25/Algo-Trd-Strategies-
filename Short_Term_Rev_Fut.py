
# The investment universe consists of 24 types of US futures contracts (4 currencies, 5 financials, 8 agricultural, 7 commodities). 
# A weekly time frame is used – a Wednesday- Wednesday interval. The contract closest to expiration is used, except within the delivery 
# month, in which the second-nearest contract is used. Rolling into the second nearest contract is done at the beginning of the delivery month.
# The contract is defined as the high- (low-) volume contract if the contract’s volume changes between period from t-1 to t and period from t-2
# to t-1 is above (below) the median volume change of all contracts (weekly trading volume is detrended by dividing the trading volume by its 
# sample mean to make the volume measure comparable across markets). All contracts are also assigned to either high-open interest (top 50% of 
# changes in open interest) or low-open interest groups (bottom 50% of changes in open interest) based on lagged changes in open interest between
# the period from t-1 to t and period from t-2 to t-1. The investor goes long (short) on futures from the high-volume, low-open interest group 
# with the lowest (greatest) returns in the previous week. The weight of each contract is proportional to the difference between the return
# of the contract over the past one week and the equal-weighted average of returns on the N (number of contracts in a group) contracts during that period.

#region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
import data_tools
#endregion

class ShortTermReversalwithFutures(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetCash(100000) 

        symbols:Dict[str, str] = {
            'CME_S1': Futures.Grains.Soybeans,
            'CME_W1': Futures.Grains.Wheat,
            'CME_BO1': Futures.Grains.SoybeanOil,
            'CME_C1': Futures.Grains.Corn,
            'CME_LC1': Futures.Meats.LiveCattle,
            'CME_FC1': Futures.Meats.FeederCattle,
            'CME_KW2': Futures.Grains.Wheat,
            'ICE_CC1': Futures.Softs.Cocoa,
            'ICE_SB1': Futures.Softs.Sugar11CME,
            
            'CME_GC1': Futures.Metals.Gold,
            'CME_SI1': Futures.Metals.Silver,
            'CME_PL1': Futures.Metals.Platinum,

            'CME_RB1': Futures.Energies.Gasoline,
            'ICE_WT1': Futures.Energies.CrudeOilWTI,
            'ICE_O1': Futures.Energies.HeatingOil,

            'CME_BP1': Futures.Currencies.GBP,
            'CME_EC1': Futures.Currencies.EUR,
            'CME_JY1': Futures.Currencies.JPY,
            'CME_SF1': Futures.Currencies.CHF,

            'CME_ES1': Futures.Indices.SP500EMini,
            'CME_TY1': Futures.Financials.Y10TreasuryNote,
            'CME_FV1': Futures.Financials.Y5TreasuryNote,
        }
                        
        self.period:int = 14

        self.futures_info:Dict = {}
        min_expiration_days:int = 2
        max_expiration_days:int = 360

        # daily close, volume and open interest data
        self.data:Dict = {}
        self.quantile:int = 2

        for qp_symbol, qc_future in symbols.items():
            # QP futures
            data:Security = self.AddData(data_tools.QuantpediaFutures, qp_symbol, Resolution.Daily)
            data.SetFeeModel(data_tools.CustomFeeModel())
            data.SetLeverage(5)
            self.data[data.Symbol] = deque(maxlen=self.period)
            
            # QC futures
            future:Future = self.AddFuture(qc_future, Resolution.Daily)
            future.SetFilter(timedelta(days=min_expiration_days), timedelta(days=max_expiration_days))
            self.futures_info[future.Symbol.Value] = data_tools.FuturesInfo(data.Symbol)

        self.recent_month:int = -1
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0.

    def find_and_update_contracts(self, futures_chain, symbol) -> None:
        near_contract:FuturesContract = None

        if symbol in futures_chain:
            contracts:List = [contract for contract in futures_chain[symbol] if contract.Expiry.date() > self.Time.date()]

            if len(contracts) >= 2:
                contracts:List = sorted(contracts, key=lambda x: x.Expiry, reverse=False)
                near_contract = contracts[0]

        self.futures_info[symbol].update_contracts(near_contract)

    def OnData(self, data: Slice) -> None:
        if data.FutureChains.Count > 0:
            for symbol, futures_info in self.futures_info.items():
                # check if near contract is expired or is not initialized
                if not futures_info.is_initialized() or \
                    (futures_info.is_initialized() and futures_info.near_contract.Expiry.date() <= self.Time.date()):
                    self.find_and_update_contracts(data.FutureChains, symbol)

        rebalance_flag:bool = False
        ret_volume_oi_data:Dict[Symbol, Tuple[float]] = {}

        # roll return calculation
        for symbol, futures_info in self.futures_info.items():
            # futures data is present in the algorithm
            if futures_info.quantpedia_future in data and data[futures_info.quantpedia_future]:
                # new month rebalance
                if self.Time.month != self.recent_month and not self.IsWarmingUp:
                    self.recent_month = self.Time.month
                    rebalance_flag = True
                
                if futures_info.is_initialized():
                    near_c:FuturesContract = futures_info.near_contract
                    if self.Securities.ContainsKey(near_c.Symbol):
                        # store daily data
                        price:float = data[futures_info.quantpedia_future].Value
                        vol:int = self.Securities[near_c.Symbol].Volume
                        oi:int = self.Securities[near_c.Symbol].OpenInterest
                        
                        if price != 0 and vol != 0 and oi != 0:
                            self.data[futures_info.quantpedia_future].append((price, vol, oi))

                    if rebalance_flag:
                        if len(self.data[futures_info.quantpedia_future]) == self.data[futures_info.quantpedia_future].maxlen:
                            # performance
                            prices:List[float] = [x[0] for x in self.data[futures_info.quantpedia_future]]
                            half:List[float] = int(len(prices)/2)
                            prices:List[float] = prices[-half:]
                            ret:float = prices[-1] / prices[0] - 1
                            
                            # volume change
                            volumes:List[int] = [x[1] for x in self.data[futures_info.quantpedia_future]]
                            volumes_t1:List[int] = volumes[-half:]
                            t1_vol_mean:float = np.mean(volumes_t1)
                            t1_vol_total:float = sum(volumes_t1) / t1_vol_mean
                            volumes_t2:List[int] = volumes[:half]
                            t2_vol_mean:float = np.mean(volumes_t2)
                            t2_vol_total:float = sum(volumes_t2) / t2_vol_mean
                            volume_weekly_diff:float = t1_vol_total - t2_vol_total
                            
                            # open interest change
                            interests:List[int] = [x[2] for x in self.data[futures_info.quantpedia_future]]
                            t1_oi:List[int] = interests[-half:]
                            t1_oi_total:float = sum(t1_oi)
                            t2_oi:List[int] = interests[:half]
                            t2_oi_total:float = sum(t2_oi)
                            oi_weekly_diff:float = t1_oi_total - t2_oi_total
                            
                            # store weekly diff data
                            ret_volume_oi_data[futures_info.quantpedia_future] = (ret, volume_weekly_diff, oi_weekly_diff)

        if rebalance_flag:
            weight:Dict[Symbol, float] = {}

            if len(ret_volume_oi_data) > self.quantile * 2:
                volume_sorted:List = sorted(ret_volume_oi_data.items(), key = lambda x: x[1][1], reverse = True)
                quantile:int = int(len(volume_sorted) / self.quantile)
                high_volume:List = [x for x in volume_sorted[:quantile]]
                
                open_interest_sorted:List = sorted(ret_volume_oi_data.items(), key = lambda x: x[1][2], reverse = True)
                quantile = int(len(open_interest_sorted) / self.quantile)
                low_oi:List = [x for x in open_interest_sorted[-quantile:]]
                
                filtered:List = [x for x in high_volume if x in low_oi]
                filtered_by_return:List = sorted(filtered, key = lambda x : x[0], reverse = True)
                quantile = int(len(filtered_by_return) / self.quantile)

                long:List[Symbol] = filtered_by_return[-quantile:]
                short:List[Symbol] = filtered_by_return[:quantile]

                if len(long + short) >= 2: 
                    # return weighting
                    diff:Dict[Symbol, float] = {}
                    avg_ret:float = np.average([x[1][0] for x in long + short])
        
                    for symbol, ret_volume_oi in long + short:
                        diff[symbol] = ret_volume_oi[0] - avg_ret
                    
                    total_diff:float = sum([abs(x[1]) for x in diff.items()])
                    long_symbols:List[Symbol] = [x[0] for x in long]
        
                    if total_diff != 0:
                        for symbol, data in long + short:
                            if symbol in long_symbols:
                                weight[symbol] = diff[symbol] / total_diff
                            else:
                                weight[symbol] = - diff[symbol] / total_diff

            # trade execution
            invested:List[Symbol] = [x.Key for x in self.Portfolio if x.Value.Invested]
            for symbol in invested:
                if symbol not in weight:
                    self.Liquidate(symbol)

            for symbol, w in weight.items():
                self.SetHoldings(symbol, w)