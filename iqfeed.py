# PyAlgoTrade
#
# Feed for data from IQfeed
#
# basis: yahoofeed.py
#
from pyalgotrade.barfeed import csvfeed
from pyalgotrade.barfeed import common
from pyalgotrade.utils import dt
from pyalgotrade import bar
from pyalgotrade import dataseries

import datetime


def parse_date(date):
    ''' Parses dates from IQfeed data files. '''
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    hour = int(date[11:13])
    minute = int(date[14:16])
    second = int(date[17:19])
    ret = datetime.datetime(year, month, day, hour, minute, second)
    return ret


class RowParser(csvfeed.RowParser):
    def __init__(self, dailyBarTime, frequency, timezone=None, sanitize=False):
        self.__dailyBarTime = dailyBarTime
        self.__frequency = frequency
        self.__timezone = timezone
        self.__sanitize = sanitize

    def __parseDate(self, dateString):
        ret = parse_date(dateString)
        if self.__timezone:
            ret = dt.localize(ret, self.__timezone)
        return ret

    def getFieldNames(self):
        # it is expected for the first row to have the field names.
        return None

    def getDelimiter(self):
        return ","

    def parseBar(self, csvRowDict):
        high = float(csvRowDict["high"])
        low = float(csvRowDict["low"])
        open_ = float(csvRowDict["open"])
        close = float(csvRowDict["close"])
        volume = int(csvRowDict["volume"])
        dateTime = self.__parseDate(csvRowDict["time"])
        adjClose = None

        if self.__sanitize:
            open_, high, low, close = common.sanitize_ohlc(
                open_, high, low, close)

        return bar.BasicBar(dateTime, open_, high, low, close, volume,
                            adjClose, self.__frequency)


class Feed(csvfeed.BarFeed):
    ''' The Feed class itself. '''

    def __init__(self, frequency=bar.Frequency.DAY, timezone=None,
                 maxLen=dataseries.DEFAULT_MAX_LEN):

        if isinstance(timezone, int):
            raise Exception("timezone as an int parameter is not supported anymore.\
                             Please use a pytz timezone instead.")

        if frequency not in [bar.Frequency.DAY, bar.Frequency.WEEK]:
            raise Exception("Invalid frequency.")

        csvfeed.BarFeed.__init__(self, frequency, maxLen)
        self.__timezone = timezone
        self.__sanitizeBars = False

    def sanitizeBars(self, sanitize):
        self.__sanitizeBars = sanitize

    def barsHaveAdjClose(self):
        return True

    def addBarsFromCSV(self, instrument, path, timezone=None):
        ''' Loads bars for a given instrument from a CSV formatted file. '''

        if isinstance(timezone, int):
            raise Exception("timezone as an int parameter is not supported anymore.\
                            Please use a pytz timezone instead.")

        if timezone is None:
            timezone = self.__timezone

        rowParser = RowParser(self.getDailyBarTime(), self.getFrequency(),
                              timezone, self.__sanitizeBars)
        csvfeed.BarFeed.addBarsFromCSV(self, instrument, path, rowParser)
