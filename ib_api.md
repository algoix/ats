### Starting

1. Download API from http://interactivebrokers.github.io/#
2. Install python API code /IBJts/source/pythonclient $ python3 setup.py install

        Note: The test cases, and the documentation refer to a python package called IBApi,but the actual package is called ibapi.
        Get the latest version of the gateway:
        https://www.interactivebrokers.com/en/?f=%2Fen%2Fcontrol%2Fsystemstandalone-ibGateway.php%3Fos%3Dunix
    
3. Run the gateway
    
    user: edemo
    pwd: demo123

### Position Handling 

http://interactivebrokers.github.io/tws-api/positions.html#position_receive

        self.reqPositions() # request
        class TestWrapper(wrapper.Ewrapper):
                def position(self, account: str, contract: Contract, position: float,
                                avgCost: float):
                        super().position(account, contract, position, avgCost)
                        print("Position.", account, "Symbol:", contract.symbol, "SecType:",
                        contract.secType, "Currency:", contract.currency,
                        "Position:", position, "Avg cost:", avgCost)

                def positionEnd(self):
                        super().positionEnd()
                        print("PositionEnd")

### Account Handling

https://stackoverflow.com/questions/43055012/ibpy-getting-my-market-position

        from time import sleep
        from ib.opt import Connection, message, ibConnection
        from ib.ext.Contract import Contract

        def acc_update(msg):
            global acc, expiry, exch, pExch, secType, symbol
            acc = msg.account
            exch = msg.contract.m_exchange
            pExch = msg.contract.m_primaryExch
            secType = msg.contract.m_secType
            expiry = msg.contract.m_expiry
            symbol = msg.contract.m_symbol
            return acc, expiry, exch, pExch, secType, symbol


        tws = ibConnection(port= 7497)
        tws.register(acc_update, message.position) 
        tws.connect()
        tws.reqPositions()
        sleep(0.5)
        tws.disconnect()
        print( [symbol, acc, expiry, exch, pExch, secType])
        https://stackoverflow.com/questions/29632515/getting-positions-of-your-portfolio-using-python-ibpy-library
        from ib.opt import ibConnection, message

        def acct_update(msg):
            print(msg)


        con = ibConnection(clientId=1)
        con.register(acct_update,
                     message.updateAccountValue,
                     message.updateAccountTime,
                     message.updatePortfolio)
        con.connect()
        con.reqAccountUpdates(True,'DU000000')
        https://stackoverflow.com/questions/34561626/ibpy-getting-portfolio-information-interactive-broker-python/34732223#34732223

        from __future__ import (absolute_import, division, print_function,)
        #                        unicode_literals)

        import collections
        import sys

        if sys.version_info.major == 2:
            import Queue as queue
            import itertools
            map = itertools.imap

        else:  # >= 3
            import queue


        import ib.opt
        import ib.ext.Contract


        class IbManager(object):
            def __init__(self, timeout=20, **kwargs):
                self.q = queue.Queue()
                self.timeout = 20

                self.con = ib.opt.ibConnection(**kwargs)
                self.con.registerAll(self.watcher)

                self.msgs = {
                    ib.opt.message.error: self.errors,
                    ib.opt.message.updatePortfolio: self.acct_update,
                    ib.opt.message.accountDownloadEnd: self.acct_update,
                }

                # Skip the registered ones plus noisy ones from acctUpdate
                self.skipmsgs = tuple(self.msgs.keys()) + (
                    ib.opt.message.updateAccountValue,
                    ib.opt.message.updateAccountTime)

                for msgtype, handler in self.msgs.items():
                    self.con.register(handler, msgtype)

                self.con.connect()

            def watcher(self, msg):
                if isinstance(msg, ib.opt.message.error):
                    if msg.errorCode > 2000:  # informative message
                        print('-' * 10, msg)

                elif not isinstance(msg, self.skipmsgs):
                    print('-' * 10, msg)

            def errors(self, msg):
                if msg.id is None:  # something is very wrong in the connection to tws
                    self.q.put((True, -1, 'Lost Connection to TWS'))
                elif msg.errorCode < 1000:
                    self.q.put((True, msg.errorCode, msg.errorMsg))

            def acct_update(self, msg):
                self.q.put((False, -1, msg))

            def get_account_update(self):
                self.con.reqAccountUpdates(True, 'D999999')

                portfolio = list()
                while True:
                    try:
                        err, mid, msg = self.q.get(block=True, timeout=self.timeout)
                    except queue.Empty:
                        err, mid, msg = True, -1, "Timeout receiving information"
                        break

                    if isinstance(msg, ib.opt.message.accountDownloadEnd):
                        break

                    if isinstance(msg, ib.opt.message.updatePortfolio):
                        c = msg.contract
                        ticker = '%s-%s-%s' % (c.m_symbol, c.m_secType, c.m_exchange)

                        entry = collections.OrderedDict(msg.items())

                        # Don't do this if contract object needs to be referenced later
                        entry['contract'] = ticker  # replace object with the ticker

                        portfolio.append(entry)

                # return list of contract details, followed by:
                #   last return code (False means no error / True Error)
                #   last error code or None if no error
                #   last error message or None if no error
                # last error message

                return portfolio, err, mid, msg


        ibm = IbManager(clientId=5001)

        portfolio, err, errid, errmsg = ibm.get_account_update()

        if portfolio:
            print(','.join(portfolio[0].keys()))

        for p in portfolio:
            print(','.join(map(str, p.values())))

        sys.exit(0)  # Ensure ib thread is terminated
