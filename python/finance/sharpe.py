import math

def ave(values):
    return float(sum(values)) / len(values)

def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2
                           for value in values)) / len(values))

def rateofreturn(capital, returns):
    ror = ((returns - capital) / capital) * 100
    return ror

def main():
    init_cash = 1000.0
    fixed_interest = 0.02
    share_prices = [100,102,102,102,102,102,102,102,102,102,102,103]
    shares = init_cash / share_prices[0]
    print "share count", shares

    rfr = init_cash
    for _ in share_prices:
        rfr = rfr + (rfr*fixed_interest)
    print "risk free rate", rfr

    exp_return = shares * share_prices[-1]
    print "expected returns", exp_return

    average = ave(share_prices)
    std_dev = std(share_prices, average)
    print average,'+-', std_dev

    ROR = rateofreturn(float(share_prices[0]),share_prices[-1])
    print "ror", ROR
    sharpe = (ROR - (fixed_interest * 100))/ std_dev
    print "sharpe", sharpe

if __name__ == "__main__":
    main()
    
