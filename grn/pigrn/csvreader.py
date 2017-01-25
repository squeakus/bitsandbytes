"""Build database from the compustat data"""
import graph
import matplotlib.mlab as mlab

class CSVReader():
    """read compustat csv file and extract data"""
    def __init__(self, csvname):
        self.table = mlab.csv2rec(csvname)
        self.fields = {'market_valuemnthly':[],
                       'priceearnings__monthly':[],
                       'pricecash_flowshare_mtly':[],
                       'price_to_book':[],
                       'beta':[],
                       'share_count_mtly':[],
                       'share_price_mtly':[]}

        for col_name in self.table.dtype.names:
            for key in self.fields:
                if col_name.startswith(key):
                    self.fields[key].append(col_name)

        #neaten down to sixty months
        for key in self.fields:
            self.fields[key] = self.fields[key][:60]
            self.fields[key].reverse()

    def filter_table(self):
        """filter by field"""
        print "filtering beta"
        cols = self.fields['beta']
        for col in cols:
            print col, 'length', len(self.table[col])
            print "mean", self.table[col].mean()
            print "std", self.table[col].std()
            mask = self.table[col] > 0.01
            self.table = self.table[mask]

        for row in self.table:
          print row.company_name

    def get_company(self, company):
        company_record = None
        shareprice = []
        beta = []
        priceearn = []
        pricebook = []
        cashflow = []

        for record in self.table:
            if record.company_name == company:
                company_record = record

        for key in self.fields:
            if key == 'share_price_mtly':
                for col in self.fields[key]:
                    shareprice.append(company_record[col])

            if key == 'beta':
                for col in self.fields[key]:
                    beta.append(company_record[col])

            if key == 'priceearnings__monthly':
                for col in self.fields[key]:
                    priceearn.append(company_record[col])

            if key == 'price_to_book':
                for col in self.fields[key]:
                    pricebook.append(company_record[col])

            if key == 'pricecash_flowshare_mtly':
                for col in self.fields[key]:
                    cashflow.append(company_record[col])

        #normalize everything
        shareprice = [i / max(shareprice) for i in shareprice]
        beta = [i / max(beta) for i in beta]
        priceearn = [i / max(priceearn) for i in priceearn]
        pricebook = [i / max(pricebook) for i in pricebook]
        cashflow = [i / max(cashflow) for i in cashflow]

        company_data = {'shareprice':shareprice,
                        'beta':beta,
                        'priceearn':priceearn,
                        'pricebook':pricebook,
                        'cashflow':cashflow}
        return company_data

    def graph_company(self, company_name):
        """all fields for a given company"""
        data = self.get_company(company_name)
        for key in data:
            graph.plot_2d([data[key]], company_name+' '+key)

    def graph_fields(self):
        """for all companies"""
        for key in self.fields:
            results_list = []
            for record in self.table:
                result = []
                for column in self.fields[key]:
                    result.append(record[column])
                results_list.append(result)
            graph.plot_2d(results_list, key)

def main():
    db = CSVReader('compustat.csv')
    db.filter_table()

if __name__ == "__main__":
    main()
