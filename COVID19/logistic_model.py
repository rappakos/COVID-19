import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import requests # download from github directly?
from datetime import date, timedelta, datetime
from tabulate import tabulate

class logistic_model():
    """ Class to model the COVID-19 confirmed cases with a simple logistic model.
        Data is downloaded from either the github repository CSSEGISandData/COVID-19:
        https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
        or from
        
        
    """
    
    def __init__(self, **kwargs):
        """ 
        
        """
        prop_defaults = {
                        'source': 'ecdc',
                        'endDate': '2020-04-15',
                        'hide': [],
                        'ylim': 150000
                        }
        
        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # constants
        self.EPS = 0.0001
        self.maxiter = 100
        self.rate = 0.5
        self.power = 1. # exponent for weighting data points

            
        # init internal containers        
        self.dataByCountry = {}
        self.rawDays = [] # needed?
        self.days = []
        self.params = {}
        self.deathsByCountry = {}

    def load_CSSEGISandData_data(self):
        """ Downloads data from github repository CSSEGISandData.
            Initializes rawDays, dataByCountry, and days containers.        
        """
        path=('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/'
              'master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
                
        data = pd.read_csv(path)
        #print(data.head())
        df = pd.DataFrame(data)
        # init days list -- exclude Province/State, Country/Region, Latitude, Longitude
        self.rawDays = df.columns[4:]
        # 
        boolfilter={}
        boolfilter['de']=df['Country/Region']=='Germany'
        boolfilter['it']=df['Country/Region']=='Italy'
        boolfilter['jp']=df['Country/Region']=='Japan'
        boolfilter['uk']=df['Province/State']=='United Kingdom' # 'UK'
        boolfilter['kr']=df['Country/Region']== 'Korea, South' #'Republic of Korea'
        #self.boolfilter['dp']=df['Province/State']=='Diamond Princess cruise ship'
        boolfilter['es']=df['Country/Region']=='Spain'
        boolfilter['hu']=df['Country/Region']=='Hungary'
        boolfilter['fr']=df['Province/State']=='France'
        boolfilter['pt']=df['Country/Region']=='Portugal'
        boolfilter['se']=df['Country/Region']=='Sweden'
        boolfilter['at']=df['Country/Region']=='Austria'
        boolfilter['ch']=df['Country/Region']=='Switzerland'
        #self.boolfilter['ny']=df['Province/State']=='New York'
        boolfilter['ro']=df['Country/Region']=='Romania'
        
        for ct in boolfilter:
            if ct not in self.hide:
                self.dataByCountry[ct]=df[boolfilter[ct]][df.columns[4:]].T
            
        self.startDate='2020-01-22'
        sdate = datetime.strptime(self.startDate, '%Y-%m-%d')
        edate = datetime.strptime(self.endDate, '%Y-%m-%d')
        delta = edate - sdate
        for i in range(delta.days + 1):
            self.days.append( (sdate + timedelta(days=i)).strftime("%m/%d/%y") )
    
    def load_ecdc_data(self):
        tstamp=(datetime.today()).strftime('%Y-%m-%d')
        path= ('https://www.ecdc.europa.eu/sites/default/files/documents/'
               'COVID-19-geographic-disbtribution-worldwide-')
        try:
            file = path + tstamp+'.xlsx'
            data = pd.read_excel(file)
        except:
            tstamp=(datetime.today()-timedelta(days=1)).strftime('%Y-%m-%d')
            file = path + tstamp+'.xlsx'
            data = pd.read_excel(file)
         
        df=pd.DataFrame(data)
        
        
        self.startDate='2020-02-01'        
         
        boolfilter = {}
        boolfilter['de']='DE'
        boolfilter['it']='IT'      
        boolfilter['uk']='UK'
        boolfilter['jp']='JP'
        boolfilter['es']='ES'
        #boolfilter['hu']='HU'
        boolfilter['fr']='FR'
        #boolfilter['pt']='PT' # PT starts at 2020-03-03 ?!
        boolfilter['se']='SE'
        boolfilter['at']='AT'
        boolfilter['ch']='CH'        
        #boolfilter['ro']='RO' # 
        boolfilter['kr*']='KR'
        
        for ct in boolfilter:
            if ct not in self.hide:
                temp = df[(df['GeoId']==boolfilter[ct]) & (df['Year'] >= 2020) & (df['Month'] >= 2)]
                if ct=='de':
                    self.rawDays = temp.sort_values(by='DateRep')['DateRep']
                self.dataByCountry[ct]=temp.sort_values(by='DateRep')['Cases'].cumsum()
                assert(len(self.rawDays)==len(self.dataByCountry[ct]))
        
        #print(self.dataByCountry['de'])
        
        
        sdate = datetime.strptime(self.startDate, '%Y-%m-%d')
        edate = datetime.strptime(self.endDate, '%Y-%m-%d')
        delta = edate - sdate
        for i in range(delta.days + 1):
            self.days.append( (sdate + timedelta(days=i)).strftime("%m/%d/%y") )
        
        
    
    def load_data(self):
        print("Loading ... ")
        if self.source=='CSSEGISandData':
            self.load_CSSEGISandData_data()
        elif self.source=='ecdc':
            self.load_ecdc_data()
        else:
            print('unsupported source')      

        self.lastday = self.rawDays.max().strftime("%m/%d/%y")
    
    def load_death_data(self):
        print('')
    
    
    def plot_raw(self, yscale='log'):
        """ Plot the raw data in logarithmic scale
        
        """
        if not self.dataByCountry:
            self.load_data()
        
        fig = plt.figure(figsize=(12,7))
        ax = plt.subplot(111)
        for country in self.dataByCountry:
            label= country+', max: '+str(self.dataByCountry[country].to_numpy().max())
            plt.scatter(self.days[:len(self.dataByCountry[country])], self.dataByCountry[country], label=label)

        ax.legend()
        plt.yscale(yscale)
        plt.ylim(1,self.ylim)
        plt.xticks(range(0,len(self.days),14), self.days[0::14])
        plt.show()

    def plot_death_cases(self, yscale='log', ylim=8000):
        print('')
        
        
    def iterateOnce(self, days, values, A, B, r):
        """ Gauss-Newton method to iterate parameters with a residual function
            R(t,N; A, B, r) =  (t - A * ln(N/(r-N) - B)        
        """
        y, x= np.array(days).flatten(), np.array(values).flatten()
        assert(len(x)==len(y))
        #if len(values) < len(days):
        #    x = np.pad(x, (len(days) - len(values), 0), 'constant')

        # Gauss-Newton
        weight= np.diag([np.power(j,self.power) for j in range(len(x))])
        beta = [[A],[B],[r]]
        residuals = [[y[i] - A*np.log(x[i]/(r-x[i]))-B] for i in range(len(y))]
        jacobi = [[np.log(x[i]/(r-x[i])), 1.0,-A/(r-x[i])] for i in range(len(y))]
        #print(np.shape(beta), np.shape(residuals), np.shape(jacobi))
        jjinv = np.linalg.inv(np.transpose(jacobi).dot(weight).dot(jacobi))
        beta = beta + self.rate*(jjinv.dot(np.transpose(jacobi).dot(weight).dot(residuals)))    
        
        return beta[0][0],beta[1][0], beta[2][0]
        
    
    def fit_logistic(self, days, values):
        """ Iterate parameter updates
        
        """
        # init
        i ,A, B, r, delta=0,10, 20, 1.2*values.max(), 10000
        # iterate
        while i < self.maxiter and delta > self.EPS*r:
            A0, B0, r0 = A,B,r
            A,B, r = self.iterateOnce(days, values, A, B, r)    
            delta=np.abs(r-r0)
            i+=1    
        return r, A, B   
        
    def process_data(self):
        """ Fit parameters for each country
        """
        if not self.dataByCountry:
            self.load_data()
        
        for country in self.dataByCountry:
                dataArr = self.dataByCountry[country].to_numpy()
                # initial condition : last day with less than 50 confirmed to exclude imported confirmed cases
                if country != 'hu':
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 50])
                    n0=dataArr[medIdx]
                elif country == 'kr*':
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 9000])
                    n0=dataArr[medIdx]
                else:
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 20])
                    n0=dataArr[medIdx]
                # truncate
                dataArr=dataArr[medIdx:]
                dayIndex=[(j[0]-medIdx) for j in enumerate(self.rawDays.to_numpy())][medIdx:]     
                # fit r
                if country == 'kr*':
                    r,A,B = 9500, 6, 16 # does not converge ?!
                    #r,A,B = self.fit_logistic(dayIndex, dataArr)
                else:
                    r,A,B = self.fit_logistic(dayIndex, dataArr)

                self.params[country] = {'country': country, 
                                        'r': r, 
                                        'A': A, 
                                        'B': B, 
                                        'medIdx': medIdx, 
                                        'current': dataArr[-1], 
                                        'n0':n0, 
                                        't0':self.rawDays.to_numpy()[medIdx]}
 
        # print data
        tbl = []
        currentIdx = self.days.index(self.lastday)
        #print(currentIdx, self.days[currentIdx])
        
        for ct in self.params:
            p = self.params[ct]
            A, B, r, medIdx = p['A'], p['B'], p['r'], p['medIdx']
            
            vtomorrow = r/(1+np.exp(-(currentIdx+1-medIdx)/A+B/A))
            vtoweek= r/(1+np.exp(-(currentIdx+7-medIdx)/A+B/A))
            tbl.append([ct, p['current'],np.round(vtomorrow),np.round(vtoweek), np.round(r), np.round(A*np.log(2.0),2) ])

        tbl.sort(key=lambda x: -x[1])
        print(tabulate(tbl, headers=['Country',
                                     'Current ('+ self.lastday + ')', 
                                     'Next day',
                                     'Next week',
                                     'Total (predicted)',
                                     'Duplication rate (days)']))
    
        
    def plot_fitted(self, yscale='log'):
        """ Plot data with fitted curves
        
        """
        if not self.dataByCountry:
            self.load_data()
            
        if not self.params:
            self.process_data()
        
        # figure for collapsing the curves
        fig = plt.figure(figsize=(15,10))
        ax = plt.subplot(121)
        for country in self.params:
            A, B = self.params[country]['A'], self.params[country]['B']
            r, medIdx = self.params[country]['r'], self.params[country]['medIdx']
            dataArr = self.dataByCountry[country].to_numpy()
            dataArr = dataArr[medIdx:]
            dayIndex = [(j[0]-medIdx) for j in enumerate(self.rawDays.to_numpy())][medIdx:]            
            dayIndex = [j  for j in dayIndex]
            plt.scatter(dayIndex ,[A*np.log(n/(r-n))+B for n in dataArr], label=country)

        # line
        plt.plot(range(50), range(50))        
        ax.legend()
        plt.ylim(-2,50)
        plt.title("A ln( N/(1-N/r))+B vs t")        
        plt.xlabel('A ln( N_i/(1-N_i/r))+B')
        plt.ylabel('t_i [days]')
        
        # figure for data on log-linear plot
        ax = plt.subplot(122)
        for country in self.params:
            A, B = self.params[country]['A'], self.params[country]['B']
            r, medIdx = self.params[country]['r'], self.params[country]['medIdx']
            # raw data points
            plt.scatter(
                self.days[:len(self.dataByCountry[country])],
                self.dataByCountry[country],
                label=country+', total: '+str(np.round(r))
            )
            # curves
            plt.plot(self.days,[r/(1+np.exp(-(j-medIdx)/A+B/A)) for j,d in enumerate(self.days)],label=country)
    
        ax.legend(bbox_to_anchor=(1.1, 1.))   
        plt.yscale(yscale)
        plt.ylim(5,self.ylim)
        plt.xticks(range(0,len(self.days),14), self.days[0::14])
        if yscale=='log':
            plt.title("N_i (log scale)")
        else:
            plt.title("N_i")
            
        
        plt.show()
        
        
if __name__ == '__main__':
    print('Starting  ... \n')
    model=logistic_model()
    model.plot_raw()
    