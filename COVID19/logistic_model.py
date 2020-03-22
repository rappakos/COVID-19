import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import requests # download from github directly?
from datetime import date, timedelta, datetime
from tabulate import tabulate

class logistic_model():
    """ Class to model the COVID-19 confirmed cases with a simple logistic model.
        Data is downloaded from the github repository CSSEGISandData/COVID-19.
    """
    
    def __init__(self, **kwargs):
        """ 
        
        """
        prop_defaults = {
                        'path': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
                        'startDate': '2020-01-22',
                        'endDate': '2020-04-15',
                        'hide': [],
                        'ylim': 100000
                        }
        
        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # constants
        self.EPS = 0.0001
        self.maxiter = 50
        self.power = 0.5 # exponent for weighting data points

            
        # init internal containers
        self.boolfilter = {}
        self.dataByCountry = {}
        self.rawDays = [] # needed?
        self.days = []
        self.params = {}
        self.deathsByCountry = {}

    
    def load_data(self):
        print("loading ... ")        
        data = pd.read_csv(self.path)
        #print(data.head())
        df = pd.DataFrame(data)
        # init days list -- exclude Province/State, Country/Region, Latitude, Longitude
        self.rawDays = df.columns[4:]
        # 
        self.boolfilter['de']=df['Country/Region']=='Germany'
        self.boolfilter['it']=df['Country/Region']=='Italy'
        self.boolfilter['jp']=df['Country/Region']=='Japan'
        self.boolfilter['uk']=df['Province/State']=='United Kingdom' # 'UK'
        self.boolfilter['kr']=df['Country/Region']== 'Korea, South' #'Republic of Korea'
        #self.boolfilter['dp']=df['Province/State']=='Diamond Princess cruise ship'
        self.boolfilter['es']=df['Country/Region']=='Spain'
        self.boolfilter['hu']=df['Country/Region']=='Hungary'
        self.boolfilter['fr']=df['Province/State']=='France'
        self.boolfilter['pt']=df['Country/Region']=='Portugal'
        self.boolfilter['se']=df['Country/Region']=='Sweden'
        self.boolfilter['at']=df['Country/Region']=='Austria'
        self.boolfilter['ch']=df['Country/Region']=='Switzerland'
        #self.boolfilter['ny']=df['Province/State']=='New York'
        self.boolfilter['ro']=df['Country/Region']=='Romania'
        
        for ct in self.boolfilter:
            if ct not in self.hide:
                self.dataByCountry[ct]=df[self.boolfilter[ct]][df.columns[4:]].T
            
        sdate = datetime.strptime(self.startDate, '%Y-%m-%d')
        edate = datetime.strptime(self.endDate, '%Y-%m-%d')
        delta = edate - sdate
        for i in range(delta.days + 1):
            self.days.append( (sdate + timedelta(days=i)).strftime("%m/%d/%y") )
        #print(self.days[-5:])
    
    def load_death_data(self):
        # TODO check existence?
        path=self.path.replace("Confirmed","Deaths")
        data = pd.read_csv(path)
        df = pd.DataFrame(data)
        
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
        #boolfilter['ny']=df['Province/State']=='New York'
        boolfilter['ro']=df['Country/Region']=='Romania'
            
        for ct in self.boolfilter:
            if ct not in self.hide:
                self.deathsByCountry[ct]=df[boolfilter[ct]][df.columns[4:]].T
    
    
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

    def plot_death_cases(self, yscale='log', ylim=5000):
        if not self.deathsByCountry:
            self.load_death_data()
        
        fig = plt.figure(figsize=(12,7))
        ax = plt.subplot(111)
        for country in self.deathsByCountry:
            label= country+', max: '+str(self.deathsByCountry[country].to_numpy().max())
            plt.scatter(self.days[:len(self.deathsByCountry[country])], self.deathsByCountry[country], label=label)

        ax.legend()
        plt.yscale(yscale)
        plt.ylim(1,ylim)
        plt.xticks(range(0,len(self.days),14), self.days[0::14])
        plt.show()
        
        
    def iterateOnce(self, days, values, A, B, r):
        """ Gauss-Newton method to iterate parameters with a residual function
            R(t,N; A, B, r) =  (t - A * ln(N/(r-N) - B)        
        """
        y, x= np.array(days).flatten(), np.array(values).flatten()
        assert(len(x)==len(y))
        # Gauss-Newton
        #weight = np.eye(len(x)) -- 
        weight= np.diag([np.power(j,self.power) for j in range(len(x))])
        beta = [[A],[B],[r]]
        residuals = [[y[i] - A*np.log(x[i]/(r-x[i]))-B] for i in range(len(y))]
        jacobi = [[np.log(x[i]/(r-x[i])), 1.0,-A/(r-x[i])] for i in range(len(y))]
        #print(np.shape(beta), np.shape(residuals), np.shape(jacobi))
        jjinv = np.linalg.inv(np.transpose(jacobi).dot(weight).dot(jacobi))
        beta = beta + (jjinv.dot(np.transpose(jacobi).dot(weight).dot(residuals)))    
        
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
                elif country == 'kr':
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 500])
                    n0=dataArr[medIdx]
                else:
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 10])
                    n0=dataArr[medIdx]
                # truncate
                dataArr=dataArr[medIdx:]
                dayIndex=[(j[0]-medIdx) for j in enumerate(self.rawDays.to_numpy())][medIdx:]     
                # fit r
                if country == 'kr':
                    r,A,B = 9500, 3, 12 # does not converge ?!
                else:
                    r,A,B = self.fit_logistic(dayIndex, dataArr)
                #print('fitting: ', country, np.round(r,2), np.round(A,2), np.round(B,2),medIdx, dataArr[-1])
                self.params[country] = {'country': country, 'r': r, 'A': A, 'B': B, 'medIdx': medIdx, 'current': dataArr[-1]}
 
        print(tabulate([[self.params[ct]['country'], self.params[ct]['current'] , np.round(self.params[ct]['r']), np.round(self.params[ct]['A'],2) ] for ct in self.params], headers=['Country','Current','Total','Rate']))

        
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
    