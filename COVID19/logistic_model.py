import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import requests # download from github directly?
from datetime import date, timedelta, datetime
from tabulate import tabulate

class logistic_model():
    """ Class to model the COVID-19 confirmed cases with a simple logistic model.
        Data is downloaded from the ECDC website.
    """
    
    def __init__(self, **kwargs):
        """ 
        
        """
        prop_defaults = {
                        'source': 'ecdc',
                        'endDate': '2020-04-15',
                        'hide': ['KR*'],
                        'show': [],
                        'ylim': 300000
                        }
        
        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # constants
        self.EPS = 0.00001
        self.maxiter = 100
        self.rate = 1.
        self.power = 0.5 # exponent for weighting data points

            
        # init internal containers        
        self.columnNames = {}
        self.dataByCountry = {}
        self.rawDays = [] # needed?
        self.days = []
        self.params = {}
        self.deathsByCountry = {}

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
        
        # at some points the columns names got lower cased
        self.columnNames = {c.lower(): c for c in df.columns.to_numpy()}
        
        
        # init day lists
        self.startDate='2020-02-15'        
        self.rawDays = df[(df[self.columnNames['geoid']]=='DE') & 
                          (df[self.columnNames['daterep']] >= self.startDate)].sort_values(by=self.columnNames['daterep'])[self.columnNames['daterep']]
        sdate = datetime.strptime(self.startDate, '%Y-%m-%d')
        edate = datetime.strptime(self.endDate, '%Y-%m-%d')
        delta = edate - sdate
        for i in range(delta.days + 1):
            self.days.append( (sdate + timedelta(days=i)).strftime("%m/%d/%y") )
         
        # init countries data
        boolfilter = {}
        boolfilter['DE']='DE'
        boolfilter['IT']='IT'      
        boolfilter['UK']='UK'
        boolfilter['JP']='JP'
        boolfilter['ES']='ES'
        boolfilter['HU']='HU'
        boolfilter['FR']='FR'
        boolfilter['PT']='PT'
        boolfilter['SE']='SE'
        boolfilter['AT']='AT'
        boolfilter['CH']='CH'        
        #boolfilter['ro']='RO'
        boolfilter['KR*']='KR'
        boolfilter['US']='US'
        boolfilter['NL']='NL'
        boolfilter['BE']='BE'
        
        
        for ct in boolfilter:
            if (self.show and ct in self.show) or (not self.show and ct not in self.hide):
                    temp = df[(df[self.columnNames['geoid']]==boolfilter[ct]) & (df[self.columnNames['daterep']] >= self.startDate)]
                
                    # confirmed cases
                    self.dataByCountry[ct]=temp.sort_values(by=self.columnNames['daterep'])[self.columnNames['cases']].cumsum()
                    # pad with 0 if data starts later
                    if(len(self.dataByCountry[ct]) < len(self.rawDays)):
                        zeros =  [0 for i in range(len(self.rawDays)-len(self.dataByCountry[ct]))]
                        self.dataByCountry[ct] = pd.concat([pd.DataFrame(zeros), self.dataByCountry[ct]], ignore_index=True)
                    
                    assert(len(self.dataByCountry[ct]) == len(self.rawDays))
            
                    # death cases
                    self.deathsByCountry[ct]=temp.sort_values(by=self.columnNames['daterep'])[self.columnNames['deaths']].cumsum()
                    if(len(self.deathsByCountry[ct]) < len(self.rawDays)):
                        zeros =  [0 for i in range(len(self.rawDays)-len(self.deathsByCountry[ct]))]
                        self.deathsByCountry[ct] = pd.concat([pd.DataFrame(zeros), self.deathsByCountry[ct]], ignore_index=True)
                
                    assert(len(self.deathsByCountry[ct]) == len(self.rawDays))
   
    def load_data(self):
        print("Loading ... ")
        if self.source=='ecdc':
            self.load_ecdc_data()
        else:
            print('unsupported source')      

        self.lastday = self.rawDays.max().strftime("%m/%d/%y")
    
    
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
        if not self.deathsByCountry:
            self.load_data()
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
        weight= np.diag([np.power(j+1.,self.power) for j in range(len(x))])
        beta = [[A],[B],[r]]
        residuals = [[y[i] - A*np.log(x[i]/(r-x[i]))-B] for i in range(len(y))]
        jacobi = [[np.log(x[i]/(r-x[i])), 1.0,-A/(r-x[i])] for i in range(len(y))]        
        #G=np.sum([[[0,0,-1.0*residuals[i,0]/(r-x[i])],[0,0,0],[-1.0*residuals[i,0]/(r-x[i]),0, residuals[i,0]*A/(r-x[i])**2]] for i in range(len(y))])
        #print(np.shape(beta), np.shape(residuals), np.shape(jacobi))        
        jj = np.transpose(jacobi).dot(weight).dot(jacobi)
        gradient = np.transpose(jacobi).dot(weight).dot(residuals)
        HessianInv = np.linalg.inv(jj)
        beta = beta + self.rate*(HessianInv.dot(gradient))    
        
        return beta[0][0],beta[1][0], beta[2][0]
        
    
    def fit_logistic(self, days, values):
        """ Iterate parameter updates until convergence        
        """
        # init
        i, A, B, r, delta = 0, 10, 20, 1.2*values.max(), 10000
        # iterate
        while i < self.maxiter and delta > self.EPS*r:
            A0, B0, r0 = A,B,r
            A, B, r = self.iterateOnce(days, values, A, B, r)    
            delta=np.abs(r-r0)
            i+=1    
        return r, A, B   
        
    def fit_logistic_with_err(self, days, values):   
        # initial values
        r, A, B = self.fit_logistic(days, values);
        
        # calculate variance of daily new cases ?
        #relVar = np.std([(r/(1+np.exp(-days[i]/A+B/A))/values[i] - 1.0) for i in range(len(values))])
        #print(relVar)
        delta_r = 0
        
        # add error to values
        #resList=[]
        #for i in range(100):
        #    ran = np.random.normal(0, 1, len(values))
        #    valuesTilde = np.array([values[i]*(1.0 + relVar*ran[i]) for i in range(len(values))])
        #    rTilde, Atilde, Btilde=self.fit_logistic(days, valuesTilde)
        #    resList.append([rTilde, Atilde, Btilde])
        #        
        #[delta_r, delta_A, delta_B] = np.std(resList, axis=0)
       
        
        return r, A, B, delta_r
        
    def process_data(self):
        """ Fit parameters for each country
        """
        if not self.dataByCountry:
            self.load_data()
        
        for country in self.dataByCountry:
                dataArr = self.dataByCountry[country].to_numpy()
                
                # initial condition : last day with less than 100 confirmed to exclude imported confirmed cases
                if country == 'HU':
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 20])
                    n0=dataArr[medIdx]
                elif country == 'KR*':
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 9000])
                    n0=dataArr[medIdx]
                elif country == 'US':
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 1000])
                    n0=dataArr[medIdx]                    
                else:
                    medIdx=max([j for j,n in enumerate(dataArr) if n < 100])
                    n0=dataArr[medIdx]

                    
                # truncate
                dataArr=dataArr[medIdx:]
                dayIndex=[(j[0]-medIdx) for j in enumerate(self.rawDays.to_numpy())][medIdx:]     
                # fit
                if country == 'KR*':
                    r,A,B = 9500, 6, 16 # does not converge ?!
                    #r,A,B = self.fit_logistic(dayIndex, dataArr)
                else:
                    r,A,B, delta_r = self.fit_logistic_with_err(dayIndex, dataArr)

                self.params[country] = {'country': country, 
                                        'r': r,
                                        'delta_r': delta_r,
                                        'A': A, 
                                        'B': B, 
                                        'medIdx': medIdx, 
                                        'current': dataArr[-1], 
                                        'n0':n0, 
                                        't0':self.rawDays.to_numpy()[medIdx]}
 
        # prepare data to print
        tbl = []
        currentIdx = self.days.index(self.lastday)
        #print(currentIdx, self.days[currentIdx])
        # sort
        self.params = {k: v for k, v in sorted(self.params.items(), key=lambda x: x[1]['current'], reverse=True)}
        
        for ct in self.params:
            p = self.params[ct]
            A, B, r, medIdx, delta_r = p['A'], p['B'], p['r'], p['medIdx'], p['delta_r']
            vtomorrow = r/(1+np.exp(-(currentIdx+1-medIdx)/A+B/A))
            vtoweek= r/(1+np.exp(-(currentIdx+7-medIdx)/A+B/A))
            tbl.append([ct, p['current'],
                        np.round(vtomorrow),
                        np.round(vtoweek), 
                        str(np.round(r)),
                        np.round(A*np.log(2.0),2)
                        #(self.rawDays.max() + timedelta(days=medIdx-int(np.round(B)))).strftime('%Y-%m-%d')
                       ])

        print(tabulate(tbl, headers=['Country',
                                     'Current ('+ self.lastday + ')', 
                                     'Next day',
                                     'Next week',
                                     'Total (predicted)',
                                     'Duplication rate (days)'
                                     #'Peak'
                                    ]))
    
        
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
            p = self.params[country]
            A, B, r, medIdx = p['A'], p['B'], p['r'], p['medIdx']
            dataArr = self.dataByCountry[country].to_numpy()[medIdx:]
            dayIndex = [(j[0]-medIdx) for j in enumerate(self.rawDays.to_numpy())][medIdx:]            
            label= country+', max: '+str(max(dataArr))
            plt.scatter(dayIndex ,[A*np.log(n/(r-n))+B for n in dataArr], label=label)

        # line
        plt.plot(range(50), range(50))        
        ax.legend(prop={'size':14})
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
    
        ax.legend(bbox_to_anchor=(1.02, 1.02), prop={'size':16}, labelspacing=0.8)   
        plt.yscale(yscale)
        plt.ylim(100,self.ylim)
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
    