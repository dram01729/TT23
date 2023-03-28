import streamlit as st
import matplotlib.pyplot as plt


import lightgbm
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

st.markdown(
	"""
	<style>
	.main {
	background-color: #F5F5F5;
	}
	<style>
	""", 
	unsafe_allow_html=True
)



@st.cache_data
def get_data(myfile):
	my_df = pd.read_csv(myfile)
	return my_df




myfilename='data/LengthOfStay.csv'
df = get_data(myfilename)

from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
    
download('https://github.com/AllenDowney/ModSimPy/raw/master/' +
         'modsim.py')


from modsim import *


def update_func1(x, t, system):

    if x > 0:
        if flip(system.mu):
            x -= 1
            
    # check for an arrival
    if flip(system.lam):
        x += 1
        
    return x


def update_func2(x, t, system):

    if x > 1 and flip(system.mu):
        x -= 1
            
    
    if x > 0 and flip(system.mu):
        x -= 1
    
    
    if flip(system.lam):
        x += 1
        
    return x


def update_func3(x1, x2, t, system):
    
    
    if x1 > 0 and flip(system.mu):
        x1 -= 1
            
    
    if x2 > 0 and flip(system.mu):
        x2 -= 1
            
    
    if flip(system.lam):
    
        if x1 < x2:
            x1 += 1
        else:
            x2 += 1
            
    return x1, x2


def run_simulation(system, update_func):
    
    x = 0
    ts = linrange(0, system.duration,endpoint=True)
    

    results = TimeSeries(ts,name='Queue length')
    results[0] = x
    
    for t in linrange(0, system.duration):
        x = update_func(x, t, system)
        results[t+1] = x

    return results



def run_simulation3(system, update_func):

    x1, x2 = 0, 0
    results = TimeSeries()
    results[0] = x1 + x2
    
    for t in linrange(0, system.duration):
        x1, x2 = update_func(x1, x2, t, system)
        results[t+1] = x1 + x2

    return results


def compute_metrics(results, system):
    L = results.mean()
    W = L / system.lam
    return L, W


def sweep_lam(lam_array, mu, update_func):
    sweep = SweepSeries()
    
    for lam in lam_array:
        system = make_system(lam, mu)
        results = run_simulation(system, update_func)
        L, W = compute_metrics(results, system)
        sweep[lam] = W
        
    return sweep

def sweep_lam3(lam_array, mu, update_func):
    sweep = SweepSeries()
    
    for lam in lam_array:
        system = make_system(lam, mu)
        results = run_simulation3(system, update_func)
        L, W = compute_metrics(results, system)
        sweep[lam] = W
        
    return sweep


tab1, tab1u, tab2, tab3, tab4 = st.tabs([" Home ", " BusyBea User Guide "," BusyBea Simulate ", " BusyBea Explore ", " BusyBea Decide "])

with tab1:
    st.header("BusyBea: Hospital :blue[**Be**]d :blue[**a**]llocation")
    st.subheader("A simulation and machine learning based software tool")

    st.write(
        '''
        Hospital resources are precious. It is important to manage the resources such as hospital beds, medical equipment, 
        medicines and human capital in order to provide a seamless service to patients. BusyBea is a simulation and machine learning based 
        software tool that enables decision making in a hospital setting to optimally allocate patients to beds in wards.

        There are some simplyfying assumptions. But this prototype can be easily expanded to create a real life operational tool in a 
        live setting. In this current version of the demonstration, the hospital is assumed to have a fixed number of beds in two wards.
        There are two possible queuing systems that can be implemented to let the two wards serve the patients. Patients can be accumulated in single
        queue and attended by medical professionals in both wards on a first come first serve basis. The other possibility is to create two queues, one each for
        attending to easier and difficult cases respectively. In essence, there can be "a one queue two wards" or "two queue two wards" setup. Also,
        it can be that the two wards can be both general or one general and one special. 

        The mode of queuing and bed allocation into wards will have an impact on the average queue size accumulating hour after hour
        in the hospital. Another important variable that affects quality of care is the average waiting time per patients measured in hours.

        The key variables that drive the decisions in the system are: 
        a)Time horizon in hours used for planning, 
        b) rate of patient arrival per hour and 
        c) rate of service measured in terms of how many patients are discharged per hour, thereby affecting the bed vacancy at any point

        The current system takes an user specified average waiting time tolerance and recommends when it is prudent to plan to add additional 
        beds in a new ward. Depending on the user setting for current expected arrival rate and discharge rate, BusyBea renders a simulation showing
        potential queue length in the hospital and corresponding waiting time. It recommends whether it is better to maintain a single queue or split
        into two queues. BusyBea can be extended to multiple queues with multiple wards with different bed capacity and treatment speeds.
        Essentially, such expansion will still boil down to real-time updating of arrival rates and discharge rates of the hospital as a whole.

        An important heuristic in bed allocation and queue management is to separate easy cases from difficult cases and segregate treatment. 
        Simulations show that this separation improves waiting time outcome for a lot of patients thereby improving averages and overall experience.
        BusyBea addresses this by allowing live data into a machine learning module where the patient attributes contributing to increased length 
        of stay is promptly identified and patients with those features separated into a different queue. 

        In this demonstration, live data is not used. But a static data (Kaggle dataset from microsoft) is used to do the classification and
        recommendation. This is not UK data either. It is to be interpreted as a placeholder for UK live data in a hospital setting. None of the model
        storage or implementation detail need to vary a lot to make use of this version of BusyBea which uses a generic lightgbm gradient boosting 
        learning model. This element can essentially be a plug and play if a different package needs to be used depending on context.

        BusyBea is built as a streamlit app, easy to modify and update in the sense that the simulation, machine learning and data components are 
        modular. It is easy to share as an open-source tool from points of view of developers as well as users. This will hopefully allow the
        scope and usability of the tool to be enhanced in a short span of time.



        '''
        )

with tab1u:

    st.header('BusyBea: Quick start user guide')
    st.write(
        '''
        
        Start with specifying parameters for the simulation. Based on simulated results, BusyBea will recommend 
        ONE queue or TWO queues. Depending on the waiting time tolerance specified by the user, BusyBea will flag if there
        is an imminent need for adding beds by opening a new ward. BusyBea Explore will allow exploration of patient attributes
        contributing to increased length of stay. Depending on which attributes of the patients are deemed critical, BusyBea will
        recommend to move patients from general to special ward. This serves to improve the service quality for all patients on an 
        average.


        There are three modules in BusyBea:
        1. BusyBea Simulate: 

        First, specify a time horizon for planning (in Hours). Accepted values range from 24 hours (one day) to 336 hours (two weeks) 
        in steps of 24. Then, specify the arrival rate of patients (Range of 1 to 10 per hour). After that, use the slider to specify 
        the discharge rate that affects bed vacancy (Range of 1 to 10 per hour). Interpret the simulated queue length graphs for the 
        two different queuing systems. Interpret the average waiting time charts to see what could happen if there is no further 
        decisions to act.

        2. BusyBea Explore

        There are two parts to exploration. One is to see how patient attributes are distributed. This allows to interpret at a 
        higher level if any specific type of patients are needing treatment. The second part is a scatter graph that shows how a
        specific attribute is correlated to length of stay. This serves visual interpretation. The actual rigourous modelling of
        this linkage is done by the machine learning algorithm and the results are displayed in the next module

        3. BusyBea Decide

        In this module, the decisions are displayed so that an appropriate strategy is pursued to improve service quality. The 
        horizontal bar chart displays the critical variables in the order of importance to length of stay impact. The red-line 
        threshold set at 10% allows medical professionals to attend to specific problems separately so that the general queue
        is not held up.




        '''
        )




with tab2:

    st.header("BusyBea Simulate")

    period_slider,lam_slider,mu_slider=st.columns(3)
    numPeriods = period_slider.slider("What is your planning time horizon? (in Hours)",min_value=24,max_value=336,value=168,step=24)
    patients_per_period = lam_slider.slider("How many patients on average are arriving per period?",min_value=1,max_value=10,value=5,step=1)
    disch_per_period = mu_slider.slider("How many patients on average are discharged per period?",min_value=1,max_value=10,value=3,step=1)

    def make_system(lam, mu):
	    return System(lam=lam, 
	                  mu=mu, 
	                  duration=numPeriods)

    interarrival_time = 10 / patients_per_period
    service_time = 10 / disch_per_period
    lam = 1 / interarrival_time
    mu = 1 / service_time

    num_vals = 101
    lam_array = linspace(0.1*mu, 2*mu, num_vals)

    system = make_system(lam, mu)

    results1 = run_simulation(system, update_func1)
    results2 = run_simulation(system, update_func2)
    results3 = run_simulation3(system, update_func3)

    if results3.mean()<=results2.mean():
        optres=results3.mean()
        worseres = results2.mean()
        q_reco_text = "Separate queues into TWO"
    else :
        optres=results2.mean()
        worseres = results3.mean()
        q_reco_text = "Merge queues into ONE"



    sweep1 = sweep_lam(lam_array, mu, update_func1)
    sweep2 = sweep_lam(lam_array, mu, update_func2)
    sweep3 = sweep_lam3(lam_array, mu, update_func3)

    st.text('****************************************************************************************')

    st.subheader("Average Queue length")

    
    fig0, ax0 = plt.subplots()
    plt.xlabel('Periods')
    plt.ylabel('Number of patients waiting to be attended')
    plt.axhline(y=worseres,color='black',linestyle='--')
    plt.plot(range(1,len(results2)+1), results2,'b-',range(1,len(results3)+1), results3,'g-')
    st.pyplot(fig0)


    st.caption('Blue line: Number of patients waiting in a **ONE** queue **TWO** ward setup')
    st.caption('Green line: Number of patients waiting in a **TWO** queue **TWO** ward setup')
    st.caption('Dotted line: Average number of patients waiting to be attended based on the worst case between two queuing systems ')

    st.text('****************************************************************************************')

    st.subheader("Average waiting time")


    waitTime_tolerance=st.selectbox("Please specify maximum acceptable average waiting time (in Hours) ", options=[2,5,10,24],index=1)

    if max(sweep2.mean(),sweep3.mean())<=waitTime_tolerance:
        ward_reco_text = "Maintain current number of wards"
    else:
        ward_reco_text = "Plan to add a new ward"

    


    fig1, ax1 = plt.subplots()
    plt.xlabel('Patient arrival rate to discharge rate ratio')
    plt.ylabel('Average waiting time per patient (in Hours)')
    plt.axhline(y=waitTime_tolerance,color='black',linestyle='--')
    plt.plot(lam_array/mu, sweep2, 'bs',lam_array/mu, sweep3, 'g^')
    st.pyplot(fig1)

    st.caption('Blue square: Average waiting time per patient in a **ONE** queue **TWO** ward setup')
    st.caption('Green Triangle: Average waiting time per patient in a **TWO** queue **TWO** ward setup')
    st.caption('Dotted line: Specified maximum acceptable average waiting time')


	
    
with tab3:
    st.header("BusyBea Explore")
    #st.subheader("Aggregate Patient Profile Viewer")

    st.text('The system is monitoring how each attribute of a patient is affecting their')
    st.text('length of stay in the hospital')

    st.text('****************************************************************************************')
    st.subheader("Attribute Distribution")

    datetime_cols = ["vdate", "discharged"]
    cat_cols = ["gender", "rcount", "facid"]
    bin_cols = ["dialysisrenalendstage", 
                "asthma", 
                "irondef", 
                "pneum", 
                "substancedependence", 
                "psychologicaldisordermajor",
                "depress",
                "psychother",
                "fibrosisandother",
                "malnutrition",
                "hemo"]
    num_cols = ["hematocrit",
                "neutrophils",
                "sodium",
                "glucose",
                "bloodureanitro",
                "creatinine",
                "bmi",
                "respiration"]

    for date_col, cat_col in zip(datetime_cols, cat_cols): 
        df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y")
        df[cat_col] = df[cat_col].astype("category")
    df.head()


    def calculate_number_of_issues(df, bin_cols):
        df["numberofissues"] = df[bin_cols].sum(axis=1)
        return df

    df = calculate_number_of_issues(df, bin_cols)

    labels, features = df[["lengthofstay"]], df.drop(["lengthofstay", "discharged", "vdate", "eid"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(features, labels)

    regressor = make_pipeline(
        ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(), cat_cols),
        ], remainder="passthrough"),
        LGBMRegressor(n_jobs=-1)
    )
    regressor.fit(x_train, y_train)
    preds = np.round(regressor.predict(x_test), 0)
    r2_score(preds, y_test)
    feature_importances = (regressor[1].feature_importances_ / sum(regressor[1].feature_importances_)) * 100
    resl = pd.DataFrame({'Features': regressor[:-1].get_feature_names_out(),
                        'Importances': feature_importances})
    resl.sort_values(by='Importances', inplace=True)

    fig2, ax2 = plt.subplots(figsize=(5, 10))
    ax2 = plt.barh(resl['Features'], resl['Importances'])
    plt.axvline(x=10,color='r',linestyle='--')
    plt.xlabel('Criticality of attributes (% contribution to length of stay)')


    inp_col,disp_col = st.columns(2)

    collist = [i for i in features.columns]
    collistPd = pd.DataFrame({'columns': collist})
    f_cols=collistPd['columns']
    

    inp_feature = inp_col.selectbox("Please select an attribute to explore?", options=f_cols,index=len(f_cols)-1)

    

    relation = pd.DataFrame({'Features': x_test[inp_feature],
                        'Length_of_Stay': y_test['lengthofstay']})


    fig25, ax25 = plt.subplots(figsize=(10, 5))
    plt.xlabel(inp_feature)
    plt.ylabel('Frequency')
    plt.hist(relation['Features'])
    st.pyplot(fig25)

    st.text('****************************************************************************************')

    st.subheader("Effect of attribute on Length of stay")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plt.xlabel(inp_feature)
    plt.ylabel('Length_of_Stay')
    #plt.plot(relation['Features'], relation['Length_of_Stay'])
    plt.scatter(relation['Features'], relation['Length_of_Stay'])
    st.pyplot(fig3)


with tab4:


    st.header("BusyBea Decide")

    st.text('The system has identified the following attributes of patients as critical in')
    st.text('contributing to increased length of stay in the hospital at the moment')
    st.text('***DECISIONS****************************************************************************')
    st.text('1. Patients with extreme readings in any critical attribute (beyond the redline),')
    st.text('are to be moved out of the General Ward and assigned to the Special Ward')
    st.write("2. "+q_reco_text)
    st.write("3. "+ward_reco_text)
    st.text('****************************************************************************************')



    st.pyplot(fig2)

