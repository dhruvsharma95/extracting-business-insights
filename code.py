# --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visual_summary(type_, df, col):
    """Summarize the Data using Visual Method.
    
    This function accepts the type of visualization, the data frame and the column to be summarized.
    It displays the chart based on the given parameters.
    
    Keyword arguments:
    type_ -- visualization method to be used
    df -- the dataframe
    col -- the column in the dataframe to be summarized
    """
    if df[col].dtype == 'object':
        df[col].value_counts().plot(kind=type_)
        plt.show()
    else:
        df[col].plot(kind=type_)
        plt.show()



def central_tendency(type_, df, col):
    """Calculate the measure of central tendency.
    
    This function accepts the type of central tendency to be calculated, the data frame and the required column.
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column in the dataframe to do the calculations
    
    Returns:
    cent_tend -- the calculated measure of central tendency
    """
    if type_ == 'mean':
        cent_tend = df[col].mean()
        print(f"the mean of {col} is {cent_tend}")
    elif type_ == 'median':
        cent_tend = df[col].median()
        print(f"the median of {col} is {cent_tend}")
    else:
        cent_tend = df[col].mode()[0]
        print(f"the mode of {col} is {cent_tend}")
    
    

def measure_of_dispersion(type_, df, col):
    """Calculate the measure of dispersion.
    
    This function accepts the measure of dispersion to be calculated, the data frame and the required column(s).
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column(s) in the dataframe to do the calculations, this is a list with 2 elements if we want to calculate covariance
    
    Returns:
    disp -- the calculated measure of dispersion
    """
    if type_ == 'range':
        min_ = df[col].min()
        max_ = df[col].max()
        range_ = max_ - min_
        disp = round(range_,3)
        print(f"Range for {col} is {disp}")

    elif type_ == 'mean absolute deviation':
        # MAD = 1/n*(summation(|x-mean|))
        mean_absolute_deviation = df[col].mad()
        disp = round(mean_absolute_deviation, 3)
        print(f"Mean Absolute Deviation for {col} is {disp}")

    elif type_ == 'standard deviation':
        # standard deviation = sqrt(1/n*(summation[(x-mean)^2]))
        standard_deviation = df[col].std()
        disp = round(standard_deviation, 3)
        print(f"Standard Deviation for {col} is {disp}")

    elif type_ == 'coefficient of variation':
        # coefficient of variation = (standard_deviation/mean)*100
        standard_deviation = df[col].std()
        mean = df[col].mean()
        coeff_of_variation = (standard_deviation/mean)*100
        disp = round(coeff_of_variation, 3)
        print(f"Coefficient of Variation for {col} is {disp}")

    elif type_ == 'iqr':
        q1 = df[col].quantile(q=0.25)
        q3 = df[col].quantile(q=0.75)
        iqr = q3 - q1
        disp = round(iqr, 3)
        print(f"IQR for {col} is {disp}")

    elif type_ == 'covariance': 
        # Covariance = 1/n*[summation(x-x_mean)*(y-y_mean)]
        col1 = col[0]
        col2 = col[1]
        mean_col1 = df[col1].mean()
        mean_col2 = df[col2].mean()

        diff_col1 = df[col1] - mean_col1
        diff_col2 = df[col2] - mean_col2

        mul = diff_col1*diff_col2
        summation = mul.sum()

        covariance = summation/mul.size
        disp = round(covariance, 3)
        print(f"Covariance for {col1} and {col2} is {disp}")



def calculate_correlation(type_, df, col1, col2):
    """Calculate the defined correlation coefficient.
    
    This function accepts the type of correlation coefficient to be calculated, the data frame and the two column.
    It returns the calculated coefficient.
    
    Keyword arguments:
    type_ -- type of correlation coefficient to be calculated
    df -- the dataframe
    col1 -- first column
    col2 -- second column
    
    Returns:
    corr -- the calculated correlation coefficient
    """
    newdf = df[[col1, col2]].copy()

    if type_ == 'pearson':
        corr = newdf.corr(method='pearson').iloc[0,1]
        print(f"Pearson correlation for {col1} and {col2} is",corr)
    elif type_ == 'spearman':
        corr = newdf.corr(method='spearman').iloc[0,1]
        print(f"Spearman correlation for {col1} and {col2} is",corr)
    


def calculate_probability_discrete(data, event):
    """Calculates the probability of an event from a discrete distribution.
    
    This function accepts the distribution of a variable and the event, and returns the probability of the event.
    
    Keyword arguments:
    data -- series that contains the distribution of the discrete variable
    event -- the event for which the probability is to be calculated
    
    Returns:
    prob -- calculated probability fo the event
    """
    newdf = df[df['country'] == data]
    total = newdf.shape[0]

    if event == 'banking_crisis':
        crisis = newdf[newdf['banking_crisis'] == 'crisis'].shape[0]
        prob = crisis/total
        return prob

    elif event == 'inflation_crises':
        crisis = newdf[newdf['inflation_crises'] == 1].shape[0]
        prob = crisis/total
        return prob

    elif event == 'currency_crises':
        crisis = newdf[newdf['currency_crises'] == 1].shape[0]
        prob = crisis/total
        return prob

    elif event == 'systemic_crisis':
        crisis = newdf[newdf['systemic_crisis'] == 1].shape[0]
        prob = crisis/total
        return prob


def event_independence_check(prob_event1, prob_event2, prob_event1_event2):
    """Checks if two events are independent.
    
    This function accepts the probability of 2 events and their joint probability.
    And prints if the events are independent or not.
    
    Keyword arguments:
    prob_event1 -- probability of event1
    prob_event2 -- probability of event2
    prob_event1_event2 -- probability of event1 and event2
    """
    p_total = prob_event1*prob_event2

    if p_total == prob_event1_event2:
        print('Events are independent')
    else:
        print('Events are not independent')
    


def bayes_theorem(df, col1, event1, col2, event2):
    """Calculates the conditional probability using Bayes Theorem.
    
    This function accepts the dataframe, two columns along with two conditions to calculate the probability, P(B|A).
    You can call the calculate_probability_discrete() to find the basic probabilities and then use them to find the conditional probability.
    
    Keyword arguments:
    df -- the dataframe
    col1 -- the first column where the first event is recorded
    event1 -- event to define the first condition
    col2 -- the second column where the second event is recorded
    event2 -- event to define the second condition
    
    Returns:
    prob -- calculated probability for the event1 given event2 has already occured
    """
    #newdf = df[df['country'] == 'Kenya']
    #p_event1 = calculate_probability_discrete('Kenya','banking_crisis')

    #if col2 == 'systemic_crisis':
    #    p_event2 = calculate_probability_discrete('Kenya','systemic_crisis')
    #    a = np.array(newdf['banking_crisis'])
    #    b = np.array(newdf['systemic_crisis'])
    #    print(pd.crosstab(a,b))
    #    table = pd.crosstab(a,b)
    #    x = table.iloc[0,1]
    #    y = table.iloc[0,:].sum()
    #    p_b_a = x/y
    #
    #    # p_a_b
    #    prob = (p_b_a*p_event1)/p_event2
    #    return prob


    #elif col2 == 'inflation_crises':
    #    p_event2 = calculate_probability_discrete('Kenya','inflation_crises')
    #    a = np.array(newdf['banking_crisis'])
    #    c = np.array(newdf['inflation_crises'])
    #    print(pd.crosstab(a,c))
    #    table = pd.crosstab(a,c)
    #    x = table.iloc[0,1]
    #    y = table.iloc[0,:].sum()
    #    p_c_a = x/y
    #
    #    # p_a_c
    #    prob = (p_c_a*p_event1)/p_event2
    #    return prob

    #elif col2 == 'currency_crises':
    #    p_event2 = calculate_probability_discrete('Kenya','currency_crises')
    #    a = np.array(newdf['banking_crisis'])
    #    d = np.array(newdf['currency_crises'])
    #    print(pd.crosstab(a,d))
    #    table = pd.crosstab(a,d)
    #    x = table.iloc[0,1]
    #    y = table.iloc[0,:].sum()
    #    p_d_a = x/y
    #
    #    # p_a_d
    #    prob = (p_d_a*p_event1)/p_event2
    #    return prob

    p1 = len(df[df[col1] == event1])/len(df)
    
    if col2 == 'systemic_crisis':
        p2 = len(df[df['systemic_crisis'] == event2])/len(df)
        a = np.array(df['banking_crisis'])
        b = np.array(df['systemic_crisis'])
        print(pd.crosstab(a,b))
        table = pd.crosstab(a,b)
        x = table.iloc[0,1]
        y = table.iloc[0,:].sum()
        p_b_a = x/y

        prob = (p_b_a*p1)/p2
        return prob

    if col2 == 'inflation_crises':
        p2 = len(df[df['inflation_crises'] == event2])/len(df)
        a = np.array(df['banking_crisis'])
        b = np.array(df['inflation_crises'])
        print(pd.crosstab(a,b))
        table = pd.crosstab(a,b)
        x = table.iloc[0,1]
        y = table.iloc[0,:].sum()
        p_b_a = x/y

        prob = (p_b_a*p1)/p2
        return prob

    if col2 == 'currency_crises':
        p2 = len(df[df['currency_crises'] == event2])/len(df)
        a = np.array(df['banking_crisis'])
        b = np.array(df['currency_crises'])
        print(pd.crosstab(a,b))
        table = pd.crosstab(a,b)
        x = table.iloc[0,1]
        y = table.iloc[0,:].sum()
        p_b_a = x/y

        prob = (p_b_a*p1)/p2
        return prob

# Load the dataset
df = pd.read_csv(path)
print(df.head())
#print(df.dtypes)

# Using the visual_summary(), visualize the distribution of the data provided.
# You can also do it at country level or based on years by passing appropriate arguments to the fuction.
visual_summary('hist', df, 'exch_usd')
visual_summary('kde', df, 'exch_usd')
visual_summary('barh', df, 'banking_crisis')
visual_summary('bar', df, 'country')

# You might also want to see the central tendency of certain variables. Call the central_tendency() to do the same.
# This can also be done at country level or based on years by passing appropriate arguments to the fuction.
central_tendency('mean', df, 'exch_usd')
central_tendency('median', df, 'year')
central_tendency('mode', df, 'cc3')


# Measures of dispersion gives a good insight about the distribution of the variable.
# Call the measure_of_dispersion() with desired parameters and see the summary of different variables.
measure_of_dispersion('range', df, 'gdp_weighted_default')
measure_of_dispersion('mean absolute deviation', df, 'inflation_annual_cpi')
measure_of_dispersion('standard deviation', df, 'exch_usd')
measure_of_dispersion('coefficient of variation', df, 'exch_usd')
measure_of_dispersion('iqr', df, 'inflation_annual_cpi')
measure_of_dispersion('covariance', df, col = ['inflation_annual_cpi','exch_usd'])


# There might exists a correlation between different variables. 
# Call the calculate_correlation() to check the correlation of the variables you desire.
calculate_correlation('pearson', df, 'gdp_weighted_default', 'exch_usd')
calculate_correlation('spearman', df, 'inflation_annual_cpi', 'exch_usd')


# From the given data, let's check the probability of banking_crisis for different countries.
# Call the calculate_probability_discrete() to check the desired probability.
# Also check which country has the maximum probability of facing the crisis.  
# You can do it by storing the probabilities in a dictionary, with country name as the key. Or you are free to use any other technique.
sa_bc = calculate_probability_discrete('South Africa', 'banking_crisis')
print('Probability of Banking Crisis in South Africe is',sa_bc)

z_bc = calculate_probability_discrete('Zimbabwe', 'banking_crisis')
print('Probability of Banking Crisis in Zimbabwe is',z_bc)

dict1 = {}
for place in df['country'].unique():
    newdf1 = df[df['country'] == place]
    total_ = newdf1.shape[0]
    crisis_ = newdf1[newdf1['banking_crisis'] == 'crisis'].shape[0]
    crisis_prob = crisis_/total_
    dict1.update({place:crisis_prob})
print(max(dict1, key=dict1.get))

# Next, let us check if banking_crisis is independent of systemic_crisis, currency_crisis & inflation_crisis.
# Calculate the probabilities of these event using calculate_probability_discrete() & joint probabilities as well.
# Then call event_independence_check() with above probabilities to check for independence.

def joint_probability_with_bankingcrisis(country, crisis_type):

    df_new = df[df['country'] == country]
    total = len(df_new)

    if crisis_type == 'systemic_crisis':
        sc = len(df_new[(df_new['banking_crisis'] == 'crisis') & (df_new['systemic_crisis'] == 1)])
        prob = sc/total
        return prob

    if crisis_type == 'inflation_crises':
        ic = len(df_new[(df_new['banking_crisis'] == 'crisis') & (df_new['inflation_crises'] == 1)])
        prob = ic/total
        return prob

    if crisis_type == 'currency_crises':
        cc = len(df_new[(df_new['banking_crisis'] == 'crisis') & (df_new['currency_crises'] == 1)])
        prob = cc/total
        return prob

prob_nigeria_bc = calculate_probability_discrete('Nigeria', 'banking_crisis')
prob_nigeria_sc = calculate_probability_discrete('Nigeria', 'systemic_crisis')
prob_nigeria_ic = calculate_probability_discrete('Nigeria', 'inflation_crises')
prob_nigeria_cc = calculate_probability_discrete('Nigeria', 'currency_crises')

print('Probability of Banking Crisis in Nigeria is',prob_nigeria_bc)
print('Probability of Systemic Crisis in Nigeria is',prob_nigeria_sc)
print('Probability of Inflation Crisis in Nigeria is',prob_nigeria_ic)
print('Probability of Currency Crisis in Nigeria is',prob_nigeria_cc)

jp1 = joint_probability_with_bankingcrisis('Nigeria', 'inflation_crises')
print('Joint prob of Banking Crisis and Inflation Crisis for Nigeria',jp1)

jp2 = joint_probability_with_bankingcrisis('Nigeria', 'currency_crises')
print('Joint prob of Banking Crisis and Currency Crisis for Nigeria',jp2)

jp3 = joint_probability_with_bankingcrisis('Nigeria', 'systemic_crisis')
print('Joint prob of Banking Crisis and Systemic Crisis for Nigeria',jp3)

event_independence_check(prob_nigeria_bc, prob_nigeria_ic, jp1)
event_independence_check(prob_nigeria_bc, prob_nigeria_cc, jp2)
event_independence_check(prob_nigeria_bc, prob_nigeria_sc, jp3)

# Finally, let us calculate the probability of banking_crisis given that other crises (systemic_crisis, currency_crisis & inflation_crisis one by one) have already occured.
# This can be done by calling the bayes_theorem() you have defined with respective parameters.

bayes1 = bayes_theorem(df, 'banking_crisis', 'crisis', 'systemic_crisis', 1)
print('prob of banking crisis given systemic crisis :',bayes1)

bayes2 = bayes_theorem(df, 'banking_crisis', 'crisis', 'currency_crises', 1)
print('prob of banking crisis given currency crisis :',bayes2)

bayes3 = bayes_theorem(df, 'banking_crisis', 'crisis', 'inflation_crises', 1)
print('prob of banking crisis given inflation crisis :',bayes3)

prob_ = []
prob_.extend((bayes1, bayes2, bayes3))
print(prob_)

# Code ends


