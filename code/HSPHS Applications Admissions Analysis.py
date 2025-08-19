#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:19:22 2020

@author: oviya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("middleSchoolData.csv")
data = np.genfromtxt(df, delimiter = ',')


#%% Q1: Correlation b/w applications and admissions

applications = df["applications"]
admissions = df["acceptances"]
app_ad_corr = applications.corr(admissions)
corr = str(app_ad_corr)

print("Question 1: The correlation between the number of applications and admissions is " + corr)

# Plot
f1 = plt.figure(1)
x = df.applications
y = df.acceptances
plt.scatter(x,y)
plt.xlabel('Number of applications')
plt.ylabel('Number of admissions')



#%% Q2: Which is better predictor for admission to HSPS: raw number of applications or application rate

df['application_rate'] = df['applications']/df['school_size']
application_rate = df['application_rate']
apprate_ad_corr = application_rate.corr(admissions)
corr2 = str(apprate_ad_corr)
print("\nQuestion 2: The correlation between the application rate and admissions is " + corr2)

## Compare correlations
if app_ad_corr > apprate_ad_corr:
    print("Raw number of applications is a better predictor for admissions.")
else:
    print ("Application rate is a better predictor for admissions.")


# Plot
f2 = plt.figure(2)
a = df.application_rate
b = df.acceptances
plt.scatter(a,b)
plt.xlabel('Application rate')
plt.ylabel('Number of admissions')




#%% Q3: Best per-student odds of sending someone to HSPHS
    
df['per_student_odds'] = df['acceptances']/df['school_size']
per_student_odds = df['per_student_odds']
max_value = per_student_odds.max()
max_val_str = str(max_value)
print("\nQuestion 3: The highest per-student odds of sending someone to HSPHS is " + max_val_str)


index = df[df['per_student_odds'] == max_value].index.values
name = df.loc[index, 'school_name']
print(name)
print("The school with the highest per-student odds of sending a student to HSPHS is shown above.")




#%% Q4: Relationship b/w students' perception of school and school performance

df['student_perception'] = df['rigorous_instruction'] + df['collaborative_teachers'] + \
df['supportive_environment'] + df['effective_school_leadership'] + df['strong_family_community_ties'] + df['trust']
student_perception = df['student_perception']
df['school_performance'] = df['student_achievement'] + df['reading_scores_exceed'] + df['math_scores_exceed']
school_performance = df['school_performance']
perc_perf_corr = student_perception.corr(school_performance)
corr3 = str(perc_perf_corr)
print("\nQuestion 4: The correlation between student perception and school performance is " + corr3)

abs_corr3 = abs(perc_perf_corr)

if abs_corr3 == 1:
    print("There is a perfect correlation between student perception and school performance.")
elif abs_corr3 >= 0.5:
    print("There is a high/strong correlation between student perception and school performance.")
elif abs_corr3 >= 0.3:
    print("There is a moderate correlation between student perception and school performance.")
elif abs_corr3 >= 0:
    print("There is a low/weak correlation between student perception and school performance.")
else:
    print("Error: Correlation is out of range.")


# Plot
f3 = plt.figure(3)
j = df.student_perception
k = df.school_performance
plt.scatter(j,k)
plt.xlabel('Student Perception')
plt.ylabel('School Performance')
    
    
    
#%% Q5: Testing a custom hypothesis: The higher the disability percent, the lower the acceptance rate

df['acceptance_rate'] = df['acceptances']/df['applications']
acceptance_rate = df['acceptance_rate']
disability = df['disability_percent']
accrate_dis_corr = acceptance_rate.corr(disability)
corr4 = str(accrate_dis_corr)
print("\nQuestion 5: The correlation between acceptance rate and disability percentage is " + corr4)

abs_corr4 = abs(accrate_dis_corr)

if abs_corr4 == 1:
    print("There is a perfect correlation between acceptance rate and disability percentage.")
elif abs_corr4 >= 0.5:
    print("There is a high/strong correlation between acceptance rate and disability percentage.")
elif abs_corr4 >= 0.3:
    print("There is a moderate correlation between acceptance rate and disability percentage.")
elif abs_corr4 >= 0:
    print("There is a low/weak correlation between acceptance rate and disability percentage.")
else:
    print("Error: Correlation is out of range.")
    
# Plot
f4 = plt.figure(4)
q = df.disability_percent
r = df.acceptance_rate
plt.scatter(q,r)
plt.xlabel('Disability Percentage')
plt.ylabel('Acceptance Rate')



#%% Q6: Finding evidence that the availibility of resources (per student spending) impacts achievement (admissions in this case)

df['acceptance_rate'] = df['acceptances']/df['applications']
acceptance_rate = df['acceptance_rate']
spending = df['per_pupil_spending']
spending_accrate_corr = acceptance_rate.corr(spending)
corr5 = str(spending_accrate_corr)
print("\nQuestion 6: The correlation between acceptance rate and per-student spending is " + corr5)

abs_corr5 = abs(spending_accrate_corr)

if abs_corr5 == 1:
    print("There is a perfect correlation between acceptance rate and disability percentage.")
elif abs_corr5 >= 0.5:
    print("There is a high/strong correlation between acceptance rate and disability percentage.")
elif abs_corr5 >= 0.3:
    print("There is a moderate correlation between acceptance rate and disability percentage.")
elif abs_corr5 >= 0:
    print("There is a low/weak correlation between acceptance rate and disability percentage.")
else:
    print("Error: Correlation is out of range.")

# Plot
f5 = plt.figure(5)
a = df.per_pupil_spending
b = df.acceptance_rate
plt.scatter(a,b)
plt.xlabel('Per Pupil Spending')
plt.ylabel('Acceptance Rate')



#%% Q7: What proportion of schools accounts for 90% of all students accepted to HSPHS?
    
# Sorted acceptances list in descending order
accept = df.sort_values('acceptances', ascending=False)['acceptances']
# Get number of acceptances that make up 90%
total_acceptances = df['acceptances']
threshold = total_acceptances.sum() * 0.9
total_schools = len(accept)
num_schools = 0
num_acceptances = 0

for i in range(len(accept)):
    if num_acceptances < threshold:
        num_acceptances += accept[i]
        num_schools += 1
    else:
        break
    prop = num_schools/ total_schools
    perc = prop * 100

prop_str = str(prop)
perc_str = str(perc)

print("\nQuestion 7: " + prop_str + " or  " + perc_str + "% of all NYC schools make up 90% of admissions to HSPHS.")



#%% Q8: Multiple regression model

from sklearn import linear_model

X_df = df[['applications', 'acceptances', 'per_pupil_spending', 'avg_class_size', 'asian_percent', 'black_percent', \
        'hispanic_percent', 'multiple_percent', 'white_percent','rigorous_instruction', 'collaborative_teachers',\
        'supportive_environment', 'effective_school_leadership', 'strong_family_community_ties', 'trust', \
        'disability_percent', 'poverty_percent', 'ESL_percent', 'school_size', 'student_achievement', \
        'reading_scores_exceed', 'math_scores_exceed']]
Y1_df = df[['acceptances']]
Y2_df = df[['school_performance']]


# Desriptives
d1 = np.mean(data,axis=0)
d2 = np.median(data,axis=0)
d3 = np.std(data,axis=0)
d4 = np.corrcoef(data[:,0],data[:,1])


# For acceptances regression
X1 = np.transpose([data[:,2],data[:,4],data[:,5], data[:,6], data[:,6], data[:,7], data[:,8], data[:,9], \
                data[:,10], data[:,11], data[:,12], data[:,13], data[:,14], data[:,15], data[:,16], \
                data[:,17], data[:,18], data[:,19], data[:20], data[:,21], data[:,22], data[:,23]]) 
Y1 = data[:,3]
regr = linear_model.LinearRegression() 
regr.fit(X1,Y1) 
r_sqr = regr.score(X1,Y1)
betas = regr.coef_
y_int = regr.intercept_  
# Plot
y_hat = betas[0]*data[:,0] + betas[1]*data[:,1] + betas[2]*data[:,2] + y_int
plt.plot(y_hat,data[:,3],'o',markersize=.75)
plt.xlabel('Prediction') 
plt.ylabel('Actual acceptances')  
plt.title('R^2: {:.3f}'.format(r_sqr))


# For achievement regression
X2 = np.transpose([data[:,2], data[:,3], data[:,4],data[:,5], data[:,6], data[:,6], data[:,7], data[:,8], \ 
                data[:,9], data[:,10], data[:,11], data[:,12], data[:,13], data[:,14], data[:,15], data[:,16], \
                data[:,17], data[:,18], data[:,19], data[:20]]) 
Y2 = np.transpose([data[:,21], data[:,22], data[:,23]])
regr = linear_model.LinearRegression() 
regr.fit(X2,Y2) 
r_sqr = regr.score(X2,Y2)
betas = regr.coef_
y_int = regr.intercept_  
# Plot
y_hat = betas[0]*data[:,0] + betas[1]*data[:,1] + betas[2]*data[:,2] + y_int
plt.plot(y_hat,data[:,21],'o',markersize=.75)
plt.xlabel('Prediction') 
plt.ylabel('Actual school performance')  
plt.title('R^2: {:.3f}'.format(r_sqr))





