import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import math, random
from sklearn.cluster import KMeans #cluster
from sklearn.decomposition import PCA #principle
from scipy.spatial.distance import cdist #cdist
from numpy import dot



_id=[]
lab_1=[]
christmas_test=[]
lab_2=[]
easter_test=[]
lab_3=[]
part_time_job=[]
exam_grade=[]
f = ("_id","lab_1","christmas_test","lab_2","easter_test","lab_3","part_time_job","exam_grade")
csvfile = open('/Users/elainechong/Desktop/Sem_5/Final_CA/Elaine_Chong/FinalProjectData.csv', newline='')
reader = csv.DictReader(csvfile, delimiter=';', fieldnames=f)
header = reader.__next__()
for row in reader:
    _id.append(int(row['_id']))
    lab_1.append(int(row['lab_1']))
    christmas_test.append(int(row['christmas_test']))
    lab_2.append(int(row['lab_2']))
    easter_test.append(int(row['easter_test']))
    lab_3.append(int(row['lab_3']))
    part_time_job.append(int(row['part_time_job']))
    exam_grade.append(int(row['exam_grade']))


x_m =[]
y = exam_grade
for i in range(len(exam_grade)):
    x_m.append([1,lab_1[i],christmas_test[i],lab_2[i],easter_test[i],lab_3[i],part_time_job[i]])
x_m_2 =[]
for i in range(len(exam_grade)):
    x_m_2.append([1,lab_1[i],christmas_test[i],lab_2[i],easter_test[i],lab_3[i]])

x=[]
for i in range(len(exam_grade)):
    x.append([lab_1[i],christmas_test[i],lab_2[i],easter_test[i],lab_3[i],part_time_job[i],exam_grade[i]])

def mean(x):
	return sum(x)/len(x)

def median(x):
	n = len(x)
	sorted_x  =sorted(x)
	midpoint = n //2
	if n% 2==1:
		return sorted_x[midpoint]
	else:
		lo = midpoint-1
		hi = midpoint //2
		return(sorted_x[lo] + sorted_x[hi]) / 2

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

def quantile(x,p):
	p_index =int(p*len(x))
	return sorted(x)[p_index]

def interQuantile_range(x):
	return quantile(x,0.75) - quantile(x,0.25)

def extreme_outliers_value(x):
	lower_outlier = quantile(x,0.25) - 3* interQuantile_range(x)
	higher_outlier = quantile(x,0.75) + 3* interQuantile_range(x)
	return lower_outlier,higher_outlier
def mild_outliers_value(x):
	lower_outlier = quantile(x,0.25) - 1.5* interQuantile_range(x)
	higher_outlier = quantile(x,0.75) + 1.5* interQuantile_range(x)
	return lower_outlier,higher_outlier

def range_(x):
	return max(x)  - min(x)

def de_mean(x):
	x_bar = mean(x)
	return [x_i - x_bar for x_i in x]

def variance(x):
	n = len(x)
	deviations = de_mean(x)
	return sum_of_square(deviations) / (n-1)

def standard_deviation(x):
	return math.sqrt(variance(x))

def covariance(x,y):
	n =len(x)
	return dot(de_mean(x),de_mean(y)) / (n-1)

def correlation(x,y):
	stdev_x = standard_deviation(x)
	stdev_y = standard_deviation(y)
	if stdev_x > 0 and stdev_y > 0:
		return covariance(x,y) / stdev_x / stdev_y
	else:
		return 0

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)
def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))
# alpha beta
def predict(beta0, beta1, x_i):
    return beta1 * x_i + beta0

#beta11*x + beta0
def least_squares_fit(x,y):
    beta1 = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    beta0 = mean(y) - beta1 * mean(x)
    return beta0, beta1
def total_sum_of_squares(y):
    return sum(v ** 2 for v in de_mean(y))
def r_squared(alpha, beta, x, y):
    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) /
                  total_sum_of_squares(y))

def scatterPlot(x,y,xLabel,yLabel,title):
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def regressionScatter(x,y,predictY,xLabel,yLabel,title):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot(x, predictY , color='red')
    plt.show()

#MLR

def sum_of_square(x):
	return dot(x,x)
def multi_squared_error(v,x,y):
    return multi_error(v,x,y)**2

def multi_predict_y(v,x_i):
    return dot(x_i,v)

def multi_error(v,x_i,y_i):
    error = y_i -multi_predict_y(v,x_i)
    return error

def partial_difference_quotient_s(f,v,i,x,y,h):
    w = [v_j +(h if j == i else 0) for j,v_j in enumerate(v)]
    return (f(w,x,y) - f(v,x,y))/h

def estimate_gradient_s(f,v,x,y,h =0.01):
    return [partial_difference_quotient_s(f,v,i,x,y,h) for i, _ in enumerate(v)]

def step(v,direction,step_size):
    return [v_i - step_size * direction_i for v_i,direction_i in zip(v,direction)]

#P-value after MLR
def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)
#CLUSTER
# def shape(M):
#     num_row=len(M)
#     num_colm =len(M[0])
    # return
def get_column(M,j):
    return [M_i[j] for M_i in M]

def checkRange(x):
    for i in x:
        if i < 0 or i > 100:
            return 1

    return 0
'''
scatterPlot(lab_1,exam_grade,"Lab 1", "Exam Grade", "Lab 1 vs Exam Grade")
scatterPlot(christmas_test,exam_grade,"Christmas Test", "Exam Grade", "Christmas Test vs Exam Grade")
scatterPlot(lab_2,exam_grade,"Lab 2", "Exam Grade", "Lab 2 vs Exam Grade")
scatterPlot(easter_test,exam_grade,"Easter Test", "Exam Grade", "Easter Test vs Exam Grade")
scatterPlot(lab_3,exam_grade,"Lab 3", "Exam Grade", "Lab 3 vs Exam Grade")
scatterPlot(part_time_job,exam_grade,"Part Time Job", "Exam Grade", "Part Time Job vs Exam Grade")

print("Check Range")
print("Lab 1          ", checkRange(lab_1))
print("Christmas Test ", checkRange(christmas_test))
print("Lab 2          ",checkRange(lab_2))
print("Easter Test    ",checkRange(easter_test))
print("Lab 3          ",checkRange(lab_3))
print("Exam Grade     ",checkRange(exam_grade))

print("Check Length to identify missing value")
print("Lab 1          ", len(lab_1))
print("Christmas Test ", len(christmas_test))
print("Lab 2          ",len(lab_2))
print("Easter Test    ",len(easter_test))
print("Lab 3          ",len(lab_3))
print("Part Time Job  ",len(part_time_job))
print("Exam Grade     ",len(exam_grade))
print("Computed sum to check integer")
print("Lab 1          ", sum(lab_1))
print("Christmas Test ", sum(christmas_test))
print("Lab 2          ",sum(lab_2))
print("Easter Test    ",sum(easter_test))
print("Lab 3          ",sum(lab_3))
print("Exam Grade     ",sum(exam_grade))

print("˜˜˜˜˜˜˜MEAN˜˜˜˜˜˜")
print ("Mean of Lab 1: ", mean(lab_1))
print ("Mean of Christmas Test: ", mean(christmas_test))
print ("Mean of Lab 2: ", mean(lab_2))
print ("Mean of Easter Test: ", mean(easter_test))
print ("Mean of Lab 3: ", mean(lab_3))
print ("Mean of Exam Grade: ", mean(exam_grade))

print ("First 3 tests")
print ((mean(lab_1)+mean(christmas_test)+mean(lab_2))/3)
print ("Last 3 tests")
print ((mean(easter_test)+mean(lab_3)+mean(exam_grade))/3)
print()

#median of all data
print("˜˜˜˜˜˜˜MEDIAN˜˜˜˜˜˜")
print ("Median of Lab 1: ", median(lab_1))
print ("Median of Christmas Test: ", median(christmas_test))
print ("Median of Lab 2: ", median(lab_2))
print ("Median of Easter Test: ", median(easter_test))
print ("Median of Lab 3: ", median(lab_3))
print ("Median of Exam Grade: ", median(exam_grade))
print()
#mode of all data
print("˜˜˜˜˜˜˜MODE˜˜˜˜˜˜")
print ("Mode of Lab 1: ", mode(lab_1))
print ("Mode of Christmas Test: ", mode(christmas_test))
print ("Mode of Lab 2: ", mode(lab_2))
print ("Mode of Easter Test: ", mode(easter_test))
print ("Mode of Lab 3: ", mode(lab_3))
print ("Mode of Part Time Job: ", mode(part_time_job))
print ("Mode of Exam Grade: ", mode(exam_grade))
print()

#range of all data
print("˜˜˜˜˜˜˜Range˜˜˜˜˜˜")
print ("Range of Lab 1: ", range_(lab_1))
print ("Range of Christmas Test: ", range_(christmas_test))
print ("Range of Lab 2: ", range_(lab_2))
print ("Range of Easter Test: ", range_(easter_test))
print ("Range of Lab 3: ", range_(lab_3))
print ("Range of Part Time Job: ", range_(part_time_job))
print ("Range of Exam Grade: ", range_(exam_grade))
print()
#quantile
print("˜˜˜˜˜˜˜Quantile˜˜˜˜˜˜")
print ("Lower quantile  of Lab 1: ", quantile(lab_1,0.25))
print ("Upper quantile  of Lab 1: ", quantile(lab_1,0.75))
print ("IQR of Lab 1: ",interQuantile_range(lab_1))
print()
print ("Lower quantile  of Christmas Test: ", quantile(christmas_test,0.25))
print ("Upper quantile  of Christmas Test: ", quantile(christmas_test,0.75))
print ("IQR of Christmas Test: ",interQuantile_range(christmas_test))
print()
print ("Lower quantile  of Lab 2: ", quantile(lab_2,0.25))
print ("Upper quantile  of Lab 2:", quantile(lab_2,0.75))
print ("IQR of Lab 2: ",interQuantile_range(lab_2))
print()
print ("Lower quantile  of Easter Test: ", quantile(easter_test,0.25))
print ("Upper quantile  of Easter Test: ", quantile(easter_test,0.75))
print ("IQR of Easter Test: ",interQuantile_range(easter_test))
print()
print ("Lower quantile  of Lab 3: ", quantile(lab_3,0.25))
print ("Upper quantile  of Lab 3: ", quantile(lab_3,0.75))
print ("IQR of Lab 3: ",interQuantile_range(lab_3))
print()
print ("Lower quantile  of Exam Grade: ", quantile(exam_grade,0.25))
print ("Upper quantile  of Exam Grade: ", quantile(exam_grade,0.75))
print ("IQR of Exam Grade: ",interQuantile_range(exam_grade))
print()

#variance of all data
print("˜˜˜˜˜˜˜Variance˜˜˜˜˜˜")
print ("Variance of Lab 1: ", variance(lab_1))
print ("Variance of Christmas Test: ", variance(christmas_test))
print ("Variance of Lab 2: ", variance(lab_2))
print ("Variance of Easter Test: ", variance(easter_test))
print ("Variance of Lab 3: ", variance(lab_3))
print ("Variance of Part Time Job: ", variance(part_time_job))
print ("Variance of Exam Grade: ", variance(exam_grade))
print()
#Standard Deviation of all data
print("˜˜˜˜˜˜˜Standard Deviation˜˜˜˜˜˜")
print ("Standard Deviation of Lab 1: ", standard_deviation(lab_1))
print ("Standard Deviation of Christmas Test: ", standard_deviation(christmas_test))
print ("Standard Deviation of Lab 2: ", standard_deviation(lab_2))
print ("Standard Deviation of Easter Test: ", standard_deviation(easter_test))
print ("Standard Deviation of Lab 3: ", standard_deviation(lab_3))
print ("Standard Deviation of Part Time Job: ", standard_deviation(part_time_job))
print ("Standard Deviation of Exam Grade: ", standard_deviation(exam_grade))
print()
def extreme_outliers_value(x):
	lower_outlier = quantile(x,0.25) - 3* interQuantile_range(x)
	higher_outlier = quantile(x,0.75) + 3* interQuantile_range(x)
	return lower_outlier,higher_outlier
def mild_outliers_value(x):
	lower_outlier = quantile(x,0.25) - 1.5* interQuantile_range(x)
	higher_outlier = quantile(x,0.75) + 1.5* interQuantile_range(x)
	return lower_outlier,higher_outlier
#Covariance
print("Covariance of christmas test and exam grade: " , covariance(christmas_test,exam_grade))
print("Covariance of easter test and exam grade:    " , covariance(easter_test,exam_grade))
print("Covariance of lab 3 and exam grade:          " , covariance(lab_3,exam_grade))
print()
#Correaltion
print("Correlation of christmas test and no of sales: " , correlation(christmas_test,exam_grade))
print("Correlation of easter test and exam grade:     " , correlation(easter_test,exam_grade))
print("Correlation of lab 3 and exam grade:           " , correlation(lab_3,exam_grade))

beta0, beta1 = least_squares_fit(christmas_test, exam_grade)
predictExamGrade =[predict(beta0,beta1,x_i) for x_i in christmas_test]
regressionScatter(christmas_test,exam_grade,predictExamGrade,"Christmas Test", "Exam Grade","Christmas Test vs Exam Grade")
print("Christmast Test vs Exam Grade")
print("R-squared: ",r_squared(beta0, beta1, christmas_test, exam_grade))
print ("beta0", beta0)
print ("beta1", beta1)
print("y=beta1(x)+beta0")
print("y=",beta1,"x+",beta0)
print()
beta0, beta1 = least_squares_fit(easter_test, exam_grade)
predictExamGrade =[predict(beta0,beta1,x_i) for x_i in easter_test]
regressionScatter(easter_test,exam_grade,predictExamGrade,"Easter Test", "Exam Grade","Easter Test vs Exam Grade")
print("Easter Test vs Exam Grade")
print("R-squared: ",r_squared(beta0, beta1, easter_test, exam_grade))
print ("beta0", beta0)
print ("beta1", beta1)
print("y=beta1(x)+beta0")
print("y=",beta1,"x+",beta0)
print()
beta0, beta1 = least_squares_fit(lab_3, exam_grade)
predictExamGrade =[predict(beta0,beta1,x_i) for x_i in lab_3]
regressionScatter(lab_3,exam_grade,predictExamGrade,"Lab 3", "Exam Grade","Lab 3 vs Exam Grade")
print("Lab 3 vs Exam Grade")
print("R-squared: ",r_squared(beta0, beta1, lab_3, exam_grade))
print ("beta0", beta0)
print ("beta1", beta1)
print("y=beta1(x)+beta0")
print("y=",beta1,"x+",beta0)
print()

def sum_of_squares(v):
	return sum(v_i**2 for v_i in v)

def multiple_r_squared(x,y,v):
    sum_of_squared_errors = sum(multi_error(v,x_i, y_i) ** 2
                                for x_i, y_i in zip(x, y))
#  sum_of_squared_errors=error of model(take away)
    return 1.0 - (sum_of_squared_errors / sum_of_squares(de_mean(y)))
def gradient_descent_stochastic(x,y,f =multi_squared_error):
    random.seed(0)
    v = [random.randint(-10,10) for x in x[0]]
    step_size_0 = 0.00001
    iterations_with_no_improvement = 0
    min_v=None
    min_value=float("inf")

    while iterations_with_no_improvement < 1000:
        value = sum( f(v,x_i, y_i) for x_i, y_i in zip(x,y))
        if value < min_value:
            min_v, min_value = v, value
            iterations_with_no_improvement = 0
            step_size = step_size_0
        else:
            iterations_with_no_improvement += 1
            step_size *= 0.9
        indexes = np.random.permutation(len(x))
        for i in indexes:
            x_i = x[i]
            y_i = y[i]
            gradient_i = estimate_gradient_s(f,v,x_i,y_i)
            v = step(v,gradient_i, step_size)
    return min_v


def bootstrap_sample_m(data):
    list_data=list(data)
    rand_data=[random.choice(list_data) for _ in list_data]
    return rand_data
def bootstrap_statistic_m(x_data,y_data, stats_fn, num_samples):
    stats=[]
    for i in range(num_samples):
        New_sample=bootstrap_sample_m(zip(x_data,y_data))
        x_sample,y_sample= zip(*New_sample)
#        separate/split it into x and y
        x=list(x_sample);
        y=list(y_sample)
        stat = stats_fn(x,y)
        stats.append(stat)
    return stats

# test1 =[3.0399, 0.1065, 0.2477, 0.1355, 0.3264, 0.1296, 0.5709]
# print("r-squared", multiple_r_squared(x_m, y, test1))


# with part time job
print("With part time job")
estimate_v=gradient_descent_stochastic(x_m,y,multi_squared_error)
print("The v is ",estimate_v)
print(multiple_r_squared(x_m,y,estimate_v))

coefficients = bootstrap_statistic_m(x_m,y,gradient_descent_stochastic,5)
bootstrap_standard_errors = [standard_deviation([coefficient[i] for coefficient in coefficients]) for i in range(len(estimate_v))]
for i in range(len(estimate_v)):
	print("i: ",i,"estimate_v",estimate_v[i],"error", bootstrap_standard_errors[i],"p-value", p_value(estimate_v[i], bootstrap_standard_errors[i]))
# print("bootstrap standard errors", bootstrap_standard_errors)

# without part time Job
print()
print("Without part_time_job")
estimate_v=gradient_descent_stochastic(x_m_2,y,multi_squared_error)
print("The v is ",estimate_v)
print(multiple_r_squared(x_m_2,y,estimate_v))
coefficients = bootstrap_statistic_m(x_m_2,y,gradient_descent_stochastic,5)
bootstrap_standard_errors = [standard_deviation([coefficient[i] for coefficient in coefficients]) for i in range(len(estimate_v))]
for i in range(len(estimate_v)):
	print("i: ",i,"estimate_v",estimate_v[i],"error", bootstrap_standard_errors[i],"p-value", p_value(estimate_v[i], bootstrap_standard_errors[i]))
'''
#CLUSTER
# determining the optimum number of clusters , choosing k
no_clusters = range(1,11)
average_dist=[]

for k in no_clusters:
    modelk = KMeans(k)
    modelk.fit(x)
    cluster_assign = modelk.predict(x)
    average_dist.append(sum(np.min(cdist(x,modelk.cluster_centers_,'euclidean'),axis=1))/len(x_m))

mean2 = modelk.cluster_centers_
plt.plot(no_clusters,average_dist)
plt.xlabel("Number of cluster")
plt.ylabel("Average ")
plt.title("Elbow plot")
plt.show()

#count

n_clusters =3;
model = KMeans(n_clusters)
model.fit(x)
cluster_assign = model.predict(x)
mean2 = model.cluster_centers_
print(mean2)
plt.scatter(x=get_column(x,0),y=get_column(x,1),c=model.labels_)
plt.title('Scatterplot for 3 Clusters')
plt.show()

# Principal Component Analysis - Screeplot
pca =PCA()
plot_data = pca.fit_transform(x)
PC = pca.components_
PCEV=pca.explained_variance_
PCEVR=pca.explained_variance_ratio_
x_pca=[i+1 for i in range(len(PC))]
plt.plot(x_pca, PCEVR)
plt.xlabel('Principal Component')
plt.ylabel('Proportion of variance explained')
plt.title('Scree-plot')
plt.show()

#PCA
pca =PCA(2)
plot_data = pca.fit_transform(x)
PC = pca.components_
PCEV=pca.explained_variance_
PCEVR=pca.explained_variance_ratio_
print("principal components are:")
print(PC)
print("variance explained by each PC is")
print(PCEV)
print("proportion of variance explained by each PC is")
print(PCEVR)
print("transformed data", plot_data)

plt.scatter(x=plot_data[:,0], y=plot_data[:,1], c=model.labels_,)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatterplot of Principal Components for 3 Clusters')
plt.show()
