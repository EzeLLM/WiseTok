import numpy as np

def calc_stats(file):
  data = np.loadtxt(file,delimiter=',')
  data = np.asarray(data, float)
  mean = np.mean(data)
  median = np.median(data)
  return round(mean,1), round(median,1)

if __name__ == '__main__':

  mean = calc_stats('data.csv')
  mean1 = calc_stats('data2.csv')
  mean2 = calc_stats('data3.csv')
  print(mean)
  print(mean1)
  print(mean2)
