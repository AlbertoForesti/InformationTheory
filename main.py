# This is a sample Python script.
import pandas as pd

from ITFirstExercise import MaxInformationFinder
import numpy as np

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "C:\\Users\\Gian Luca Foresti\\Desktop\\Materiale Uni\\4 - anno\\IT"
    """MaxInformationFinder.max_entropy_vector_plot()
    MaxInformationFinder.average_vector_entropy_plot()
    costs = np.asarray( [ 4, 7, 10, 20, 15 ] )
    MaxInformationFinder.meal_pmf(costs, 9, path+"\\Ex2A.png")
    print(np.average(costs))
    MaxInformationFinder.meal_pmf(costs, np.average(costs), path+"\\Ex2B.png" )
    MaxInformationFinder.meal_pmf( costs, 13, path + "\\Ex2B13.png" )
    MaxInformationFinder.meal_pmf( costs, 5, path + "\\Ex2B5.png" )"""
    juve_data = np.asarray([[14./38, 6./38, 0], [5./38, 3./38, 2./38], [1./38, 1./38, 6./38]])
    MaxInformationFinder.joint_distribution_stats(juve_data)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
