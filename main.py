# This is a sample Python script.
from ITFirstExercise import MaxInformationFinder
import numpy as np

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #MaxInformationFinder.max_entropy_vector_plot()
    #MaxInformationFinder.average_vector_entropy_plot()
    costs = np.asarray( [ 3.97542, 12.72573, 8.1325, 5, -2.1625 ] )
    MaxInformationFinder.meal_pmf(costs, np.average(costs))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
