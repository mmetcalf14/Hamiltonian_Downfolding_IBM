import matplotlib.pyplot as plt


class Plotting:

    def sort_x(self, l1, l2):
        """
        :param l1: (list) the list representing the x coordinates.
        :param l2: (list) the list representing the y coordinates.
        :return: a nested list where the first element is l1 sorted in ascending order, and the second element is l2 sorted
        based on l1.
        """

        def helper(unsorted1, unsorted2, sorted1, sorted2):
            to_be_added1, to_be_added2 = [], []
            if len(unsorted1) == 0:
                return [sorted1, sorted2]
            else:
                minimum = min(unsorted1)
                corresponding = unsorted2[unsorted1.index(minimum)]
                unsorted1.remove(minimum)
                unsorted2.remove(corresponding)
                to_be_added1.append(minimum)
                to_be_added2.append(corresponding)
                return helper(unsorted1, unsorted2, sorted1 + to_be_added1, sorted2 + to_be_added2)

        return helper(l1, l2, [], [])

    def plot(self, file, save=False):
        """
        :param file: .dat file where data is saved.
        :param save: (boolean) if the plot will be saved or not.
        """
        num_figure = input("Enter the number of data sets you are plotting: ")
        index = 0
        while index < int(num_figure):
            x_index = input("Enter the index of the column for the x axis: ")
            y_index = input("Enter the index of the column for the y axis: ")
            line_color = input("Enter the color you would like to use (blue, "
                               "green, red, cyan, magenta, yellow, black, white) ")
            line_label = input("Enter a label: ")

            x, y = [], []
            for line in open(file, 'r'):
                values = [float(s) for s in line.split()]

                x.append(self.sort_x(values[int(x_index)], values[int(y_index)])[0])
                y.append(self.sort_x(values[int(x_index)], values[int(y_index)])[1])

            plt.plot(x, y, color=line_color, marker='o', label=line_label)
            index += 1
        plt.legend()
        plt.show()
        if save:
            name = input("Enter the name you would like to save it as ")
            plt.savefig(name)
