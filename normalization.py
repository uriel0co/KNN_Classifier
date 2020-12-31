from point import Point
from numpy import mean, var


class DummyNormalizer:
    # this class doesnt normalize the data.
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class SumNormalizer:
    # this class normalize the data using sum normalizer.
    def __init__(self):
        self.l1_list = []

    def fit(self, points):
        # this function adds to the l1 list lists with the sum of every cordinate in all the point.
        all_coordinates = [p.coordinates for p in points]
        self.l1_list = []
        for i in range(len(all_coordinates[0])):
            values = [abs((x[i])) for x in all_coordinates]
            divider = sum(values)
            if divider == 0:
                divider = 1
            self.l1_list.append([divider])

    def transform(self, points):
        # this function returns a list of the data after normalizing it with sum normalizer.
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [new_coordinates[i] / self.l1_list[i][0] for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class MinMaxNormalizer:
    # this class normalize the data using min-max normalizer.
    def __init__(self):
        self.min_max_list = []

    def fit(self, points):
        # this function adds to the min_max_list lists with the min of every cordinate for all the point and
        # a parameter named divider which is the max-min.
        all_coordinates = [p.coordinates for p in points]
        self.min_max_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            divider = max(values)-min(values)
            if divider == 0:
                divider = max(values)
            self.min_max_list.append([min(values), divider])

    def transform(self, points):
        # this function returns a list of the data after normalizing it with min-max normalizer.
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.min_max_list[i][0]) / self.min_max_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class ZNormalizer:
    # this class normalize the data using z normalizer.
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        # this function adds to the mean_variance_list lists with the min and varience of every cordinate.
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1)**0.5])

    def transform(self, points):
        # this function returns a list of the data after normalizing it with z normalizer.
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new
