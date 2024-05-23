import numpy as np


class Point:
    def __init__(self, position: float, value: float, delta: float):
        self.position = position
        self.value = value
        self.delta = delta


class Points:
    def __init__(self):
        self.points = []


class Polynom:
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


class Spline:
    def __init__(self, poly: Polynom, min: float, max: float):
        self.poly = poly
        self.min = min
        self.max = max


class Splines:
    def __init__(self):
        self.splines = []


class PolySpline:
    def __init__(self):
        self._points = []
        self._splines = []

    def add_point(self, position: float, value: float, delta: float):
        if len(self._points) > 0 and position <= self._points[-1].position:
            raise Exception(
                "Trying to add a point in a cublic spline before a previous one"
            )
        self._points.append(Point(position, value, delta))

        self.compute_splines()

    def copy(self):
        poly_spline = PolySpline()
        for point in self._points:
            poly_spline.add_point(point.position, point.value, point.delta)
        return poly_spline

    def get(self, x: float):
        return self.interpolation(x, "value")

    def get_vel(self, x: float):
        return self.interpolation(x, "speed")

    def get_mod(self, x: float):
        if x < 0.0:
            x = 1.0 + (x - ((int(x) / 1)))
        elif x > 1.0:
            x = x - ((int(x) / 1))
        return self.get(x)

    def clear(self):
        self._points = []
        self._splines = []

    def compute_splines(self):
        self._splines = []
        if len(self._points) < 2:
            return

        for i in range(1, len(self._points)):
            if (
                np.abs(self._points[i - 1].position - self._points[i].position)
                < 0.00001
            ):
                continue
            poly = self.polynom_fit(
                self._points[i - 1].value,
                self._points[i - 1].delta,
                self._points[i].value,
                self._points[i].delta,
            )
            spline = Spline(
                poly, self._points[i - 1].position, self._points[i].position
            )
            self._splines.append(spline)

    def polynom_fit(self, val1: float, delta1: float, val2: float, delta2: float):
        a = 2.0 * val1 + delta1 + delta2 - 2.0 * val2
        b = 3.0 * val2 - 2.0 * delta1 - 3.0 * val1 - delta2
        c = delta1
        d = val1
        return Polynom(a, b, c, d)

    def interpolation(
        self, x: float, value_type="value"
    ):  # value_type can be "value" or "speed"
        if value_type not in ["value", "speed"]:
            raise Exception("Invalid value_type")

        if len(self._points) == 0:
            return 0.0
        elif len(self._points) == 1:
            if value_type == "value":
                return self._points[0].value
            else:
                return self._points[0].delta
        else:
            if x < self._splines[0].min:
                x = self._splines[0].min
            if x > self._splines[-1].max:
                x = self._splines[-1].max

            for i in range(len(self._splines)):
                if x >= self._splines[i].min and x <= self._splines[i].max:
                    xi = (x - self._splines[i].min) / (
                        self._splines[i].max - self._splines[i].min
                    )
                    if value_type == "value":
                        return self.polynom_value(xi, self._splines[i].poly)
                    elif value_type == "speed":
                        return self.polynom_diff(xi, self._splines[i].poly)
            return 0.0

    def polynom_value(self, t: float, p: Polynom):
        return p.d + t * (t * (p.a * t + p.b) + p.c)

    def polynom_diff(self, t: float, p: Polynom):
        return t * (3 * p.a * t + 2 * p.b) + p.c


if __name__ == "__main__":
    poly_spline = PolySpline()
    poly_spline.add_point(0.0, 0.0, 0.0)
    poly_spline.add_point(1.0, 1.0, 0.0)
    poly_spline.add_point(2.0, 0.0, 0.0)
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 3, 100)
    y = [poly_spline.get(i) for i in x]
    plt.plot(x, y)
    plt.show()
