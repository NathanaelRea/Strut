from collections import InlineArray


struct ThermalProfile2d(ImplicitlyCopyable, Movable):
    var count: Int
    var locations: InlineArray[Float64, 9]
    var temperatures: InlineArray[Float64, 9]

    fn __init__(out self):
        self.count = 0
        self.locations = InlineArray[Float64, 9](fill=0.0)
        self.temperatures = InlineArray[Float64, 9](fill=0.0)

    fn temperature_at(self, y: Float64) -> Float64:
        if self.count == 0 or y <= self.locations[0]:
            return 0.0
        for i in range(1, self.count):
            if y <= self.locations[i]:
                var fraction = (y - self.locations[i - 1]) / (
                    self.locations[i] - self.locations[i - 1]
                )
                return self.temperatures[i - 1] + fraction * (
                    self.temperatures[i] - self.temperatures[i - 1]
                )
        return 0.0

    fn scale(mut self, factor: Float64):
        for i in range(self.count):
            self.temperatures[i] *= factor
