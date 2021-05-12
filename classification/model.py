from data.labeled import RISKS_MAPPING


class Model:
    def fit(self):
        return self

    def predict(x_coord, y_coord):
        # @TODO implement
        pred = {x: "Dummy" for x in RISKS_MAPPING}
        return pred
