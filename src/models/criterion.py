class Criterion:
    def __init__(self, name: str, criterion: str):
        self.name = name
        self.criterion = criterion

    @staticmethod
    def from_json(json_obj: dict):
        return Criterion(json_obj["name"], json_obj["criterion"])

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "criterion": self.criterion
        }

    def __str__(self):
        return f"Criterion(name={self.name}, criterion={self.criterion})"
