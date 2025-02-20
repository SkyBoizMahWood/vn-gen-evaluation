from loguru import logger

from src.config import CRITERIA_DIR_PATH
from src.models.criterion import Criterion


class CriterionRepository:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CriterionRepository, cls).__new__(cls)
            cls._instance._initialize()
            logger.info("CriterionRepository instance created")
        return cls._instance

    def _initialize(self):
        self.criterion_objs: list[Criterion] = []
        criteria_dir_list = [c for c in CRITERIA_DIR_PATH.iterdir() if c.name.endswith(".txt")]
        for criteria_dir in criteria_dir_list:
            with open(criteria_dir, "r") as file:
                criterion_text = file.read()
            criterion_name = criteria_dir.name.split(".")[0]
            criterion_obj = Criterion(criterion_name, criterion_text)
            self.criterion_objs.append(criterion_obj)

    def list_criterion(self) -> list[Criterion]:
        return self.criterion_objs
