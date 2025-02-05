class StageOutputs:
    def __init__(self):
        self._outputs = {}

    def store(self, stage_name: str, result):
        self._outputs[stage_name] = result

    def get(self, stage_name: str, default=None):
        return self._outputs.get(stage_name, default)

    def has(self, stage_name: str):
        return stage_name in self._outputs

    def all_outputs(self):
        return self._outputs.copy()