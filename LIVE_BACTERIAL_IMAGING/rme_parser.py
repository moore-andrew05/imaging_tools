import numpy as np

class rme_parser:
    def __init__(self, filepath):
        self.infos = []

        self.PARSE1 = "*--*"
        self.PARSE2 = "*-*"
        self.header = []

        self.parser(filepath=filepath)
        self.infos = np.array(self.infos)

    def parser(self, filepath:str):
        common = []
        data = []

        with open(filepath, "r") as f:
            lines = f.readlines()

            level = 0
            first_line = False

            for line in lines:
                line = line.strip("\n")
                if level == 1:
                    if line.strip().startswith(self.PARSE1):
                        level = 0
                        continue
                    if first_line:
                        self.header = line.split(sep=",")
                        first_line = False
                    else:
                        common = line.split(sep=",")
                    
                if level == 2:
                    if line.strip().startswith(self.PARSE2):
                        self.infos.append(data)
                        data = []
                        level = 0
                        continue
                    if first_line:
                        self.header = [*self.header, *line.split(sep=",")]
                        for i, w in enumerate(self.header):
                            if w == "feed":
                                self.header[i] = "diet"
                        first_line = False
                    else:
                        data.append([*common, *line.split(sep=",")])
                    
                if line.strip().startswith(self.PARSE1):
                    level = 1
                    first_line = True
                if line.strip().startswith(self.PARSE2):
                    level = 2
                    first_line = True
        
