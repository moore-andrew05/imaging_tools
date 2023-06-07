import os

class UserIn:
    def __init__(self):
        self.date = input("Enter the date in the form year/month/day [e.g. 220510]: ").strip()
        self.worm_type = input("What Strain of worm is being imaged today?").strip().upper()
        self.strains = []
        self.strain = input("What strain(s) of bacteria are being imaged today?").strip()
        while(not self.strain.strip().lower().startswith("ex") or self.strain == ""):
              self.strains.append(self.strain)
              self.strain = input("Enter another strain if desired, or type (ex)it if all strains have been entered")
        
        self.temp = int(input("What temperature were these worms grown up at? (please enter as an integer [e.g. 25])").strip())
        self.N = int(input("What N will you be collecting on each group today? (please enter as an integer [e.g. 25])").strip())
        self.name = input("What will your file names be?")

    def make_dirs(self):
        os.mkdir(self.date)
        os.mkdir(f"{self.date}/stitched")

        for s in self.strains:
            os.mkdir(f"{self.date}/{s}") 
            os.mkdir(f"{self.date}/stitched/{s}")
        

    def write_log_template(self):
        with open(f"./{self.date}/README.txt", 'w') as file:
            file.write(f"{self.date}\n")
            file.write("Comments: \n\
*-* \n\
Laser Settings: \n\
\n\
405: \n\
exposure: \n\
intensity: \n\
\n\
561: \n\
exposure: \n\
intensity: \n\n")
            for s in self.strains:
                file.write("*--*\n")
                file.write("date,strain,feed,temp\n")
                file.write(f"{self.date},{self.worm_type},{s},{self.temp}\n\n")
                for i in range(self.N):
                    file.write(f"{self.name + str(i + 1)},stage,rating,comments\n")

                file.write("\n\n\n")


if __name__ == "__main__":
    fields = UserIn()
    fields.make_dirs()
    fields.write_log_template()