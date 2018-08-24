class Exam:
    'defines individual exams'
    def __init__(self, matnr, study, lvnumber, name, date, semester, ects, grade):
        self.matnr    = matnr
        self.study    = study
        self.lvnumber = lvnumber
        self.name     = name
        self.date     = date
        self.semester = semester
        self.ects     = float(ects)
        self.grade    = int(grade)

    def __str__(self):
        return '[' + self.name + ']'

    def __hash__(self):
        return (self.matnr, self.study, self.lvnumber, self.name, self.date, \
            self.semester, self.ects, self.grade).__hash__()
